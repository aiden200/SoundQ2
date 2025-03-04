
import os
import cv2
import torch
import numpy as np
import supervision as sv
import shutil
import csv
import copy

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from grounded_sam_2.utils.track_utils import sample_points_from_masks
from grounded_sam_2.utils.video_utils import create_video_from_images
from grounded_sam_2.utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from synth_data_gen.utils import *
from synth_data_gen.scene_transition_detector import SceneTransitionDetector


class SegmentObjectsWithBoundaries:
    def __init__(self, config, device, log=None, verbose=False, histogram_threshold=0.05, parallel=False):

        generate_paths(config)
        self.model_id=config["gd_sam2"]["MODEL_ID"]
        self.video_dir=config["dataset_paths"]["video_dir"]
        self.classes, self.classes_dino_prompted=get_classes(config)
        self.prompt_type=config["gd_sam2"]["PROMPT_TYPE_FOR_VIDEO"]
        self.working_dir=config["paths"]["working_dir"]
        self.working_dir_gd_sam=config["paths"]["working_dir_gd_sam"]
        self.working_dir_gd_sam_img=config["paths"]["working_dir_gd_sam_img"]
        self.working_dir_gd_sam_masks_only=config["paths"]["working_dir_gd_sam_masks_only"]
        self.results_path=config["paths"]["results_path"]
        self.segmented_class_result_path=config["paths"]["segmented_class_results"]
        self.sam2_checkpoint=config["gd_sam2"]["sam2_checkpoint"]
        self.model_cfg=config["gd_sam2"]["model_cfg"]
        self.result_file=config["gd_sam2"]["gd_sam2_generate_results"]
        self.scene_transition_detector=SceneTransitionDetector(histogram_diff_threshold=histogram_threshold)
        self.verbose=verbose
        self.device=device

        # use bfloat16
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


        self.video_predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint)
        sam2_image_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)

        # build grounding dino from huggingface
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(device)
        if parallel:
            # self.grounding_model = torch.nn.DataParallel(self.grounding_model)
            self.grounding_device = "cuda:1"
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.grounding_device)

        self.scene_transition_detector = SceneTransitionDetector(histogram_diff_threshold=histogram_threshold)


    def process_frame_with_dino(self, frame: Image):

        categories = self.classes_dino_prompted
        # VERY important: text queries need to be lowercased + end with a dot
        categories = categories.lower()

        if self.verbose:
            print(categories)
        # categories = "smoke. train. railroad. car. building. fire. boxes."

        # Draw bounding boxes over detected objects
        inputs = self.processor(images=frame, text=categories, return_tensors="pt").to(self.device)
        # inputs = self.processor(images=frame, text=categories, return_tensors="pt").to(self.grounding_device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[frame.size[::-1]]
        )

        # input_boxes = results[0]["boxes"].cpu().numpy()
        input_boxes = results[0]["boxes"].cpu().numpy()
        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]


        self.image_predictor.set_image(np.array(frame.convert("RGB")))


        return [input_boxes, confidences, class_names]


    def predict_with_sam2(self, prompts, inference_state, ann_frame_idx):

        input_boxes, confidences, class_names = prompts
        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = self.sam2.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        assert self.prompt_type == "box", "SAM 2 video predictor only support box prompt"

        for object_id, (label, box) in enumerate(zip(class_names, input_boxes), start=1):
            _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )

        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        return video_segments


    def visualize_and_save(self, frame_names, video_segments, objects, output_path, masked_output_path, video_name):

        # Visualize the segment results across the video and save them
        if os.path.exists(self.working_dir_gd_sam):
            shutil.rmtree(self.working_dir_gd_sam)

        if not os.path.exists(self.working_dir_gd_sam):
            os.makedirs(self.working_dir_gd_sam)

        if os.path.exists(self.working_dir_gd_sam_masks_only):
            shutil.rmtree(self.working_dir_gd_sam_masks_only)

        if not os.path.exists(self.working_dir_gd_sam_masks_only):
            os.makedirs(self.working_dir_gd_sam_masks_only)

        ID_TO_OBJECTS = {i: obj for i, obj in enumerate(objects, start=1)}

        segmented_objects_dirs = {}

        for frame_idx, segments in video_segments.items():
            img = cv2.imread(os.path.join(self.working_dir_gd_sam_img, frame_names[frame_idx]))
            
            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)
            
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                mask=masks, # (n, h, w)
                class_id=np.array(object_ids, dtype=np.int32),
            )
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
            
            # mask_annotator = sv.MaskAnnotator()
            # annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(os.path.join(self.working_dir_gd_sam, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

            for obj_id, mask in zip(object_ids, masks):
                segmented_img = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
                segmented_object_dir = os.path.join(self.working_dir_gd_sam_masks_only, str(obj_id))
                if segmented_object_dir not in segmented_objects_dirs:
                    os.mkdir(segmented_object_dir)
                    segmented_objects_dirs[segmented_object_dir] = str(obj_id)
                cv2.imwrite(os.path.join(segmented_object_dir, f"annotated_frame_{frame_idx:05d}.jpg"), segmented_img)


        create_video_from_images(self.working_dir_gd_sam, output_path)
        
        for segmented_path in segmented_objects_dirs:
            save_path = os.path.join(masked_output_path, segmented_objects_dirs[segmented_path])
            if not os.path.exsits(save_path):
                os.mkdir(save_path)
            
            create_video_from_images(self.working_dir_gd_sam_masks_only, os.path.join(save_path, f"{video_name}.mp4"))
    

    def process_videos(self, video_dir):
        for filename in os.listdir(video_dir):
            if filename.endswith('.mp4'):
                video_path = os.path.join(video_dir, filename)
                video_name = filename[:-4]
                output_path = os.path.join(self.result_file, video_name)
                output_video = os.path.join(output_path, filename)
                self.process_video_with_scene_boundaries(video_path, output_video, video_name, output_path)
    
    def test_video(self, video_path):
   
        video_name = video_path[:-4]
        output_path = os.path.join(self.result_file, video_name)
        output_video = os.path.join(output_path, video_path)
        self.process_video_with_scene_boundaries(video_path, output_video, video_name, output_path)

    def process_video_with_scene_boundaries(self, video_path, output_video, video_name, output_path):
        """
        Process the video using scene boundaries. For each scene segment,
        generate annotated frames and, for each detected class (object), create a
        masked video that only shows that object. Also, generate an overall annotated video.
        """
        
        for d in [self.working_dir_gd_sam, self.working_dir_gd_sam_img, self.working_dir_gd_sam_masks_only]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

        mask_data_dir = os.path.join(output_path, "mask_data")
        json_data_dir = os.path.join(output_path, "json_data")
        CommonUtils.creat_dirs(output_path)
        CommonUtils.creat_dirs(mask_data_dir)
        CommonUtils.creat_dirs(json_data_dir)


        scene_boundaries, _ = self.scene_transition_detector.detect_scene_changes(video_path)
        scene_boundaries = sorted(scene_boundaries)
        
        video_info = sv.VideoInfo.from_video_path(video_path)
        frame_generator = sv.get_video_frames_generator(video_path, stride=1, start=0, end=None)

        source_frames = Path(self.working_dir_gd_sam_img)

        with sv.ImageSink(
            target_dir_path=source_frames, 
            overwrite=True, 
            image_name_pattern="{:05d}.jpg"
        ) as sink:
            for frame in tqdm(frame_generator, desc="Saving Video Frames"):
                sink.save_image(frame)

        frame_names = [
            p for p in os.listdir(self.working_dir_gd_sam_img)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


        inference_state = self.video_predictor.init_state(video_path=self.working_dir_gd_sam_img)
        sam2_masks = MaskDictionaryModel()
        objects_count = 0

        scene_boundaries.append(len(frame_names))
        if scene_boundaries[0] != 0:
            scene_boundaries = [0] + scene_boundaries

        annotated_frames_dir = os.path.join(self.working_dir_gd_sam, "annotated_frames")
        os.makedirs(annotated_frames_dir, exist_ok=True)

        for scene_idx in range(len(scene_boundaries) - 1):
            start_frame = scene_boundaries[scene_idx]
            end_frame = scene_boundaries[scene_idx + 1]

            # Process the starting frame for detection
            img_path = os.path.join(self.working_dir_gd_sam_img, frame_names[start_frame])
            image = Image.open(img_path)
            image_base_name = frame_names[start_frame].split(".")[0]
            mask_dict = MaskDictionaryModel(promote_type=self.prompt_type, mask_name=f"mask_{image_base_name}.npy")

            # Run DINO to get initial detections
            results = self.process_frame_with_dino(image)
            input_boxes, confidences, labels = results
            self.image_predictor.set_image(np.array(image.convert("RGB")))

            if input_boxes.shape[0] != 0:
                masks, scores, logits = self.image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
                # Ensure mask dimensions are correct
                if masks.ndim == 2:
                    masks = masks[None]
                    scores = scores[None]
                    logits = logits[None]
                elif masks.ndim == 4:
                    masks = masks.squeeze(1)

                mask_dict.add_new_frame_annotation(
                    mask_list=torch.tensor(masks),
                    box_list=torch.tensor(input_boxes),
                    label_list=labels
                )
                objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
            else:
                # If no detections, fallback to previous masks
                mask_dict = sam2_masks

            # If still no objects, save empty mask info and continue to next scene
            if len(mask_dict.labels) == 0:
                mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir,
                                                    image_name_list=frame_names[start_frame:end_frame])
                continue
            else:
                self.video_predictor.reset_state(inference_state)
                for object_id, object_info in mask_dict.labels.items():
                    _, _, _ = self.video_predictor.add_new_mask(
                        inference_state,
                        start_frame,
                        object_id,
                        object_info.mask,
                    )

                # Propagate masks over frames in this scene segment
                video_segments = {}  # {frame_index: MaskDictionaryModel}
                for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
                    inference_state, max_frame_num_to_track=(end_frame - start_frame), start_frame_idx=start_frame):
                    
                    frame_masks = MaskDictionaryModel()
                    for i, out_obj_id in enumerate(out_obj_ids):
                        out_mask = (out_mask_logits[i] > 0.0).cpu() #.numpy()  # binary mask
                        # Create an ObjectInfo instance; note that out_mask[0] is used (adjust if needed)
                        object_info = ObjectInfo(
                            instance_id=out_obj_id, 
                            mask=out_mask[0], 
                            class_name=mask_dict.get_target_class_name(out_obj_id)
                        )
                        object_info.update_box()
                        frame_masks.labels[out_obj_id] = object_info
                        image_base_name = frame_names[out_frame_idx].split(".")[0]
                        frame_masks.mask_name = f"mask_{image_base_name}.npy"
                        frame_masks.mask_height = out_mask.shape[-2]
                        frame_masks.mask_width = out_mask.shape[-1]
                    video_segments[out_frame_idx] = frame_masks
                    sam2_masks = copy.deepcopy(frame_masks)

            # Save mask data and metadata for this scene segment
            for frame_idx, frame_masks_info in video_segments.items():
                mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
                for obj_id, obj_info in frame_masks_info.labels.items():
                    mask_img[obj_info.mask == True] = obj_id
                mask_img = mask_img.numpy().astype(np.uint16)
                np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

                json_data = frame_masks_info.to_dict()
                json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
                with open(json_data_path, "w") as f:
                    json.dump(json_data, f)

            # -------------------------------
            # Visualize frames and save per-class masked images for this scene segment
            # -------------------------------
            # Dictionary to track directories per (scene, class)
            scene_masked_dirs = {}
            for frame_idx, frame_masks_info in video_segments.items():
                # Read the original frame image
                img_path = os.path.join(self.working_dir_gd_sam_img, frame_names[frame_idx])
                img = cv2.imread(img_path)

                # Build lists of detected object IDs and corresponding masks from this frame
                object_ids = list(frame_masks_info.labels.keys())
                masks = [frame_masks_info.labels[obj_id].mask for obj_id in object_ids]

                # (Optional) Annotate the frame with bounding boxes and class labels
                # For annotation we need the masks as numpy arrays
                masks_np = []
                for mask in masks:
                    if hasattr(mask, "numpy"):
                        masks_np.append(mask.numpy().astype(np.uint8))
                    else:
                        masks_np.append(mask.astype(np.uint8))
                # Concatenate masks so that we can compute bounding boxes, etc.
                if len(masks_np) > 0:
                    masks_concat = np.concatenate(masks_np, axis=0)
                else:
                    masks_concat = np.array([])

                detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks_concat),
                    mask=masks_concat,
                    class_id=np.array(object_ids, dtype=np.int32),
                )
                box_annotator = sv.BoxAnnotator()
                annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
                label_annotator = sv.LabelAnnotator()
                labels = [frame_masks_info.labels[obj_id].class_name for obj_id in object_ids]
                annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
                # Save annotated frame (for overall video)
                cv2.imwrite(os.path.join(annotated_frames_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

                # For each detected object, create a masked image (only that object)
                for obj_id, mask in zip(object_ids, masks_np):
                    class_name = frame_masks_info.labels[obj_id].class_name
                    # Create a directory for this scene and object (using class name and object id)
                    scene_class_dir = os.path.join(self.working_dir_gd_sam_masks_only, f"scene_{scene_idx}", f"{class_name}_{obj_id}")
                    if scene_class_dir not in scene_masked_dirs:
                        os.makedirs(scene_class_dir, exist_ok=True)
                        scene_masked_dirs[scene_class_dir] = class_name
                    segmented_img = cv2.bitwise_and(img, img, mask=mask)
                    cv2.imwrite(os.path.join(scene_class_dir, f"annotated_frame_{frame_idx:05d}.jpg"), segmented_img)

            # -------------------------------
            # Create a video for each class (object) in this scene segment
            # -------------------------------
            for scene_class_dir in scene_masked_dirs:
                # Build an output path for this masked video.
                # (For example: masked_output_path/<scene_class_dir>_<video_name>_scene<scene_idx>.mp4)
                class_name = scene_masked_dirs[scene_class_dir]
                class_path = os.path.join(self.masked_output_path, class_name)
                if not os.path.exists(class_path):
                    os.mkdir(class_path)
                output_class_video = os.path.join(
                    class_path, f"_{video_name}_scene{scene_idx}.mp4"
                )
                create_video_from_images(scene_class_dir, output_class_video)

        # -------------------------------
        # Create overall annotated video from all annotated frames
        # -------------------------------
        create_video_from_images(annotated_frames_dir, output_video)


def write_to_check_file(status_document, status, new_element):
    with open(status_document, "a") as file:
        file.write(f"{new_element},{str(status)}\n")


