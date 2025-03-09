
import os
import cv2
import torch
import numpy as np
import supervision as sv
import shutil
import csv
import copy
import pandas as pd
import subprocess


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
        self.config = config
        self.model_id=config["gd_sam2"]["MODEL_ID"]
        # self.video_dir=config["dataset_paths"]["video_dir"]
        self.classes, self.classes_dino_prompted=get_classes(config)
        self.prompt_type=config["gd_sam2"]["PROMPT_TYPE_FOR_VIDEO"]
        self.working_dir=config["paths"]["working_dir"]
        self.working_dir_gd_sam=config["paths"]["working_dir_gd_sam"]
        self.working_dir_gd_sam_img=config["paths"]["working_dir_gd_sam_img"]
        self.working_dir_gd_sam_masks_only=config["paths"]["working_dir_gd_sam_masks_only"]
        self.results_path=config["paths"]["results_path"]
        self.max_vid_seg=config["max_vid_seg"]
        self.video_results_path=config["paths"]["video_results_path"]
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
            self.grounding_model = torch.nn.DataParallel(self.grounding_model)
            # self.grounding_device = "cuda:1"
            # self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.grounding_device)

        self.scene_transition_detector = SceneTransitionDetector(histogram_diff_threshold=histogram_threshold)


    def process_frame_with_dino(self, frame: Image, category: str):

        # categories = self.classes_dino_prompted
        # VERY important: text queries need to be lowercased + end with a dot
        # categories = categories.lower()

        if self.verbose:
            print(category)
        # categories = "smoke. train. railroad. car. building. fire. boxes."

        # Draw bounding boxes over detected objects
        inputs = self.processor(images=frame, text=category, return_tensors="pt").to(self.device)
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

    def process_videos(self, video_dir):
        for filename in os.listdir(video_dir):
            if filename.endswith('.mp4'):
                video_path = os.path.join(video_dir, filename)
                video_name = filename[:-4]
                output_path = os.path.join(self.video_results_path, video_name)
                output_video = os.path.join(output_path, filename)
                self.process_video_with_scene_boundaries(video_path, output_video, video_name, output_path)
    
    def test_video(self, video_path):
        
        video_name = os.path.basename(video_path)[:-4]
        self.process_video_with_scene_boundaries(video_path, video_name)
    

    def run_sam_dino_on_video(self, video_path, video_name, output_path, class_num, class_name, scene_boundaries):

        category = self.classes[str(class_num)]
        # For DINO prompting purpose
        category_formatted = category.replace("_", " ") + "."
    
        class_dir = os.path.join(self.segmented_class_result_path, f"{str(class_num)}_{class_name}")
        class_video_dir = os.path.join(class_dir, video_name)
        output_video = os.path.join(class_dir, f"{video_name}.mp4")
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        if not os.path.exists(class_video_dir):
            os.mkdir(class_video_dir)
    
    
        for d in [self.working_dir_gd_sam, self.working_dir_gd_sam_img, self.working_dir_gd_sam_masks_only]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

        mask_data_dir = os.path.join(class_video_dir, "mask_data")
        json_data_dir = os.path.join(class_video_dir, "json_data")
        CommonUtils.creat_dirs(output_path)
        CommonUtils.creat_dirs(mask_data_dir)
        CommonUtils.creat_dirs(json_data_dir)

        
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

        if scene_boundaries:
            scene_boundaries, _ = self.scene_transition_detector.detect_scene_changes(video_path)
            scene_boundaries = sorted(scene_boundaries)
            scene_boundaries.append(len(frame_names))
            if scene_boundaries[0] != 0:
                scene_boundaries = [0] + scene_boundaries
        else:
            # Process entire video
            scene_boundaries = [0, len(frame_names)]


        inference_state = self.video_predictor.init_state(video_path=self.working_dir_gd_sam_img)
        sam2_masks = MaskDictionaryModel()
        objects_count = 0

        annotated_frames_dir = os.path.join(self.working_dir_gd_sam, "annotated_frames")
        os.makedirs(annotated_frames_dir, exist_ok=True)
    
        
        video_segments = None
        for scene_idx in range(len(scene_boundaries) - 1):
            start_frame = scene_boundaries[scene_idx]
            end_frame = scene_boundaries[scene_idx + 1]

            # Process the starting frame for detection
            img_path = os.path.join(self.working_dir_gd_sam_img, frame_names[start_frame])
            image = Image.open(img_path)
            image_base_name = frame_names[start_frame].split(".")[0]
            mask_dict = MaskDictionaryModel(promote_type="mask", mask_name=f"mask_{image_base_name}.npy")

            # Run DINO to get initial detections
            results = self.process_frame_with_dino(image, category_formatted)
            input_boxes, confidences, labels = results
            # print(labels)
            # exit(0)
            # self.image_predictor.set_image(np.array(image.convert("RGB")))


            if input_boxes.shape[0] != 0:
                masks, scores, logits = self.image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
                # convert the mask shape to (n, H, W)
                if masks.ndim == 2:
                    masks = masks[None]
                    scores = scores[None]
                    logits = logits[None]
                elif masks.ndim == 4:
                    masks = masks.squeeze(1)

                # mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(self.device), box_list=torch.tensor(input_boxes), label_list=labels)
                mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks), box_list=torch.tensor(input_boxes), label_list=labels)

                objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)

            else:
                # No objects detected in the frame, skip merge
                mask_dict = sam2_masks
            
            if len(mask_dict.labels) == 0:
                mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list = frame_names[start_frame:end_frame])
                # print("No object detected in the frame, skip the frame {}".format(start_frame))
                continue
            else: 
                self.video_predictor.reset_state(inference_state)

                for object_id, object_info in mask_dict.labels.items():
                    frame_idx, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                            inference_state,
                            start_frame,
                            object_id,
                            object_info.mask,
                        )
                
                video_segments = {}  # output the following {step} frames tracking masks
                for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=(end_frame-start_frame), start_frame_idx=start_frame):
                    frame_masks = MaskDictionaryModel()
                    
                    for i, out_obj_id in enumerate(out_obj_ids):
                        out_mask = (out_mask_logits[i] > 0.0).cpu() #.numpy()
                        object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id))
                        object_info.update_box()
                        frame_masks.labels[out_obj_id] = object_info
                        image_base_name = frame_names[out_frame_idx].split(".")[0]
                        frame_masks.mask_name = f"mask_{image_base_name}.npy"
                        frame_masks.mask_height = out_mask.shape[-2]
                        frame_masks.mask_width = out_mask.shape[-1]

                    video_segments[out_frame_idx] = frame_masks
                    sam2_masks = copy.deepcopy(frame_masks)
            
            for frame_idx, frame_masks_info in video_segments.items():
                mask = frame_masks_info.labels
                mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
                for obj_id, obj_info in mask.items():
                    mask_img[obj_info.mask == True] = obj_id

                mask_img = mask_img.numpy().astype(np.uint16)
                np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

                json_data = frame_masks_info.to_dict()
                json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
                with open(json_data_path, "w") as f:
                    json.dump(json_data, f)

            
            CommonUtils.draw_masks_only(self.working_dir_gd_sam_img, mask_data_dir, json_data_dir, self.working_dir_gd_sam)

        if not os.listdir(self.working_dir_gd_sam):
            #no detection
            shutil.rmtree(mask_data_dir)
            shutil.rmtree(json_data_dir)
        else:
            create_video_from_images(self.working_dir_gd_sam, output_video)
            
        
           
        



    def process_video_with_scene_boundaries(self, video_path, video_name, scene_boundaries=False):
        """
        Process the video using scene boundaries. For each scene segment,
        generate annotated frames and, for each detected class (object), create a
        masked video that only shows that object. Also, generate an overall annotated video.
        """
        
        segment_paths = [(video_path, video_name)]

        video_info = sv.VideoInfo.from_video_path(video_path)
        video_duration = video_info.total_frames / video_info.fps

        # Check if video duration is greater than 15 seconds
        if video_duration > self.max_vid_seg:
            # Calculate the number of segments
            num_segments = int(video_duration // self.max_vid_seg)
            segment_paths = []

            # Split the video into self.max_vid_seg-second segments
            for i in range(num_segments + 1):
                start_time = i * self.max_vid_seg
                end_time = min((i + 1) * self.max_vid_seg, video_duration)
                segment_name = f"{video_name}_segment_{i}"
                segment_dir = os.path.join(self.video_results_path, segment_name)
                if not os.path.exists(segment_dir):
                    os.mkdir(segment_dir)
                segment_output_path = os.path.join(segment_dir, f"{video_name}_segment_{i}.mp4")
                
                if not os.path.exists(segment_output_path):
                    # Use ffmpeg or similar tool to split the video
                    os.system(f'ffmpeg -i "{video_path}" -ss {start_time} -to {end_time} -c copy "{segment_output_path}"')
                if self.verbose:
                    print(f"creating new segment video: {segment_name} from {start_time}:{end_time}")
                segment_paths.append((segment_output_path, segment_name))

        for video_path, video_name in segment_paths:
            output_path = os.path.join(self.video_results_path, video_name)
            
            done = []
            
            if os.path.exists(self.result_file):
                with open(self.result_file, mode='r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        # video, number, status
                        v, n = row
                        done.append((v, n))
                        
            
            for class_num in self.classes:
                if (video_name, self.classes[class_num]) in done:
                    continue
                else:
                    done.append((video_name, self.classes[class_num]))
                
                
                
                self.run_sam_dino_on_video(video_path, video_name, output_path, class_num, self.classes[class_num], scene_boundaries)
                
                with open(self.result_file, mode='w', encoding='utf-8', newline='') as file:
                    writer = csv.writer(file)
                    for row in done:
                        writer.writerow(row)
    


    def process_annotated_videos(self, download=False, scene_boundaries=False):
        """
        Process the video using scene boundaries. For each scene segment,
        generate annotated frames and, for each detected class (object), create a
        masked video that only shows that object. Also, generate an overall annotated video.
        """

        fragment_yt_dir = self.config["paths"]["fragment_yt_path"]
        yt_csv_vid = self.config["filepaths"]["audio_updated_csv_file"]
        if download:
            # Download the respective videos
            cookie_path = self.config["filepaths"]["cookie_filepath"]

            download_vids(fragment_yt_dir, yt_csv_vid, cookie_path)
        
        df = pd.read_csv(yt_csv_vid)

        fragments = []
        for index, row in df.iterrows():
            # Access data in each row using the column names
            video_link = row['link']
            start_time = row['start']
            end_time = row['end']
            video_class = row['class']

            video_name = generate_name_from_yt_vids(video_link)
            
            def time_to_seconds(time_str):
                minutes, seconds = map(int, time_str.split(':'))
                return minutes * 60 + seconds

            start_seconds = time_to_seconds(start_time)
            end_seconds = time_to_seconds(end_time)
            video_location = os.path.join(fragment_yt_dir, f"{video_name}.mp4")
            fragments.append((video_location, start_seconds, end_seconds, video_class, video_name))

        
        for v_path, s, e, v_class, v_name in fragments:
            if not os.path.exists(v_path):
                continue

            segment_paths = [(v_path, s, e, v_class, v_name)]
            
            

            video_info = sv.VideoInfo.from_video_path(v_path)
            
            video_duration = e - s

            # Check if video duration is greater than 15 seconds
            if video_duration > self.max_vid_seg:
                # Calculate the number of segments
                num_segments = int(video_duration // self.max_vid_seg)
                segment_paths = []

                # Split the video into self.max_vid_seg-second segments
                start_time = s
                for i in range(num_segments + 1):
                    start_time = s + i * self.max_vid_seg
                    end_time = min(s + (i + 1) * self.max_vid_seg, video_duration)
                    segment_name = f"{v_name}_segment_{i}"
                    segment_dir = os.path.join(self.video_results_path, segment_name)
                    if not os.path.exists(segment_dir):
                        os.mkdir(segment_dir)
                    segment_output_path = os.path.join(segment_dir, f"{v_name}_segment_{i}.mp4")
                    
                    if not os.path.exists(segment_output_path):
                        # Use ffmpeg or similar tool to split the video
                        # os.system(f'ffmpeg -i "{v_path}" -ss {start_time} -to {end_time} -c copy "{segment_output_path}"')
                        subprocess.run(
                            ['ffmpeg', '-i', v_path, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', segment_output_path],
                            check=True
                        )
                    if self.verbose:
                        print(f"creating new segment video: {segment_name} from {start_time}:{end_time}")
                    segment_paths.append((segment_output_path, start_time, end_time, v_class, segment_name))
            
            
            for video_location, start_seconds, end_seconds, video_class, video_name in segment_paths:
                output_path = os.path.join(self.video_results_path, video_name)
                
                done = []
                
                if os.path.exists(self.result_file):
                    with open(self.result_file, mode='r', encoding='utf-8') as file:
                        reader = csv.reader(file)
                        for row in reader:
                            # video, number, status
                            v, n = row
                            done.append((v, n))
                
                            
                if (video_name, self.classes[str(video_class)]) in done:
                    continue
                else:
                    done.append((video_name, self.classes[str(video_class)]))
                
                
                
                self.run_sam_dino_on_video(video_location, video_name, output_path, video_class, self.classes[str(video_class)], scene_boundaries)
                
                with open(self.result_file, mode='w', encoding='utf-8', newline='') as file:
                    writer = csv.writer(file)
                    for row in done:
                        writer.writerow(row)





