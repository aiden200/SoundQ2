import math
from typing import Tuple
import os
import json
import cv2
import numpy as np
import yt_dlp
import supervision as sv
import random
import pandas as pd


# Reference METU outter trayectory:  bottom outter trayectory
REF_OUT_TRAJ = ["034", "024", "014", "004", "104", "204",
			"304", "404", "504", "604", "614", "624",
			"634", "644", "654", "664", "564", "464",
			"364", "264", "164", "064", "054", "044"]
# Reference METU inner trayectory:  bottom inner trayectory
REF_IN_TRAJ = ["134", "124", "114", "214","314", "414", "514", "524",
				"534", "544", "554", "454", "354", "254", "154", "145"]

def get_mic_xyz():
	"""
	Get em32 microphone coordinates in 3D space
	"""
	return [(3 - 3) * 0.5, (3 - 3) * 0.5, (2 - 2) * 0.3 + 1.5]

def az_ele_from_source_radians(ref_point, src_point):
	"""
	Calculates the azimuth and elevation between a reference point and a source point in 3D space
	Args:
		ref_point (list): A list of three floats representing the x, y, and z coordinates of the reference point
		src_point (list): A list of three floats representing the x, y, and z coordinates of the other point
	Returns:
		A tuple of two floats representing the azimuth and elevation angles in radians plus distance between reference and source point
	"""
	dx = src_point[0] - ref_point[0]
	dy = src_point[1] - ref_point[1]
	dz = src_point[2] - ref_point[2]
	azimuth = math.atan2(dy, dx)
	distance = math.sqrt(dx**2 + dy**2 + dz**2)
	elevation = math.asin(dz/distance)
	return azimuth, elevation, distance

def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def az_ele_from_source(ref_point, src_point):
	"""
	Calculates the azimuth and elevation between a reference point and a source point in 3D space.

	Args:
		ref_point (list): A list of three floats representing the x, y, and z coordinates of the reference point.
		src_point (list): A list of three floats representing the x, y, and z coordinates of the other point.

	Returns:
		A tuple of two floats representing the azimuth and elevation angles in degrees plus the distance between the reference and source points.
	"""
	dx = src_point[0] - ref_point[0]
	dy = src_point[1] - ref_point[1]
	dz = src_point[2] - ref_point[2]

	azimuth = math.degrees(math.atan2(dy, dx))
	distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
	elevation = math.degrees(math.asin(dz / distance))

	return azimuth, elevation, distance


def get_classes(config) -> Tuple[dict, str]:
	classes_file = config["dataset_paths"]["class_dir"]
	classes = {}
	formatted_classes =""
	with open(classes_file, 'r') as f:
		for line in f:
			number, class_name = line.strip().split('.')
			classes[number] = class_name.replace(" ", "_")
			formatted_classes = formatted_classes + class_name + ". "
	
	
	return classes, formatted_classes

def dwl_vid(video_url, save_path, filename, cookie_filepath) -> None:
    
    if os.path.exists(f"{save_path}/{filename}.mp4"):
        print("Skipping, video exists")
        return
    
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # Combine the best video and audio streams
        'outtmpl': f'{save_path}/{filename}.%(ext)s',       # Custom filename template
        'cookiefile': cookie_filepath, # "data/yt_cookies.txt",
        "cachedir": False,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',    # Convert video format
            'preferedformat': 'mp4',          # Convert to MP4
        }],
        'quiet': True,             # Suppresses all output messages
        'no_warnings': True,  
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


def generate_name_from_yt_vids(yt_link:str) -> str:
    if "/shorts/" in yt_link:
        video_name = yt_link.split("/shorts/")[1]
    else:
        if "si=" in yt_link:
            video_name = yt_link.split("si=")[1]
        elif "v=" in yt_link:
            video_name = yt_link.split("v=")[1]
        else:
             print(yt_link)

    video_name = video_name.split("&")[0]
    
    return video_name


def download_vids(save_dir, dataset_path, cookie_filepath):
    df = pd.read_csv(dataset_path)
    success_vids = 0
    error_vids = 0
    
    for index, row in df.iterrows():
        # Access data in each row using the column names
        video_id = row['link']
        start_time = row['start']
        end_time = row['end']
        video_class = row['class']


        video_name = generate_name_from_yt_vids(video_id)
        
        # print(f"Video ID: {video_id}, Start: {start_time}, End: {end_time}, Class: {video_class}")

        # You can perform additional processing here
        # For example, convert start and end times to seconds
        def time_to_seconds(time_str):
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds

        start_seconds = time_to_seconds(start_time)
        end_seconds = time_to_seconds(end_time)
        try:
            dwl_vid(video_id, save_dir, video_name, cookie_filepath)
            success_vids += 1
        except Exception as e:
             print(e)
             error_vids += 1
    
    print(f"Success vids: {success_vids}, error vids: {error_vids}")



def generate_paths(config) -> None:
	for path in config["paths"]:
		if not os.path.exists(config["paths"][path]):
			os.mkdir(config["paths"][path])



class CommonUtils:
    @staticmethod
    def creat_dirs(path):
        """
        Ensure the given path exists. If it does not exist, create it using os.makedirs.

        :param path: The directory path to check or create.
        """
        try: 
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"Path '{path}' did not exist and has been created.")
            else:
                print(f"Path '{path}' already exists.")
        except Exception as e:
            print(f"An error occurred while creating the path: {e}")
    

    

    @staticmethod
    def draw_masks_only(raw_image_path, mask_path, json_path, output_path):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = sorted(os.listdir(raw_image_path))
        
        for raw_image_name in raw_image_name_list:
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Load mask
            mask_npy_path = os.path.join(mask_path, "mask_" + raw_image_name.split(".")[0] + ".npy")
            mask = np.load(mask_npy_path)

            # Create a blank black image
            black_background = np.zeros_like(image)

            # Get unique mask IDs (skip background)
            unique_ids = np.unique(mask)

            # Apply masks to extract objects and place them on a black background
            for uid in unique_ids:
                if uid == 0:
                    continue  # Skip background

                object_mask = (mask == uid)
                
                # Use the mask to extract object pixels from the original image
                black_background[object_mask] = image[object_mask]

            # Save the result
            output_image_path = os.path.join(output_path, raw_image_name)
            cv2.imwrite(output_image_path, black_background)
            


    @staticmethod
    def draw_masks_and_box_with_supervision(raw_image_path, mask_path, json_path, output_path):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = os.listdir(raw_image_path)
        raw_image_name_list.sort()
        for raw_image_name in raw_image_name_list:
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")
            # load mask
            mask_npy_path = os.path.join(mask_path, "mask_"+raw_image_name.split(".")[0]+".npy")
            mask = np.load(mask_npy_path)
            # color map
            unique_ids = np.unique(mask)
            
            # get each mask from unique mask file
            all_object_masks = []
            for uid in unique_ids:
                if uid == 0: # skip background id
                    continue
                else:
                    object_mask = (mask == uid)
                    all_object_masks.append(object_mask[None])
            
            if len(all_object_masks) == 0:
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, image)
                continue
            # get n masks: (n, h, w)
            all_object_masks = np.concatenate(all_object_masks, axis=0)
            
            # load box information
            file_path = os.path.join(json_path, "mask_"+raw_image_name.split(".")[0]+".json")
            
            all_object_boxes = []
            all_object_ids = []
            all_class_names = []
            object_id_to_name = {}
            with open(file_path, "r") as file:
                json_data = json.load(file)
                for obj_id, obj_item in json_data["labels"].items():
                    # box id
                    instance_id = obj_item["instance_id"]
                    if instance_id not in unique_ids: # not a valid box
                        continue
                    # box coordinates
                    x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                    all_object_boxes.append([x1, y1, x2, y2])
                    # box name
                    class_name = obj_item["class_name"]
                    
                    # build id list and id2name mapping
                    all_object_ids.append(instance_id)
                    all_class_names.append(class_name)
                    object_id_to_name[instance_id] = class_name
            
            # Adjust object id and boxes to ascending order
            paired_id_and_box = zip(all_object_ids, all_object_boxes)
            sorted_pair = sorted(paired_id_and_box, key=lambda pair: pair[0])
            
            # Because we get the mask data as ascending order, so we also need to ascend box and ids
            all_object_ids = [pair[0] for pair in sorted_pair]
            all_object_boxes = [pair[1] for pair in sorted_pair]
            
            detections = sv.Detections(
                xyxy=np.array(all_object_boxes),
                mask=all_object_masks,
                class_id=np.array(all_object_ids, dtype=np.int32),
            )
            
            # custom label to show both id and class name
            labels = [
                f"{instance_id}: {class_name}" for instance_id, class_name in zip(all_object_ids, all_class_names)
            ]
            
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            
            output_image_path = os.path.join(output_path, raw_image_name)
            cv2.imwrite(output_image_path, annotated_frame)
            print(f"Annotated image saved as {output_image_path}")

    @staticmethod
    def draw_masks_and_box(raw_image_path, mask_path, json_path, output_path):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = os.listdir(raw_image_path)
        raw_image_name_list.sort()
        for raw_image_name in raw_image_name_list:
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")
            # load mask
            mask_npy_path = os.path.join(mask_path, "mask_"+raw_image_name.split(".")[0]+".npy")
            mask = np.load(mask_npy_path)
            # color map
            unique_ids = np.unique(mask)
            colors = {uid: CommonUtils.random_color() for uid in unique_ids}
            colors[0] = (0, 0, 0)  # background color

            # apply mask to image in RBG channels
            colored_mask = np.zeros_like(image)
            for uid in unique_ids:
                colored_mask[mask == uid] = colors[uid]
            alpha = 0.5  # 调整 alpha 值以改变透明度
            output_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)


            file_path = os.path.join(json_path, "mask_"+raw_image_name.split(".")[0]+".json")
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                # Draw bounding boxes and labels
                for obj_id, obj_item in json_data["labels"].items():
                    # Extract data from JSON
                    x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                    instance_id = obj_item["instance_id"]
                    class_name = obj_item["class_name"]

                    # Draw rectangle
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Put text
                    label = f"{instance_id}: {class_name}"
                    cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Save the modified image
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, output_image)

                print(f"Annotated image saved as {output_image_path}")

    @staticmethod
    def random_color():
        """random color generator"""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
