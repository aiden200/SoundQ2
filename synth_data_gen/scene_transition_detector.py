import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class SceneTransitionDetector:
    def __init__(self, histogram_diff_threshold=0.05):
        self.histogram_diff_threshold = histogram_diff_threshold


    def extract_frames_from_video(self, video_path, frame_skip=1):
        """
        Extract frames from a video file.
        
        Args:
            video_path (str): Path to the video file.
            frame_skip (int): Number of frames to skip to reduce processing.
        
        Returns:
            frames (list): List of extracted frames as numpy arrays.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_skip == 0:
                frames.append(frame)
            frame_index += 1

        cap.release()
        return frames


    def detect_scene_changes(
        self,
        video_path, 
        frame_skip=1, 
        threshold=0.4  # interpret as "if difference > 0.4 => scene boundary"
    ):
        """
        Compare histograms of consecutive frames and detect scene changes
        if the difference exceeds `threshold`.

        Returns:
            scene_boundaries: List of indices where a new scene starts
                            e.g. [0, 15, 40, 80, ...]
            histogram_diffs:  List of difference/correlation scores for debugging
        """
        frames = self.extract_frames_from_video(video_path, frame_skip)
        scene_boundaries = [0]  # always start with frame 0 as a scene boundary
        histogram_diffs = []

        prev_hist = None

        # print(f"Threshold: " + str(threshold))

        for i, frame in enumerate(frames):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Build histogram for each channel
            # Note: 16 or 32 bins are typical. Adjust if needed.
            h_bins = 16
            s_bins = 16
            v_bins = 16
            hist_h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])

            # Normalize each histogram
            cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX)

            # You can either compare them separately or concatenate them
            # For simplicity, let's just add them up:
            combined_hist = hist_h + hist_s + hist_v

            if prev_hist is not None:
                # Use OpenCV's compareHist for correlation
                # correlation = 1 => identical hist, 0 => different, -1 => inverted.
                corr = cv2.compareHist(prev_hist, combined_hist, cv2.HISTCMP_CORREL)

                # Convert correlation to a difference measure:
                difference = 1.0 - corr  # difference = 0 => same, near 1 => very different
                histogram_diffs.append(difference)

                # Compare difference to threshold => if difference > threshold => new scene
                if difference > threshold:
                    scene_boundaries.append(i)
            else:
                histogram_diffs.append(0.0)  # No diff for the first frame

            prev_hist = combined_hist

        return scene_boundaries, histogram_diffs