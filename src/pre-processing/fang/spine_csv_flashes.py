import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import json

# Load metadata
with open(
    "/home/thomas/washu/behavior_clustering/data/Fang/processed_csvs/metadata.json", "r"
) as file:
    metadata = json.load(file)

# Iterate over each video in the metadata
for video_name, video_info in metadata["videos"].items():
    light_source_point = video_info["light_source_point"]
    light_on_rgb = np.array([255, 243, 197])
    light_off_rgb = np.array([250, 160, 135])

    def is_light_on(light_source_point, frame):
        """
        Determine if the light is on based on the RGB value at the light source point.

        Args:
            light_source_point (tuple): (x, y) coordinates of the light source point.
            frame (np.ndarray): The video frame.

        Returns:
            bool: True if the light is on, False otherwise.
        """
        x, y = light_source_point
        pixel_rgb = frame[y, x][::-1]  # Convert BGR to RGB
        dist_to_on = np.linalg.norm(pixel_rgb - light_on_rgb)
        dist_to_off = np.linalg.norm(pixel_rgb - light_off_rgb)
        return dist_to_on < dist_to_off

    # Create a list to store light status for each frame
    light_status_list = []

    # Open the video file
    cap = cv2.VideoCapture(video_info["video_path"])
    success, frame = cap.read()
    frame_count = 1

    while success:
        frame_count += 1
        light_status = is_light_on(light_source_point, frame)
        light_status_list.append(light_status)
        success, frame = cap.read()

    cap.release()

    # Load the corresponding CSV file
    csv_df = pd.read_csv(video_info["csv_path"], index_col=0)

    # Add light status to the CSV DataFrame
    csv_df["Light On"] = light_status_list

    # Process light status to identify continuous light on periods
    progressive_index = 0
    light_on_periods = [False] * len(csv_df)
    light_on_indices = csv_df[csv_df["Light On"] == True].index

    print(light_on_indices)
    for idx in light_on_indices:
        if progressive_index == 0:
            progressive_index = idx
            light_on_periods[idx] = True
        elif idx - progressive_index < 10:
            progressive_index = idx
        else:
            light_on_periods[idx] = True
            progressive_index = idx

    csv_df["Light On"] = light_on_periods

    # Save the updated CSV file
    csv_df.to_csv(video_info["csv_path"])
