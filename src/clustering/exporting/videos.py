import glob
from pathlib import Path
import json
import os
import cv2
import pandas as pd
from tqdm import tqdm


def put_outlined_text(
    frame, text, position, font, font_scale, font_color, outline_color, thickness
):
    # Draw outline (4 passes)
    x, y = position
    cv2.putText(
        frame,
        text,
        (x - 1, y - 1),
        font,
        font_scale,
        outline_color,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x + 1, y - 1),
        font,
        font_scale,
        outline_color,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x - 1, y + 1),
        font,
        font_scale,
        outline_color,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x + 1, y + 1),
        font,
        font_scale,
        outline_color,
        thickness + 2,
        cv2.LINE_AA,
    )
    # Draw main text
    cv2.putText(
        frame, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA
    )
    return frame


def precompute_next_prev_events(event_dicts, frames_to_process):
    """
    Precompute, for each event and each frame i, the indices of the previous
    and next occurrences of that event. Returns two dictionaries:
      prev_events[event_name][i] -> frame index of previous event or None
      next_events[event_name][i] -> frame index of next event or None
    """
    prev_events = {}
    next_events = {}

    for event_name, frame_list in event_dicts.items():
        # Sort the frame_list just in case
        sorted_frames = sorted(frame_list)

        # Initialize arrays
        prev_arr = [None] * frames_to_process
        next_arr = [None] * frames_to_process

        # Single pass to fill prev_arr:
        #   We'll walk through sorted_frames and mark that for every
        #   frame from sorted_frames[k] up to the next event, the "prev" is sorted_frames[k].
        current_event_idx = 0
        last_event_frame = None

        for i in range(frames_to_process):
            # Move forward in sorted_frames if we've passed the current event
            while (
                current_event_idx < len(sorted_frames)
                and i >= sorted_frames[current_event_idx]
            ):
                last_event_frame = sorted_frames[current_event_idx]
                current_event_idx += 1
            prev_arr[i] = last_event_frame

        # Single pass to fill next_arr (go from end -> start):
        current_event_idx = len(sorted_frames) - 1
        next_event_frame = None

        for i in reversed(range(frames_to_process)):
            while current_event_idx >= 0 and i <= sorted_frames[current_event_idx]:
                next_event_frame = sorted_frames[current_event_idx]
                current_event_idx -= 1
            next_arr[i] = next_event_frame

        prev_events[event_name] = prev_arr
        next_events[event_name] = next_arr

    return prev_events, next_events


def generate_videos(meta_data):
    for index, video in meta_data["videos"].items():

        csv = video["df"]
        video_path = video["video_path"]

        # Make output directory
        out_dir = f'{meta_data["output_path"]}/videos/{video["trial"]}/{index}'
        os.makedirs(out_dir, exist_ok=True)
        print(f"Output directory: {out_dir}")
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue

        # Gather event frames
        event_dicts = {}
        for event in meta_data["event_columns"]:
            mask = csv[event]  # boolean mask for that event
            results = csv[mask]  # rows where that event is True
            event_dicts[event] = list(results.index)  # frame indices

        # Input columns
        bouts = csv["Group"].values
        numbers = csv["Cluster"].values

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine how many frames we actually need
        frames_to_process = min(len(numbers), frame_count)
        print(f"Processing {frames_to_process} frames out of {frame_count}.")

        # Precompute next/previous events for each frame
        prev_events, next_events = precompute_next_prev_events(
            event_dicts, frames_to_process
        )

        # Set up the main video writer
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_path = os.path.join(out_dir, "output_video.avi")
        out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

        # One writer per cluster
        num_clusters = csv["Cluster"].max() + 1
        cluster_writers = []
        for i in range(num_clusters):
            cluster_path = os.path.join(out_dir, f"{i}.avi")
            writer = cv2.VideoWriter(
                cluster_path, fourcc, fps, (frame_width, frame_height)
            )
            cluster_writers.append(writer)

        # Font params for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        thickness = 3
        outline_color = (0, 0, 0)

        # Read and process frames
        frame_index = 0
        while cap.isOpened() and frame_index < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            # Overlay cluster/bout text
            frame = put_outlined_text(
                frame,
                f"Cluster {numbers[frame_index]}",
                (50, 60),
                font,
                font_scale,
                font_color,
                outline_color,
                thickness,
            )
            frame = put_outlined_text(
                frame,
                f"Bout {bouts[frame_index]}",
                (50, 30),
                font,
                font_scale,
                font_color,
                outline_color,
                thickness,
            )

            # Overlay next/previous events
            event_counter = 0
            for event_name in meta_data["event_columns"]:
                prev_idx = prev_events[event_name][frame_index]
                next_idx = next_events[event_name][frame_index]

                if next_idx is None:
                    next_string = f"Next {event_name}: NA"
                else:
                    sec_until = int((next_idx - frame_index) / fps)
                    next_string = f"Next {event_name} {sec_until} Seconds"

                if prev_idx is None:
                    prev_string = f"Previous {event_name}: NA"
                else:
                    sec_ago = int((frame_index - prev_idx) / fps)
                    prev_string = f"Previous {event_name} in {sec_ago} Seconds Ago"

                y_next = 90 + event_counter * 60
                y_prev = 120 + event_counter * 60

                frame = put_outlined_text(
                    frame,
                    next_string,
                    (50, y_next),
                    font,
                    font_scale,
                    font_color,
                    outline_color,
                    thickness,
                )
                frame = put_outlined_text(
                    frame,
                    prev_string,
                    (50, y_prev),
                    font,
                    font_scale,
                    font_color,
                    outline_color,
                    thickness,
                )
                event_counter += 1

            # Write frame to the appropriate cluster file and to output
            cluster_idx = numbers[frame_index]
            cluster_writers[cluster_idx].write(frame)
            out.write(frame)

            frame_index += 1

        # Release resources
        cap.release()
        out.release()
        for w in cluster_writers:
            w.release()

        print(f"Video saved to {out_path}")


import pickle

if __name__ == "__main__":
    file_path = "/home/thomas/washu/behavior_clustering_rewrite/src/outputs/fear_voiding_absolute_10_clusters/meta_data.pkl"

    with open(file_path, "rb") as file:
        meta_data = pickle.load(file)

    generate_videos(meta_data)
