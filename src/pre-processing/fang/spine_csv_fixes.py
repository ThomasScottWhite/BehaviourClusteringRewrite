import glob
import re
import os
import shutil
import pandas as pd
import json

src = "/home/thomas/washu/behavior_clustering/data/Fang/unprocessed_csvs"
target = "/home/thomas/washu/behavior_clustering/data/Fang/processed_csvs"

import pandas as pd
import numpy as np


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calculate_angle(x1, y1, x2, y2, x3, y3):
    vec1 = np.array([x1 - x2, y1 - y2]).T  # Transpose to ensure correct shape
    vec2 = np.array([x3 - x2, y3 - y2]).T  # Transpose to ensure correct shape

    norm1 = np.linalg.norm(vec1, axis=1)
    norm2 = np.linalg.norm(vec2, axis=1)

    dot_product = np.einsum("ij,ij->i", vec1, vec2)  # Element-wise dot product
    cosine_angle = dot_product / (norm1 * norm2)

    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))  # Convert to degrees


def fix_pose_data(df_path):
    df = pd.read_csv(df_path)
    body_parts = df.iloc[0, 1:]
    coords = df.iloc[1, 1:]

    new_columns = ["Image"]
    for part, coord in zip(body_parts, coords):
        new_columns.append(f"{part}_{coord}")
    df.columns = new_columns
    df = df[2:].reset_index(drop=True).drop(["Image"], axis=1)
    df = df.apply(pd.to_numeric, errors="coerce")

    likelihood_threshold = 0.75
    for column_group in df.columns[::3]:
        base_name = column_group[:-2]
        x_col, y_col, likelihood_col = (
            f"{base_name}_x",
            f"{base_name}_y",
            f"{base_name}_likelihood",
        )
        mask = df[likelihood_col] < likelihood_threshold
        df.loc[mask, [x_col, y_col]] = pd.NA

    df.interpolate(method="linear", inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    # Normalize relative to nose
    center_x = (df["spinal_front_x"] + df["spinal_mid_x"] + df["spinal_low_x"]) / 3
    center_y = (df["spinal_front_y"] + df["spinal_mid_y"] + df["spinal_low_y"]) / 3

    x_cols = [col for col in df.columns if col.endswith("_x")]
    y_cols = [col for col in df.columns if col.endswith("_y")]
    x_df, y_df = df[x_cols], df[y_cols]
    relative_x_df = x_df.subtract(center_x, axis=0)
    relative_y_df = y_df.subtract(center_y, axis=0)
    relative_coordinates = pd.concat([relative_x_df, relative_y_df], axis=1)

    relative_coordinates["distance_nose_abdomen"] = euclidean_distance(
        df["nose_x"], df["nose_y"], df["ab_mid_x"], df["ab_mid_y"]
    )
    relative_coordinates["distance_nose_paw"] = euclidean_distance(
        df["nose_x"], df["nose_y"], df["hand_R_x"], df["hand_R_y"]
    )
    relative_coordinates["distance_nose_tail"] = euclidean_distance(
        df["nose_x"], df["nose_y"], df["tail_end_x"], df["tail_end_y"]
    )
    relative_coordinates["distance_nose_spine"] = euclidean_distance(
        df["nose_x"], df["nose_y"], df["spinal_mid_x"], df["spinal_mid_y"]
    )
    relative_coordinates["distance_paw_ear"] = euclidean_distance(
        df["hand_R_x"], df["hand_R_y"], df["ear_R_x"], df["ear_R_y"]
    )
    relative_coordinates["distance_paw_spinal"] = euclidean_distance(
        df["hand_R_x"], df["hand_R_y"], df["spinal_mid_x"], df["spinal_mid_y"]
    )

    relative_coordinates["spinal_angle"] = calculate_angle(
        df["spinal_front_x"],
        df["spinal_front_y"],
        df["spinal_mid_x"],
        df["spinal_mid_y"],
        df["spinal_low_x"],
        df["spinal_low_y"],
    )

    # Compute speeds (difference between consecutive frames)
    relative_coordinates["paw_speed"] = euclidean_distance(
        df["hand_R_x"].diff(), df["hand_R_y"].diff(), 0, 0
    )
    relative_coordinates["nose_speed"] = euclidean_distance(
        df["nose_x"].diff(), df["nose_y"].diff(), 0, 0
    )

    relative_coordinates["Frame"] = relative_coordinates.index
    relative_coordinates.to_csv(df_path, index=False)

    keep = []
    keep += [col for col in relative_coordinates.columns if "distance" in col]
    keep += [col for col in relative_coordinates.columns if "angle" in col]
    keep += [col for col in relative_coordinates.columns if "speed" in col]
    keep += [col for col in relative_coordinates.columns if "_x" in col or "_y" in col]
    return keep


trials = ["Spine_Trial"]
for trial in trials:
    shutil.rmtree(f"{target}/{trial}/")
    os.makedirs(f"{target}/{trial}/")

metadata_json = {"videos": {}, "experiment": "Fang"}

file_path = "/home/thomas/washu/behavior_clustering/data/Fang/unprocessed_csvs"
avi_files = glob.glob(f"{file_path}/*.avi")

for file in glob.glob(f"{file_path}/*filtered.csv"):
    file_stem = file[file.index("Pde1c(+)") :]
    pattern = r"^(.*?)DLC_Resnet"

    # Apply the regex
    match = re.match(pattern, file_stem)
    if match:
        result = match.group(1)

    print(result)
    sucess = False

    avi_path = None
    for avi in avi_files:
        if result in avi:
            sucess = True
            avi_path = avi
            print(avi)
            break

    trial = trials[0]
    new_directory = f"{target}/{trial}/{result}"
    os.makedirs(f"{new_directory}/", exist_ok=True)
    new_csv_file_path = f"{new_directory}/{result}_Pose_Data.csv"
    new_avi_path = f"{new_directory}/{result}.avi"

    shutil.copy(file, new_csv_file_path)

    shutil.copy(avi_path, new_avi_path)

    metadata_json["videos"][result] = {
        "csv_path": new_csv_file_path,
        "video_path": new_avi_path,
        "trial": result,
    }

for video, data in metadata_json["videos"].items():
    data_columns = fix_pose_data(data["csv_path"])

df = pd.read_csv(metadata_json["videos"]["Pde1c(+) SDGC #21 (1)"]["csv_path"])
regex = r"(_x|_y)$"
matching_column_names = [
    col for col in df.columns if pd.Series(col).str.contains(regex).any()
]
metadata_json["data_columns"] = data_columns
metadata_json["event_columns"] = []

with open(f"{target}/metadata.json", "w") as f:
    json.dump(metadata_json, f)

print("Done")
