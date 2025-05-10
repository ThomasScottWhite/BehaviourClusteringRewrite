import os
import shutil
import re
import pandas as pd
from pathlib import Path
import json
import numpy as np
from scipy.signal import savgol_filter
import json
import pandas as pd 
import os
import shutil
from pathlib import Path

def process_csv(df):
    kept_points = ['RightEar', 'LeftEar', 'forehead','Nose', 'shoulder', 'Spine1', 'Spine2', 'Spine3', 'Hipbone', 'TailBase', 'Tail2']

    kept_points_x = [col + "_x" for col in kept_points]
    kept_points_y = [col + "_y" for col in kept_points]

    x_cols = [col for col in df.columns if col.endswith("_x")]
    y_cols = [col for col in df.columns if col.endswith("_y")]

    # columns_to_keep = df.columns.tolist()
    # columns_to_drop = [col for col in columns_to_keep if col not in (x_cols + y_cols)]

    # df = df.drop(columns=columns_to_drop)

    df.interpolate(method="linear", inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    # Smoothing the data using Savitzky-Golay filter
    window_length = 5  # pick an odd window
    polyorder = 2
    for col in x_cols + y_cols:
        df[col] = savgol_filter(
            df[col], window_length=window_length, polyorder=polyorder
        )

    # Relative coordinates to the nose, The relative cordinates are used in all future analysis, except freezing analysis\
    x_cols = kept_points_x
    y_cols = kept_points_y

    reference_part = "Nose"
    reference_x = df[f"{reference_part}_x"]
    reference_y = df[f"{reference_part}_y"]

    relative_x_df = df[x_cols].subtract(reference_x, axis=0)
    relative_y_df = df[y_cols].subtract(reference_y, axis=0)

    relative_x_df.columns = [f"{col}_rel" for col in relative_x_df.columns]
    relative_y_df.columns = [f"{col}_rel" for col in relative_y_df.columns]

    relative_coordinates = pd.concat([relative_x_df, relative_y_df], axis=1)
    relative_column_names = relative_coordinates.columns.tolist()
    original_column_names = df.columns.tolist()

    df = pd.concat([df, relative_coordinates], axis=1)

    speed_cols = []

    for part in kept_points:
        if "Ear_" in part or "Spine1_" in part:
            continue
        df[f"{part}_speed"] = np.sqrt(df[f"{part}_x"] ** 2 + df[f"{part}_y"] ** 2)
        speed_cols.append(f"{part}_speed")

    df["Ear_symmetry"] = np.abs(df["RightEar_x_rel"] - df["LeftEar_x_rel"]) + np.abs(
        df["RightEar_y_rel"] - df["LeftEar_y_rel"]
    )

    df["Tail_elevation"] = (
        df["TailBase_y_rel"] - df["Hipbone_y_rel"]
    )  # Change tailbase to another point

    df["Nose_to_TailRoot_dist"] = np.sqrt(
        (df["Nose_x"] - df["TailBase_x"]) ** 2 + (df["Nose_y"] - df["TailBase_y"]) ** 2
    )

    def calculate_angle(df, p1, p2, p3):
        v1 = df[[f"{p1}_x", f"{p1}_y"]].values - df[[f"{p2}_x", f"{p2}_y"]].values
        v2 = df[[f"{p3}_x", f"{p3}_y"]].values - df[[f"{p2}_x", f"{p2}_y"]].values
        dot = np.sum(v1 * v2, axis=1)
        norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        return np.degrees(np.arccos(dot / norm))

    print(df.columns)

    df["Body_angle"] = calculate_angle(df, "Nose", "Hipbone", "TailBase")
    df["Trunk_angle"] = calculate_angle(df, "Spine1", "Spine2", "Spine3")
    df["Body_tail_angle"] = calculate_angle(df, "Hipbone", "TailBase", "Tail2")
    df["tail_angle"] = calculate_angle(df, "TailBase", "Tail3", "Tail4")


    keep = []
    keep += relative_column_names
    keep += ["Nose_to_TailRoot_dist"]
    keep += ["Body_angle"]
    keep += speed_cols
    keep += ["Ear_symmetry", "Tail_elevation"]

    return df, keep, original_column_names

if __name__ == "__main__":

    
    metadata_path = "/home/thomas/washu/behavior_clustering_rewrite/data/structured_data/fear_voiding/metadata.json"
    output_dir = Path("/home/thomas/washu/behavior_clustering_rewrite/data/data_mined_data/fear_voiding_removed_bad_datapoints")

    os.makedirs(output_dir, exist_ok=True)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)



    for videoname in metadata["videos"]:
        video_dict = metadata["videos"][videoname]
        csv_path = video_dict["csv_path"]
        trial = video_dict["trial"]
        video_path = video_dict["video_path"]


        df = pd.read_csv(csv_path)

        new_csv_path = output_dir / trial / f"{videoname}.csv"
        new_video_path = output_dir / trial / f"{videoname}.avi"

        os.makedirs(new_csv_path.parent, exist_ok=True)

        metadata["videos"][videoname]["original_csv"] = metadata["videos"][videoname]["csv_path"]
        metadata["videos"][videoname]["csv_path"] = str(new_csv_path)
        metadata["videos"][videoname]["video_path"] = str(new_video_path)
        
        df, keep, original_column_names = process_csv(df)

        metadata["data_columns"] = keep
        metadata["original_columns"] = original_column_names
        
        df.to_csv(new_csv_path, index=False)
        if not os.path.exists(new_video_path):
            shutil.copy(video_path, new_video_path)

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)