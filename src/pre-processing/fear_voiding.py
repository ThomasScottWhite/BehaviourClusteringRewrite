# %%%
import os
import shutil
import re
import pandas as pd
from pathlib import Path
import json
import numpy as np
from scipy.signal import savgol_filter

# Define source and destination directories

# %%%
# Loop through all files in the source directory
def move_data_to_new_directory(src_dir, dst_dir):

    # Ensure the destination directory exists
    os.makedirs(dst_dir, exist_ok=True)


    for folder in os.listdir(src_dir):
        folder_path = os.path.join(src_dir, folder)
        folder_dst_dir = os.path.join(dst_dir, folder)

        os.makedirs(folder_dst_dir, exist_ok=True)

        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if "filtered" in filename:
                continue
            

            if "Ai213" in filename and (".CSV" in filename or ".csv" in filename):

                start_index = filename.find("Ai213")
                # Remove all characters before 'Ai213' and change .CSV to .csv
                new_filename = filename[start_index:].replace(".CSV", ".csv")

                # Extract the grouping pattern Ai213_x=x_#x using regex
                match = re.search(r"Ai213_\d+-\d+_#\d+", new_filename)
                if match:
                    if "Side_viewDLC_Resnet50" in filename:
                        new_filename = match.group(0) + "_Pose_Data.csv"

                    # Create a subdirectory based on the matched pattern
                    subfolder_name = match.group(0)
                    subfolder_path = os.path.join(folder_dst_dir, subfolder_name)
                    os.makedirs(subfolder_path, exist_ok=True)

                    # Construct full file paths
                    old_filepath = os.path.join(folder_path, filename)
                    new_filepath = os.path.join(subfolder_path, new_filename)

                    # Move and rename the file
                    shutil.copy(old_filepath, new_filepath)
            
            elif ".AVI" in filename:
                start_index = filename.find("Ai213")
                # Remove all characters before 'Ai213' and change .CSV to .csv
                new_filename = filename[start_index:].replace(".AVI", ".avi")

                # Extract the grouping pattern Ai213_x=x_#x using regex
                match = re.search(r"Ai213_\d+-\d+_#\d+", new_filename)
                if match:
                    if "Side_viewDLC_Resnet50" in filename:
                        new_filename = match.group(0) + "_Pose_Data.csv"

                    # Create a subdirectory based on the matched pattern
                    subfolder_name = match.group(0)
                    subfolder_path = os.path.join(folder_dst_dir, subfolder_name)
                    os.makedirs(subfolder_path, exist_ok=True)

                    # Construct full file paths
                    old_filepath = os.path.join(folder_path, filename)
                    new_filepath = os.path.join(subfolder_path, new_filename)

                    # Move and rename the file
                    shutil.copy(old_filepath, new_filepath)

                print(filename)
                old_filepath = os.path.join(folder_path, filename)
                new_filepath = os.path.join(subfolder_path, new_filename)
                shutil.copy(old_filepath, new_filepath)


        print(
            "Files have been renamed, grouped into folders, and moved to the new directory."
        )

def reduce_df(df_path):
    df = pd.read_csv(df_path, index_col=0)
    likelihood_cols = [col for col in df.columns if "likelihood" in col]
    df["total_likelihood"] = df[likelihood_cols].sum(axis=1)

    # reduced_df = df.groupby(df.index // 8).first().reset_index(drop=True)
    # Group rows into groups of 8 and select the one with the highest total likelihood in each group
    grouped = df.groupby(df.index // 8)

    reduced_df = grouped.apply(
        lambda group: group.loc[group["total_likelihood"].idxmax()]
    )

    bool_columns = ["Is_Voiding", "Shock_Start", "Shock_End", "Tone_Start", "Tone_End"]
    for col in bool_columns:
        reduced_df[col] = grouped[col].any()

    # Drop the 'total_likelihood' column used for selection
    reduced_df = reduced_df.drop(columns=["total_likelihood"])
    reduced_df = reduced_df.reset_index(drop=True)
    reduced_df.to_csv(df_path)


def fix_void_timing(df_path):
    df = pd.read_csv(df_path)
    df["seconds"] = df["Var4"].str.extract(r"(\d+\.\d+)")
    df = df[["seconds"]]
    df.to_csv(df_path)


def fix_pose_data(df_path):
    df = pd.read_csv(df_path)

    body_parts = df.iloc[0, 1:]
    coords = df.iloc[1, 1:]

    new_columns = ["Image"]
    for part, coord in zip(body_parts, coords):
        new_columns.append(f"{part}_{coord}")
    df.columns = new_columns
    df = df[2:].reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors="coerce")

    df.to_csv(df_path, index=False)

    return df.columns.tolist()


# def fix_pose_data(df_path):
#     df = pd.read_csv(df_path)

#     body_parts = df.iloc[0, 1:]
#     coords = df.iloc[1, 1:]

#     new_columns = ["Image"]
#     for part, coord in zip(body_parts, coords):
#         new_columns.append(f"{part}_{coord}")
#     df.columns = new_columns
#     df = df[2:].reset_index(drop=True)
#     df = df.apply(pd.to_numeric, errors="coerce")

#     freezing_points = [
#         "Nose",
#         "RightEar",
#         "LeftEar",
#         "Spine1",
#         "Hipbone",
#         "TailBase",
#         "R_hindpaw",
#         "L_hindpaw",
#         "R_forepaw",
#         "L_forepaw",
#     ]

#     freezing_points_x = [col + "_x" for col in freezing_points]
#     freezing_points_y = [col + "_y" for col in freezing_points]

#     x_cols = [col for col in df.columns if col.endswith("_x")]
#     y_cols = [col for col in df.columns if col.endswith("_y")]

#     # columns_to_keep = df.columns.tolist()
#     # columns_to_drop = [col for col in columns_to_keep if col not in (x_cols + y_cols)]

#     # df = df.drop(columns=columns_to_drop)

#     df.interpolate(method="linear", inplace=True)
#     df.fillna(method="ffill", inplace=True)
#     df.fillna(method="bfill", inplace=True)

#     # Smoothing the data using Savitzky-Golay filter
#     window_length = 5  # pick an odd window
#     polyorder = 2
#     for col in x_cols + y_cols:
#         df[col] = savgol_filter(
#             df[col], window_length=window_length, polyorder=polyorder
#         )

#     # Relative coordinates to the nose, The relative cordinates are used in all future analysis, except freezing analysis\
#     x_cols = freezing_points_x
#     y_cols = freezing_points_y

#     reference_part = "Nose"
#     reference_x = df[f"{reference_part}_x"]
#     reference_y = df[f"{reference_part}_y"]

#     relative_x_df = df[x_cols].subtract(reference_x, axis=0)
#     relative_y_df = df[y_cols].subtract(reference_y, axis=0)

#     relative_x_df.columns = [f"{col}_rel" for col in relative_x_df.columns]
#     relative_y_df.columns = [f"{col}_rel" for col in relative_y_df.columns]

#     relative_coordinates = pd.concat([relative_x_df, relative_y_df], axis=1)
#     relative_column_names = relative_coordinates.columns.tolist()
#     original_column_names = df.columns.tolist()

#     df = pd.concat([df, relative_coordinates], axis=1)

#     print(df.columns)
#     speed_cols = []

#     for part in freezing_points:
#         if "Ear_" in part or "Spine1_" in part:
#             continue
#         df[f"{part}_speed"] = np.sqrt(df[f"{part}_x"] ** 2 + df[f"{part}_y"] ** 2)
#         speed_cols.append(f"{part}_speed")

#     df["Rigidity"] = 1 / (
#         1
#         + df[[f"{pt}_speed" for pt in ["Nose", "R_hindpaw", "L_hindpaw"]]].mean(axis=1)
#     )

#     df["Ear_symmetry"] = np.abs(df["RightEar_x_rel"] - df["LeftEar_x_rel"]) + np.abs(
#         df["RightEar_y_rel"] - df["LeftEar_y_rel"]
#     )

#     df["Tail_elevation"] = (
#         df["TailBase_y_rel"] - df["Hipbone_y_rel"]
#     )  # Change tailbase to another point

#     df["Forepaw_dist"] = np.sqrt(
#         (df["R_forepaw_x"] - df["L_forepaw_x"]) ** 2
#         + (df["R_forepaw_y"] - df["L_forepaw_y"]) ** 2
#     )
#     df["Hindpaw_dist"] = np.sqrt(
#         (df["R_hindpaw_x"] - df["L_hindpaw_x"]) ** 2
#         + (df["R_hindpaw_y"] - df["L_hindpaw_y"]) ** 2
#     )

#     df["Nose_to_TailRoot_dist"] = np.sqrt(
#         (df["Nose_x"] - df["TailBase_x"]) ** 2 + (df["Nose_y"] - df["TailBase_y"]) ** 2
#     )

#     def calculate_angle(df, p1, p2, p3):
#         v1 = df[[f"{p1}_x", f"{p1}_y"]].values - df[[f"{p2}_x", f"{p2}_y"]].values
#         v2 = df[[f"{p3}_x", f"{p3}_y"]].values - df[[f"{p2}_x", f"{p2}_y"]].values
#         dot = np.sum(v1 * v2, axis=1)
#         norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
#         return np.degrees(np.arccos(dot / norm))

#     print(df.columns)
#     df["Body_angle"] = calculate_angle(df, "Nose", "Hipbone", "TailBase")
#     df["Trunk_angle"] = calculate_angle(df, "Spine1", "Spine2", "Spine3")
#     df["Body_tail_angle"] = calculate_angle(df, "Hipbone", "TailBase", "Tail2")
#     df["tail_angle"] = calculate_angle(df, "TailBase", "Tail3", "Tail4")

#     df.to_csv(df_path, index=False)

#     keep = []
#     # keep += relative_column_names
#     keep += ["Nose_to_TailRoot_dist"]
#     keep += ["Body_angle"]
#     keep += speed_cols
#     keep += ["Rigidity", "Ear_symmetry", "Tail_elevation"]
#     keep += ["Forepaw_dist", "Hindpaw_dist"]

#     return original_column_names, keep


def fix_time_df(df_path):
    df = pd.read_csv(df_path, header=None, names=["DateTime", "Seconds"])
    df = df.replace(r"[()]", "", regex=True)
    df.to_csv(df_path)


def combine_dfs(
    pose_path,
    side_path,
    void_path,
    shock_on_path,
    shock_off_path,
    tone_on_path,
    tone_off_path,
    new_path,
):
    pose_data_df = pd.read_csv(pose_path, index_col=0)
    side_view_df = pd.read_csv(side_path, index_col=0)
    void_data_df = pd.read_csv(void_path, index_col=0)
    shock_on_df = pd.read_csv(shock_on_path, index_col=0)
    shock_off_df = pd.read_csv(shock_off_path, index_col=0)
    tone_on_df = pd.read_csv(tone_on_path, index_col=0)
    tone_off_df = pd.read_csv(tone_off_path, index_col=0)

    pose_time_df = pd.merge(
        pose_data_df, side_view_df, left_index=True, right_index=True
    )
    pose_time_df["Is_Voiding"] = False

    for voidtime in void_data_df["seconds"]:
        pose_time_df["difference"] = (pose_time_df["Seconds"] - voidtime).abs()

        closest_index = pose_time_df["difference"].idxmin()

        pose_time_df.loc[closest_index, "Is_Voiding"] = True
        pose_time_df = pose_time_df.drop(columns=["difference"])

    pose_time_df["Shock_Start"] = False
    for shock_on in shock_on_df["side_Shock_frame"]:

        pose_time_df.loc[shock_on, "Shock_Start"] = True

    pose_time_df["Shock_End"] = False
    for shock_off in shock_off_df["side_Shock_frame"]:

        pose_time_df.loc[shock_off, "Shock_End"] = True

    pose_time_df["Tone_Start"] = False
    pose_time_df["Tone_Start"] = False
    for tone_on in tone_on_df["side_Tone_frame"]:

        pose_time_df.loc[tone_on, "Tone_Start"] = True

    pose_time_df["Tone_End"] = False
    for tone_off in tone_off_df["side_Tone_frame"]:

        pose_time_df.loc[tone_off, "Tone_End"] = True

    pose_time_df["Frame"] = pose_time_df.index

    pose_time_df.to_csv(new_path)


def main(src_dir, dst_dir):
    move_data_to_new_directory(src_dir, dst_dir)

    metadata_json = {"videos": {}, "experiment": "fear_voiding"}

    for trial in os.listdir(dst_dir):
        if not os.path.isdir(os.path.join(dst_dir, trial)):
            continue

        trail_path = os.path.join(dst_dir, trial)

        folders = [
            os.path.join(trail_path, d)
            for d in os.listdir(trail_path)
            if os.path.isdir(os.path.join(trail_path, d))
        ]

        for folder in folders:

            for filename in os.listdir(folder):
                if "Bottom_camera" in filename:
                    bottom_path = os.path.join(folder, filename)

                if "Pose_Data" in filename:
                    pose_path = os.path.join(folder, filename)

                if "ShockOffset" in filename:
                    shock_off_path = os.path.join(folder, filename)

                if "ShockONset" in filename:
                    shock_on_path = os.path.join(folder, filename)

                if "Side_view" in filename:
                    if ".csv" in filename:
                        side_path = os.path.join(folder, filename)
                    else:
                        video_path = os.path.join(folder, filename)

                if "ToneOffset" in filename:
                    tone_off_path = os.path.join(folder, filename)

                if "ToneONset" in filename:
                    tone_on_path = os.path.join(folder, filename)

                if "VoidTiming" in filename:
                    void_path = os.path.join(folder, filename)

            columns = fix_pose_data(pose_path)
            fix_void_timing(void_path)
            fix_time_df(bottom_path)
            fix_time_df(side_path)

            new_path = os.path.join(folder, "pose_void_tone_shock_combined.csv")
            
            combine_dfs(
                pose_path,
                side_path,
                void_path,
                shock_on_path,
                shock_off_path,
                tone_on_path,
                tone_off_path,
                new_path,
            )
            # reduce_df(new_path)

            metadata_json["videos"][trial + "_" + folder[-12:]] = {
                "csv_path": new_path,
                "trial": trial,
                "video_path": video_path,
            }

    metadata_json["Columns"] = columns
    metadata_json["event_columns"] = [
        "Is_Voiding",
        "Shock_Start",
        "Shock_End",
        "Tone_Start",
        "Tone_End",
    ]
    dst_dir = Path(dst_dir)

    with open(dst_dir / "metadata.json", "w") as f:
        json.dump(metadata_json, f)

if __name__ == "__main__":
    src_dir = Path("/home/thomas/washu/behavior_clustering_rewrite/data/unstructured_data/fear_voiding")
    dst_dir = Path("/home/thomas/washu/behavior_clustering_rewrite/data/structured_data/fear_voiding")

    main(src_dir, dst_dir)
