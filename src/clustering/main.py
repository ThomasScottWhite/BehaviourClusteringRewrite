# %%%
import json
import os
import numpy as np
import pandas as pd
import shutil
from clustering import clustering
from exporting import graphs, videos
import pickle
from pathlib import Path


def load_metadata(file_path):
    """
    Load metadata from a JSON file and read associated CSV files into DataFrames.

    Args:
        file_path (str): Path to the metadata JSON file.

    Returns:
        dict: Metadata dictionary with DataFrames added for each video.
    """
    with open(file_path, "r") as file:
        meta_data = json.load(file)

    for video in meta_data["videos"].values():
        video["df"] = pd.read_csv(video["csv_path"], index_col=0)
    return meta_data


def rotate_points_global(df, ref_points=["Nose", "Spine1", "Hipbone"]):
    """
    Rotate data to align the plane defined by 3 points (e.g., nose, spine, hip) with the x-axis.

    Args:
        df (pd.DataFrame): DataFrame containing the coordinates.
        ref_points (list): List of reference points to define the plane.

    Returns:
        pd.DataFrame: Rotated DataFrame.
    """
    # Extract coordinates for the 3 reference points
    p1 = df[[f"{ref_points[0]}_x", f"{ref_points[0]}_y"]].values
    p2 = df[[f"{ref_points[1]}_x", f"{ref_points[1]}_y"]].values
    p3 = df[[f"{ref_points[2]}_x", f"{ref_points[2]}_y"]].values

    # Compute the primary axis using PCA on the 3 points
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(np.vstack([p1, p2, p3]))
    main_axis = pca.components_[0]  # Direction of maximum variance

    # Calculate rotation angle to align main_axis with x-axis
    angle = np.arctan2(main_axis[1], main_axis[0])
    cos_theta, sin_theta = np.cos(-angle), np.sin(-angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Rotate all points
    rotated_data = {}
    for col in df.columns:
        if "_x" in col:
            y_col = col.replace("_x", "_y")
            x_vals = df[col].values
            y_vals = df[y_col].values
            rotated_coords = rotation_matrix @ np.vstack([x_vals, y_vals])
            rotated_data[col] = rotated_coords[0]
            rotated_data[y_col] = rotated_coords[1]
        else:
            rotated_data[col] = df[col]

    return pd.DataFrame(rotated_data)


def rotate_all(meta_data, ref_points=["Nose", "Spine1", "Hipbone"]):
    """
    Rotate all videos' data to align with the x-axis.

    Args:
        meta_data (dict): Metadata dictionary containing DataFrames for each video.
        ref_points (list): List of reference points to define the plane.

    Returns:
        dict: Updated metadata dictionary with rotated DataFrames.
    """
    for video in meta_data["videos"].values():
        video["df"] = rotate_points_global(video["df"], ref_points)
    return meta_data


def add_bouts(meta_data, bout_frames=4):
    """
    Add bout information to each video's DataFrame.

    Args:
        meta_data (dict): Metadata dictionary containing DataFrames for each video.
        bout_frames (int): Number of frames in each bout.

    Returns:
        dict: Updated metadata dictionary with bout information added.
    """
    for video in meta_data["videos"].values():

        total_frames = len(video["df"])
        complete_bouts = total_frames // bout_frames
        cutoff = complete_bouts * bout_frames
        video["df"] = video["df"].iloc[:cutoff].reset_index(drop=True)

        video["df"]["Group"] = video["df"].index // bout_frames

    return meta_data


def save_csvs(meta_data):
    """
    Save the processed DataFrames to CSV files.

    Args:
        meta_data (dict): Metadata dictionary containing DataFrames for each video.
    """
    for video_name, video in meta_data["videos"].items():
        os.makedirs(f'{meta_data['output_path']}/csvs/{video["trial"]}', exist_ok=True)
        video["df"].to_csv(
            f'{meta_data['output_path']}/csvs/{video["trial"]}/{video_name}.csv'
        )


def save_tsne_results(meta_data):
    """
    Save the t-SNE results to a CSV file.

    Args:
        meta_data (dict): Metadata dictionary containing t-SNE results.
    """
    meta_data["tsne_results"].to_csv(
        f"{meta_data["output_path"]}/csvs/tsne_results.csv"
    )


def reduce_dfs(meta_data, factor=4):
    """
    Reduce the DataFrames by a specified factor.

    Args:
        meta_data (dict): Metadata dictionary containing DataFrames for each video.
        factor (int): Factor by which to reduce the DataFrames.

    Returns:
        dict: Updated metadata dictionary with reduced DataFrames.
    """
    event_columns = meta_data["event_columns"]
    for video in meta_data["videos"].values():
        df = video["df"]

        # Create a new DataFrame to store the reduced data
        reduced_df = df.iloc[::factor].copy()

        # Iterate over each event column and apply the logic
        for col in event_columns:
            # Create a boolean mask for each chunk of `factor` rows
            mask = (
                df[col]
                .rolling(window=factor, min_periods=1)
                .max()
                .iloc[::factor]
                .astype(bool)
            )
            # Update the event column in the reduced DataFrame
            reduced_df[col] = mask.values

        # Update the video's DataFrame
        video["df"] = reduced_df

    return meta_data


def increase_dfs(meta_data, factor=4):
    """
    Increase the DataFrames by a specified factor.

    Args:
        meta_data (dict): Metadata dictionary containing DataFrames for each video.
        factor (int): Factor by which to increase the DataFrames.

    Returns:
        dict: Updated metadata dictionary with increased DataFrames.
    """
    for video in meta_data["videos"].values():
        video["df"] = (
            video["df"].loc[np.repeat(video["df"].index, factor)].reset_index(drop=True)
        )
    return meta_data


def make_output_directory(meta_data, metadatasource, cluster_method):
    """
    Create an output directory for storing results.

    Args:
        meta_data (dict): Metadata dictionary.
        metadatasource (str): Path to the metadata source file.

    Returns:
        dict: Updated metadata dictionary with the output path added.
    """

    output_path_name = f"{meta_data["experiment"]}_{bout_frames}_frames"
    if reduction_factor != 1:
        output_path_name += f"_reduced{reduction_factor}x"
    output_path_name += f"_{cluster_method}"
    if rotation:
        output_path_name += "_rotated"

    base_output_dir = (Path(__file__).resolve().parent / "../outputs").resolve()
    output_path = base_output_dir / output_path_name

    counter = 0
    # If the path already exists, append a counter to create a unique directory.
    while output_path.exists():
        counter += 1
        output_path = base_output_dir / f"{output_path_name}_{counter}"

    meta_data["output_path"] = output_path
    os.makedirs(output_path, exist_ok=True)
    shutil.copy(metadatasource, output_path / "metadata.json")

    return meta_data


def pickle_meta_data(meta_data):
    """
    Save the metadata dictionary to a pickle file.

    Args:
        meta_data (dict): Metadata dictionary.
    """
    with open(f"{meta_data['output_path']}/meta_data.pkl", "wb") as f:
        pickle.dump(meta_data, f)


def main(
    metadatasource,
    bout_frames=16,
    reduction_factor=4,
    cluster_method="tsne",
    rotation=True,
    video=False,
    determine_n_clusters=False,
    n_clusters=5,
):
    """
    Main function to process metadata and generate outputs.

    Args:
        metadatasource (str): Path to the metadata source file.
        bout_frames (int): Number of frames in a bout.
        reduction_factor (int): Factor by which to reduce the DataFrames.
        cluster_method (str): Clustering method to use ('tsne', 'pca', 'pre_group').
        rotation (bool): Whether to rotate the data.
        video (bool): Whether to generate videos.
    """
    # Metadata source

    print("Step 1: Load and process metadata")
    meta_data = load_metadata(metadatasource)
    meta_data = reduce_dfs(meta_data, reduction_factor)

    if rotation:
        if meta_data["experiment"] == "fear_voiding":
            meta_data = rotate_all(meta_data)
        else:
            meta_data = rotate_all(meta_data, ["nose", "spinal_mid", "tail_base"])

    meta_data = make_output_directory(meta_data, metadatasource, cluster_method)

    # Save parameters to a config file in the output directory
    config_path = os.path.join(meta_data["output_path"], "args.json")
    config = {
        "bout_frames": bout_frames,
        "reduction_factor": reduction_factor,
        "cluster_method": cluster_method,
        "rotation": rotation,
    }
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

    print("Step 2: TSNE Clustering")
    # Add bouts and cluster videos
    meta_data = add_bouts(meta_data, bout_frames=bout_frames)

    if cluster_method == "umap_pca":
        meta_data = clustering.cluster_videos_with_pca(
            meta_data, bout_frames=bout_frames, reduction="umap", determine_n_clusters=determine_n_clusters, n_clusters=n_clusters
        )
    elif cluster_method == "tsne_pca":
        meta_data = clustering.cluster_videos_with_pca(
            meta_data, bout_frames=bout_frames, reduction="tsne", determine_n_clusters=determine_n_clusters, n_clusters=n_clusters
        )
    elif cluster_method == "pre_group":
        meta_data = clustering.cluster_videos_pre_group(
            meta_data, bout_frames=bout_frames
        )

    print("Step 3: Generate outputs")
    # Revert dataframes and save outputs
    meta_data = increase_dfs(meta_data, reduction_factor)
    save_csvs(meta_data)
    save_tsne_results(meta_data)
    pickle_meta_data(meta_data)

    print("Step 4: Generate graphs and videos")
    # Generate graphs and videos
    graphs.graph_all(meta_data)

    if video:
        videos.generate_videos(meta_data)


if __name__ == "__main__":

    metadatasource = (
        "/home/thomas/washu/behavior_clustering_rewrite/data/data_mined_data/fear_voiding_absolute/metadata.json"
    )

    bout_frames = 8
    reduction_factor = 4
    cluster_method = "umap_pca"
    rotation = False
    video = True
    n_clusters = 10
    determine_n_clusters = False
    main(
        metadatasource=metadatasource,
        bout_frames=bout_frames,
        reduction_factor=reduction_factor,
        cluster_method=cluster_method,
        rotation=rotation,
        video=video,
        determine_n_clusters=determine_n_clusters,
        n_clusters=n_clusters,
    )
