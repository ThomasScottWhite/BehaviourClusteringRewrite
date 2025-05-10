import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

if __name__ == "__main__":
    file_path = "/home/thomas/washu/behavior_clustering/outputs/fear_voiding_8_frames_reduced4x_umap/meta_data.pkl"

    with open(file_path, "rb") as file:
        meta_data = pickle.load(file)


def combine_freezing_dataframe(meta_data):
    # Combines Video DFs into one dataframe
    dfs = []

    for video_name, video_dict in meta_data["videos"].items():

        df = video_dict["df"]

        keep_columns = meta_data["original_columns"]
        keep_columns = [
            column
            for column in keep_columns
            if column in df.columns and (column.endswith("_x") or column.endswith("_y"))
        ]

        keep_columns += ["Frame", "Cluster", "Group"]
        keep_columns += meta_data["event_columns"]
        df = df[keep_columns].copy()

        # Extract all x and y coordinates separately
        x_columns = [col for col in df.columns if col.endswith("_x")]
        y_columns = [col for col in df.columns if col.endswith("_y")]

        # Calculate centroid coordinates
        centroid_x = df[x_columns].mean(axis=1)
        centroid_y = df[y_columns].mean(axis=1)

        centroid_df = pd.DataFrame(
            {
                "Centroid_x": centroid_x,
                "Centroid_y": centroid_y,
                "Frame": df["Frame"],
                "Cluster": df["Cluster"],
                "Group": df["Group"],
            }
        )

        # Calculate Euclidean distance (rate of change) for each cluster
        df["Delta_x"] = centroid_df["Centroid_x"].diff()
        df["Delta_y"] = centroid_df["Centroid_y"].diff()
        df["Rate_of_Change"] = np.sqrt(df["Delta_x"] ** 2 + df["Delta_y"] ** 2)
        df["Trial"] = video_dict["trial"]
        # Remove NaN and infinite values from rate of change
        df["Rate_of_Change"] = df["Rate_of_Change"].replace([np.inf, -np.inf], np.nan)
        df.dropna(subset=["Rate_of_Change"], inplace=True)

        df["Video"] = video_name
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    event_dict = {"Cluster": "mean", "Rate_of_Change": "sum"}

    for event in meta_data["event_columns"]:
        event_dict[event] = "max"

    combined_df = (
        combined_df.loc[
            :,
            ["Rate_of_Change", "Group", "Video", "Cluster", "Trial"]
            + meta_data["event_columns"],
        ]
        .groupby(["Group", "Video", "Cluster", "Trial"], as_index=False)
        .agg(event_dict)
    )

    freezeing_threshold = 4
    combined_df["Is_Frozen"] = combined_df["Rate_of_Change"] < freezeing_threshold

    return combined_df


def graph_trial_rate_of_change_distrubution(meta_data):

    df = combine_freezing_dataframe(meta_data)
    graph_base_path = Path(meta_data["output_path"]) / "graphs" / "freezing"

    for trail in df.Trial.unique():
        trail_df = df[df.Trial == trail]
        data = trail_df["Rate_of_Change"]

        # Ensure only strictly positive values remain
        data = data[data > 0]

        # Define log-spaced bins
        bin_min, bin_max = data.min(), data.max()
        if bin_min == bin_max:  # Prevent logspace error
            bin_min, bin_max = bin_min * 0.9, bin_max * 1.1
        bins = np.logspace(np.log10(bin_min), np.log10(bin_max), num=100)

        sns.histplot(data, bins=bins, kde=False)
        plt.title(f"Rate of Change Distribution In {trail} (Log Scale)")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.xscale("log")

        graph_path = graph_base_path / "logarithmic"
        graph_path.mkdir(parents=True, exist_ok=True)
        plt.show()


def generate_freezing_event_markers(meta_data, combined_df, heatmap_data, ax):
    """
    Generate event markers on the heatmap based on the experiment type.

    Args:
        meta_data (dict): Metadata containing experiment type and event information.
        combined_df (DataFrame): Combined DataFrame of all videos.
        heatmap_data (DataFrame): Data for the heatmap.
        ax (Axes): Matplotlib Axes object for the heatmap.
    """
    if meta_data["experiment"] == "Fang":
        for _, row in combined_df.iterrows():
            trial_video_idx = heatmap_data.index.tolist().index(row["Video"])
            group = row["Group"]  # Get the group (x-axis)

            if row["Light On"]:
                ax.plot(
                    group + 0.5,
                    trial_video_idx + 0.5,
                    "rx",
                    markersize=12,
                    label="Light",
                )

    if meta_data["experiment"] == "fear_voiding":
        for _, row in combined_df.iterrows():
            trial_video_idx = heatmap_data.index.tolist().index(row["Video"])
            group = row["Group"]  # Get the group (x-axis)

            if row["Is_Voiding"]:
                ax.plot(
                    group + 0.5,
                    trial_video_idx + 0.5,
                    "rx",
                    markersize=12,
                    label="Is_Voiding",
                )
            if row["Shock_Start"]:
                ax.plot(
                    group + 0.5,
                    trial_video_idx + 0.5,
                    "bx",
                    markersize=12,
                    label="Shock_Start",
                )
            if row["Shock_End"]:
                ax.plot(
                    group + 0.5,
                    trial_video_idx + 0.5,
                    "yx",
                    markersize=12,
                    label="Shock_End",
                )
            if row["Tone_Start"]:
                ax.plot(
                    group + 0.5,
                    trial_video_idx + 0.5,
                    "gx",
                    markersize=12,
                    label="Tone_Start",
                )
            if row["Tone_End"]:
                ax.plot(
                    group + 0.5,
                    trial_video_idx + 0.5,
                    "mx",
                    markersize=12,
                    label="Tone_End",
                )


def _graph_freezing_heatmap(
    meta_data, combined_df, graph_path, title="Heatmap of freezing across all trials"
):
    """
    Generate a heatmap of clusters with event markers.

    Args:
        meta_data (dict): Metadata containing experiment type and event information.
        combined_df (DataFrame): Combined DataFrame of all videos.
        graph_path (Path): Path to save the heatmap.
    """
    heatmap_data = combined_df.pivot_table(
        index="Video",  # Combined Trial and Video as the row index
        columns="Group",  # Group as the x-axis
        values="Is_Frozen",  # Cluster value for the heatmap
        aggfunc="mean",  # Aggregating by mean (or other suitable method)
    ).fillna(
        0
    )  # Fill missing values with 0 if necessary
    plt.figure(figsize=(60, 15))

    colors = ["white", "#ADD8E6"]
    cmap = LinearSegmentedColormap.from_list("custom_white_blue", colors, N=2)

    ax = sns.heatmap(
        heatmap_data,
        cmap=cmap,
        linewidths=0,
        cbar_kws={"label": "Is Frozen (1 = True, 0 = False)"},
    )

    generate_freezing_event_markers(meta_data, combined_df, heatmap_data, ax)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))

    ax.legend(
        unique_labels.values(),
        unique_labels.keys(),
        bbox_to_anchor=(1.10, 1),
        loc="upper left",
        borderaxespad=0,
        title="Events",
    )
    # Adjust labels and title
    plt.xlabel("Group")
    plt.ylabel("Trial - Video")
    plt.title(title)
    plt.savefig(graph_path)
    plt.show()
    plt.close()  # Close the figure


import os


def graph_freezing_heatmap(meta_data):
    df = combine_freezing_dataframe(meta_data)

    trial_names = df["Trial"].unique()
    for trial in trial_names:
        video_df = df[df["Trial"] == trial]
        path = (
            Path(meta_data["output_path"])
            / "graphs"
            / "freeze_heatmaps"
            / f"{trial}_freeze_heatmap.png"
        )
        os.makedirs(path.parent, exist_ok=True)
        _graph_freezing_heatmap(
            meta_data, video_df, path, f"Heatmap of freezing across {trial}"
        )

    path = Path(meta_data["output_path"]) / "graphs" / "heatmaps" / f"heatmap.png"
    os.makedirs(path.parent, exist_ok=True)

    _graph_freezing_heatmap(
        meta_data, df, path, "Heatmap of freezing across all trials"
    )
