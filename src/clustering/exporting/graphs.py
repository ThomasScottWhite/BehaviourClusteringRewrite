import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


def tsne_plot(meta_data):
    """
    Generate t-SNE plots for the given metadata.

    Args:
        meta_data (dict): Metadata containing t-SNE results and output path.
    """
    tsne_df = meta_data["tsne_results"]

    plt.figure(figsize=(10, 8))
    for trial in tsne_df["trial"].unique():
        cluster_data = tsne_df[tsne_df["trial"] == trial]
        plt.scatter(
            cluster_data["TSNE_1"], cluster_data["TSNE_2"], label=trial, alpha=0.6
        )

    plt.xlabel("TSNE_1")
    plt.ylabel("TSNE_2")
    plt.title("t-SNE with K-means Clustering")
    plt.legend()
    graph_path = Path(meta_data["output_path"]) / "graphs"
    os.makedirs(graph_path, exist_ok=True)

    plt.savefig(graph_path / "tsne_results_by_video.png")
    plt.close()  # Close the figure

    plt.figure(figsize=(10, 8))
    for cluster in range(tsne_df["Cluster"].nunique()):
        cluster_data = tsne_df[tsne_df["Cluster"] == cluster]
        plt.scatter(
            cluster_data["TSNE_1"],
            cluster_data["TSNE_2"],
            label=f"Cluster {cluster}",
            alpha=0.6,
        )

    plt.xlabel("TSNE_1")
    plt.ylabel("TSNE_2")
    plt.title("t-SNE with K-means Clustering")
    plt.legend()
    graph_path = Path(meta_data["output_path"]) / "graphs"
    os.makedirs(graph_path, exist_ok=True)
    plt.savefig(graph_path / "tsne_results_by_cluster.png")
    plt.close()  # Close the figure


def generate_event_markers(meta_data, combined_df, heatmap_data, ax):
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


def graph_heatmap(meta_data, combined_df, graph_path):
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
        values="Cluster",  # Cluster value for the heatmap
        aggfunc="mean",  # Aggregating by mean (or other suitable method)
    ).fillna(
        0
    )  # Fill missing values with 0 if necessary
    plt.figure(figsize=(60, 15))

    ax = sns.heatmap(
        heatmap_data,
        cmap="viridis",
        linewidths=0,
        cbar_kws={"label": "Cluster"},
    )

    generate_event_markers(meta_data, combined_df, heatmap_data, ax)
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
    plt.title("Heatmap of Clusters with Event Markers Across Trial and Video")
    plt.savefig(graph_path)
    plt.close()  # Close the figure


def combine_dfs(meta_data):
    """
    Combine DataFrames from all videos into a single DataFrame.

    Args:
        meta_data (dict): Metadata containing video information.

    Returns:
        DataFrame: Combined DataFrame of all videos.
    """
    event_dict = {"Cluster": "mean"}
    for event in meta_data["event_columns"]:
        event_dict[event] = "max"
    event_dict["Cluster"] = "mean"

    grouped_dfs = []
    for video_name, video in meta_data["videos"].items():

        if not event_dict:  # Check if event_dict is empty
            # Perform groupby without aggregation
            video_df = (
                video["df"]
                .groupby("Group", as_index=False)
                .apply(lambda x: x)  # No aggregation, just keep the grouped data
                .assign(Index=lambda x: x.index)
            )
        else:
            # Perform groupby with aggregation
            video_df = (
                video["df"]
                .groupby("Group", as_index=False)
                .agg(event_dict)
                .assign(Index=lambda x: x.index)
            )

        video_df["Video"] = video_name
        video_df["Trial"] = video["trial"]
        grouped_dfs.append(video_df)

    return pd.concat(grouped_dfs, ignore_index=True)


def create_heatmap_plot(meta_data):
    """
    Create heatmap plots for each trial and a combined heatmap.

    Args:
        meta_data (dict): Metadata containing video information and output path.
    """
    combined_df = combine_dfs(meta_data)
    trial_names = combined_df["Trial"].unique()
    for trial in trial_names:
        video_df = combined_df[combined_df["Trial"] == trial]
        path = (
            Path(meta_data["output_path"])
            / "graphs"
            / "heatmaps"
            / f"{trial}_heatmap.png"
        )
        os.makedirs(path.parent, exist_ok=True)
        graph_heatmap(meta_data, video_df, path)

    path = Path(meta_data["output_path"]) / "graphs" / "heatmaps" / f"heatmap.png"
    os.makedirs(path.parent, exist_ok=True)

    graph_heatmap(meta_data, combined_df, path)


def graph_cluster_percentage_pie_chart(meta_data):
    """
    Generate pie charts showing the percentage of each cluster for individual videos and trials.

    Args:
        meta_data (dict): Metadata containing video information and output path.
    """
    df = combine_dfs(meta_data)

    # Graphs Individual Video Cluster Percentage
    for group_name, group_df in df.groupby("Trial"):
        for video_name, video_df in group_df.groupby("Video"):

            cluster_counts = video_df["Cluster"].value_counts(normalize=True) * 100

            def autopct_format(pct):
                total = sum(cluster_counts)
                count = int(round(pct * total / 100))
                return f"{count}%"

            labels = [f"Cluster {int(label)}" for label in cluster_counts.index]

            # Plotting the updated pie chart
            plt.figure(figsize=(8, 8))
            plt.pie(
                cluster_counts, labels=labels, autopct=autopct_format, startangle=140
            )
            plt.title(f"Percentage of Each Cluster in {video_name}")

            graph_path = (
                Path(meta_data["output_path"])
                / "graphs"
                / "cluster_percentage_graphs"
                / group_name
            )
            os.makedirs(graph_path, exist_ok=True)
            plt.savefig(graph_path / f"{video_name}.png")
            plt.close()  # Close the figure

    # Graphs Collective Cluster Percentage
    for group_name, group_df in df.groupby("Trial"):

        cluster_counts = group_df["Cluster"].value_counts(normalize=True) * 100

        def autopct_format(pct):
            total = sum(cluster_counts)
            count = int(round(pct * total / 100))
            return f"{count}%"

        labels = [f"Cluster {int(label)}" for label in cluster_counts.index]

        # Plotting the updated pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(cluster_counts, labels=labels, autopct=autopct_format, startangle=140)
        plt.title(f"Percentage of Each Cluster in {group_name}")
        graph_path = (
            Path(meta_data["output_path"])
            / "graphs"
            / "cluster_percentage_graphs"
            / group_name
        )
        os.makedirs(graph_path, exist_ok=True)
        plt.savefig(graph_path / f"{group_name}.png")
        plt.close()  # Close the figure


def graph_cluster_percentage_trial_bar_chart(meta_data):
    """
    Generate bar charts showing the percentage of each cluster for individual trials.

    Args:
        meta_data (dict): Metadata containing video information and output path.
    """
    df = combine_dfs(meta_data)

    # Iterate through groups in "Trial"
    for group_name, group_df in df.groupby("Trial"):
        # Collect data for stacked bar chart
        cluster_percentages = []
        video_names = []

        for video_name, video_df in group_df.groupby("Video"):
            video_names.append(video_name)
            # Get cluster percentages for the current video
            cluster_counts = (
                video_df["Cluster"].value_counts(normalize=True).sort_index() * 100
            )
            cluster_percentages.append(cluster_counts)

        # Convert collected data to a DataFrame for easier plotting
        stacked_data = pd.DataFrame(cluster_percentages, index=video_names).fillna(0).T

        # Plot the stacked bar chart
        plt.figure(figsize=(12, 8))
        bottom = None
        for cluster_label, percentages in stacked_data.iterrows():
            plt.bar(
                stacked_data.columns,
                percentages,
                bottom=bottom,
                label=f"Cluster {int(cluster_label)}",
            )
            bottom = percentages if bottom is None else bottom + percentages

        # Customize the chart
        plt.xlabel("Videos")
        plt.ylabel("Percentage")
        plt.title(f"Cluster Distribution per Video in {group_name}")
        plt.ylim(0, 100)  # Ensure y-axis is 0-100%
        plt.xticks(rotation=45, ha="right")  # Rotate video names for clarity
        plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Save the graph
        graph_path = (
            Path(meta_data["output_path"])
            / "graphs"
            / "cluster_percentage_graphs"
            / group_name
        )
        plt.savefig(graph_path / f"{group_name}_bar.png", bbox_inches="tight")
        os.makedirs(graph_path, exist_ok=True)
        plt.close()  # Close the figure


def graph_cluster_percentage_bar_char(meta_data):
    """
    Generate a bar chart showing the percentage of each cluster for all trials combined.

    Args:
        meta_data (dict): Metadata containing video information and output path.
    """
    df = combine_dfs(meta_data)
    cluster_percentages = []
    video_names = []

    # Iterate through groups in "Trial"
    for group_name, group_df in df.groupby("Trial"):
        # Collect data for stacked bar chart

        video_names.append(group_name)
        # Get cluster percentages for the current video
        cluster_counts = (
            group_df["Cluster"].value_counts(normalize=True).sort_index() * 100
        )
        cluster_percentages.append(cluster_counts)

    # Convert collected data to a DataFrame for easier plotting
    stacked_data = pd.DataFrame(cluster_percentages, index=video_names).fillna(0).T

    # Plot the stacked bar chart
    plt.figure(figsize=(12, 8))
    bottom = None
    for cluster_label, percentages in stacked_data.iterrows():
        plt.bar(
            stacked_data.columns,
            percentages,
            bottom=bottom,
            label=f"Cluster {int(cluster_label)}",
        )
        bottom = percentages if bottom is None else bottom + percentages

    # Customize the chart
    plt.xlabel("Videos")
    plt.ylabel("Percentage")
    plt.title(f"Cluster Distribution per Trial")
    plt.ylim(0, 100)  # Ensure y-axis is 0-100%
    plt.xticks(rotation=45, ha="right")  # Rotate video names for clarity
    plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the graph
    graph_path = Path(meta_data["output_path"]) / "graphs" / "cluster_percentage_graphs"
    os.makedirs(graph_path, exist_ok=True)

    plt.savefig(
        graph_path / "collective_bar_chart.png",
        bbox_inches="tight",
    )
    plt.close()  # Close the figure


def collect_dfs_for_freezing_graphs(meta_data):
    """
    Collect DataFrames for freezing graphs.

    Args:
        meta_data (dict): Metadata containing video information.

    Returns:
        DataFrame: Combined DataFrame of all videos with rate of change information.
    """
    dfs = []

    # Combines Video DFs into one dataframe
    for video_name, video_dict in meta_data["videos"].items():
        df = video_dict["df"]
        original_columns = meta_data["original_columns"]
        original_columns = [
            column
            for column in original_columns
            if column in df.columns and (column.endswith("_x") or column.endswith("_y"))
        ]

        original_columns += ["Frame", "Cluster", "Group"]
        df = df[original_columns].copy()

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

        # Remove NaN and infinite values from rate of change
        df["Rate_of_Change"] = df["Rate_of_Change"].replace([np.inf, -np.inf], np.nan)
        df.dropna(subset=["Rate_of_Change"], inplace=True)

        df["Video_Name"] = video_name
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    df = (
        combined_df[["Rate_of_Change", "Group", "Video_Name", "Cluster"]]
        .groupby(["Group", "Video_Name", "Cluster"])
        .sum()
        .reset_index()
    )

    return df


def plot_linear_scale(df, graph_base_path):
    """
    Plot the rate of change distribution for each cluster on a linear scale.

    Args:
        df (DataFrame): DataFrame containing rate of change information.
        graph_base_path (Path): Base path to save the graphs.
    """
    for cluster in df["Cluster"].unique():
        cluster_df = df[df["Cluster"] == cluster]
        data = cluster_df["Rate_of_Change"]

        if data.empty:
            continue  # Skip empty clusters

        sns.histplot(data, bins=30, kde=True)
        plt.title(f"Cluster {cluster} Rate of Change Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        graph_path = graph_base_path / "linear"
        graph_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(graph_path / f"cluster_{cluster}_rate_of_change_distribution.png")
        plt.close()  # Close the figure


def plot_log_scale(df, graph_base_path):
    """
    Plot the rate of change distribution for each cluster on a logarithmic scale.

    Args:
        df (DataFrame): DataFrame containing rate of change information.
        graph_base_path (Path): Base path to save the graphs.
    """
    for cluster in df["Cluster"].unique():
        cluster_df = df[df["Cluster"] == cluster]
        data = cluster_df["Rate_of_Change"]

        # Ensure only strictly positive values remain
        data = data[data > 0]

        if data.empty or len(data) < 2:  # Log scale needs at least two positive values
            continue

        # Define log-spaced bins
        bin_min, bin_max = data.min(), data.max()
        if bin_min == bin_max:  # Prevent logspace error
            bin_min, bin_max = bin_min * 0.9, bin_max * 1.1
        bins = np.logspace(np.log10(bin_min), np.log10(bin_max), num=30)

        sns.histplot(data, bins=bins, kde=False)
        plt.title(f"Cluster {cluster} Rate of Change Distribution (Log Scale)")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.xscale("log")

        graph_path = graph_base_path / "logarithmic"
        graph_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            graph_path
            / f"logarithmic_cluster_{cluster}_rate_of_change_distribution.png"
        )
        plt.close()  # Close the figure


def plot_mean_rate_of_change(df, graph_base_path):
    """
    Plot the mean rate of change for each cluster.

    Args:
        df (DataFrame): DataFrame containing rate of change information.
        graph_base_path (Path): Base path to save the graphs.
    """
    mean_rates = df.groupby("Cluster")["Rate_of_Change"].mean()

    if not mean_rates.empty:
        sns.barplot(x=mean_rates.index, y=mean_rates.values)
        plt.xlabel("Cluster")
        plt.ylabel("Mean Rate of Change")
        plt.title("Mean Rate of Change by Cluster")
        plt.savefig(graph_base_path / "mean_rate_of_change_by_cluster.png")
        plt.close()  # Close the figure


def freezing_graphs(meta_data):
    """
    Generate freezing graphs for the given metadata.

    Args:
        meta_data (dict): Metadata containing video information and output path.
    """
    df = collect_dfs_for_freezing_graphs(meta_data)
    graph_base_path = Path(meta_data["output_path"]) / "graphs" / "freezing"

    plot_linear_scale(df, graph_base_path)
    plot_log_scale(df, graph_base_path)
    plot_mean_rate_of_change(df, graph_base_path)


def graph_all(meta_data):
    """
    Generate all graphs for the given metadata.

    Args:
        meta_data (dict): Metadata containing video information and output path.
    """
    os.makedirs(Path(meta_data["output_path"]) / "graphs", exist_ok=True)

    tsne_plot(meta_data)
    create_heatmap_plot(meta_data)

    graph_cluster_percentage_pie_chart(meta_data)
    graph_cluster_percentage_trial_bar_chart(meta_data)

    if meta_data["experiment"] == "fear_voiding":
        graph_cluster_percentage_bar_char(meta_data)
        freezing_graphs(meta_data)


if __name__ == "__main__":
    file_path = "/home/thomas/washu/behavior_clustering/outputs/fear_voiding_8_frames_reduced4x_pca_rotated_4/meta_data.pkl"

    with open(file_path, "rb") as file:
        meta_data = pickle.load(file)

    graph_all(meta_data)
