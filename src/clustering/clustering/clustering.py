# %%%
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from umap import UMAP
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def cluster_videos_with_frame_recluster(
    meta_data: dict,
    bout_frames,
    append_results_to_df=True,
) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    import numpy as np
    import pandas as pd

    start_index = 0
    tsne_input = []
    frame_clusters = []  # To store frame-wise clustering results

    for video in meta_data["videos"].values():
        video["tsne_start"] = start_index
        video["tsne_end"] = start_index + video["df"]["Group"].nunique()
        start_index = video["tsne_end"]

        # Extract frame-wise feature data
        df_xy = video["df"][meta_data["data_columns"]]
        frame_values = df_xy.values

        # Perform clustering on individual frames
        kmeans_frame = KMeans(n_clusters=8, random_state=42)
        frame_labels = kmeans_frame.fit_predict(frame_values)
        frame_clusters.append(frame_labels)

        # Reshape for t-SNE input (combine frames into bout-level data)
        bout_values = frame_values.reshape(-1, len(df_xy.columns) * bout_frames)
        tsne_input.append(bout_values)

    # Combine all bout-level data for t-SNE
    tsne_input = np.vstack(tsne_input)

    # Apply t-SNE on bout-level data
    tsne_results = TSNE(
        n_components=2, perplexity=30, method="barnes_hut", random_state=42
    ).fit_transform(tsne_input)

    # Perform clustering on t-SNE results
    kmeans_labels = KMeans(n_clusters=8, random_state=42).fit_predict(tsne_results)

    # Create DataFrame for t-SNE results and assign cluster labels
    tsne_df = pd.DataFrame(tsne_results, columns=["TSNE_1", "TSNE_2"])
    tsne_df["Cluster"] = kmeans_labels

    for video in meta_data["videos"].values():
        for index, group in video["df"].groupby("Group"):
            if "Seconds" in group.columns:
                tsne_df.loc[video["tsne_start"] + index, "Seconds"] = group.iloc[0][
                    "Seconds"
                ]
                tsne_df.loc[video["tsne_start"] + index, "Video_Frame"] = group[
                    "Frame"
                ].iloc[0]

    # Adds clustering results to each group in the dataframe
    for video_name, video in meta_data["videos"].items():
        tsne_df.loc[video["tsne_start"] : video["tsne_end"], "video"] = video_name
        tsne_df.loc[video["tsne_start"] : video["tsne_end"], "trial"] = video["trial"]

        if append_results_to_df:
            video_results = kmeans_labels[video["tsne_start"] : video["tsne_end"]]
            video["df"]["Cluster"] = np.repeat(video_results, bout_frames)

            # Append frame-wise clusters to the dataframe
            video["df"]["Frame_Cluster"] = frame_clusters.pop(0)

    meta_data["tsne_results"] = tsne_df

    return meta_data


def cluster_videos_pre_group(
    meta_data: dict,
    bout_frames,
    append_results_to_df=True,
) -> dict:
    start_index = 0
    tsne_input = []

    for video in meta_data["videos"].values():
        video["tsne_start"] = start_index
        video["tsne_end"] = start_index + video["df"]["Group"].nunique()
        start_index = video["tsne_end"]
        df_xy = video["df"][meta_data["data_columns"]]

        new_values = df_xy.values.reshape(-1, len(df_xy.columns) * bout_frames)
        tsne_input.append(new_values)
    tsne_input = np.vstack(tsne_input)

    tsne_results = TSNE(
        n_components=2, perplexity=30, method="barnes_hut", random_state=42
    ).fit_transform(tsne_input)
    kmeans_labels = KMeans(n_clusters=8, random_state=42).fit_predict(tsne_results)

    # Create DataFrame for t-SNE results and assign cluster labels
    tsne_df = pd.DataFrame(tsne_results, columns=["TSNE_1", "TSNE_2"])
    tsne_df["Cluster"] = kmeans_labels

    for video in meta_data["videos"].values():
        for index, group in video["df"].groupby("Group"):
            if "Seconds" in group.columns:
                tsne_df.loc[video["tsne_start"] + index, "Seconds"] = group.iloc[0][
                    "Seconds"
                ]
                tsne_df.loc[video["tsne_start"] + index, "Video_Frame"] = group[
                    "Frame"
                ].iloc[0]

    # Adds clustering results to each group in the dataframe
    for video_name, video in meta_data["videos"].items():
        tsne_df.loc[video["tsne_start"] : video["tsne_end"], "video"] = video_name
        tsne_df.loc[video["tsne_start"] : video["tsne_end"], "trial"] = video["trial"]
        if append_results_to_df:
            video_results = kmeans_labels[video["tsne_start"] : video["tsne_end"]]
            video["df"]["Cluster"] = np.repeat(video_results, bout_frames)

    meta_data["tsne_results"] = tsne_df

    return meta_data


def cluster_videos_with_pca(
    meta_data: dict,
    bout_frames: int,
    append_results_to_df: bool = True,
    pca_variance: float = 0.99,  # Fraction of variance to retain in PCA
    reduction: str = "umap",  # "umap" or "tsne"
    n_clusters: int = 15,  # Number of clusters for KMeans
    determine_n_clusters: bool = False,  # Whether to graph n_clusters vs distance squared
) -> dict:
    # Prepare The Input Data
    start_index = 0
    tsne_input = []

    for video in meta_data["videos"].values():
        video["tsne_start"] = start_index
        video["tsne_end"] = start_index + video["df"]["Group"].nunique()
        start_index = video["tsne_end"]        
        # Extract (_x, _y) columns and reshape per bout_frames
        df_xy = video["df"][meta_data["data_columns"]]
        new_values = df_xy.values.reshape(-1, len(df_xy.columns) * bout_frames)
        tsne_input.append(new_values)

    tsne_input = np.vstack(tsne_input)

    columns = []
    for f in range(bout_frames):
        for col in meta_data["data_columns"]:
            columns.append(f"{col}_frame{f}")

    df_bouts = pd.DataFrame(tsne_input, columns=columns)

    # Preprocessing
    speed_cols = [
        c for c in df_bouts.columns if "Nose_speed" in c or "Hindpaw_speed" in c
    ]
    posture_cols = [c for c in df_bouts.columns if "Body_angle" in c or "Rigidity" in c]
    coords_cols = [c for c in df_bouts.columns if "_rel" in c]

    preprocessor = ColumnTransformer(
        [
            ("speed", RobustScaler(), speed_cols),
            ("posture", RobustScaler(), posture_cols),
            ("coords", RobustScaler(), coords_cols),
        ],
        remainder="passthrough",  # so you don't lose other columns
    )

    # Dimensionality Reduction
    if reduction == "umap":
        reducer = UMAP(
            n_components=2,
            n_neighbors=100,
            min_dist=0.1,
            random_state=42,
            densmap=True,
            metric="cosine",
        )
    else:
        reducer = TSNE(
            n_components=2,
            perplexity=15,
            learning_rate=150,
            max_iter=2500,
            init="pca",
            method="barnes_hut",
            random_state=42,
        )

    # Create a pipeline
    pipeline = Pipeline(
        [
            ("pre", preprocessor),
            ("pca", PCA(n_components=pca_variance, svd_solver="full")),
            ("reduce", reducer),
        ]
    )

    # Fit and transform the data
    tsne_results = pipeline.fit_transform(df_bouts)

    if determine_n_clusters:
        inertias = []
        k_values = range(1, 30)  # Try different k values

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(tsne_results)
            inertias.append(kmeans.inertia_)

        # Plot the elbow curve
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, inertias, marker='o')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (Sum of Squared Distances)')
        plt.title('Elbow Method for Optimal k')
        plt.show()

        n_clusters = int(input("Enter the optimal number of clusters: "))
        print(f"Using {n_clusters} clusters for KMeans.")

    # Cluster the data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(tsne_results)

    # Store The Results
    tsne_df = pd.DataFrame(tsne_results, columns=["TSNE_1", "TSNE_2"])
    tsne_df["Cluster"] = kmeans_labels

    # Add time and frame data for interpretability
    for video in meta_data["videos"].values():
        for index, group in video["df"].groupby("Group"):
            if "Seconds" in group.columns:
                tsne_df.loc[video["tsne_start"] + index, "Seconds"] = group.iloc[0][
                    "Seconds"
                ]
                tsne_df.loc[video["tsne_start"] + index, "Video_Frame"] = group[
                    "Frame"
                ].iloc[0]

    # Add clustering results to each video's DataFrame
    for video_name, video in meta_data["videos"].items():
        tsne_df.loc[video["tsne_start"] : video["tsne_end"], "video"] = video_name
        tsne_df.loc[video["tsne_start"] : video["tsne_end"], "trial"] = video["trial"]
        if append_results_to_df:
            video_results = kmeans_labels[video["tsne_start"] : video["tsne_end"]]
            video["df"]["Cluster"] = np.repeat(video_results, bout_frames)

    # Store results in meta_data
    meta_data["tsne_results"] = tsne_df

    return meta_data
