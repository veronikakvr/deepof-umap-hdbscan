### created by V. Kovarova August 2025
import os

# Set environment variables for single-threading BEFORE importing numpy/scipy or related libs
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Now import the rest of your libraries that depend on numpy/scipy
import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import kruskal
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'   # Just the message, no extra metadata
)


# --- Configuration Parameters ---

DATA_PATH_FEMALES = "path/to/females.csv"
DATA_PATH_MALES = "path/to/males.csv"
INTERVAL_LENGTH_SEC = 2

UMAP_PARAMS = dict(
    n_components=2,
    n_neighbors=25,
    min_dist=0.1,
    metric='euclidean',
    random_state=42,
    verbose=True,
)

HDBSCAN_PARAMS = dict(
    min_cluster_size=500,
    min_samples=90,
    gen_min_span_tree=True  # <-- Enable MST generation for DBCV calculation
)


# --- Functions ---

def load_and_combine_data(female_path, male_path):
    """
    Load female and male CSVs, add Sex column, combine and convert Time column to timedelta.
    Handles file existence and basic errors.
    """
    if not os.path.isfile(female_path):
        logger.error(f"Female data file not found: {female_path}")
        raise FileNotFoundError(f"Female data file not found: {female_path}")
    if not os.path.isfile(male_path):
        logger.error(f"Male data file not found: {male_path}")
        raise FileNotFoundError(f"Male data file not found: {male_path}")

    df_females = pd.read_csv(female_path)
    df_males = pd.read_csv(male_path)

    for df, sex in [(df_females, 'Female'), (df_males, 'Male')]:
        if 'Time' not in df.columns:
            logger.error(f"Column 'Time' not found in {sex} data")
            raise ValueError(f"Column 'Time' not found in {sex} data")

    df_females['Sex'] = 'Female'
    df_males['Sex'] = 'Male'

    combined_df = pd.concat([df_females, df_males], ignore_index=True)

    # Ensure 'Time' is convertable to timedelta
    try:
        combined_df['Time'] = pd.to_timedelta(combined_df['Time'])
    except Exception as e:
        logger.error(f"Error converting 'Time' to timedelta: {e}")
        raise

    return combined_df


def create_intervals(df, interval_seconds=2):
    if 'Time' not in df.columns:
        logger.error("Column 'Time' not found in dataframe")
        raise ValueError("Column 'Time' not found in dataframe")

    df['Interval_bin'] = (df['Time'].dt.total_seconds() // interval_seconds).astype(int)
    df['Interval_start'] = pd.to_timedelta(df['Interval_bin'] * interval_seconds, unit='s')
    df['Interval_end'] = pd.to_timedelta((df['Interval_bin'] + 1) * interval_seconds, unit='s')
    df['Interval_label'] = df['Interval_start'].astype(str) + ' - ' + df['Interval_end'].astype(str)
    return df


def aggregate_behaviors(df, behavior_cols):
    group_cols = ['Interval_bin', 'Interval_label', 'Sex', 'experimental_id', 'Geno']
    for col in group_cols + behavior_cols:
        if col not in df.columns:
            logger.error(f"Column '{col}' expected for aggregation not found in dataframe")
            raise ValueError(f"Column '{col}' expected for aggregation not found in dataframe")
    agg_df = df.groupby(group_cols)[behavior_cols].mean().reset_index()
    return agg_df


def preprocess_and_impute(df, exclude_ids=None, exclude_geno=None, last_label=None):
    filtered_df = df.copy()
    if exclude_ids:
        filtered_df = filtered_df[~filtered_df['experimental_id'].isin(exclude_ids)]
    if exclude_geno:
        filtered_df = filtered_df[filtered_df['Geno'] != exclude_geno]
    if last_label:
        filtered_df = filtered_df[filtered_df['Interval_label'] <= last_label]

    filtered_df = filtered_df.sort_values(['experimental_id', 'Interval_label']).reset_index(drop=True)

    excluded_cols = ['Interval_label', 'experimental_id', 'Interval_bin', 'B_speed', 'Geno', 'Sex']
    behavior_cols = [col for col in filtered_df.columns if col not in excluded_cols]

    if not behavior_cols:
        logger.error("No behavior columns found for imputation")
        raise ValueError("No behavior columns found for imputation")

    imputer = IterativeImputer(random_state=42)
    filtered_df[behavior_cols] = imputer.fit_transform(filtered_df[behavior_cols])

    return filtered_df, behavior_cols


def scale_features(df, behavior_cols):
    for col in behavior_cols:
        if col not in df.columns:
            logger.error(f"Behavior column '{col}' missing for scaling")
            raise ValueError(f"Behavior column '{col}' missing for scaling")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[behavior_cols])
    return scaled


def compute_umap_embedding(scaled_data, umap_params):
    reducer = umap.UMAP(**umap_params)
    embedding = reducer.fit_transform(scaled_data)
    return embedding


def perform_hdbscan_clustering(embedding, hdbscan_params):
    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    labels = clusterer.fit_predict(embedding)
    return labels, clusterer


def add_cluster_labels(df, labels):
    df['Cluster'] = labels
    return df


def save_results(df, output_path):
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {e}")
        raise


# --- Additional Features ---

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def assess_cluster_quality(clusterer, embedding):
    """
    Assess cluster quality using multiple metrics:
    - HDBSCAN cluster persistence
    - Silhouette Score (excluding noise)
    - Davies-Bouldin Index (excluding noise)
    - Calinski-Harabasz Index (excluding noise)
    - HDBSCAN Density-Based Clustering Validation (DBCV) index via relative_validity_

    Requires that the clusterer is fit with gen_min_span_tree=True to enable DBCV computation.
    """
    persistence = clusterer.cluster_persistence_
    logger.info(f"Cluster persistence scores: {persistence}")

    labels = clusterer.labels_
    mask = labels != -1  # filter out noise points
    labels_non_noise = labels[mask]
    embedding_non_noise = embedding[mask]

    if len(set(labels_non_noise)) > 1:
        sil_score = silhouette_score(embedding_non_noise, labels_non_noise)
        dbi_score = davies_bouldin_score(embedding_non_noise, labels_non_noise)
        ch_score = calinski_harabasz_score(embedding_non_noise, labels_non_noise)

        logger.info(f"Silhouette Score (excluding noise): {sil_score:.4f}")
        logger.info(f"Davies-Bouldin Index (excluding noise): {dbi_score:.4f} (lower is better)")
        logger.info(f"Calinski-Harabasz Index (excluding noise): {ch_score:.4f} (higher is better)")
    else:
        logger.info("Cluster quality metrics: Not applicable (only one cluster excluding noise)")

    # Attempt to log HDBSCAN DBCV index
    if hasattr(clusterer, 'relative_validity_') and clusterer.relative_validity_ is not None:
        logger.info(f"HDBSCAN DBCV (relative_validity_): {clusterer.relative_validity_:.4f}")
    else:
        logger.warning(
            "HDBSCAN DBCV score (relative_validity_) not available. "
            "Ensure you set 'gen_min_span_tree=True' in HDBSCAN parameters."
        )

    return persistence


def perform_stat_tests(df, behavior_cols):
    clusters = df['Cluster'].unique()
    clusters = [c for c in clusters if c != -1]
    logger.info("Performing Kruskal-Wallis and Dunn's tests:")

    kruskal_results = {}
    dunn_results = {}

    for behavior in behavior_cols:
        groups = [df[df['Cluster'] == c][behavior].values for c in clusters]
        kw_stat, kw_p = kruskal(*groups)
        kruskal_results[behavior] = (kw_stat, kw_p)
        logger.info(f"Kruskal-Wallis for {behavior}: stat={kw_stat:.3f}, p={kw_p:.4f}")

        if kw_p < 0.05:
            data_for_dunn = df[df['Cluster'].isin(clusters)][[behavior, 'Cluster']]
            dunn = sp.posthoc_dunn(data_for_dunn, val_col=behavior, group_col='Cluster', p_adjust='bonferroni')
            dunn_results[behavior] = dunn
            logger.info(f"Dunn's test results for {behavior}:\n{dunn}\n")

    return kruskal_results, dunn_results


def plot_cluster_profiles(df, behavior_cols):
    sns.set(style="whitegrid")

    # Ensure 'Cluster' is categorical for clean plotting
    df['Cluster'] = df['Cluster'].astype('category')

    # Prepare palette dict with noise (-1) in grey, others from palette
    clusters = sorted(df['Cluster'].cat.categories.tolist(), key=lambda x: (x == -1, x))
    has_noise = (-1 in clusters)

    # Remove noise to assign palette colors to other clusters first
    non_noise_clusters = [c for c in clusters if c != -1]

    # Choose palette name depending on number of clusters
    palette_name = "tab10" if len(non_noise_clusters) <= 10 else "tab20"
    base_colors = sns.color_palette(palette_name, n_colors=len(non_noise_clusters))

    # Map cluster label to color
    palette_dict = {cluster: color for cluster, color in zip(non_noise_clusters, base_colors)}

    # Assign grey color to noise (-1)
    if has_noise:
        palette_dict[-1] = "gray"

    # Plot violin plots
    for behavior in behavior_cols:
        plt.figure(figsize=(10, 6))
        sns.violinplot(
            x='Cluster', y=behavior, data=df, hue='Cluster',
            palette=palette_dict, inner='quartile', dodge=False, legend=False
        )
        plt.title(f"Violin Plot of {behavior} by Cluster")
        plt.xlabel("Cluster")
        plt.ylabel(behavior)
        plt.tight_layout()
        plt.show()

    # Plot UMAP scatter with same palette
    if 'UMAP_1' in df.columns and 'UMAP_2' in df.columns:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x='UMAP_1', y='UMAP_2', hue='Cluster', data=df,
            palette=palette_dict, legend='full', alpha=0.7, s=50
        )

        cluster_centers = df.groupby('Cluster')[['UMAP_1', 'UMAP_2']].mean()
        for cluster_label, (x, y) in cluster_centers.iterrows():
            plt.text(x, y, f"Cluster {cluster_label}", fontsize=12, weight='bold')

        plt.title("UMAP Embedding Colored by Cluster")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    else:
        logger.warning("UMAP embedding columns not found in dataframe, skipping scatter plot.")

def plot_clusters_without_noise(df):
    cluster_df = df[df['Cluster'] != -1]

    if cluster_df.empty:
        logger.warning("No non-noise clusters found to plot.")
        return

    plt.figure(figsize=(10, 8))
    n_clusters = len(cluster_df['Cluster'].unique())

    palette_name = "tab10" if n_clusters <= 10 else "tab20"
    palette = sns.color_palette(palette_name, n_clusters)

    sns.scatterplot(
        x='UMAP_1',
        y='UMAP_2',
        hue='Cluster',
        data=cluster_df,
        palette=palette,
        legend='full',
        alpha=0.8,
        s=40
    )

    cluster_centers = cluster_df.groupby('Cluster')[['UMAP_1', 'UMAP_2']].mean()
    for cluster_label, (x, y) in cluster_centers.iterrows():
        plt.text(x, y, f"Cluster {cluster_label}", fontsize=12, weight='bold')

    plt.title(f"UMAP Embedding of Clusters (excluding Noise), Clusters: {n_clusters}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# --- Main pipeline ---

def main(exclude_ids=None, exclude_geno=None, last_label=None):
    logger.info("Loading data...")
    combined_df = load_and_combine_data(DATA_PATH_FEMALES, DATA_PATH_MALES)

    logger.info("Creating intervals...")
    combined_df = create_intervals(combined_df, INTERVAL_LENGTH_SEC)

    behavior_cols = [
        'B_W_nose2nose', 'B_W_sidebyside', 'B_W_sidereside', 'B_W_nose2tail', 'B_W_nose2body',
        'B_W_following', 'B_climb-arena', 'B_sniff-arena', 'B_immobility', 'B_stat-lookaround',
        'B_stat-active', 'B_stat-passive', 'B_moving', 'B_sniffing', 'B_speed'
    ]

    logger.info("Aggregating behaviors...")
    agg_df = aggregate_behaviors(combined_df, behavior_cols)

    logger.info("Preprocessing and imputing missing data...")
    filtered_df, behavior_cols_filtered = preprocess_and_impute(
        agg_df,
        exclude_ids=exclude_ids or ['ID63', 'ID214'],
        exclude_geno=exclude_geno or 'atg7OE',
        last_label=last_label or "0 days 00:09:58 - 0 days 00:10:00"
    )

    logger.info("Scaling features...")
    scaled_data = scale_features(filtered_df, behavior_cols_filtered)

    logger.info("Computing UMAP embedding...")
    embedding = compute_umap_embedding(scaled_data, UMAP_PARAMS)

    filtered_df['UMAP_1'] = embedding[:, 0]
    filtered_df['UMAP_2'] = embedding[:, 1]

    logger.info("Performing HDBSCAN clustering...")
    labels, clusterer = perform_hdbscan_clustering(embedding, HDBSCAN_PARAMS)

    logger.info("Adding cluster labels to dataframe...")
    filtered_df = add_cluster_labels(filtered_df, labels)

    logger.info("Plotting clusters excluding noise...")
    plot_clusters_without_noise(filtered_df)

    logger.info("Assessing cluster quality...")
    assess_cluster_quality(clusterer, embedding)

    logger.info("Performing statistical tests on clusters...")
    kruskal_results, dunn_results = perform_stat_tests(filtered_df, behavior_cols_filtered)

    logger.info("Plotting clustered behavioral profiles...")
    plot_cluster_profiles(filtered_df, behavior_cols_filtered)

    save_results(filtered_df, "clustered_results_enriched.csv")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
