import os
import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# Configuration / Defaults
# ------------------------

DEFAULT_BEHAVIOR_COLS = [
    'B_W_nose2nose', 'B_W_sidebyside', 'B_W_sidereside', 'B_W_nose2tail', 'B_W_nose2body',
    'B_W_following', 'B_climb-arena', 'B_sniff-arena', 'B_immobility', 'B_stat-lookaround',
    'B_stat-active', 'B_stat-passive', 'B_moving', 'B_sniffing', 'B_speed'
]

# ------------------------
# Functions
# ------------------------

def load_and_combine_data(female_path, male_path):
    """
    Load female and male CSV files, add Sex column, convert Time column to timedelta, combine.
    """
    df_females = pd.read_csv(female_path)
    df_males = pd.read_csv(male_path)
    df_females['Sex'] = 'Female'
    df_males['Sex'] = 'Male'
    combined_df = pd.concat([df_females, df_males], ignore_index=True)
    # Convert 'Time' to timedelta if it's not already
    if not np.issubdtype(combined_df['Time'].dtype, np.timedelta64):
        combined_df['Time'] = pd.to_timedelta(combined_df['Time'])
    return combined_df

def create_intervals(df, interval_seconds=2):
    """
    Create 2-second interval bins and labels.
    """
    df['Interval_bin'] = (df['Time'].dt.total_seconds() // interval_seconds).astype(int)
    df['Interval_start'] = pd.to_timedelta(df['Interval_bin'] * interval_seconds, unit='s')
    df['Interval_end'] = pd.to_timedelta((df['Interval_bin'] + 1) * interval_seconds, unit='s')
    df['Interval_label'] = df['Interval_start'].astype(str) + ' - ' + df['Interval_end'].astype(str)
    return df

def aggregate_behaviors(df, behavior_cols):
    """
    Aggregate behavior columns by means over groups of Interval_bin, Interval_label, Sex, experimental_id, Geno.
    """
    grouped = df.groupby(['Interval_bin', 'Interval_label', 'Sex', 'experimental_id', 'Geno'])[behavior_cols]
    agg_df = grouped.mean().reset_index()
    return agg_df

def preprocess_and_impute(df, 
                         behavior_cols=None, 
                         exclude_ids=None, 
                         exclude_geno=None, 
                         last_label=None):
    """
    Filter data by excluding experimental_ids and genotypes, limit data up to last_label,
    then impute missing values iteratively.
    """
    filtered_df = df.copy()
    if exclude_ids:
        filtered_df = filtered_df[~filtered_df['experimental_id'].isin(exclude_ids)]
    if exclude_geno:
        filtered_df = filtered_df[filtered_df['Geno'] != exclude_geno]
    if last_label:
        filtered_df = filtered_df[filtered_df['Interval_label'] <= last_label]
    filtered_df = filtered_df.sort_values(['experimental_id', 'Interval_label']).reset_index(drop=True)
    
    if behavior_cols is None:
        # Infer behavior columns by excluding known metadata columns
        exclude_cols = ['Interval_label', 'experimental_id', 'Interval_bin', 'B_speed','Geno','Sex']
        behavior_cols = [col for col in filtered_df.columns if col not in exclude_cols]
    
    imputer = IterativeImputer(random_state=42)
    filtered_df.loc[:, behavior_cols] = imputer.fit_transform(filtered_df[behavior_cols])
    return filtered_df, behavior_cols

def scale_features(df, behavior_cols):
    """
    Standard scale features columns.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[behavior_cols])
    return scaled, scaler

def compute_umap_embedding(scaled_data, umap_params=None):
    """
    Compute UMAP embedding with given parameters.
    """
    default_params = dict(
        n_components=2,
        n_neighbors=25,
        min_dist=0.1,
        metric='euclidean',
        random_state=42,
        verbose=True,
    )
    if umap_params is not None:
        default_params.update(umap_params)
    reducer = umap.UMAP(**default_params)
    embedding = reducer.fit_transform(scaled_data)
    return embedding, reducer

def perform_hdbscan_clustering(embedding, hdbscan_params=None):
    """
    Perform HDBSCAN clustering on embeddings with given params.
    """
    default_params = dict(
        min_cluster_size=500,
        min_samples=90,
    )
    if hdbscan_params is not None:
        default_params.update(hdbscan_params)
    clusterer = hdbscan.HDBSCAN(**default_params)
    labels = clusterer.fit_predict(embedding)
    return labels, clusterer

def add_cluster_labels(df, labels, label_colname='Cluster'):
    """
    Add cluster labels as a new column to the dataframe.
    """
    df[label_colname] = labels
    return df

def plot_umap_clusters(embedding, labels, title="UMAP with HDBSCAN clusters"):
    """
    Scatter plot of UMAP embeddings colored by cluster labels.
    """
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette('tab20', np.unique(labels).max() + 2)
    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1],
        hue=labels,
        palette=palette,
        legend='full',
        s=10,
        linewidth=0,
        alpha=0.8
    )
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def save_results(df, output_path):
    """
    Save dataframe to CSV.
    """
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

# ------------------------
# Main pipeline function
# ------------------------

def run_pipeline(female_csv_path,
                 male_csv_path,
                 interval_sec=2,
                 exclude_ids=None,
                 exclude_geno=None,
                 last_interval_label=None,
                 behavior_cols=None,
                 umap_params=None,
                 hdbscan_params=None,
                 save_path=None,
                 visualize=True):
    """
    Run the processing, clustering and optionally save and visualize results.
    
    Returns the processed dataframe with cluster labels and the UMAP embedding (numpy array).
    """
    print("Loading and combining data...")
    df = load_and_combine_data(female_csv_path, male_csv_path)

    print("Creating time intervals...")
    df = create_intervals(df, interval_sec)

    if behavior_cols is None:
        behavior_cols = DEFAULT_BEHAVIOR_COLS

    print("Aggregating behavior data...")
    agg_df = aggregate_behaviors(df, behavior_cols)

    print("Filtering and imputing missing data...")
    filtered_df, behavior_cols_filtered = preprocess_and_impute(
        agg_df,
        behavior_cols=behavior_cols,
        exclude_ids=exclude_ids,
        exclude_geno=exclude_geno,
        last_label=last_interval_label,
    )

    print("Scaling features...")
    scaled_data, scaler = scale_features(filtered_df, behavior_cols_filtered)

    print("Computing UMAP embedding...")
    embedding, umap_model = compute_umap_embedding(scaled_data, umap_params)

    print("Clustering with HDBSCAN...")
    labels, hdbscan_model = perform_hdbscan_clustering(embedding, hdbscan_params)

    print(f"Number of clusters found (excluding noise): {len(set(labels)) - (1 if -1 in labels else 0)}")
    filtered_df = add_cluster_labels(filtered_df, labels)

    if save_path:
        save_results(filtered_df, save_path)

    if visualize:
        plot_umap_clusters(embedding, labels)

    return filtered_df, embedding, umap_model, hdbscan_model

# ------------------------
# CLI entry point
# ------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UMAP+HDBSCAN Clustering Pipeline")
    parser.add_argument("--female_csv", required=True, help="Path to females CSV file")
    parser.add_argument("--male_csv", required=True, help="Path to males CSV file")
    parser.add_argument("--output_csv", default="clustered_results.csv", help="Path to save clustered results CSV")
    parser.add_argument("--interval_sec", type=int, default=2, help="Interval length in seconds")
    parser.add_argument("--exclude_ids", nargs='*', default=['ID63','ID214'], help="List of experimental_ids to exclude")
    parser.add_argument("--exclude_geno", default="atg7OE", help="Genotype to exclude")
    parser.add_argument("--last_label", default="0 days 00:09:58 - 0 days 00:10:00", help="Last interval label to include")
    args = parser.parse_args()

    run_pipeline(
        female_csv_path=args.female_csv,
        male_csv_path=args.male_csv,
        interval_sec=args.interval_sec,
        exclude_ids=args.exclude_ids,
        exclude_geno=args.exclude_geno,
        last_interval_label=args.last_label,
        save_path=args.output_csv,
        visualize=True,
    )
