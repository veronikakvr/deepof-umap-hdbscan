import os
import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
)

# --- Functions ---

def load_and_combine_data(female_path, male_path):
    df_females = pd.read_csv(female_path)
    df_males = pd.read_csv(male_path)
    df_females['Sex'] = 'Female'
    df_males['Sex'] = 'Male'
    combined_df = pd.concat([df_females, df_males], ignore_index=True)
    combined_df['Time'] = pd.to_timedelta(combined_df['Time'])
    return combined_df

def create_intervals(df, interval_seconds=2):
    df['Interval_bin'] = (df['Time'].dt.total_seconds() // interval_seconds).astype(int)
    df['Interval_start'] = pd.to_timedelta(df['Interval_bin'] * interval_seconds, unit='s')
    df['Interval_end'] = pd.to_timedelta((df['Interval_bin'] + 1) * interval_seconds, unit='s')
    df['Interval_label'] = df['Interval_start'].astype(str) + ' - ' + df['Interval_end'].astype(str)
    return df

def aggregate_behaviors(df, behavior_cols):
    agg_df = df.groupby(['Interval_bin', 'Interval_label', 'Sex', 'experimental_id', 'Geno'])[behavior_cols].mean().reset_index()
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
    behavior_cols = [col for col in filtered_df.columns if col not in ['Interval_label', 'experimental_id', 'Interval_bin','B_speed','Geno','Sex']]
    imputer = IterativeImputer(random_state=42)
    filtered_df[behavior_cols] = imputer.fit_transform(filtered_df[behavior_cols])
    return filtered_df, behavior_cols

def scale_features(df, behavior_cols):
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
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

# Add your visualization, statistical analysis functions here...

# --- Main pipeline ---

def main():
    print("Loading data...")
    combined_df = load_and_combine_data(DATA_PATH_FEMALES, DATA_PATH_MALES)

    print("Creating intervals...")
    combined_df = create_intervals(combined_df, INTERVAL_LENGTH_SEC)

    behavior_cols = [ 
        # list your behavior columns here, e.g.,
        'B_W_nose2nose', 'B_W_sidebyside', 'B_W_sidereside', 'B_W_nose2tail', 'B_W_nose2body',
        'B_W_following', 'B_climb-arena', 'B_sniff-arena', 'B_immobility', 'B_stat-lookaround',
        'B_stat-active', 'B_stat-passive', 'B_moving', 'B_sniffing', 'B_speed'
    ]
    print("Aggregating behaviors...")
    agg_df = aggregate_behaviors(combined_df, behavior_cols)

    print("Preprocessing and imputing missing data...")
    filtered_df, behavior_cols_filtered = preprocess_and_impute(
        agg_df,
        exclude_ids=['ID63', 'ID214'],
        exclude_geno='atg7OE',
        last_label="0 days 00:09:58 - 0 days 00:10:00"
    )

    print("Scaling features...")
    scaled_data = scale_features(filtered_df, behavior_cols_filtered)

    print("Computing UMAP embedding...")
    embedding = compute_umap_embedding(scaled_data, UMAP_PARAMS)

    print("Performing HDBSCAN clustering...")
    labels, clusterer = perform_hdbscan_clustering(embedding, HDBSCAN_PARAMS)

    print("Adding cluster labels to dataframe...")
    filtered_df = add_cluster_labels(filtered_df, labels)

    # Save or visualize results here as needed
    save_results(filtered_df, "clustered_results.csv")

    print("Pipeline complete.")

if __name__ == "__main__":
    main()
