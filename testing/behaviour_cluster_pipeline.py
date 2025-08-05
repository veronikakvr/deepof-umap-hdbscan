"""
Animal Behavioral Clustering Pipeline
------------------------------------
A modular pipeline for behavioral time-series clustering, interpretation,
and visualization using UMAP + HDBSCAN, with statistical assessment and
plots for visualisation.

Author: Veronika Kovarova
Date: 2024-06-16
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Set single-thread threading for reproducibility
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# -----------------------------
# 1. LOAD AND PREPARE DATASETS
# -----------------------------
def load_and_combine_data(female_path, male_path):
    """
    Loads and combines female and male datasets, adds a 'Sex' column.
    """
    df_f = pd.read_csv(female_path)
    df_m = pd.read_csv(male_path)
    df_f['Sex'] = 'Female'
    df_m['Sex'] = 'Male'
    df = pd.concat([df_f, df_m], ignore_index=True)
    return df

def bin_time_column(df, interval_length=2):
    """
    Adds time-bin columns (Interval_bin, Interval_start, Interval_end, Interval_label)
    """
    df['Time'] = pd.to_timedelta(df['Time'])
    df['Interval_bin'] = (df['Time'].dt.total_seconds() // interval_length).astype(int)
    df['Interval_start'] = pd.to_timedelta(df['Interval_bin'] * interval_length, unit='s')
    df['Interval_end'] = pd.to_timedelta((df['Interval_bin'] + 1) * interval_length, unit='s')
    df['Interval_label'] = (df['Interval_start'].astype(str) + ' - ' +
                            df['Interval_end'].astype(str))
    return df

# --------
# 2. AGGREGATION AND IMPUTATION
# --------
def aggregate_behaviors(df, behavior_cols):
    """
    Aggregates behaviors by time bin, experimental_id, Geno, Sex.
    """
    dfagg = (
        df.groupby(['Interval_bin', 'Interval_label', 'Sex', 'experimental_id', 'Geno'])[behavior_cols]
        .mean()
        .reset_index()
    )
    return dfagg

def filter_and_impute(dfagg, behavior_cols, last_label, drop_ids=None, drop_gens=None):
    """
    Filters, sorts, and imputes missing data in behavior columns.
    """
    if drop_ids is None:
        drop_ids = []
    if drop_gens is None:
        drop_gens = []
    df = dfagg[
        (~dfagg['experimental_id'].isin(drop_ids))
        & (~dfagg['Geno'].isin(drop_gens))
        & (dfagg['Interval_label'] <= last_label)
    ].sort_values(['experimental_id', 'Interval_label']).reset_index(drop=True)
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    imputer = IterativeImputer(random_state=42)
    df[behavior_cols] = imputer.fit_transform(df[behavior_cols])
    return df

def scale_features(df, behavior_cols):
    """
    Standardizes feature columns.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[behavior_cols])
    return df_scaled

# --------
# 3. DIMENSIONALITY REDUCTION AND CLUSTERING
# --------
def cluster_behaviors(df_scaled, min_cluster_size=500, min_samples=90):
    """
    Embeds and clusters data using UMAP and HDBSCAN.
    Returns embedding, labels, clusterer object.
    """
    import umap.umap_ as umap
    import hdbscan
    reducer = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.1,
                        metric='euclidean', random_state=42, verbose=True)
    embedding = reducer.fit_transform(df_scaled)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(embedding)
    return embedding, labels, clusterer

# --------
# 4. CLUSTER STATISTICS & VALIDATION
# --------
def assess_clusters(embedding, labels):
    """
    Computes cluster quality indices.
    """
    from hdbscan.validity import validity_index
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    mask = labels != -1
    d64 = np.asarray(embedding, dtype=np.float64)
    n_clusters = len(np.unique(labels[mask]))
    stats = {}
    stats['dbcv'] = validity_index(d64, labels)
    if n_clusters > 1:
        stats['silhouette'] = silhouette_score(embedding[mask], labels[mask])
        stats['davies_bouldin'] = davies_bouldin_score(embedding[mask], labels[mask])
        stats['calinski_harabasz'] = calinski_harabasz_score(embedding[mask], labels[mask])
    else:
        stats['silhouette'] = np.nan
        stats['davies_bouldin'] = np.nan
        stats['calinski_harabasz'] = np.nan
    return stats

# --------
# 5. STATISTICAL TESTS FOR CLUSTER DIFFERENCES
# --------
def kruskal_dunn_tests(df, behavior_cols):
    """
    Kruskal-Wallis + Dunn's post hoc for all behavioral variables.
    Returns a dataframe of results.
    """
    from scipy.stats import kruskal
    import scikit_posthocs as sp
    from statsmodels.stats.multitest import multipletests
    results = []
    clusters = sorted(df['Cluster'].unique())
    for col in behavior_cols:
        # Kruskal-Wallis
        groups = [group[col].values for name, group in df.groupby('Cluster')]
        _, p_kruskal = kruskal(*groups)
        # Dunn's
        dunn = sp.posthoc_dunn(df, val_col=col, group_col='Cluster', p_adjust='bonferroni')
        sig_pairs = [(i, j) for i in dunn.index for j in dunn.columns if i < j and dunn.loc[i, j] < 0.05]
        # save
        results.append({
            'feature': col,
            'kruskal_p': p_kruskal,
            'dunn_sig_pairs': sig_pairs
        })
    result_df = pd.DataFrame(results)
    # Bonferroni correction for Kruskal-Wallis
    _, p_bonf, _, _ = multipletests(result_df['kruskal_p'], method='bonferroni')
    result_df['kruskal_p_bonf'] = p_bonf
    return result_df

# --------
# 6. VISUALIZATION (EXAMPLES)
# --------
def plot_violin_dunn(df, col, sig_pairs, dunn_results=None):
    """
    Violin plot for a single variable with annotation for significant Dunn's pairs.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    ax = sns.violinplot(x='Cluster', y=col, data=df, inner='box', palette='Set2')
    plt.title(f'Distribution of {col} across clusters')
    plt.xlabel('Cluster')
    plt.ylabel(col)
    # Annotate significant pairs
    if sig_pairs:
        y_max = df[col].max()
        y_offset = (y_max - df[col].min()) * 0.07
        for idx, (i, j) in enumerate(sig_pairs):
            x1, x2 = i, j
            y = y_max + y_offset * (idx + 1)
            ax.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], lw=1.5, c='k')
            ax.text((x1 + x2) / 2, y + 0.01, "*", ha='center', va='bottom', color='k')
    plt.tight_layout()
    plt.show()

def plot_umap_clusters(embedding, df, cluster_to_color):
    """
    Scatterplot of UMAP embedding colored by cluster and/or condition.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    mask = df['Cluster'] != -1
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding[mask, 0],
        y=embedding[mask, 1],
        hue=df.loc[mask, 'Cluster'],
        style=df.loc[mask, 'Geno'],
        palette=cluster_to_color,
        alpha=0.7
    )
    plt.title('Clusters by Condition in UMAP space (Outliers Excluded)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.show()

# More plotting utilities could be added similarly...

# ----------------
# 7. MAIN PIPELINE
# ----------------
if __name__ == "__main__":
    # ---- User input ----
    female_path = "data/female_data.csv"  # <-- update with your paths
    male_path = "data/male_data.csv"

    behavior_cols = [
        'B_W_nose2nose', 'B_W_sidebyside', 'B_W_sidereside', 'B_W_nose2tail',
        'B_W_nose2body', 'B_W_following', 'B_climb-arena', 'B_sniff-arena',
        'B_immobility', 'B_stat-lookaround', 'B_stat-active', 'B_stat-passive',
        'B_moving', 'B_sniffing', 'B_speed'
    ]

    last_label = "0 days 00:09:58 - 0 days 00:10:00"
    flagged_ids = ['ID63', 'ID214']
    flagged_gens = ['atg7OE']

    # ---- 1. Load and preprocess ----
    df_raw = load_and_combine_data(female_path, male_path)
    df_binned = bin_time_column(df_raw)
    dfagg = aggregate_behaviors(df_binned, behavior_cols)

    # ---- 2. Filter, impute, scale ----
    df_imputed = filter_and_impute(dfagg, behavior_cols, last_label, drop_ids=flagged_ids, drop_gens=flagged_gens)
    df_scaled = scale_features(df_imputed, behavior_cols)

    # ---- 3. UMAP + HDBSCAN ----
    embedding, labels, clusterer = cluster_behaviors(df_scaled)
    df_imputed['Cluster'] = labels

    # ---- 4. Assess clusters ----
    metrics = assess_clusters(embedding, labels)
    print("Cluster validation indices:", metrics)

    # ---- 5. Kruskal-Wallis/Dunn ----
    kruskal_results = kruskal_dunn_tests(df_imputed[df_imputed['Cluster'] != -1], behavior_cols)
    print("Kruskal-Wallis/Dunn results:\n", kruskal_results[['feature', 'kruskal_p', 'kruskal_p_bonf']])

    # ---- 6. Example plot for one variable ----
    first_row = kruskal_results.iloc[0]
    plot_violin_dunn(df_imputed[df_imputed['Cluster'] != -1], first_row['feature'], first_row['dunn_sig_pairs'])

    # ---- 7. Additional analysis and plots ----
    # Add calls to additional plotting/statistics as needed...

    print("Pipeline complete. Examine plot windows and results above.")
