# %%
import os
# 1. Set single-threading environment variables *before* importing numpy or related libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# %%
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import umap
import umap.umap_ as umap
import hdbscan

# %%
# Load your dataset
#df = pd.read_csv(r"/Users/veronika/ownCloud/PCA_analysis2025/SocialOF/chow/females/master_combined_females_chow_SocialOF_withoutCD1.csv")
#df_females = pd.read_csv(r"C:\Users\veronika_kovarova\ownCloud\PCA_analysis2025\SocialOF\chow\females\atg7KO\atg7KO_females_chow_SocialOF\master_combined_females_atg7KO_chow_SocialOF_FINAL.csv")
#df_males = df = pd.read_csv(r"C:\Users\veronika_kovarova\ownCloud\PCA_analysis2025\SocialOF\chow\males\atg7KO\master_combined_males_atg7KO_chow_SocialOF_FINAL.csv")

df_females = pd.read_csv(r"/Users/veronika/ownCloud/PCA_analysis2025/SocialOF/HFD/females/atg7KO/Social/master_combined_females_atg7KO_HFD_SocialOF_FINAL.csv")
df_males = df = pd.read_csv(r"/Users/veronika/ownCloud/PCA_analysis2025/SocialOF/HFD/males/atg7KO/master_combined_males_atg7KO_HFD_SocialOF_FINAL.csv")


# %%
df_females['Sex'] = 'Female'
df_males['Sex'] = 'Male'
df_combined = pd.concat([df_females, df_males], ignore_index=True)


# %%
# --- Convert 'Time' to Timedelta format ---
df_combined['Time'] = pd.to_timedelta(df_combined['Time'])

# %%
print(df_combined.tail())

# %%

# --- Define minute interval bins (in seconds) ---
#df_combined['Interval_bin'] = (df_combined['Time'].dt.total_seconds() // 2).astype(int)

# Create a label column with the start time of each 2.5-minute interval
#df_combined['Interval_label'] = pd.to_timedelta(df_combined['Interval_bin'] * 150, unit='s')

# Define the interval length in seconds
#interval_length = 2  # 2 seconds

# Compute start and end times as timedeltas
#df_combined['Interval_start'] = pd.to_timedelta(df_combined['Interval_bin'] * interval_length, unit='s')
#df_combined['Interval_end'] = pd.to_timedelta((df_combined['Interval_bin'] + 1) * interval_length, unit='s')

# Create a combined interval label as string
#df_combined['Interval_label'] = df_combined['Interval_start'].astype(str) + ' - ' + df_combined['Interval_end'].astype(str)


# %%
# Define the interval length in seconds
interval_length = 2  # 2 seconds

# Create interval bin index by dividing total seconds by interval length
df_combined['Interval_bin'] = (df_combined['Time'].dt.total_seconds() // interval_length).astype(int)

# Compute start and end of each interval
df_combined['Interval_start'] = pd.to_timedelta(df_combined['Interval_bin'] * interval_length, unit='s')
df_combined['Interval_end'] = pd.to_timedelta((df_combined['Interval_bin'] + 1) * interval_length, unit='s')

# Create human-readable interval label
df_combined['Interval_label'] = df_combined['Interval_start'].astype(str) + ' - ' + df_combined['Interval_end'].astype(str)

# %%
# --- List of columns with behaviors ---
behavior_cols = [
    'B_W_nose2nose', 'B_W_sidebyside', 'B_W_sidereside', 'B_W_nose2tail', 'B_W_nose2body',
    'B_W_following', 'B_climb-arena', 'B_sniff-arena', 'B_immobility', 'B_stat-lookaround',
    'B_stat-active', 'B_stat-passive', 'B_moving', 'B_sniffing', 'B_speed'
]


# --- Group by 'Interval_bin', 'experimental_id', 'Geno' and calculate means ---
agg_df = df_combined.groupby(['Interval_bin', 'Interval_label', 'Sex', 'experimental_id', 'Geno'])[behavior_cols].mean().reset_index()


# %%
# --- Preview ---
#print(agg_df.head()
print(agg_df.Interval_label.unique())
# --- Save to variable or export ---
averaged_behaviors = agg_df

# %%
# Example: Save agg_df to a specific path
#agg_df.to_csv(r'C:\Users\veronika_kovarova\ownCloud\PCA_analysis2025\SocialOF\chow\my_output_atg7KO_males&females_chowSocialOF.csv', index=False)


# %%
# Optional: Check for missing values
print(agg_df.isnull().sum())

# %%
df_imputed=agg_df.dropna()
print(df_imputed.isnull().sum())

# %%
###MULTIPLE IMPUTATION FOR THE MISSING DATA
# Define the last interval label to include
#last_label = "0 days 00:09:58 - 0 days 00:10:00"

#filtered_df = agg_df[
 #   (~agg_df['experimental_id'].isin(['ID63', 'ID214'])) &
  #  (agg_df['Geno'] != 'atg7OE') &
   # (agg_df['Interval_label'] <= last_label)
#]
#(agg_df['Interval_bin'] != 4), (~agg_df['Interval_bin'].isin([4, 0])), (agg_df['Geno'] != 'atg7OE')

# Get all columns except 'ID', 'Condition', 'Hour'
#behavior_cols = filtered_df.columns.difference(['Interval_label', 'experimental_id', 'Interval_bin','B_speed','Geno','Sex'])

# Or, if you want to preserve column order:
#behavior_cols = [col for col in filtered_df.columns if col not in ['Interval_label', 'experimental_id', 'Interval_bin','B_speed','Geno','Sex']]

#imputer = IterativeImputer(random_state=42)
#df_imputed = filtered_df.copy()
#df_imputed[behavior_cols] = imputer.fit_transform(filtered_df[behavior_cols])

# %%
# 2. Filter your data deterministically and sort by a stable column (recommended: 'experimental_id' and/or 'Interval_label')
last_label = "0 days 00:09:58 - 0 days 00:10:00"
filtered_df = agg_df[
    (~agg_df['experimental_id'].isin(['ID63', 'ID214'])) &
    (agg_df['Geno'] != 'atg7OE') &
    (agg_df['Interval_label'] <= last_label)
].copy()

# %%
# Deterministic sort. Replace or add columns for the most stable order possible.
filtered_df = filtered_df.sort_values(['experimental_id', 'Interval_label']).reset_index(drop=True)

# %%
# 3. Define behavioral columns for imputation/scaling
behavior_cols = [col for col in filtered_df.columns if col not in ['Interval_label', 'experimental_id', 'Interval_bin','B_speed','Geno','Sex']]


# %%
# 4. Multiple imputation
imputer = IterativeImputer(random_state=42)
df_imputed = filtered_df.copy()
df_imputed[behavior_cols] = imputer.fit_transform(filtered_df[behavior_cols])

# %%
df_imputed.tail()

# %%
# 5. Standard scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_imputed[behavior_cols])

# %%
###DIMENSIONALITY REDUCTION (UMAP)
# 6. UMAP (on sorted, imputed, and scaled features)
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=25,
    min_dist=0.1,
    metric='euclidean',
    verbose=True,
    random_state=42,
)
embedding = reducer.fit_transform(df_scaled)

# %%
# 7. HDBSCAN clustering (on same, ordered embedding)
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=500,
    min_samples=90,
    #approx_min_span_tree=True,
)
labels = clusterer.fit_predict(embedding)

# 8. Add cluster labels to (sorted) imputed DataFrame
df_imputed['Cluster'] = labels

# %%
####clustering HDBSCAN
# Cluster based on the UMAP embedding
#clusterer = hdbscan.HDBSCAN(min_cluster_size=500, min_samples=90, approx_min_span_tree=True) #(min_cluster_size=500, min_samples=90) # approx_min_span_tree=his disables a stochastic approximation step and can improve reproducibility
#labels = clusterer.fit_predict(embedding)

# Add cluster labels to dataframe
#df_imputed['Cluster'] = labels

# %%
####Behavioural profile analysis
# Group by Cluster and Condition to examine behavioral profiles
profiles = df_imputed.groupby(['Cluster', 'Geno', 'Sex', 'Interval_bin'])[behavior_cols].mean()

print(profiles)

# %%
# Contingency table: clusters vs. genotype
cluster_geno_table = pd.crosstab(df_imputed['Cluster'], df_imputed['Geno'])
print(cluster_geno_table)

# %%
# Contingency table: clusters vs. genotype
cluster_geno_table = pd.crosstab(df_imputed['Cluster'], df_imputed['Sex'])
print(cluster_geno_table)

# %%
# Contingency table: clusters vs. sex (counts for each Geno/Sex combination)
cluster_sex_table = pd.crosstab([df_imputed['Cluster'], df_imputed['Geno']], df_imputed['Sex'])
print(cluster_sex_table)

# %%
# Create the contingency table as before
cluster_sex_table = pd.crosstab([df_imputed['Cluster'], df_imputed['Geno']], df_imputed['Sex'])

# Normalize by row to get proportions within each (Cluster, Geno) group
cluster_sex_prop = cluster_sex_table.div(cluster_sex_table.sum(axis=1), axis=0)

print(cluster_sex_prop)  # Shows the proportion of each sex in each cluster/genotype group


# %%
#Print cluster characteristics
df_imputed['Cluster'] = labels  # labels from HDBSCAN

# %%
# Exclude outliers (Cluster == -1)
df_no_outliers = df_imputed[df_imputed['Cluster'] != -1]


# %%
##Summarise cluster characteristics
# Calculate mean (or median) of behavioral features per cluster
cluster_summary = df_no_outliers.groupby('Cluster')[behavior_cols].mean()
print(cluster_summary)


# %%
# Get unique cluster labels (ensure consistent order)
from pypalettes import load_cmap
cluster_labels = sorted(df_no_outliers['Cluster'].unique())
colors = [load_cmap("Tableau_10", cmap_type='discrete')(i) for i in range(len(cluster_labels))] #Tableau_10 is a discrete colormap with 10 distinct colors, Dark2
cluster_to_color = dict(zip(cluster_labels, colors))

# %%
import numpy as np
import matplotlib.pyplot as plt
from pypalettes import load_cmap


# Use df_no_outliers to compute the cluster summary
# (Assuming behavior_cols is a list of your behavioral columns)
cluster_summary = df_no_outliers.groupby('Cluster')[behavior_cols].mean()

categories = behavior_cols
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]


fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))

for idx, cluster in enumerate(cluster_summary.index):
    values = cluster_summary.loc[cluster].tolist()
    values += values[:1]
    ax.plot(angles, values, label=f'Cluster {cluster}', color=cluster_to_color[cluster], linewidth=2.5)
    ax.fill(angles, values, color=cluster_to_color[cluster], alpha=0.10)

# Set category labels and adjust their position and alignment
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, color='grey', size=10)

# Adjust label position and rotation to prevent overlap with the graph [1]
for label, angle in zip(ax.get_xticklabels(), angles):
    x_pos, y_pos = label.get_position()
    # Move labels further out by increasing the y-coordinate or using rotation
    # Adjust y_offset as needed for your specific plot [1]
    y_offset = 0.12 # Increased offset for more space

    # For angles around pi/2 and 3*pi/2 (top and bottom), adjust alignment
    if angle > np.pi/2 and angle < 3*np.pi/2:
        label.set_horizontalalignment('right')
    else:
        label.set_horizontalalignment('left')

    # Calculate new position and rotate the labels for better fit [1]
    # The rotation here is based on the angle of the spoke
    rotation_angle = np.rad2deg(angle)
    if rotation_angle > 120 and rotation_angle < 270:
        rotation_angle += 180 # Rotate 180 degrees for upside-down labels

    label.set_rotation(rotation_angle)
    label.set_verticalalignment('center') # Center vertically to rotate around its center

    # Set the position of the labels relative to the circular boundary
    ax.text(angle, ax.get_rmax() + y_offset, label.get_text(),
            ha=label.get_horizontalalignment(), va=label.get_verticalalignment(),
            rotation=rotation_angle, color='grey', size=13, weight='bold')

ax.set_xticklabels([]) # Hide the original xticklabels as we are using custom text labels

# Move legend outside the plot
ax.legend(loc='upper left', bbox_to_anchor=(1.20, 1.20), fontsize=15, frameon=False)

#plt.title(' ', size=17, y=1.10)
plt.tight_layout()
#plt.savefig("PolarPlot_malesatg7KO_chow.svg", format="svg", dpi=600)
plt.show()




# %%
import seaborn as sns
import matplotlib.pyplot as plt

#palette4clusters = ['#1571b0','#209b0c','#8E5EB9','#e273c0','#babb1c','#15bdcf'] #green purple orange dark

# Reset index to get positional indices
df_no_outliers_reset = df_no_outliers.reset_index(drop=True)
# Get the mask of non-outliers from the original DataFrame
mask_no_outliers = df_imputed['Cluster'] != -1
# Filter embedding using this boolean mask
embedding_no_outliers = embedding[mask_no_outliers]

plt.figure(figsize=(10,8))
sns.scatterplot(
    x=embedding_no_outliers[:,0], 
    y=embedding_no_outliers[:,1],
    hue=df_no_outliers_reset['Cluster'],
    style=df_no_outliers_reset['Geno'],
    palette=cluster_to_color,
    alpha=0.7
)

sns.set_style("ticks")
plt.title('Clusters by Condition in UMAP space (Outliers Excluded)')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.tight_layout()
#plt.savefig("UMAP_malesatg7KO_chow.svg", format="svg", dpi=800)
plt.show()


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Heatmap
fig, ax=plt.subplots(figsize=(18,9))
ax.set_xticklabels(categories, color='grey', size=15, weight='bold')
sns.heatmap(cluster_summary, annot=True, cmap='viridis')
plt.title('Cluster Behavioral Profiles (Heatmap)')
plt.xlabel('Behavior')
plt.ylabel('Cluster')
plt.tight_layout()
plt.savefig("Heatmap_malesatg7KO_chow.svg", format="svg", dpi=800)
plt.show()


# %%
###FIRST CHECK OF THE DBCV VALIDITY INDEX
# Ensure embedding is float64
embedding_float64 = np.asarray(embedding, dtype=np.float64)
dbcv_score = validity_index(embedding_float64, clusterer.labels_)
print(f"HDBSCAN DBCV (validity index): {dbcv_score:.3f}")

# %%
import pandas as pd
from scipy.stats import chi2_contingency

# Contingency table: clusters vs. genotype
#Comparison cluster composition by genotype
ct = pd.crosstab(df_no_outliers['Cluster'], df_no_outliers['Geno'])
chi2, p, dof, expected = chi2_contingency(ct)
print(f"Chi-square statistic for Geno: {chi2}, p-value: {p}")
# Interpretation of p-value
def print_significance(p_value, alpha=0.05):
    if p_value < alpha:
        print(f"Significant difference (p < {alpha}): {p_value}")
    else:
        print(f"No significant difference (p >= {alpha}): {p_value}")
print_significance(p)

# %%
import pandas as pd
from scipy.stats import chi2_contingency

# Contingency table: clusters vs. genotype
#Comparison cluster composition by genotype
ct = pd.crosstab(df_no_outliers['Cluster'], df_no_outliers['Sex'])
chi2, p, dof, expected = chi2_contingency(ct)
print(f"Chi-square statistic for Sex: {chi2}, p-value: {p}")
# Interpretation of p-value
def print_significance(p_value, alpha=0.05):
    if p_value < alpha:
        print(f"Significant difference (p < {alpha}): {p_value}")
    else:
        print(f"No significant difference (p >= {alpha}): {p_value}")
print_significance(p)

# %%
import pandas as pd
from scipy.stats import chi2_contingency

# Assume df_no_outliers has columns: 'Cluster', 'Geno'

clusters = df_no_outliers['Cluster'].unique()

for cluster in clusters:
    print(f"\n--- Chi-square test with Geno for Cluster {cluster} ---")
    # Create 2xN contingency table: rows = 'in cluster' vs. 'not in cluster', columns = genotypes
    in_cluster = df_no_outliers['Cluster'] == cluster
    ct = pd.crosstab(in_cluster, df_no_outliers['Geno'])

    # Check that the table has dimensions > 1
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        chi2, p, dof, expected = chi2_contingency(ct)
        print(f"Chi-square statistic: {chi2:.2f}, p-value: {p:.4g}")
        if p < 0.05:
            print("Significant difference in genotype proportions for this cluster.")
        else:
            print("No significant difference in genotype proportions for this cluster.")
    else:
        print("Not enough group/label diversity to perform test in this cluster.")


# %%
import pandas as pd
from scipy.stats import chi2_contingency

# Assume df_no_outliers has columns: 'Cluster', 'Geno'

clusters = df_no_outliers['Cluster'].unique()

for cluster in clusters:
    print(f"\n--- Chi-square test with Sex for Cluster {cluster} ---")
    # Create 2xN contingency table: rows = 'in cluster' vs. 'not in cluster', columns = genotypes
    in_cluster = df_no_outliers['Cluster'] == cluster
    ct = pd.crosstab(in_cluster, df_no_outliers['Sex'])

    # Check that the table has dimensions > 1
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        chi2, p, dof, expected = chi2_contingency(ct)
        print(f"Chi-square statistic: {chi2:.2f}, p-value: {p:.4g}")
        if p < 0.05:
            print("Significant difference in sex proportions for this cluster.")
        else:
            print("No significant difference in sex proportions for this cluster.")
    else:
        print("Not enough group/label diversity to perform test in this cluster.")


# %%
import pandas as pd
from scipy.stats import chi2_contingency

# Make sure you have a combined variable for genotype and sex
df_no_outliers['Geno_Sex'] = df_no_outliers['Geno'] + '-' + df_no_outliers['Sex']

clusters = df_no_outliers['Cluster'].unique()

for cluster in clusters:
    print(f"\n--- Chi-square test for Cluster {cluster} (Geno & Sex) ---")
    in_cluster = df_no_outliers['Cluster'] == cluster
    # contingency table: rows = in/out of cluster, columns = Geno_Sex combinations
    ct = pd.crosstab(in_cluster, df_no_outliers['Geno_Sex'])
    
    # Only do the test if both groups are present
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        chi2, p, dof, expected = chi2_contingency(ct)
        print(f"Chi-square statistic: {chi2:.2f}, p-value: {p:.4g}")
        if p < 0.05:
            print("Significant difference in Geno/Sex proportions for this cluster.")
        else:
            print("No significant difference in Geno/Sex proportions for this cluster.")
    else:
        print("Not enough diversity to perform test in this cluster.")


# %%
####TESTING FOR NORMALITY AND VARIANCE HOMOGENEITY - IF either is False use non-parametric tests

from scipy.stats import shapiro, levene
import pandas as pd

normality_results = []
variance_results = []

for col in behavior_cols:
    groups = [group[col].dropna().values for _, group in df_no_outliers.groupby('Cluster')]
    
    # Normality test for each group (Shapiro-Wilk)
    shapiro_pvals = [shapiro(g)[1] for g in groups if len(g) >= 3]  # Shapiro needs at least 3 data points
    normal = all(p > 0.05 for p in shapiro_pvals)
    
    # Equal variance test across groups (Levene's test)
    levene_stat, levene_p = levene(*groups)
    equal_var = levene_p > 0.05
    
    normality_results.append(normal)
    variance_results.append(equal_var)
    
    print(f"{col}: normal={normal}, equal_var={equal_var}, min_shapiro_p={min(shapiro_pvals):.4g}, levene_p={levene_p:.4g}")

#What This Code Does:
#    - Loops through each behavioral feature.
#    - Splits values by cluster (groupby('Cluster')).
#    - Runs Shapiro-Wilk test per cluster for normality.
#    - Runs Levene’s test across all clusters for equal variance.
#       
#       Prints:
#     Whether all groups are normal, whether variances are queal, The lowest Shapiro-Wilk p-value and Levene's p-value.


# %%
###KRUSKAL-WALLIS TEST FOR NON-PARAMETRIC COMPARISON (non-parametric test)
from scipy.stats import f_oneway, kruskal

alpha = 0.05  # significance threshold
pvals = []

# Compare behavioral features across clusters
for col in behavior_cols:
    groups = [group[col].values for name, group in df_no_outliers.groupby('Cluster')]
    stat, p = kruskal(*groups)
    pvals.append(p)
    if p < alpha:
        signif = "SIGNIFICANT"
    else:
        signif = "not significant"
    print(f"{col}: Kruskal-Wallis p-value = {p:.4g} ({signif})")

#Features with significant p-values are statistically different across clusters.


# %%
##one-way ANOVA for normally distributed data (parametric test)
#from scipy.stats import f_oneway, kruskal

#pvals = []
#for col in behavior_cols:
 #   groups = [group[col].values for name, group in df_no_outliers.groupby('Cluster')]
  #  stat, p = f_oneway(*groups)  # or use kruskal(*groups) for non-normal data
   # pvals.append(p)


# %%
#BONFERRONI CORRECTION FOR MULTIPLE TESTING
from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
# Correct p-values using FDR (Benjamini-Hochberg)
#rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')


# %%
###SUMMARISING THE RESULTS IN A TABLE - FROM PREVIOUS COMPAIRSON AND MULTIPLE TESTING CORRECTION
##Feature differences between cluster with multiple testing correction
results = pd.DataFrame({
    'feature': behavior_cols,
    'raw_p': pvals,
    'p_corrected': pvals_corrected,
    'significant': reject
})
print(results.sort_values('p_corrected'))


# %%
import scikit_posthocs as sp
import pandas as pd

# Assume df_no_outliers is your DataFrame and 'Cluster' is the cluster label column
for col in behavior_cols:
    # Dunn's test for pairwise comparisons between clusters
    p_values = sp.posthoc_dunn(df_no_outliers, val_col=col, group_col='Cluster', p_adjust='bonferroni')
    print(f"Pairwise Dunn's test for {col}:\n", p_values)


# %%
import pandas as pd
import scikit_posthocs as sp

alpha = 0.05  # significance threshold

cluster_numbers = sorted(df_no_outliers['Cluster'].unique())  # Dynamically get found clusters

for col in behavior_cols:
    dunn_results = sp.posthoc_dunn(df_no_outliers, val_col=col, group_col='Cluster', p_adjust='bonferroni')
    print(f"\nPairwise Dunn's with bonferroni correction comparisons for {col}:")

    # Compare Cluster 0 against all other clusters dynamically
    for target_cluster in cluster_numbers:
        if target_cluster == 0:
            continue
        p = dunn_results.loc[0, target_cluster]
        signif = "SIGNIFICANT" if p < alpha else "not significant"
        print(f"Cluster 0 vs Cluster {target_cluster}: p-value = {p:.4g} ({signif})")


# %%
import pandas as pd
import scikit_posthocs as sp

alpha = 0.05  # significance threshold

for col in behavior_cols:
    dunn_results = sp.posthoc_dunn(df_no_outliers, val_col=col, group_col='Cluster', p_adjust='bonferroni')
    print(f"\nAll pairwise comparisons for {col}:")
    print(dunn_results)  # This prints the full matrix of adjusted p-values

    # Optionally, highlight significant differences:
    sig = dunn_results < alpha
    print("\nSignificant pairwise comparisons (alpha=0.05):")
    print(sig)


# %%
import matplotlib.pyplot as plt
import seaborn as sns

for col in behavior_cols:
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Cluster', y=col, data=df_no_outliers, inner='box', palette='Set2')
    plt.title(f'Distribution of {col} across clusters')
    plt.xlabel('Cluster')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


# %%
def add_sig_markers(ax, pairs, y_max, y_offset=0.1):
    """
    ax: matplotlib Axes
    pairs: list of tuples (i, j), cluster indices to annotate
    y_max: maximum y value across violins (for annotation placement)
    """
    for idx, (i, j) in enumerate(pairs):
        x1, x2 = i, j
        y = y_max + y_offset * (idx + 1)
        ax.plot([x1, x1, x2, x2], [y, y+0.02, y+0.02, y], lw=1.5, c='k')
        ax.text((x1+x2)/2, y+0.01, "*", ha='center', va='bottom', color='k')

for col in behavior_cols:
    plt.figure(figsize=(8, 6))
    ax = sns.violinplot(x='Cluster', y=col, data=df_no_outliers, inner='box', palette='Set2')
    plt.title(f'Distribution of {col} across clusters')
    plt.xlabel('Cluster')
    plt.ylabel(col)

    # ---- Annotation part ----
    # Get significant pairs from your Dunn's test results:
    # dunn_results is a DataFrame with clusters as index/column and p-values as entries
    dunn_results = sp.posthoc_dunn(df_no_outliers, val_col=col, group_col='Cluster', p_adjust='bonferroni')

    # Extract significant pairs (i < j to avoid duplicates)
    sig_pairs = [(i, j) for i in dunn_results.index for j in dunn_results.columns
                 if i < j and dunn_results.loc[i, j] < 0.05]

    # Find maximum y to place the annotations above the violins
    y_max = df_no_outliers[col].max()
    add_sig_markers(ax, sig_pairs, y_max)

    plt.tight_layout()
    plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
import math

variables_to_plot = behavior_cols  # Use your full list

def add_sig_markers(ax, pairs, y_max, y_offset=0.05):
    # annotate significant pairs with horizontal bars and stars.
    for idx, (i, j) in enumerate(pairs):
        x1, x2 = i, j
        y = y_max + y_offset * (idx + 1)
        ax.plot([x1, x1, x2, x2], [y, y + y_offset/2, y + y_offset/2, y], lw=1.5, c='k')
        ax.text((x1 + x2) / 2, y + y_offset/2, "*", ha='center', va='bottom', color='k', fontsize=16)

# Set subplot layout: 5 per row, enough rows to fit all
n_cols = 5
n_vars = len(variables_to_plot)
n_rows = math.ceil(n_vars / n_cols)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=False)

# Support axs always as a flat array for arbitrary shape
axs = axs.flatten()

cluster_labels = sorted(df_no_outliers['Cluster'].unique())

for idx, col in enumerate(variables_to_plot):
    ax = axs[idx]
    sns.violinplot(x='Cluster', y=col, data=df_no_outliers, inner='box', palette='Set2', ax=ax)
    ax.set_title(col)
    ax.set_xlabel("Cluster")
    if idx % n_cols == 0:
        ax.set_ylabel("Value")
    else:
        ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    dunn_results = sp.posthoc_dunn(df_no_outliers, val_col=col, group_col='Cluster', p_adjust='bonferroni')
    sig_pairs = [
        (i, j)
        for i in range(len(cluster_labels))
        for j in range(len(cluster_labels))
        if i < j and dunn_results.loc[cluster_labels[i], cluster_labels[j]] < 0.05
    ]
    y_max = df_no_outliers[col].max()
    add_sig_markers(ax, sig_pairs, y_max)

# Remove empty subplots (if total variables < n_rows * n_cols)
for i in range(n_vars, n_rows * n_cols):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.show()


# %%
import pandas as pd
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial.distance import pdist, squareform

# Extract behavioral variables and cluster assignments
X = df_no_outliers[behavior_cols].values
labels = df_no_outliers['Cluster'].astype(str).values

# Compute the Euclidean distance matrix
dist_matrix = squareform(pdist(X, metric='euclidean'))
dm = DistanceMatrix(dist_matrix, ids=df_no_outliers.index.astype(str))

# Run PERMANOVA
result = permanova(dm, labels, permutations=999)
print(result)


# %%
import numpy as np
import pandas as pd
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests

clusters = df_no_outliers['Cluster'].unique()
pairs = [(i, j) for idx, i in enumerate(clusters) for j in clusters[idx+1:]]
pvals = []

for i, j in pairs:
    subset = df_no_outliers[df_no_outliers['Cluster'].isin([i, j])]
    X = subset[behavior_cols].values
    labels = subset['Cluster'].astype(str).values
    dist_matrix = squareform(pdist(X, metric='euclidean'))
    dm = DistanceMatrix(dist_matrix, ids=subset.index.astype(str))
    result = permanova(dm, labels, permutations=999)
    pvals.append(result['p-value'])

# Adjust for multiple comparisons
_, pvals_adj, _, _ = multipletests(pvals, method='fdr_bh')

for idx, (i, j) in enumerate(pairs):
    print(f"PERMANOVA with fdr_bh corr: Cluster {i} vs Cluster {j}: adjusted p-value = {pvals_adj[idx]:.4g}")


# %%
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial.distance import pdist, squareform

for cluster_id in df_no_outliers['Cluster'].unique():
    cluster_df = df_no_outliers[df_no_outliers['Cluster'] == cluster_id]
    
    if cluster_df['Geno'].nunique() < 2:
        continue  # Skip clusters with only one condition present
    
    X = cluster_df[behavior_cols].values
    labels = cluster_df['Geno'].values
    
    dist_matrix = squareform(pdist(X, metric='euclidean'))
    dm = DistanceMatrix(dist_matrix, ids=cluster_df.index.astype(str))
    
    result = permanova(dm, labels, permutations=999)
    print(f"Cluster {cluster_id} (GENO):\n{result}\n")


# %%
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial.distance import pdist, squareform

for cluster_id in df_no_outliers['Cluster'].unique():
    cluster_df = df_no_outliers[df_no_outliers['Cluster'] == cluster_id]
    
    if cluster_df['Sex'].nunique() < 2:
        continue  # Skip clusters with only one condition present
    
    X = cluster_df[behavior_cols].values
    labels = cluster_df['Sex'].values
    
    dist_matrix = squareform(pdist(X, metric='euclidean'))
    dm = DistanceMatrix(dist_matrix, ids=cluster_df.index.astype(str))
    
    result = permanova(dm, labels, permutations=999)
    print(f"Cluster {cluster_id} (SEX):\n{result}\n")


# %%
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests
import pandas as pd

permanova_results = []

for cluster_id in df_no_outliers['Cluster'].unique():
    cluster_df = df_no_outliers[df_no_outliers['Cluster'] == cluster_id]
    
    if cluster_df['Geno'].nunique() < 2:
        continue  # Skip clusters with only one condition present
    
    X = cluster_df[behavior_cols].values
    labels = cluster_df['Geno'].values
    
    dist_matrix = squareform(pdist(X, metric='euclidean'))
    dm = DistanceMatrix(dist_matrix, ids=cluster_df.index.astype(str))
    
    result = permanova(dm, labels, permutations=999)
    permanova_results.append({
        'Cluster': cluster_id,
        'pseudo-F': result['test statistic'],
        'p-value': result['p-value'],
        'n': len(cluster_df)
    })

# Convert to DataFrame for easier handling
permanova_df = pd.DataFrame(permanova_results)

# Apply multiple testing correction (FDR is usually preferred)
rejected, pvals_corrected, _, _ = multipletests(permanova_df['p-value'], method='fdr_bh')
permanova_df['p-value_corrected'] = pvals_corrected
permanova_df['significant'] = rejected

print(permanova_df)


# %%
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests
import pandas as pd

permanova_results = []

for cluster_id in df_no_outliers['Cluster'].unique():
    cluster_df = df_no_outliers[df_no_outliers['Cluster'] == cluster_id]
    
    if cluster_df['Sex'].nunique() < 2:
        continue  # Skip clusters with only one condition present
    
    X = cluster_df[behavior_cols].values
    labels = cluster_df['Sex'].values
    
    dist_matrix = squareform(pdist(X, metric='euclidean'))
    dm = DistanceMatrix(dist_matrix, ids=cluster_df.index.astype(str))
    
    result = permanova(dm, labels, permutations=999)
    permanova_results.append({
        'Cluster': cluster_id,
        'pseudo-F': result['test statistic'],
        'p-value': result['p-value'],
        'n': len(cluster_df)
    })

# Convert to DataFrame for easier handling
permanova_df = pd.DataFrame(permanova_results)

# Apply multiple testing correction (FDR is usually preferred)
rejected, pvals_corrected, _, _ = multipletests(permanova_df['p-value'], method='fdr_bh')
permanova_df['p-value_corrected'] = pvals_corrected
permanova_df['significant'] = rejected

print(permanova_df)


# %%
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial.distance import pdist, squareform
import pandas as pd

permanova_results = []

for cluster_id in df_no_outliers['Cluster'].unique():
    cluster_df = df_no_outliers[df_no_outliers['Cluster'] == cluster_id].copy()
    cluster_df.index = cluster_df.index.astype(str)  # Ensure index is string
    assert cluster_df.index.is_unique  # Ensure no duplicates

    X = cluster_df[behavior_cols].values
    dist_matrix = squareform(pdist(X, metric='euclidean'))
    dm = DistanceMatrix(dist_matrix, ids=cluster_df.index)

    # Test Geno
    if cluster_df['Geno'].nunique() > 1:
        grouping_geno = cluster_df.loc[list(dm.ids), 'Geno']
        result_geno = permanova(dm, grouping_geno, permutations=999)
        permanova_results.append({
            'Cluster': cluster_id,
            'Factor': 'Geno',
            'pseudo-F': result_geno['test statistic'],
            'p-value': result_geno['p-value'],
            'n': len(cluster_df)
        })

    # Test Sex
    if cluster_df['Sex'].nunique() > 1:
        grouping_sex = cluster_df.loc[list(dm.ids), 'Sex']
        result_sex = permanova(dm, grouping_sex, permutations=999)
        permanova_results.append({
            'Cluster': cluster_id,
            'Factor': 'Sex',
            'pseudo-F': result_sex['test statistic'],
            'p-value': result_sex['p-value'],
            'n': len(cluster_df)
        })

permanova_df = pd.DataFrame(permanova_results)
from statsmodels.stats.multitest import multipletests
rejected, pvals_corrected, _, _ = multipletests(permanova_df['p-value'], method='fdr_bh') #The Benjamini-Hochberg method is commonly used for multiple testing correction
permanova_df['p-value_corrected'] = pvals_corrected
permanova_df['significant'] = rejected

print(permanova_df)


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Sample data (replace with your df_no_outliers)
counts = df_no_outliers.groupby(['Cluster', 'Geno']).size().reset_index(name='Count')

clusters = counts['Cluster'].unique()
conditions = ['control', 'atg7KO']  # Adjust if you have other condition names
#count = counts['Count'].unique()

# Prepare data matrix for plotting
plot_data = counts.pivot(index='Cluster', columns='Geno', values='Count').fillna(0)

# Bar width and positions
bar_width = 0.4
x = np.arange(len(clusters))

fig, ax = plt.subplots(figsize=(10,6))


# Plot Chow bars (grey solid)
bars_chow = ax.bar(x - bar_width/2, plot_data['control'], width=bar_width, color='black', label='control')

# Plot HFD bars (black with hatch)
bars_hfd = ax.bar(x + bar_width/2, plot_data['atg7KO'], width=bar_width, color='grey', label='atg7KO', hatch='///')

# Add labels and legend
ax.set_xticks(x)
ax.set_xticklabels(clusters, size=12)
#ax.set_yticklabels(count, size=12)
ax.set_xlabel('Cluster', size=15)
ax.set_ylabel('Count', size=15)
ax.set_title('Distribution of Conditions Across Clusters')
ax.legend()
plt.tight_layout()
plt.savefig("DistributionConditions_atg7KO_males_chow.svg", format="svg", dpi=600)
plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Sample data (replace with your df_no_outliers)
counts = df_no_outliers.groupby(['Cluster', 'Sex']).size().reset_index(name='Count')

clusters = counts['Cluster'].unique()
conditions = ['Male', 'Female']  # Adjust if you have other condition names
#count = counts['Count'].unique()

# Prepare data matrix for plotting
plot_data = counts.pivot(index='Cluster', columns='Sex', values='Count').fillna(0)

# Bar width and positions
bar_width = 0.4
x = np.arange(len(clusters))

fig, ax = plt.subplots(figsize=(10,6))


# Plot Chow bars (grey solid)
bars_chow = ax.bar(x - bar_width/2, plot_data['Male'], width=bar_width, color='black', label='male')

# Plot HFD bars (black with hatch)
bars_hfd = ax.bar(x + bar_width/2, plot_data['Female'], width=bar_width, color='grey', label='female', hatch='///')

# Add labels and legend
ax.set_xticks(x)
ax.set_xticklabels(clusters, size=12)
#ax.set_yticklabels(count, size=12)
ax.set_xlabel('Cluster', size=15)
ax.set_ylabel('Count', size=15)
ax.set_title('Distribution of Conditions Across Clusters')
ax.legend()
plt.tight_layout()
#plt.savefig("DistributionConditions_atg7KO_males_chow.svg", format="svg", dpi=600)
plt.show()


# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Calculate counts per (Cluster, Geno)
counts = df_no_outliers.groupby(['Cluster', 'Geno']).size().reset_index(name='Count')

# Calculate total counts per cluster
counts['Total'] = counts.groupby('Cluster')['Count'].transform('sum')

# Calculate proportion
counts['Proportion'] = counts['Count'] / counts['Total']

# Prepare data for plotting
plot_data = counts.pivot(index='Cluster', columns='Geno', values='Proportion').fillna(0)

clusters = plot_data.index.values
bar_width = 0.4
x = np.arange(len(clusters))

fig, ax = plt.subplots(figsize=(10,6))

# Plot bars for each genotype as proportions
bars_control = ax.bar(x - bar_width/2, plot_data['control'], width=bar_width, color='black', label='control')
bars_atg7KO = ax.bar(x + bar_width/2, plot_data['atg7KO'], width=bar_width, color='grey', label='atg7KO', hatch='///')

# Add labels and legend
ax.set_xticks(x)
ax.set_xticklabels(clusters, size=12)
ax.set_xlabel('Cluster', size=15)
ax.set_ylabel('Proportion', size=15)
ax.set_title('Distribution of Conditions Across Clusters (Proportion)')
ax.legend()
plt.tight_layout()
#plt.savefig("DistributionConditions_atg7KO_males_chow_proportion.svg", format="svg", dpi=600)
plt.show()


# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Matplotlib hatches include: '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*

# Calculate counts per (Cluster, Geno)
counts = df_no_outliers.groupby(['Cluster', 'Sex']).size().reset_index(name='Count')

# Calculate total counts per cluster
counts['Total'] = counts.groupby('Cluster')['Count'].transform('sum')

# Calculate proportion
counts['Proportion'] = counts['Count'] / counts['Total']

# Prepare data for plotting
plot_data = counts.pivot(index='Cluster', columns='Sex', values='Proportion').fillna(0)

clusters = plot_data.index.values
bar_width = 0.4
x = np.arange(len(clusters))

fig, ax = plt.subplots(figsize=(10,6))

# Plot bars for each genotype as proportions
bars_control = ax.bar(x - bar_width/2, plot_data['Male'], width=bar_width, color='black', label='Males')
bars_atg7KO = ax.bar(x + bar_width/2, plot_data['Female'], width=bar_width, color='darkred', label='Females', hatch='///')

# Add labels and legend
ax.set_xticks(x)
ax.set_xticklabels(clusters, size=12)
ax.set_xlabel('Cluster', size=15)
ax.set_ylabel('Proportion', size=15)
ax.set_title('Distribution of Conditions Across Clusters (Proportion)')
ax.legend()
plt.tight_layout()
#plt.savefig("DistributionConditions_atg7KO_males_chow_proportion.svg", format="svg", dpi=600)
plt.show()


# %%
print(plot_data.columns)


# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

# Calculate counts, totals, proportions
counts = df_no_outliers.groupby(['Cluster', 'Geno', 'Sex']).size().reset_index(name='Count')
counts['Total'] = counts.groupby('Cluster')['Count'].transform('sum')
counts['Proportion'] = counts['Count'] / counts['Total']
counts['Geno_Sex'] = counts['Geno'] + '-' + counts['Sex']

# Define order, colors, and labels (IMPORTANT: match naming here and in data)
group_order = ['control-Male', 'control-Female', 'atg7KO-Male', 'atg7KO-Female']
group_colors = {
    'control-Male': "#000000",
    'control-Female': "#9D9D9D",
    'atg7KO-Male': "#444B29",
    'atg7KO-Female': "#A44316"
}
group_labels = [
    "Control Male",
    "Control Female",
    "ATG7 KO Male",
    "ATG7 KO Female"
]
color_list = [group_colors[key] for key in group_order]

# Pivot so rows=clusters, columns=Geno_Sex, values are proportions
plot_data = counts.pivot(index='Cluster', columns='Geno_Sex', values='Proportion').fillna(0)
# Reorder columns
plot_data = plot_data[group_order]  # This sets both the bar and legend stacking order[1]

ax = plot_data.plot(
    kind='bar',
    stacked=True,
    figsize=(10,6),
    color=color_list
)

# Custom legend in defined order[1][2]
handles = [
    Patch(facecolor=group_colors[key], label=label)
    for key, label in zip(group_order, group_labels)
]
ax.legend(handles=handles, title='Geno-Sex', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xlabel('Cluster', fontsize=15)
ax.set_ylabel('Proportion', fontsize=15)
ax.set_title('Proportion of Geno and Sex per Cluster', fontsize=16)
plt.tight_layout()
plt.show()


# %%
import pandas as pd

#df_bw_females = pd.read_csv(r'C:\Users\veronika_kovarova\ownCloud\PCA_analysis2025\SocialOF\chow\females\atg7KO\atg7KO_females_chow_SocialOF/bw_atg7KO_chow.csv')
#df_bw_males = pd.read_csv(r'C:\Users\veronika_kovarova\ownCloud\PCA_analysis2025\SocialOF\chow\males\atg7KO/bw_atg7KO_chow_males.csv')

df_bw_females = pd.read_csv(r'/Users/veronika/ownCloud/PCA_analysis2025/SocialOF/HFD/females/atg7KO/Social/bw_atg7KO_HFD.csv')
df_bw_males = pd.read_csv(r'/Users/veronika/ownCloud/PCA_analysis2025/SocialOF/HFD/males/atg7KO/bw_atg7KO_HFD.csv')


# %%
df_bw_females['Sex'] = 'Female'
df_bw_males['Sex'] = 'Male'
df_bw= pd.concat([df_bw_females, df_bw_males], ignore_index=True)

# %%
df_no_outliers['experimental_id'] = df_no_outliers['experimental_id'].astype(str)
df_bw['experimental_id'] = df_bw['experimental_id'].astype(str)

# %%
###TRUOBLESHOOTING
cluster_labels = df_no_outliers['Cluster']
print(cluster_labels)

# %%
import pandas as pd

# Example: Calculate the proportion of time each animal spends in each cluster
cluster_summary = (
    df_no_outliers
    .groupby(['experimental_id', df_no_outliers['Cluster']])
    .size()
    .reset_index(name='count')
)

# Calculate total intervals per animal
total_intervals = (
    df_no_outliers
    .groupby('experimental_id')
    .size()
    .reset_index(name='total')
)

# Merge totals into summary
cluster_summary = cluster_summary.merge(total_intervals, on='experimental_id')
cluster_summary['cluster_prop'] = cluster_summary['count'] / cluster_summary['total']


# %%
# Assuming df_subjects has columns: experimental_id, BW, Geno, etc.
df_merged = cluster_summary.merge(df_bw, on='experimental_id', how='left')
print(df_merged)


# %%
import statsmodels.formula.api as smf

model = smf.ols('cluster_prop ~ Geno + BW + Sex', data=df_merged).fit()
print(model.summary())


# %%
import statsmodels.formula.api as smf

# Suppose your DataFrame is final_df and has columns: experimental_id, Cluster, cluster_prop, BW, Geno

clusters = df_merged['Cluster'].unique()

for cluster_id in clusters:
    df_cluster = df_merged[df_merged['Cluster'] == cluster_id]
    print(f"\n--- Cluster {cluster_id} ---")
    model = smf.ols('cluster_prop ~ Geno + BW + Sex', data=df_cluster).fit()
    print(model.summary())


# %%
import matplotlib.pyplot as plt
import numpy as np

# Gather coefficients and CIs for each cluster
cluster_coefs = []
cluster_names = []
for cluster_id in clusters:
    df_cluster = df_merged[df_merged['Cluster'] == cluster_id]
    model = smf.ols('cluster_prop ~ Geno + BW + Sex', data=df_cluster).fit()
    params = model.params
    conf = model.conf_int()
    # Save Geno, Sex, BW only
    for var in ['Geno[T.control]', 'BW', 'Sex[T.Male]']:
        cluster_coefs.append({
            'cluster': cluster_id,
            'variable': var,
            'coef': params.get(var, np.nan),
            'err_low': conf.loc[var, 0],
            'err_high': conf.loc[var, 1]
        })
    cluster_names.append(cluster_id)

# Convert to a DataFrame for plotting
import pandas as pd
dfplot = pd.DataFrame(cluster_coefs)

variables = ['Geno[T.control]', 'BW', 'Sex[T.Male]']
fig, ax = plt.subplots(figsize=(8, 6))
for i, var in enumerate(variables):
    sub = dfplot[dfplot['variable'] == var]
    x = np.arange(len(cluster_names)) + i*0.15  # offset for clarity
    ax.errorbar(
        x, sub['coef'], 
        yerr=[sub['coef'] - sub['err_low'], sub['err_high'] - sub['coef']], 
        fmt='o', label=var)
ax.axhline(0, color='grey', linestyle='--')
ax.set_xticks(np.arange(len(cluster_names)) + 0.15)
ax.set_xticklabels(cluster_names)
ax.set_xlabel('Cluster')
ax.set_ylabel('Regression Coefficient')
ax.legend()
plt.title('Effect Size of Predictors by Cluster')
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Gather coefficients, CIs, and p-values for each cluster
cluster_stats = []
cluster_names = []
for cluster_id in clusters:
    df_cluster = df_merged[df_merged['Cluster'] == cluster_id]
    model = smf.ols('cluster_prop ~ Geno + BW + Sex', data=df_cluster).fit()
    params = model.params
    conf = model.conf_int()
    pvalues = model.pvalues
    # Save Geno, Sex, BW only
    for var in ['Geno[T.control]', 'BW', 'Sex[T.Male]']:
        cluster_stats.append({
            'cluster': cluster_id,
            'variable': var,
            'coef': params.get(var, np.nan),
            'err_low': conf.loc[var, 0],
            'err_high': conf.loc[var, 1],
            'pvalue': pvalues.get(var, np.nan)
        })
    cluster_names.append(cluster_id)
dfplot = pd.DataFrame(cluster_stats)

# Pivot for heatmap/bar plotting
pval_mat = dfplot.pivot(index='variable', columns='cluster', values='pvalue')

pval_log = -np.log10(pval_mat)

plt.figure(figsize=(10, 3))
sns.heatmap(pval_log, annot=pval_mat.round(3), fmt='', cmap='Blues', cbar_kws={"label": '-log10(p-value)'})
plt.title('-log10(p-value) of Predictors by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Predictor')
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt

fitted_vals = model.fittedvalues
residuals = model.resid

plt.scatter(fitted_vals, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


# %%
import statsmodels.api as sm
sm.graphics.plot_ccpr_grid(model)
plt.show()


# %%
import statsmodels.api as sm

rainbow_statistic, rainbow_p_value = sm.stats.diagnostic.linear_rainbow(model)
print(f'Rainbow test p-value: {rainbow_p_value}')

# If p-value < 0.05, the linearity assumption may be violated
if rainbow_p_value < 0.05:
    print("Linearity assumption may be violated (p < 0.05)")


# %%
import pandas as pd

# Calculate the proportion of each cluster at each time bin
df_counts = df_no_outliers.groupby(['Interval_bin', 'Cluster']).size().reset_index(name='count')
df_total = df_no_outliers.groupby(['Interval_bin']).size().reset_index(name='total')
df_merged = df_counts.merge(df_total, on='Interval_bin')
df_merged['cluster_prop'] = df_merged['count'] / df_merged['total']


# %%
import numpy as np
import matplotlib.pyplot as plt

# Get unique clusters and time bins
clusters = df_merged['Cluster'].unique()
intervals = sorted(df_merged['Interval_bin'].unique())

# Convert Interval_bin to angles (radians)
theta = 2 * np.pi * np.array(intervals) / len(intervals)


# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Assume df_merged has columns: ['Interval_bin', 'Cluster', 'cluster_prop']
clusters = sorted(df_merged['Cluster'].unique())
intervals = sorted(df_merged['Interval_bin'].unique())
n_intervals = len(intervals)

# Calculate total duration in seconds, assuming each bin is 2 seconds
total_seconds = (intervals[-1] + 1) * 2

# Evenly distribute theta over the full circle
theta = 2 * np.pi * np.arange(n_intervals) / n_intervals

# Color palette
#cmap = mpl.colormaps['Dark2']
cmap = load_cmap("Tableau_10", cmap_type='discrete')
colors = cmap(np.linspace(0, 1, len(clusters)))

# Set up grid: e.g., 2 rows, ceil(n_clusters/2) columns
n_clusters = len(clusters)
ncols = int(np.ceil(n_clusters / 2))
nrows = 2

fig, axs = plt.subplots(nrows, ncols, subplot_kw={'projection': 'polar'}, figsize=(7 * ncols, 7 * nrows))
axs = axs.flatten()

for idx, cluster in enumerate(clusters):
    y = np.zeros(n_intervals)
    cluster_data = df_merged[df_merged['Cluster'] == cluster].sort_values('Interval_bin')
    for _, row in cluster_data.iterrows():
        bin_idx = intervals.index(row['Interval_bin'])
        y[bin_idx] = row['cluster_prop']
    theta_closed = np.append(theta, theta[0])
    y_closed = np.append(y, y[0])
    axs[idx].plot(theta_closed, y_closed, color=cluster_to_color[cluster], alpha=0.7) #cluster_to_color
    axs[idx].set_title(f'Cluster {cluster}', va='bottom', fontsize=15)
    axs[idx].set_theta_zero_location('N')
    axs[idx].set_theta_direction(-1)

    # --- Custom major ticks at 0,2,4,6,8 min ---
    major_minutes = np.arange(0, 9, 2)  # [0, 2, 4, 6, 8]
    major_seconds = major_minutes * 60
    # Map these to theta using total_seconds
    major_tick_locs = 2 * np.pi * (major_seconds / total_seconds)
    major_tick_labels = [f"{m} min" for m in major_minutes]
    axs[idx].set_xticks(major_tick_locs)
    axs[idx].set_xticklabels(major_tick_labels, fontsize=15, color='black')

    axs[idx].tick_params(axis='x', pad=10)
    axs[idx].grid(True)
    axs[idx].tick_params(axis='y', labelsize=13, labelcolor='grey')
    axs[idx].set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])

# Hide unused subplots if clusters < nrows*ncols
for ax in axs[len(clusters):]:
    ax.axis('off')

fig.suptitle('Cluster Representation Across Time (Polar Plots)', y=1.00, fontsize=16)
plt.tight_layout()
plt.savefig("Cluster_Representation_Across_Time_moving.svg", format="svg", dpi=600)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- 1. Bin the data into 30s bins ---
df_merged['Interval_bin_30s'] = (df_merged['Interval_bin'] // 15) * 30  # 15*2s = 30s

# --- 2. Aggregate by 30s bin and cluster ---
df_agg = df_merged.groupby(['Interval_bin_30s', 'Cluster'], as_index=False)['cluster_prop'].mean()

# --- 3. Define intervals and clusters from your aggregated data ---
intervals = sorted(df_agg['Interval_bin_30s'].unique())
clusters = sorted(df_agg['Cluster'].unique())
n_intervals = len(intervals)

# Evenly distribute theta over the full circle
theta = 2 * np.pi * np.arange(n_intervals) / n_intervals

# --- 4. Prepare the plot ---
#cmap = mpl.colormaps['Dark2']
cmap = load_cmap("Tableau_10", cmap_type='discrete')
colors = cmap(np.linspace(0, 1, len(clusters)))

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))

for idx, cluster in enumerate(clusters):
    y = np.zeros(n_intervals)
    cluster_data = df_agg[df_agg['Cluster'] == cluster].sort_values('Interval_bin_30s')
    for _, row in cluster_data.iterrows():
        bin_idx = intervals.index(row['Interval_bin_30s'])
        y[bin_idx] = row['cluster_prop']
    theta_closed = np.append(theta, theta[0])
    y_closed = np.append(y, y[0])
    ax.plot(theta_closed, y_closed, color=cluster_to_color[cluster], linewidth=3, label=f'Cluster {cluster}', alpha=0.9)
    ax.fill(theta_closed, y_closed, color=cluster_to_color[cluster], alpha=0.15)

# --- 5. Set major ticks at 0,2,4,6,8 min based on real data ---

# Find the maximum time in seconds from your 30s bins
max_time_sec = intervals[-1] + 30  # Last bin's end (since bins are labeled by start)
total_minutes = max_time_sec // 60

# Define major tick minutes (0,2,4,6,8) within your data's range
major_minutes = np.arange(0, min(10, total_minutes + 1), 2)  # [0, 2, 4, 6, 8] (up to your data's end)
major_seconds = major_minutes * 60

# Map these to theta using the total duration
tick_locs = 2 * np.pi * (major_seconds / max_time_sec)
tick_labels = [f"{m} min" for m in major_minutes]

ax.set_xticks(tick_locs)
ax.set_xticklabels(tick_labels, fontsize=15, color='black')

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.tick_params(axis='x', pad=10)
ax.grid(True)
ax.tick_params(axis='y', labelsize=13, labelcolor='grey')
ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=13)
ax.set_title('Cluster Representation Across Time (Polar Plot)', va='bottom', fontsize=16)
plt.savefig("Cluster_Representation_Across_Time_SUMMARY_moving.svg", format="svg", dpi=600)
plt.tight_layout()
plt.show()


# %%
print("Intervals (30s bins):", intervals)
print("Number of intervals:", n_intervals)
print("Total seconds in data:", total_seconds)


# %%
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from hdbscan.validity import validity_index

# %%
# Ensure embedding is float64
embedding_float64 = np.asarray(embedding, dtype=np.float64)
dbcv_score = validity_index(embedding_float64, clusterer.labels_)
print(f"HDBSCAN DBCV (validity index): {dbcv_score:.3f}")

# %%
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np

# Use the embedding and labels from your full dataset
# embedding: shape (n_samples, 2) from UMAP
# labels: shape (n_samples,) from HDBSCAN

# Exclude noise points (label == -1)
mask = df_imputed['Cluster'] != -1
X_clustered = embedding[mask]
labels_clustered = df_imputed['Cluster'][mask]

unique_clusters = np.unique(labels_clustered)
n_clusters = len(unique_clusters)
print(f"Clusters present (excluding noise): {unique_clusters}")

if n_clusters > 1:
    sil_score = silhouette_score(X_clustered, labels_clustered)
    db_score = davies_bouldin_score(X_clustered, labels_clustered)
    ch_score = calinski_harabasz_score(X_clustered, labels_clustered)
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Davies-Bouldin Index: {db_score:.3f}")
    print(f"Calinski-Harabasz Index: {ch_score:.3f}")
else:
    print("Not enough clusters (after noise removal) for Silhouette/DB/CH scores.")


# %%
# --- 4. Proportion of Noise Points and Membership Scores ---
n_noise = np.sum(df_imputed['Cluster'] == -1)
prop_noise = n_noise / len(df_imputed['Cluster'])
print(f"Proportion of noise points: {prop_noise:.2%}")

if hasattr(clusterer, 'probabilities_'):
    print("Membership probabilities summary:")
    print(f"  Mean:   {np.mean(clusterer.probabilities_):.3f}")
    print(f"  Median: {np.median(clusterer.probabilities_):.3f}")
    print(f"  Min:    {np.min(clusterer.probabilities_):.3f}")
    print(f"  Max:    {np.max(clusterer.probabilities_):.3f}")
else:
    print("No membership probabilities found in the clusterer object.")

# %%
###CALCULATION OF CLUSTER TRANSITIONS - df_no_outliers

# Ensure your data is sorted by experimental_id and time (Interval_bin)
df = df_no_outliers.sort_values(['experimental_id', 'Interval_bin'])

# %%
# Function to compute transition matrix for a single experimental_id
def get_transitions(subdf):
    # Get consecutive pairs of clusters
    transitions = list(zip(subdf['Cluster'], subdf['Cluster'].shift(-1)))
    # Remove the last NaN transition
    transitions = transitions[:-1]
    return transitions

# %%
# Collect all transitions per experimental_id, including Geno and Sex
all_transitions = []
for exp_id, subdf in df.groupby('experimental_id'):
    transitions = get_transitions(subdf)
    geno = subdf['Geno'].iloc[0]
    sex = subdf['Sex'].iloc[0]
    all_transitions.extend([(exp_id, geno, sex, t[0], t[1]) for t in transitions])


# %%
# Create a DataFrame of transitions
transitions_df = pd.DataFrame(all_transitions, columns=['experimental_id', 'Geno', 'Sex', 'from_cluster', 'to_cluster'])

# %%
# To get transition counts per Geno group:
transition_counts_geno = transitions_df.groupby(['Geno', 'from_cluster', 'to_cluster']).size().reset_index(name='count')

# %%
# To get transition probabilities per Geno group:
transition_probs_geno = transition_counts_geno.copy()
transition_probs_geno['prob'] = transition_probs_geno.groupby(['Geno', 'from_cluster'])['count'].transform(lambda x: x / x.sum())

# %%
# To get transition counts per Sex group:
transition_counts_sex = transitions_df.groupby(['Sex', 'from_cluster', 'to_cluster']).size().reset_index(name='count')

# %%
# To get transition probabilities per Geno group:
transition_probs_sex = transition_counts_sex.copy()
transition_probs_sex['prob'] = transition_probs_sex.groupby(['Sex', 'from_cluster'])['count'].transform(lambda x: x / x.sum())

# %%
# Pivot to get a matrix format for a specific Geno group (e.g., 'atg7KO')
geno = 'control'
matrix = transition_probs_geno[transition_probs_geno['Geno'] == geno].pivot(index='from_cluster', columns='to_cluster', values='prob').fillna(0)

print(matrix)

# %%
# Pivot to get a matrix format for a specific Geno group (e.g., 'atg7KO')
geno = 'Male'
matrix = transition_probs_sex[transition_probs_sex['Sex'] == geno].pivot(index='from_cluster', columns='to_cluster', values='prob').fillna(0)

print(matrix)

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Group data
grouped_geno = transition_counts_geno.groupby('Geno')['count'].sum().reset_index()
geno_order = ['control', 'atg7KO']
grouped_geno = grouped_geno.set_index('Geno').loc[geno_order].reset_index()

grouped_sex = transition_counts_sex.groupby('Sex')['count'].sum().reset_index()
sex_order = ['Female', 'Male']
grouped_sex = grouped_sex.set_index('Sex').loc[sex_order].reset_index()

# Define colors and hatches
geno_colors = ['black' if g == 'control' else 'darkgrey' for g in grouped_geno['Geno']]
geno_hatches = ['' if g == 'control' else '//' for g in grouped_geno['Geno']]
sex_colors = ['black' if s == 'Female' else 'darkgrey' for s in grouped_sex['Sex']]
sex_hatches = ['' if s == 'Female' else '//' for s in grouped_sex['Sex']]

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), sharex=False)

# Plot Geno
bars1 = axes[0].bar(
    grouped_geno['Geno'],
    grouped_geno['count'],
    color=geno_colors,
    edgecolor='white',
    hatch=geno_hatches,
    width=0.6
)
axes[0].set_ylabel('Number of Transitions')
axes[0].set_title('Transitions by Geno')

# Custom legend for Geno (outside plot)
legend_elements_geno = [
    Patch(facecolor='black', edgecolor='black', label='control'),
    Patch(facecolor='darkgrey', edgecolor='white', hatch='//', label='atg7KO')
]
axes[0].legend(
    handles=legend_elements_geno, 
    title='Geno', 
    loc='center left', 
    bbox_to_anchor=(1.01, 0.5), 
    fontsize=12
)

# Plot Sex
bars2 = axes[1].bar(
    grouped_sex['Sex'],
    grouped_sex['count'],
    color=sex_colors,
    edgecolor='white',
    hatch=sex_hatches,
    width=0.6
)
axes[1].set_xlabel('Group')
axes[1].set_ylabel('Number of Transitions')
axes[1].set_title('Transitions by Sex')

# Custom legend for Sex (outside plot)
legend_elements_sex = [
    Patch(facecolor='black', edgecolor='black', label='Female'),
    Patch(facecolor='darkgrey', edgecolor='white', hatch='//', label='Male')
]
axes[1].legend(
    handles=legend_elements_sex, 
    title='Sex', 
    loc='center left', 
    bbox_to_anchor=(1.01, 0.5), 
    fontsize=12
)

plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Make space for the legends
plt.show()


# %%
# Collect all transitions per experimental_id, including Geno and Sex
all_transitions = []
for exp_id, subdf in df.groupby('experimental_id'):
    transitions = get_transitions(subdf)
    geno = subdf['Geno'].iloc[0]
    sex = subdf['Sex'].iloc[0]
    all_transitions.extend([(exp_id, geno, sex, t[0], t[1]) for t in transitions])

# Create a DataFrame of transitions
transitions_df = pd.DataFrame(all_transitions, columns=['experimental_id', 'Geno', 'Sex', 'from_cluster', 'to_cluster'])

# To get transition counts per Geno group:
transition_counts = transitions_df.groupby(['Geno', 'Sex', 'from_cluster', 'to_cluster']).size().reset_index(name='count')

# To get transition probabilities per Geno group:
transition_probs = transition_counts.copy()
transition_probs['prob'] = transition_probs.groupby(['Geno', 'Sex', 'from_cluster'])['count'].transform(lambda x: x / x.sum())


# %%
# Combine Sex and Geno for grouping
transition_counts['Sex_Geno'] = transition_counts['Sex'] + '-' + transition_counts['Geno']

# Group by the combined column and sum counts
grouped_sex_geno = transition_counts.groupby('Sex_Geno')['count'].sum().reset_index()

# Optional: Set order for plotting
group_order = ['Female-control', 'Female-atg7KO', 'Male-control', 'Male-atg7KO']
grouped_sex_geno = grouped_sex_geno.set_index('Sex_Geno').loc[group_order].reset_index()

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# Define colors and hatches in the same order as group_order
#group_colors = ['#000000', '#9D9D9D', '#444B29', '#A44316']  # adjust as needed
group_colors = ['#9D9D9D', '#A44316', '#000000', '#444B29']
group_hatches = ['', '//', '', '//']


fig, ax = plt.subplots(figsize=(7, 6))

bars = ax.bar(
    grouped_sex_geno['Sex_Geno'],
    grouped_sex_geno['count'],
    color=group_colors,
    edgecolor='white',
    width=0.6
)

# Set hatches for each bar
for bar, hatch in zip(bars, group_hatches):
    bar.set_hatch(hatch)

ax.set_xlabel('Sex-Geno Group')
ax.set_ylabel('Number of Transitions')
ax.set_title('Transitions by Sex and Geno')

# Custom legend
legend_elements = [
    Patch(facecolor='#9D9D9D', edgecolor='black', label='Female-control'),
    Patch(facecolor='#A44316', edgecolor='white', hatch='//', label='Female-atg7KO'),
    Patch(facecolor='#000000', edgecolor='black', label='Male-control'),
    Patch(facecolor='#444B29', edgecolor='white', hatch='//', label='Male-atg7KO')
]
ax.legend(
    handles=legend_elements, 
    title='Sex-Geno', 
    loc='center left', 
    bbox_to_anchor=(1.01, 0.5), 
    fontsize=12
)

plt.tight_layout()
plt.subplots_adjust(right=0.75)
plt.show()


# %%
for geno in transition_counts_geno['Geno'].unique():
    subset = transition_counts_geno[transition_counts_geno['Geno'] == geno]
    matrix = subset.pivot(index='from_cluster', columns='to_cluster', values='count').fillna(0)
    plt.figure(figsize=(6,5))
    sns.heatmap(matrix, annot=True, fmt='g', cmap='viridis')
    plt.title(f'Cluster Transition Counts - {geno}')
    plt.ylabel('From Cluster')
    plt.xlabel('To Cluster')
    plt.tight_layout()
    plt.savefig("Heatmap_transition_femalesatg7KO_chow_MOVING.svg", format="svg", dpi=600)
    plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Combine Sex and Geno for grouping (already done)
transition_counts['Sex_Geno'] = transition_counts['Sex'] + '-' + transition_counts['Geno']

# Set group order and titles
group_order = ['Female-control', 'Female-atg7KO', 'Male-control', 'Male-atg7KO']
titles = [
    'Female - Control',
    'Female - atg7KO',
    'Male - Control',
    'Male - atg7KO'
]

# Create the transition matrices for each group in the specified order
matrices = []
for group in group_order:
    matrix = transition_counts[transition_counts['Sex_Geno'] == group] \
        .pivot(index='from_cluster', columns='to_cluster', values='count').fillna(0)
    matrices.append(matrix)

# Find the overall maximum value for unified color scaling
overall_max = max(m.values.max() for m in matrices if not m.empty)

fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)

# Plot each heatmap with unified vmin/vmax and no individual colorbars
for ax, matrix, title in zip(axes.flat, matrices, titles):
    if matrix.empty:
        clusters = sorted(transition_counts['from_cluster'].unique())
        matrix = pd.DataFrame(0, index=clusters, columns=clusters)
    sns.heatmap(
        matrix,
        annot=True,
        fmt='g',
        cmap='viridis',
        ax=ax,
        vmin=0,
        vmax=overall_max,
        cbar=False
    )
    ax.set_title(title)
    ax.set_ylabel('From Cluster')
    ax.set_xlabel('To Cluster')

# Add a single, unified colorbar for all subplots
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])

# Create a ScalarMappable and colorbar
norm = mpl.colors.Normalize(vmin=0, vmax=overall_max)
sm = mpl.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])  # Only needed for older matplotlib versions

cbar = fig.colorbar(sm, cax=cbar_ax)
cbar_ax.set_ylabel('Transition Count', fontsize=14)
cbar_ax.set_xlabel('')  # No label at the bottom

plt.tight_layout(rect=[0, 0, 0.85, 1])
# plt.savefig("Heatmap_transition_by_Sex_Geno.svg", format="svg", dpi=600)
plt.show()


# %%
from pypalettes import load_cmap
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Get unique cluster labels in consistent order
cluster_labels = sorted(df_no_outliers['Cluster'].unique())
colors = [load_cmap("Tableau_10", cmap_type='discrete')(i) for i in range(len(cluster_labels))]
cluster_to_color = dict(zip(cluster_labels, colors))

# Use the same group_order as before
group_order = ['Female-control', 'Female-atg7KO', 'Male-control', 'Male-atg7KO']

for group in group_order:
    subset = transition_counts[transition_counts['Sex_Geno'] == group]

    # Build directed graph
    G = nx.DiGraph()
    for _, row in subset.iterrows():
        G.add_edge(row['from_cluster'], row['to_cluster'], weight=row['count'])

    # Node incidence: total transitions involving each cluster
    node_incidence = {}
    for node in G.nodes():
        in_weight = sum([d['weight'] for u, v, d in G.in_edges(node, data=True)])
        out_weight = sum([d['weight'] for u, v, d in G.out_edges(node, data=True)])
        node_incidence[node] = in_weight + out_weight

    # Assign node colors using the cluster_to_color mapping
    node_colors = [cluster_to_color[node] for node in G.nodes()]

    # Node sizes (same as before)
    incidences = np.array(list(node_incidence.values()))
    min_size = 500
    max_size = 3000
    if incidences.size == 0 or incidences.max() == incidences.min():
        scaled_sizes = [min_size for _ in incidences]
    else:
        scaled_sizes = min_size + (max_size - min_size) * (incidences - incidences.min()) / (incidences.max() - incidences.min())

    # Edge widths (EXACT CODE YOU PROVIDED)
    edge_weights = np.array([d['weight'] for u, v, d in G.edges(data=True)])
    min_width = 1
    max_width = 16
    if edge_weights.size == 0 or edge_weights.max() == edge_weights.min():
        edge_widths = [min_width for _ in edge_weights]
    else:
        edge_widths = min_width + (max_width - min_width) * (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())

    # Layout
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(9, 8))
    nx.draw_networkx_nodes(
        G, pos,
        node_size=scaled_sizes,
        node_color=node_colors,
        alpha=0.9
    )
    # Draw edges (no self-loops)
    edges_no_selfloops = [(u, v) for u, v in G.edges() if u != v]
    # Match edge widths to the filtered edgelist
    edge_widths_no_selfloops = [edge_widths[list(G.edges()).index(e)] for e in edges_no_selfloops]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges_no_selfloops,
        width=edge_widths_no_selfloops,
        alpha=0.7,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=22
    )
    nx.draw_networkx_labels(G, pos, font_size=13, font_weight='bold')
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=11)

    plt.title(f'Cluster Transition Network - {group}', fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# %%
# 1. Get all unique clusters (states), and sort for consistency
clusters = sorted(set(transitions_df['from_cluster']).union(set(transitions_df['to_cluster'])))
n_clusters = len(clusters)

# Create index mapping for easy access
cluster_to_idx = {c: i for i, c in enumerate(clusters)}
idx_to_cluster = {i: c for i, c in enumerate(clusters)}

# 2. Build transition matrices for each (Geno, Sex) group
groups = transitions_df.groupby(['Geno', 'Sex'])
transition_matrices = {}

for (geno, sex), group_df in groups:
    # Initialize transition count matrix
    count_matrix = np.zeros((n_clusters, n_clusters), dtype=int)
    for _, row in group_df.iterrows():
        i = cluster_to_idx[row['from_cluster']]
        j = cluster_to_idx[row['to_cluster']]
        count_matrix[i, j] += 1

    # Convert counts to probabilities per row (Markov property: ∑row = 1 for existing transitions)
    prob_matrix = np.zeros_like(count_matrix, dtype=float)
    # Avoid division by zero for states with no outgoing transitions
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = count_matrix.sum(axis=1, keepdims=True)
        prob_matrix = np.divide(count_matrix, row_sums, out=np.zeros_like(count_matrix, dtype=float), where=row_sums > 0)

    # Store (optional: as DataFrame for readability)
    prob_df = pd.DataFrame(prob_matrix, index=clusters, columns=clusters)
    transition_matrices[(geno, sex)] = prob_df

    print(f"\nTransition probability matrix for Geno={geno}, Sex={sex}:")
    print(prob_df.round(3))

# 3. (Optional) Simulate Markov chain for a group
import random

def simulate_markov_chain(prob_matrix, start_state, n_steps):
    state = start_state
    states = [state]
    for _ in range(n_steps - 1):
        next_probs = prob_matrix.loc[state].values
        if next_probs.sum() == 0:  # If no outgoing transitions, stay in place
            states.append(state)
        else:
            state = random.choices(prob_matrix.columns, weights=next_probs)[0]
            states.append(state)
    return states

# Example: simulate for one group
geno, sex = list(transition_matrices.keys())[0]
prob_df = transition_matrices[(geno, sex)]
start_cluster = clusters[0]
simulated_sequence = simulate_markov_chain(prob_df, start_cluster, 10)
print(f"\nSimulated Markov chain for {geno}, {sex}: {simulated_sequence}")

# %%
# Simulate for all (Geno, Sex) groups
n_steps = 10  # or any desired simulation length

simulated_sequences = {}  # Store results for each group

for (geno, sex), prob_df in transition_matrices.items():
    # Choose a starting state (e.g., first cluster in sorted order)
    start_cluster = clusters[0]
    sequence = simulate_markov_chain(prob_df, start_cluster, n_steps)
    simulated_sequences[(geno, sex)] = sequence

    print(f"\nSimulated Markov chain for Geno={geno}, Sex={sex}:")
    print(sequence)


# %%
import matplotlib.pyplot as plt
import seaborn as sns

for (geno, sex), matrix in transition_matrices.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f'Transition Probability Matrix: Geno={geno}, Sex={sex}')
    plt.ylabel('From Cluster')
    plt.xlabel('To Cluster')
    plt.show()


# %%
#HEATMAP OF CLUSTER REPRESENTATIONS
# CALCULATION OF THE MEANS
cluster_averages_print = df.groupby('Cluster')[behavior_cols].mean().reset_index()
print(cluster_averages_print)

# %% [markdown]
# Exploring non-linear relationships in the dataset

# %% [markdown]
# To model non-linear relationships between your response (cluster_prop) and all predictors (Geno, Sex, BW), a Generalized Additive Model (GAM) is recommended. GAMs allow you to include all variables—categoricals as linear/factor terms, and continuous variables as smooth (non-linear) splines—providing flexibility and interpretability

# %% [markdown]
# Background Context
# In your code, you're using a Generalized Additive Model (GAM) with a spline for the variable BW (Body Weight) to model how the proportion of observations in each cluster (cluster_prop) is related to BW, while also including Geno and Sex as linear (categorical) effects.

# %% [markdown]
# What Is a Spline?
# A spline is a mathematical tool for modeling complex, smooth non-linear relationships between a predictor (here: BW) and the outcome (here: cluster proportion). Unlike fitting a straight line (linear regression) or a single curve (polynomial regression), splines allow the fitted relationship to be piecewise and flexible, adapting to changing trends in your data.
# 
# A spline works by dividing the BW range into several segments ("knots").
# 
# Within each segment, it fits low-degree (usually cubic) polynomials and smoothly joins them together.
# 
# The number of degrees of freedom (df=) and the degree (degree=) determine how "wiggly" the curve is—which in your case means up to 6 pieces joined with cubic smoothness.

# %% [markdown]
# What Do the Splines Do Here?
# They let the model estimate a curved, flexible association between body weight and cluster membership.
# 
# BW does not have to affect cluster proportion in a simple linear way; the relationship can increase, decrease, flatten, or curve non-monotonically.
# 
# The spline for BW helps capture patterns like thresholds, plateaus, or inflection points—the sort of structure a linear regression would miss.

# %%
#Makes sure this variable is correctly defined
# Assuming df_subjects has columns: experimental_id, BW, Geno, etc.
df_merged = cluster_summary.merge(df_bw, on='experimental_id', how='left')
print(df_merged)


# %%
from statsmodels.gam.api import GLMGam, BSplines
import pandas as pd
import numpy as np

clusters = df_merged['Cluster'].unique()

for cluster_id in clusters:
    df_cluster = df_merged[df_merged['Cluster'] == cluster_id]
    print(f"\n--- Cluster {cluster_id} ---")

    # Convert categorical variables to dummy codes
    exog = pd.get_dummies(df_cluster[['Geno', 'Sex']], drop_first=True)

    # Spline for BW
    bs = BSplines(df_cluster[['BW']], df=[6], degree=[3])  # 6 degrees of freedom for BW
    y = df_cluster['cluster_prop']

    model = GLMGam(y, exog=exog, smoother=bs)
    result = model.fit()
    print(result.summary())


# %%
print(df_merged.columns)


# %%
import matplotlib.pyplot as plt

for cluster_id in clusters:
    df_cluster = df_merged[df_merged['Cluster'] == cluster_id]
    print(f"\n--- Cluster {cluster_id} ---")

    # Convert categorical variables to dummy codes
    exog = pd.get_dummies(df_cluster[['Geno', 'Sex']], drop_first=True)
    bs = BSplines(df_cluster[['BW']], df=[6], degree=[3])
    y = df_cluster['cluster_prop']

    model = GLMGam(y, exog=exog, smoother=bs)
    result = model.fit()
    print(result.summary())

    # Plot the partial effect of the spline on BW
    fig = result.plot_partial(0, plot_se=True)  # 0 because BW is the first (and only) smooth term
    plt.title(f'Effect of BW (spline) for Cluster {cluster_id}')
    plt.xlabel('BW')
    plt.ylabel('Partial prediction')
    plt.show()


# %% [markdown]
# Looking at non-linear relationship with respect to teh TimeBins

# %%
# Grouping at the level of animal × timebin × cluster
cluster_summary_2 = (
    df_no_outliers
    .groupby(['experimental_id', 'Interval_bin', 'Cluster'])
    .size()
    .reset_index(name='count')
)

# Total counts per Interval_bin per animal
total_intervals_2 = (
    df_no_outliers
    .groupby(['experimental_id', 'Interval_bin'])
    .size()
    .reset_index(name='total')
)

# Merge to calculate cluster_prop per animal × time × cluster
cluster_summary_2 = cluster_summary_2.merge(total_intervals_2, on=['experimental_id', 'Interval_bin'])
cluster_summary_2['cluster_prop'] = cluster_summary_2['count'] / cluster_summary_2['total']

# %%
#Makes sure this variable is correctly defined
# Assuming df_subjects has columns: experimental_id, BW, Geno, etc.
df_merged_2 = cluster_summary_2.merge(df_bw, on='experimental_id', how='left')
print(df_merged_2)


# %%

for cluster_id in clusters:
    df_cluster = df_merged_2[df_merged_2['Cluster'] == cluster_id]
    print(f"\n--- Cluster {cluster_id} ---")

    # Convert categorical variables to dummy codes
    exog = pd.get_dummies(df_cluster[['Geno', 'Sex']], drop_first=True)
    bs = BSplines(df_cluster[['Interval_bin']], df=[6], degree=[3])
    y = df_cluster['cluster_prop']

    model = GLMGam(y, exog=exog, smoother=bs)
    result = model.fit()
    print(result.summary())

    # Plot the partial effect of the spline on BW
    fig = result.plot_partial(0, plot_se=True)  # 0 because BW is the first (and only) smooth term
    plt.title(f'Effect of BW (spline) for Cluster {cluster_id}')
    plt.xlabel('Time Interval (spline)')
    plt.ylabel('Partial prediction')
    plt.show()


# %%
from statsmodels.gam.api import GLMGam, BSplines
import pandas as pd
import numpy as np

clusters = df_merged_2['Cluster'].unique()

for cluster_id in clusters:
    df_cluster = df_merged_2[df_merged_2['Cluster'] == cluster_id]
    print(f"\n--- Cluster {cluster_id} ---")

    # Convert categorical variables to dummy codes
    exog = pd.get_dummies(df_cluster[['Geno', 'Sex']], drop_first=True)

    # Spline for BW
    bs = BSplines(df_cluster[['Interval_bin']], df=[6], degree=[3])  # 6 degrees of freedom for BW
    y = df_cluster['cluster_prop']

    model = GLMGam(y, exog=exog, smoother=bs)
    result = model.fit()
    print(result.summary())


# %%



