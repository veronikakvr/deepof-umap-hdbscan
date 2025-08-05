# %%
import pandas as pd
import glob
from functools import reduce
from pathlib import Path
import pandas as pd
import os
from collections import defaultdict
import re
from datetime import timedelta
from tqdm import tqdm

# %%
# Set your specific directory here
csv_dir = r'/Users/veronika/ownCloud/PCA_analysis2025/SocialOF/HFD/males/atg7KO/Single/rawdata' #load your raw deepof 0.8 csv data from a specific directory
file_list = glob.glob(f'{csv_dir}/*.csv')
output_path = r'/Users/veronika/ownCloud/PCA_analysis2025/SocialOF/HFD/males/atg7KO/Single' # Path to save the merged CSV file

print(f"Found {len(file_list)} CSV files in {csv_dir}") # this will print the number of CSV files found in the first specified directory - check if the files are loaded correctly


# %%
##Merge all CSV files into a single DataFrame and renames the first column to "Time" and adds an "experimental_id" column with the file name (without extension)
dfs = []
for file in file_list:
    df = pd.read_csv(file)
    # Rename the first column to "Time"
    if len(df.columns) > 0:
        df.rename(columns={df.columns[0]: "Time"}, inplace=True)
    # Add experimental_id column with file name (without extension)
    df['experimental_id'] = file.split('/')[-1].split('.')[0]
    dfs.append(df)

master_df = pd.concat(dfs, ignore_index=True)

# %%
out_file = os.path.join(output_path, 'master_combined_males_atg7KO_HFD_Single.csv') # Save the merged DataFrame to a CSV file
master_df.to_csv(out_file, index=False)
print(f"✅ Dataset saved to: {out_file}")

# %%
######################################### Set your directory and condition file paths
csv_file = Path(r'/Users/veronika/ownCloud/PCA_analysis2025/SocialOF/HFD/males/atg7KO/Single/master_combined_males_atg7KO_HFD_Single.csv') #this is the path to the merged CSV file, we created above
condition_file = Path(r'/Users/veronika/ownCloud/PCA_analysis2025/SocialOF/HFD/males/atg7KO/exp_condi_atg7KOmales_HFD.csv') #this is the path to the condition file, which contains the experimental conditions for each sample

# %%
# Load condition mapping file
condition_df = pd.read_csv(condition_file)

# Create genotype map based on experimental ID
geno_map = dict(zip(condition_df['experimental_id'], condition_df['Geno']))

# %%
# Load data file
df = pd.read_csv(csv_file)

# %%
#Rename first column to "Time" if necessary - this is to ensure consistency
if df.columns[0] != 'Time':
    df.rename(columns={df.columns[0]: 'Time'}, inplace=True)

# Extract just the ID (e.g., "ID196") from the 'experimental_id' path
df['experimental_id'] = df['experimental_id'].apply(lambda x: Path(x).name.split('+')[0])

# Map to Geno
df['Geno'] = df['experimental_id'].map(geno_map).fillna('Unknown')

# %%
# Save final output
output_file = csv_file.parent / (csv_file.stem + '_FINAL.csv')
df.to_csv(output_file, index=False)

print(f"✅ Saved cleaned file to: {output_file}")

# %% [markdown]
# ##  A little code check if experimental conditions were loaded correctly

# %%
#This will print the unique genotypes found in the dataset - useful for debugging
unique_genotypes = df['Geno'].unique()
print(f"Unique genotypes found: {unique_genotypes}")

# %%
# Another check to see if there are any unknown genotypes (should have been filled with 'Unknown' and already checked above)
geno_check = df[df["Geno"] == "Unknown"]
print(f"Number of unknown genotypes: {len(geno_check)}")

# %% [markdown]
# #   Proceed only if you have anytroubles with geno matching or time in your dataset

# %% [markdown]
# Checking for the Unknown genotype and to what experimental ID it matches - if there are inconsisencies in loading, check if the experimental file at the beginning is formatted correctly

# %%
# Filter the dataframe for rows where genotype is 'Unknown'
unknown_geno_df = df[df['Geno'] == 'Unknown']

# List all unique experimental_id values with Unknown genotype
experimental_ids_with_unknown = unknown_geno_df['experimental_id'].unique()

print(experimental_ids_with_unknown)

# %% [markdown]
# Checking if the Time corresponds to a lenght we would expect - e.g. no funny timestamps - it should look like e.g.: '00:00:00' ... '00:09:59.0665'

# %%
unique_Times = df.sort_values(['Time', 'experimental_id']).reset_index(drop=True)
unique_Times = unique_Times['Time'].unique()
print(f"Unique time points found: {unique_Times}")

# %% [markdown]
# Checking the whole dataset as a table - if the behaviours look right, experimental id make sense and the mapping to Geno looks ok - check the Time column here!! Excel can be funny and additional forcing to the right format might be neccessary

# %%
df_sorted = df.sort_values(by=['experimental_id', 'Time']).reset_index(drop=True)
df_sorted.tail()  # Display the first few rows of the sorted DataFrame


