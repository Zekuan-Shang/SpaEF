import pandas as pd
from scipy.io import mmread
import numpy as np

# Input and Output paths 
mtx_path = YOUR_PATH_TO_(matrix.mtx)
feats_path = YOUR_PATH_TO_(features.tsv)
bacds_path = YOUR_PATH_TO_(barcodes.tsv)
output_path = YOUR_PATH_TO_(expression.csv)

gene_rate = 0.01
spot_rate = 0.01

# Keep specific genes regardless of filtering criteria (optional)
required_genes = []

print("Reading data...")
sparse_matrix = mmread(mtx_path).tocsr() 
features = pd.read_csv(feats_path, header=None, sep='\t')
barcodes = pd.read_csv(bacds_path, header=None, sep='\t')
print(f"Raw data shape: {sparse_matrix.shape[0]} spots, {sparse_matrix.shape[1]} genes")


gene_ids = features.iloc[:, 1].values
barcode_ids = barcodes.iloc[:, 0].values
print("Starting gene filtering...")
total_spots = sparse_matrix.shape[0]
min_spots_for_gene = max(1, int(total_spots * gene_rate))  
gene_counts = np.diff(sparse_matrix.indptr) 
selected_gene_indices_by_freq = np.where(gene_counts >= min_spots_for_gene)[0]
selected_gene_indices_by_freq_set = set(selected_gene_indices_by_freq)
required_gene_indices = []
if required_genes:
    gene_dict = {gene: idx for idx, gene in enumerate(gene_ids)}
    for gene in required_genes:
        if gene in gene_dict:
            idx = gene_dict[gene]
            if idx not in selected_gene_indices_by_freq_set:
                required_gene_indices.append(idx)
        else:
            print(f"Warning: Required gene '{gene}' is not in the dataset")

final_gene_indices = sorted(set(selected_gene_indices_by_freq).union(required_gene_indices))
print(f"After filtering, {len(final_gene_indices)} genes are retained (including {len(required_gene_indices)} required genes)")

print("Starting spot filtering...")
filtered_matrix = sparse_matrix[:, final_gene_indices]
spot_counts = filtered_matrix.getnnz(axis=1)  
total_genes_filtered = filtered_matrix.shape[1]
min_genes_for_spot = max(1, int(total_genes_filtered * spot_rate))  
selected_spot_indices = np.where(spot_counts >= min_genes_for_spot)[0]
final_sparse_matrix = filtered_matrix[selected_spot_indices, :]
print(f"After filtering, {len(selected_spot_indices)} spots are retained")

print("Calculating CPM and applying log1p transformation...")
spot_sums = final_sparse_matrix.sum(axis=1).A.ravel()  
with np.errstate(divide='ignore', invalid='ignore'):
    cpm_matrix = final_sparse_matrix.astype(np.float64).toarray()
    cpm_matrix = (cpm_matrix / spot_sums[:, np.newaxis]) * 1e6
    cpm_matrix[np.isnan(cpm_matrix)] = 0
    log1p_matrix = np.log1p(cpm_matrix)

print("Creating DataFrame and saving...")
selected_genes = gene_ids[final_gene_indices]
selected_barcodes = barcode_ids[selected_spot_indices]
log1p_df = pd.DataFrame(log1p_matrix, index=selected_barcodes, columns=selected_genes)
log1p_df.to_csv(output_path)
print(f"Conversion complete! Final dataset size: {log1p_df.shape[0]} spots, {log1p_df.shape[1]} genes")
print(f"Results saved to: {output_path}")
