import umap
import os 
import numpy as np
import scanpy as sc
import scipy.stats as stats
import scipy.sparse  
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import (adjusted_rand_score, roc_auc_score,
                             jaccard_score)
from scib.metrics import ilisi_graph
import celltypist
#from celltypist import models
#from celltypist.models import Model
from scipy.sparse import issparse




def filter_low_quality_cells_and_genes(adata, min_counts=10, min_cells=3):
    adata = adata.copy() 
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    return adata


def is_sparse(adata):
    if scipy.sparse.issparse(adata.X):
        return adata.X.toarray()  # Convert sparse to dense
    return adata.X  # Already dense
    
def to_dense(adata):
    return adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X


def check_for_inf_nan(adata, label):
    X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
    print(f"Checking {label} dataset:")
    print(f"NaNs? {np.isnan(X).any()}")
    print(f"Infs? {np.isinf(X).any()}")
    print(f"Min: {np.min(X)}, Max: {np.max(X)}\n")



def check_missing_genes(real_data, synthetic_data):
    # Convert to sets for easy comparison
    real_genes = set(real_data.var_names)
    synthetic_genes = set(synthetic_data.var_names)

    # Find missing genes
    missing_in_real = synthetic_genes - real_genes
    missing_in_synthetic = real_genes - synthetic_genes

    print(f"Genes in synthetic but not in real: {len(missing_in_real)}")
    print(f"Genes in real but not in synthetic: {len(missing_in_synthetic)}")

    # Print some missing genes
    print(f"Example missing in real: {list(missing_in_real)[:10]}")
    print(f"Example missing in synthetic: {list(missing_in_synthetic)[:10]}")

    print(f"real_data.var_names dtype: {real_data.var_names.dtype}")
    print(f"synthetic_data.var_names dtype: {synthetic_data.var_names.dtype}")


class Statistics:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def compute_scc(self, real_data, synthetic_data, n_hvgs=5000):
        np.random.seed(self.random_seed)
        check_missing_genes(real_data, synthetic_data )
        real_data = real_data[:, synthetic_data.var_names]
        synthetic_data = synthetic_data[:, real_data.var_names]

        check_for_inf_nan(real_data, "Real")
        check_for_inf_nan(synthetic_data, "Synthetic")

        ## Combine real and synthetic
        combined_adata = real_data.concatenate( 
            synthetic_data, batch_key="source", batch_categories=["real", "synthetic"]
            )
        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)

        #  Identify HVGs
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)
        hvg_mask = combined_adata.var["highly_variable"].index
        real_hvg = combined_adata[combined_adata.obs["source"] == "real"][:, hvg_mask]
        synth_hvg = combined_adata[combined_adata.obs["source"] == "synthetic"][:, hvg_mask]

        # Convert to dense format
        real_exp = is_sparse(real_hvg)
        synth_exp = is_sparse(synth_hvg)

        #  Compute mean expression per gene
        real_means = real_exp.mean(axis=0)
        synth_means = synth_exp.mean(axis=0)

        scc, _ = stats.spearmanr(real_means, synth_means)
        pcc, _ = stats.pearsonr(real_means, synth_means)

        return scc, pcc
    

    def compute_mmd_optimized(self, real_data, synthetic_data, sample_size=20000,
                               n_pca=50, gamma=1.0, n_hvgs=5000):
        np.random.seed(self.random_seed)
        real_data, synthetic_data = real_data[:, synthetic_data.var_names], synthetic_data[:, real_data.var_names]

        # Identify HVGs
        combined_adata = real_data.concatenate(synthetic_data)
        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)

        # Subset to HVGs
        real_hvg = real_data[:, combined_adata.var["highly_variable"]]
        synth_hvg = synthetic_data[:, combined_adata.var["highly_variable"]]

        # Convert sparse to dense
        real_dense = real_hvg.X.toarray() if scipy.sparse.issparse(real_hvg.X) else real_hvg.X
        synth_dense = synth_hvg.X.toarray() if scipy.sparse.issparse(synth_hvg.X) else synth_hvg.X

        # Subsample
        real_idx = np.random.choice(real_dense.shape[0], 
                                    min(sample_size, real_dense.shape[0]), 
                                    replace=False)
        synth_idx = np.random.choice(synth_dense.shape[0], 
                                     min(sample_size, synth_dense.shape[0]), 
                                     replace=False)

        real_sample = real_dense[real_idx]
        synth_sample = synth_dense[synth_idx]

        # Combine for PCA
        combined_sample = np.vstack([real_sample, synth_sample])
        pca = PCA(n_components=n_pca, random_state=self.random_seed)
        combined_pca = pca.fit_transform(combined_sample)

        # Split PCA results
        real_pca = combined_pca[: len(real_sample)]
        synth_pca = combined_pca[len(real_sample) :]

        # Compute MMD
        K_xx = rbf_kernel(real_pca, real_pca, gamma=gamma).mean()
        K_yy = rbf_kernel(synth_pca, synth_pca, gamma=gamma).mean()
        K_xy = rbf_kernel(real_pca, synth_pca, gamma=gamma).mean()

        return K_xx + K_yy - 2 * K_xy
    


    ## Computing Earth Movers Distance (EMD)
    def compute_emd(self, real_data, synthetic_data, n_hvgs=5000):
        real_data = real_data[:, synthetic_data.var_names]
        synthetic_data = synthetic_data[:, real_data.var_names]
        check_for_inf_nan(real_data, "Real")
        check_for_inf_nan(synthetic_data, "Synthetic")

        # Combine and normalize
        combined_adata = real_data.concatenate(
            synthetic_data, batch_key="source", batch_categories=["real", "synthetic"]
        )
        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)
        hvgs = combined_adata.var[combined_adata.var['highly_variable']].index

        real_expr = combined_adata[combined_adata.obs["source"] == "real"][:, hvgs].X.toarray()
        synth_expr = combined_adata[combined_adata.obs["source"] == "synthetic"][:, hvgs].X.toarray()

        emd_values = []
        for gene_idx in range(len(hvgs)):
            real_gene = real_expr[:, gene_idx]
            synth_gene = synth_expr[:, gene_idx]

            emd = stats.wasserstein_distance(real_gene, synth_gene)
            emd_values.append(emd)

        return np.mean(emd_values)
    



   
    # Goal: Measure the mixing of real and synthetic cells in a shared space.
    def compute_lisi(self, real_data, synthetic_data, n_hvgs=5000):
        # Ensure both datasets have the same genes
        np.random.seed(self.random_seed)
        real_data = real_data[:, synthetic_data.var_names]
        synthetic_data = synthetic_data[:, real_data.var_names]
        combined_adata = real_data.concatenate(
            synthetic_data, batch_key="source", batch_categories=["real", "synthetic"]
        )
        # Assign batch labels (0 = real, 1 = synthetic)
        combined_adata.obs["batch"] = (combined_adata.obs["source"] == "synthetic").astype(int)

        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)
        combined_adata = combined_adata[:, combined_adata.var['highly_variable']]

         # ----  Downsampling (if enabled) ----
        #if use_downsample:
        #    sample_size = int(downsample_ratio * combined_adata.shape[0])  # Compute sample size
        #    sampled_idx = np.random.choice(combined_adata.shape[0], size=sample_size, replace=False)
        #    combined_adata = combined_adata[sampled_idx, :]

        # Perform PCA
        sc.pp.pca(combined_adata,  n_comps=100, random_state=self.random_seed)
        sc.pp.neighbors(combined_adata, n_neighbors=10, method='umap')


        return ilisi_graph(combined_adata, batch_key="batch", type_="knn")



    # Goal: Measure how well real & synthetic cells cluster into the same types.
    def compute_ari(self, real_data, synthetic_data, cell_type_col, n_hvgs=5000):
        np.random.seed(self.random_seed)
        real_data = real_data[:, synthetic_data.var_names]
        synthetic_data = synthetic_data[:, real_data.var_names]
        combined_adata = real_data.concatenate(
            synthetic_data, batch_key="source", batch_categories=["real", "synthetic"]
        )

        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)
        combined_adata = combined_adata[:, combined_adata.var['highly_variable']]


        # Perform PCA
        sc.pp.pca(combined_adata,  n_comps=100, random_state=self.random_seed)
        sc.pp.neighbors(combined_adata, n_neighbors=10, method='umap')
        sc.tl.louvain(combined_adata)

        # Convert Louvain clusters to numerical labels
        combined_adata.obs["louvain"] = combined_adata.obs["louvain"].astype("category").cat.codes
        real_clusters = combined_adata.obs.loc[
                            combined_adata.obs["source"] == "real", "louvain"].values
        synthetic_clusters = combined_adata.obs.loc[
                            combined_adata.obs["source"] == "synthetic", "louvain"].values
        ari_real_vs_syn = adjusted_rand_score(real_clusters, synthetic_clusters)
        ari_gt_vs_comb = adjusted_rand_score(combined_adata.obs[cell_type_col], 
                                             combined_adata.obs["louvain"])

        return ari_real_vs_syn, ari_gt_vs_comb
    


    def compute_gene_correlation_similarity(self, real_data, synthetic_data, n_hvgs=5000):
        real_data = real_data[:, synthetic_data.var_names]
        synthetic_data = synthetic_data[:, real_data.var_names]
        check_for_inf_nan(real_data, "Real")
        check_for_inf_nan(synthetic_data, "Synthetic")

        # Combine and normalize
        combined_adata = real_data.concatenate(
            synthetic_data, batch_key="source", batch_categories=["real", "synthetic"]
        )
        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)
        hvgs = combined_adata.var[combined_adata.var['highly_variable']].index

        real_expr = combined_adata[combined_adata.obs["source"] == "real"][:, hvgs].X.toarray()
        synth_expr = combined_adata[combined_adata.obs["source"] == "synthetic"][:, hvgs].X.toarray()

        # Compute gene-gene correlation matrices
        real_corr, _ = stats.spearmanr(real_expr, axis=0)
        synth_corr, _ = stats.spearmanr(synth_expr, axis=0)


        # Extract upper triangle only (since correlation matrices are symmetric)
        triu_idx = np.triu_indices_from(real_corr, k=1)
        real_vals = real_corr[triu_idx]
        synth_vals = synth_corr[triu_idx]

        # Compute similarity 
        correlation_of_correlations = np.corrcoef(real_vals, synth_vals)[0, 1]

        return correlation_of_correlations






class VisualizeClassify:
    ### add figures_dir = 
    def __init__(self, sc_figures_dir, random_seed=42):
        self.random_seed = random_seed
        self.sc_figures_dir = sc_figures_dir
        np.random.seed(self.random_seed)
       # self.figures_dir = figures_dir

    ## get example name instead of sc_figures dir 
    def plot_umap(self, real_data, synthetic_data, n_hvgs=5000):
        sc.settings.figdir = self.sc_figures_dir
        np.random.seed(self.random_seed)
        # Combine datasets with batch labels
        check_for_inf_nan(real_data, "Real")
        check_for_inf_nan(synthetic_data, "Synthetic")
        combined_adata = real_data.concatenate(
            synthetic_data, batch_key="source", batch_categories=["real", "synthetic"]
        )

        sc.pp.normalize_total(combined_adata, target_sum=1e4 )
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)
        combined_adata = combined_adata[:, combined_adata.var['highly_variable']]

        # Perform PCA & UMAP
        sc.pp.pca(combined_adata, random_state=self.random_seed)
        sc.pp.neighbors(combined_adata)
        sc.tl.umap(combined_adata, random_state=self.random_seed)

        # Plot UMAP
        sc.pl.umap(combined_adata, 
                   color=["source"], 
                   title="UMAP of Real vs Synthetic Data",
                   save = f"syn_test_PCA_HVG={n_hvgs}.png")


    def celltypist_classification(self, real_data_test, synthetic_data, celltypist_model, n_hvgs=5000):
        np.random.seed(self.random_seed)

        # Combine datasets for HVG selection
        combined_adata = real_data_test.concatenate(synthetic_data)

        # Normalize before selecting HVGs
        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)

        ### Normalize and logp
        sc.pp.normalize_total(real_data_test, target_sum=1e4)
        sc.pp.log1p(real_data_test)

        ### Normalize and logp
        sc.pp.normalize_total(synthetic_data, target_sum=1e4)
        sc.pp.log1p(synthetic_data)


        # Subset both datasets to HVGs
        real_data_test = real_data_test[:, combined_adata.var['highly_variable']]
        synthetic_data = synthetic_data[:, combined_adata.var['highly_variable']]

        # Load CellTypist model
        model = celltypist.models.Model.load(celltypist_model)
        real_predictions = celltypist.annotate(real_data_test, model=model)
        synthetic_predictions = celltypist.annotate(synthetic_data, model=model)

        # Extract predicted labels
        real_labels = real_predictions.predicted_labels.values.ravel()
        synthetic_labels = synthetic_predictions.predicted_labels.values.ravel()

        # Compute ARI score
        ari_score = adjusted_rand_score(real_labels, synthetic_labels)

        # Compute Jaccard score for multi-class labels
        lb = LabelBinarizer()
        real_onehot = lb.fit_transform(real_labels)
        synthetic_onehot = lb.transform(synthetic_labels)

        jaccard_scores = [
            jaccard_score(real_onehot[:, i], synthetic_onehot[:, i]) 
            for i in range(real_onehot.shape[1])
        ]
        jaccard = np.mean(jaccard_scores)

        return ari_score, jaccard


    ## whether it can separate synthetic vs real
    def random_forest_eval(self, real_data, synthetic_data, n_hvgs=5000):
        np.random.seed(self.random_seed)

        # Explicitly label real vs. synthetic
        real_data.obs["source"] = "real"
        synthetic_data.obs["source"] = "synthetic"

        # Concatenate datasets
        combined_adata = real_data.concatenate(
            synthetic_data, batch_key="source", batch_categories=["real", "synthetic"]
        )

        # Normalize & log transform
        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)

        # Select highly variable genes (HVGs)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)
        combined_adata = combined_adata[:, combined_adata.var['highly_variable']]

        # **Batch correction using Combat**
        sc.pp.combat(combined_adata, key="source")  # Removes batch effect

        # Convert sparse to dense if needed
        X = combined_adata.X.A if hasattr(combined_adata.X, "A") else combined_adata.X

        # Assign labels: 0 = real, 1 = synthetic
        y = (combined_adata.obs["source"] == "synthetic").astype(int).values

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=self.random_seed)

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=self.random_seed)
        rf.fit(X_train, y_train)

        # Predict probabilities and compute AUC
        pred_probs = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, pred_probs)

        return auc, pred_probs

    






