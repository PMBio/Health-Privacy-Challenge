from typing import Optional, Tuple, Dict, Any
import sys
import os
import pandas as pd
import scanpy as sc
import numpy as np
import torch
import torch.nn.functional as F

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

from mia.models.base import BaseMIAModel
#from domias.baselines import (MC, GAN_leaks)


#Adapted from https://github.com/holarissun/DOMIAS/blob/main/src/domias/baselines.py
## code is modified to handle batch processing due to memory-extensive needs of scrna-seq data

def d(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute squared L2 distances between two sets of vectors X and Y."""
    X_sq = np.sum(X**2, axis=1, keepdims=True)  # (m, 1)
    Y_sq = np.sum(Y**2, axis=1)  # (n,)
    XY = np.dot(X, Y.T)  # (m, n)
    distances = X_sq + Y_sq - 2 * XY
    return np.maximum(distances, 0)  # Ensure non-negative distances
    

def batch_d(X: np.ndarray, Y: np.ndarray, batch_size=1000) -> np.ndarray:
    results = []
    for i in range(0, X.shape[0], batch_size):  # Iterate by batch size
        batch = X[i:i + batch_size]
        results.append(d(batch, Y))  # Use the batch_d() function for the whole batch
    return np.vstack(results)  # Combine results from all batches


def d_min(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.min(d(X, Y))


def batch_d_min(X: np.ndarray, Y: np.ndarray, batch_size=1000) -> np.ndarray:
    min_dists = []
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:i + batch_size]
        distances = batch_d(batch, Y, batch_size)
        min_dists.append(np.min(distances, axis=1))
    return np.concatenate(min_dists)



def batch_GAN_leaks(X_test: np.ndarray, X_G: np.ndarray, batch_size=1000) -> np.ndarray:
    min_dists = batch_d_min(X_test, X_G, batch_size)
    scores = np.exp(-min_dists)
    assert not np.any(np.isinf(scores)), "Infinity values found in scores."
    return scores

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))



def pre_filtering_before_split(adata):
    ## prefiltering before separation 
    sc.pp.filter_genes(adata, min_cells=3) # Remove genes that are detected in less than 3 cells.
    print(adata.shape)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(adata,
                 ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
                 jitter=0.4, multi_panel=True,
                 save="")

    adata = adata[adata.obs.total_counts > 10,:]
    adata = adata[adata.obs.total_counts < 40000,:]
    adata.shape # Checking number of cells remaining
    sc.pl.violin(adata,
                 ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
                 jitter=0.4,multi_panel=True,
                 save="")
    
    return adata


class DOMIASSingleCellBaselineModels(BaseMIAModel):
    def __init__(self, 
                 config: Dict[str, Any], 
                 synthetic_file: str,
                 membership_test_file: str,
                 membership_lbl_file: str,
                 mia_experiment_name:str,
                 reference_file:str = None):
        super().__init__(config, 
                         synthetic_file, 
                         membership_test_file, 
                         membership_lbl_file,
                         mia_experiment_name,
                         reference_file)
        self.donors = None

    def run_attack(self):
        data_loader = MIASingleCellDataLoader(
            synthetic_file=self.synthetic_file,
            membership_test_file=self.membership_test_file,
            membership_lbl_file = self.membership_lbl_file,
            membership_label_col=self.membership_label_col,
            generator_model=self.generator_model,
            reference_file=self.reference_file
        )
        synthetic_data = data_loader.load_synthetic_data()
        X_test = data_loader.load_membership_dataset(filter=False)
        y_test, barcodes, self.donors = data_loader.load_membership_labels(X_test.obs)

        #same = X_test.obs["barcode_col"].values.equals(barcodes)
        same = np.array_equal(X_test.obs["barcode_col"].to_numpy(), barcodes.to_numpy())

        if not same:
            raise Exception(f"Barcode orders do not match for test and membership labels. ")

        if y_test is not None:
            assert len(X_test) == len(y_test), "mismatch in test data and label lengths."
        
        #reference = data_loader.load_reference_data()
        #X_test_dense = X_test.X.toarray() if hasattr(X_test.X, "toarray") else X_test.X
        #syn_dense = synthetic_data.X.toarray() if hasattr(synthetic_data.X, "toarray") else synthetic_data.X

        ## Testing if using HVG have an effect 
        combined_adata = X_test.concatenate(synthetic_data)
        # Normalize before selecting HVGs
        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=5000)

        ### Normalize and logp
        sc.pp.normalize_total(X_test, target_sum=1e4)
        sc.pp.log1p(X_test)

        ### Normalize and logp
        sc.pp.normalize_total(synthetic_data, target_sum=1e4)
        sc.pp.log1p(synthetic_data)

        X_test = X_test[:, combined_adata.var['highly_variable']]
        synthetic_data = synthetic_data[:, combined_adata.var['highly_variable']]

        X_test_dense = X_test.X.toarray() if hasattr(X_test.X, "toarray") else X_test.X
        syn_dense = synthetic_data.X.toarray() if hasattr(synthetic_data.X, "toarray") else synthetic_data.X

        scores = {}
        scores["gan_leaks"] = batch_GAN_leaks(X_test_dense, syn_dense)

        return scores, y_test
    


    def perform_donor_level_avg(self, predictions, labels):
       
       predictions = predictions["gan_leaks"]
       assert len(predictions) == len(self.donors), f"Scores {len(predictions)} and donors {len(self.donors)} must have the same length."

        # Create a DataFrame to easily group scores by donor
       scores_df = pd.DataFrame({
            'donor': self.donors,
            'score': predictions,
            'y_test': labels,
        })
       
       grp_predictions = scores_df.groupby('donor')['score'].mean().to_numpy()
       grp_labels = scores_df.groupby('donor')['y_test'].mean().to_numpy()

       #print(grp_predictions)
       #print(grp_labels)

       return grp_predictions, grp_labels




class MIASingleCellDataLoader:
    def __init__(self, 
                synthetic_file: str,
                membership_test_file: str,
                membership_lbl_file:str,
                membership_label_col: str,
                generator_model: str,
                reference_file: str = None):
        self.generator_model = generator_model
        self.membership_test_file = membership_test_file
        self.membership_lbl_file = membership_lbl_file
        self.membership_label_col = membership_label_col
        self.synthetic_file = synthetic_file
        self.reference_file = reference_file
        self.membership_data = {}


    def load_synthetic_data(self):
        if not os.path.exists(self.synthetic_file):
            raise FileNotFoundError("Synthetic dataset is missing.")

        try:
            syn_data = sc.read_h5ad(self.synthetic_file)
            print(f"Synthetic dataset is loaded. Size {syn_data.shape}")
            return syn_data
        except:
            raise Exception(f"Failed to load Synthetic anndata.")
    
    
    def load_membership_dataset(self, filter=False):
        if not os.path.exists(self.membership_test_file):
            raise FileNotFoundError("Membership test dataset is missing.")

        try:
            test_data = sc.read_h5ad(self.membership_test_file)
            if filter:
                test_data = pre_filtering_before_split(test_data)
            print(f"Membership test set is loaded. Size {test_data.shape}")
            return test_data
        except:
            raise Exception(f"Failed to load membership test anndata.")



    def load_membership_labels(self, org_barcodes):
        labels = None
        if self.membership_lbl_file is not None:
            labels = pd.read_csv(self.membership_lbl_file)[
                                        self.membership_label_col].values
            
            barcodes = pd.read_csv(self.membership_lbl_file)[
                                        "barcode_col"]
            
            donors = pd.read_csv(self.membership_lbl_file)[
                                        "individual"].values

            print(f"Membership test labels are loaded. Size {len(labels)}")
            
        return labels, barcodes, donors


    def initialize_datasets(self, real_data, synthetic_data):
        print(f"Initial gene count - Real: {real_data.n_vars}, Synthetic: {synthetic_data.n_vars}")

        sc.pp.filter_cells(real_data, min_counts=10)
        sc.pp.filter_genes(real_data, min_cells=3)

        sc.pp.filter_cells(synthetic_data, min_counts=10)
        sc.pp.filter_genes(synthetic_data, min_cells=3)

        print(f"After filtering - Real: {real_data.n_vars}, Synthetic: {synthetic_data.n_vars}")

        # make sure both datasets have the same genes after filter
        common_genes = real_data.var_names.intersection(synthetic_data.var_names)
        real_data = real_data[:, common_genes]
        synthetic_data = synthetic_data[:, common_genes]

        print(f"After gene alignment - Real: {real_data.n_vars}, Synthetic: {synthetic_data.n_vars}")

        combined_adata = real_data.concatenate(synthetic_data)

        # Identify HVGs
        sc.pp.normalize_total(combined_adata, target_sum=1e4)
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat", n_top_genes=n_hvgs)

        # Subset to HVGs
        real_hvg = real_data[:, combined_adata.var["highly_variable"]]
        synth_hvg = synthetic_data[:, combined_adata.var["highly_variable"]]

        # Convert sparse to dense
        real_dense = real_hvg.X.toarray() if scipy.sparse.issparse(real_hvg.X) else real_hvg.X
        synth_dense = synth_hvg.X.toarray() if scipy.sparse.issparse(synth_hvg.X) else synth_hvg.X



        return real_data, synthetic_data


    
    def load_reference_data(self):
        if self.reference_file:
            try:
                reference = sc.read_h5ad(self.reference_file)
                print(f"Membership test set is loaded. Size {reference.shape}")
                return reference
            except:
                raise Exception(f"Failed to load reference anndata")
                
        else:
            return None
    
    
    @staticmethod
    def save_files(save_dir, file_name_list, array_list):

        assert len(file_name_list) == len(array_list)

        for i in range(len(file_name_list)):
            np.save(os.path.join(save_dir, file_name_list[i]), array_list[i], allow_pickle=False)


