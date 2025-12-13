import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as st
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
from typing import Dict, Any

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

from generators.models.sc_base import BaseSingleCellDataGenerator

class ScDistributionDataGenerator(BaseSingleCellDataGenerator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.noise_level = self.generator_config["noise_level"]
        self.random_seed = self.generator_config["random_seed"]
        self.distribution = self.generator_config["distribution"]  # Either 'NB' or 'Poisson'
        self.cell_type_col_name = self.dataset_config["cell_type_col_name"]
        self.cell_label_col_name = self.dataset_config["cell_label_col_name"]

        # Parameters for the data generation
        self.gene_means = None
        self.num_samples = None
        self.X_train_features = None
        self.cell_type_params = {}

        self.initialize_random_seeds()

    def initialize_random_seeds(self):
        np.random.seed(self.random_seed)

    def train(self):
        """Compute gene expression parameters for each cell type from training data."""
        X_train_adata = self.load_train_anndata()

        counts = X_train_adata.X.toarray() if isinstance(X_train_adata.X, np.ndarray) else X_train_adata.X.A
        cell_types = X_train_adata.obs[self.cell_type_col_name].values
        cell_labels = X_train_adata.obs[self.cell_label_col_name].values

        self.cell_type_to_label = dict(set(zip(cell_types, cell_labels)))

        print("Cell Type to Label Mapping:", self.cell_type_to_label)

        # Store the max gene expression value from training
        self.max_real_value = counts.max()
        print(f"Max real expression value from training: {self.max_real_value}")

        for cell_type in np.unique(cell_types):
            print(f"Training on Cell Type: {cell_type}")

            cell_type_mask = cell_types == cell_type
            cell_type_counts = counts[cell_type_mask, :]

            means = cell_type_counts.mean(axis=0)
            means = np.clip(means, 1e-6, None)  # Avoid zero means

            if self.distribution == 'NB':
                variances = cell_type_counts.var(axis=0)

                # variance >= mean to prevent negative dispersions
                variances = np.maximum(variances, means)

                dispersions = (variances - means) / (means ** 2)
                dispersions = np.clip(dispersions, 1e-3, 10)  # Avoid extreme values

                # debugging
                print(f"Dispersion values for {cell_type}: min={dispersions.min()}, max={dispersions.max()}")

                if np.any(np.isnan(dispersions)):
                    raise ValueError(f"NaN detected in dispersions for {cell_type}!")

                self.cell_type_params[str(cell_type)] = {
                    'means': means.astype(np.float32),
                    'dispersions': dispersions.astype(np.float32)
                }

            elif self.distribution == 'Poisson':
                self.cell_type_params[str(cell_type)] = {
                    'means': means.astype(np.float32)
                }

        print("Training completed successfully!")



    def generate(self):
        if self.max_real_value is None:
            raise ValueError("Training must be completed before generating data!")

        #X_test_adata = self.load_test_anndata()
        X_test_adata = self.load_external_anndata()

        counts = X_test_adata.X.toarray() if isinstance(X_test_adata.X, np.ndarray) else X_test_adata.X.A
        print("Original counts shape:", counts.shape)

        cell_types = X_test_adata.obs[self.cell_type_col_name].values
        synthetic_counts = sp.lil_matrix(counts.shape, dtype=np.int64)
        synthetic_cell_types = []

        for cell_type in np.unique(cell_types):
            print(f"Generating for Cell Type: {cell_type}")

            if str(cell_type) not in self.cell_type_params:
                print(f"Cell type {cell_type} not found in training data! Skipping...")
                continue

            cell_type_mask = cell_types == cell_type
            cell_indices = np.where(cell_type_mask)[0]
            num_cells = len(cell_indices)

            means = self.cell_type_params[str(cell_type)]['means'].astype(np.float64)
            means = np.clip(means, 1e-6, None)  # Avoid zeros

            if self.distribution == 'NB':
                dispersions = self.cell_type_params[str(cell_type)]['dispersions'].astype(np.float64)
                dispersions = np.clip(dispersions, 1e-3, 10)  # Prevent extreme values

                # Compute Negative Binomial parameters
                n_param = np.clip(1 / (dispersions + 1e-6), 1e-2, 10) 
                p_param = np.clip(means / (means + n_param), 0.01, 0.99)  

                # Debugging prints
                print(f"n_param range for {cell_type}: min={n_param.min()}, max={n_param.max()}")
                print(f"p_param range for {cell_type}: min={p_param.min()}, max={p_param.max()}")

                expected_variance = means + (means ** 2) / n_param
                print(f"Expected variance for {cell_type}: min={expected_variance.min()}, max={expected_variance.max()}")

                # Generate Negative Binomial samples
                generated_data = st.nbinom.rvs(n=n_param, p=p_param, size=(num_cells, means.shape[0])).astype(np.int64)

            elif self.distribution == 'Poisson':
                generated_data = st.poisson.rvs(means, size=(num_cells, means.shape[0])).astype(np.int64)

            # Limit extreme values to prevent memory explosion
            upper_clip = np.percentile(generated_data, 99.5)
            generated_data = np.clip(generated_data, 0, min(upper_clip, self.max_real_value * 2))

            # Store generated data
            synthetic_counts[cell_indices, :] = generated_data
            synthetic_cell_types.extend([cell_type] * num_cells)

        synthetic_counts_csr = synthetic_counts.tocsr().astype(np.int64)
        synthetic_adata = ad.AnnData(X=synthetic_counts_csr)
        synthetic_adata.obs[self.cell_type_col_name] = synthetic_cell_types
        synthetic_adata.var_names = X_test_adata.var_names

        return synthetic_adata


    def load_from_checkpoint(self):
        pass

