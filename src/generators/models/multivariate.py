import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

from generators.models.base import BaseDataGenerator


class MultivariateDataGenerator(BaseDataGenerator):
    def __init__(self, config: Dict[str, Any], split_no: int = 1):
        super().__init__(config, split_no)
        self.noise_level = self.generator_config["noise_level"]
        self.random_seed = self.generator_config["random_seed"]

        self.gene_means = None
        self.gene_covs = None
        self.num_samples = None
        self.X_train_features = None

        self.initialize_random_seeds()

    def initialize_random_seeds(self):
        np.random.seed(self.random_seed)

    def compute_gene_statistics(self,  X_train, y_train):
        #original_data_df = pd.DataFrame(X_train, columns=self.X_train_features)
        #self.labels_df = pd.DataFrame({self.subtype_col_name: y_train})
        original_data_df = X_train.copy()
        self.labels_df = y_train.copy()

        merged_data = original_data_df.join(self.labels_df)
        subtype_column = merged_data[self.subtype_col_name]
        
        # Select only numeric data
        numeric_data = merged_data.select_dtypes(include=[np.number])
        self.gene_means = numeric_data.groupby(subtype_column).mean()

        def compute_cov(group):
            return group.cov()

        self.gene_covs = numeric_data.groupby(subtype_column).apply(compute_cov)
        

    def run_multivariate_normal(self):
        if self.gene_means is None or self.gene_covs is None:
            raise ValueError("Gene statistics not computed.")
            
        synthetic_data_list = []
        for subtype in self.gene_means.index:
            n_samples = sum(self.labels_df.iloc[:, 0] == subtype) #???
            cov_matrix = self.gene_covs.loc[subtype].values
            mean_vector = self.gene_means.loc[subtype]

            # the covariance matrix is symmetric 
            cov_matrix = (cov_matrix + cov_matrix.T) / 2 
            # the covariance matrix is positive semi-definite 
            eigvals, eigvecs = np.linalg.eigh(cov_matrix) 
            eigvals = np.maximum(eigvals, 0) 
            # set negative eigenvalues to zero 
            cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # adding a small value to the diagonal
            min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
            if min_eig < 0:
                cov_matrix -= 10 * min_eig * np.eye(*cov_matrix.shape)

            synthetic_data_subtype = np.random.multivariate_normal(
                        mean=mean_vector, 
                        cov = cov_matrix,
                        size=n_samples
                        )
            #  multivariate gaussian noise
            noise_cov_matrix = cov_matrix * self.noise_level  
            noise = np.random.multivariate_normal(
                mean=np.zeros(len(mean_vector)), 
                cov=noise_cov_matrix,  
                size=n_samples
            )

            noisy_synthetic_data_subtype = synthetic_data_subtype + noise
            synthetic_data_subtype_df = pd.DataFrame(noisy_synthetic_data_subtype, 
                                                     columns=self.X_train_features)
            synthetic_data_subtype_df['Subtype'] = subtype
            synthetic_data_list.append(synthetic_data_subtype_df)

        synthetic_data = pd.concat(synthetic_data_list, ignore_index=True)
        synthetic_data = synthetic_data.sample(frac=1).reset_index(drop=True)

        return synthetic_data

    def generate(self):
        data_save_dir = os.path.join(self.config["dir_list"]["home"],
                                     self.config["dir_list"]["data_save_dir"])

        real_save_dir = os.path.join(data_save_dir, self.dataset_name, "real")


        # read the training data
        X_train = pd.read_csv(os.path.join(real_save_dir, f"X_train_real_split_{self.split_no}.csv"))#.values
        y_train = pd.read_csv(os.path.join(real_save_dir, f"y_train_real_split_{self.split_no}.csv"))#.values
        self.X_train_features = pd.read_csv(os.path.join(real_save_dir, "column_names.csv")).values.flatten()

        self.compute_gene_statistics(X_train, y_train)
        synthetic_data = self.run_multivariate_normal()
        synthetic_features = synthetic_data[synthetic_data.columns.difference(['Subtype'])]#.values
        synthetic_labels = synthetic_data['Subtype']#.values

        return synthetic_features, synthetic_labels
    
    #def save_synthetic_data(self, 
    #                        synthetic_features: pd.DataFrame, 
    #                        synthetic_labels: pd.DataFrame, 
    #                       experiment_name: str):
    #   BaseDataGenerator.save_synthetic_data(self, 
    #                                          synthetic_features, 
    #                                          synthetic_labels, 
    #                                          experiment_name)


    def generate_for_type(self, subtype: str, num_samples: int = None) -> pd.DataFrame:
        if self.gene_means is None or self.gene_covs is None:
            ## compute_gene_statistics() without throwing an error 
            data_save_dir = os.path.join(self.config["dir_list"]["home"],
                                     self.config["dir_list"]["data_save_dir"])

            real_save_dir = os.path.join(data_save_dir, self.dataset_name, "real")

            # read the training data
            X_train = pd.read_csv(os.path.join(real_save_dir, f"X_train_real_split_{self.split_no}.csv"))#.values
            y_train = pd.read_csv(os.path.join(real_save_dir, f"y_train_real_split_{self.split_no}.csv"))#.values
            self.X_train_features = pd.read_csv(os.path.join(real_save_dir, "column_names.csv")).values.flatten()

            self.compute_gene_statistics(X_train, y_train)

        if subtype not in self.gene_means.index:
            raise ValueError(f"Subtype '{subtype}' not found in training data.")

        if num_samples is None:
            num_samples = sum(self.labels_df.iloc[:, 0] == subtype)

        cov_matrix = self.gene_covs.loc[subtype].values
        mean_vector = self.gene_means.loc[subtype]

        # Ensure symmetry and positive semi-definite
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.maximum(eigvals, 0)
        cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

        min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
        if min_eig < 0:
            cov_matrix -= 10 * min_eig * np.eye(*cov_matrix.shape)

        synthetic_data = np.random.multivariate_normal(
            mean=mean_vector,
            cov=cov_matrix,
            size=num_samples
        )

        # Add Gaussian noise
        noise_cov_matrix = cov_matrix * self.noise_level
        noise = np.random.multivariate_normal(
            mean=np.zeros(len(mean_vector)),
            cov=noise_cov_matrix,
            size=num_samples
        )

        noisy_data = synthetic_data + noise
        synthetic_df = pd.DataFrame(noisy_data, columns=self.X_train_features)
        synthetic_df['Subtype'] = subtype

        synthetic_features = synthetic_df[synthetic_df.columns.difference(['Subtype'])]#.values
        synthetic_labels = synthetic_df['Subtype']#.values

        return synthetic_features, synthetic_labels



    def train(self):
        pass

    def load_from_checkpoint(self):
        pass


