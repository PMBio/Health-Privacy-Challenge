import os
import click
import yaml
import sys
import fnmatch
import numpy as np
import pandas as pd
import scanpy as sc

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

from evaluation.utils.sc_metrics import (filter_low_quality_cells_and_genes,
                                         Statistics, VisualizeClassify)




def check_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

class SingleCellEvaluator:
    def __init__(self, config):
        self.config = config
        self.home_dir = config["dir_list"]["home"]
        self.dataset_config = config["dataset_config"]
        self.dataset_name = self.dataset_config["name"]
        self.cell_type_col = self.dataset_config["cell_type_col_name"]
        self.cell_label_col = self.dataset_config["cell_label_col_name"]

        self.save_dir = os.path.join(self.home_dir, "data_splits")
        self.random_seed = config["evaluator_config"]["random_seed"]


        ## experiment name
        self.experiment_name = self.config['generator_config']['experiment_name']
        self.generator_name = self.config['generator_config']['name']
        self.res_figures_dir = os.path.join(self.home_dir, 
                                            config["dir_list"]["figures"], 
                                            self.dataset_name, 
                                            self.generator_name, 
                                            self.experiment_name
                                            )
        self.res_files_dir = os.path.join(self.home_dir, 
                                          config["dir_list"]["res_files"], 
                                          self.dataset_name, 
                                          self.generator_name, 
                                          self.experiment_name)
        check_dirs( self.res_figures_dir)
        check_dirs( self.res_files_dir)

        self.synthetic_data_path = os.path.join(self.save_dir, 
                                                self.dataset_name, 
                                                "synthetic",
                                                self.generator_name,
                                                self.experiment_name, 
                                                )
        self.celltypist_model_path = os.path.join(self.home_dir,
                                                 self.dataset_config["celltypist_model"])
        self.results = {}
    

    @staticmethod
    def save_split_results(results, output_file):
        df = pd.DataFrame([results])
        df.to_csv(output_file, index=False)


    def load_test_anndata(self):
        try:
            test_data_pth = os.path.join(self.home_dir, self.dataset_config["test_count_file"])
            test_data = sc.read_h5ad(test_data_pth)
            #cell_types = test_data.obs[self.cell_type_col].values
            #cell_labels = test_data.obs[self.cell_label_col].values

            #self.cell_type_to_label = dict(set(zip(cell_types, cell_labels)))

            test_data.obs[self.cell_label_col] = (
                test_data.obs[self.cell_label_col]
                .astype(str)
                .str.replace(" ", "_", regex=True)
            )

            X_dense = test_data.X.toarray() if hasattr(test_data.X, "toarray") else test_data.X
            # Count NaN and Inf values
            nan_count = np.isnan(X_dense).sum()
            inf_count = np.isinf(X_dense).sum()

            if nan_count > 0 or inf_count > 0:
                raise ValueError(f"Test data contains {nan_count} NaN values and {inf_count} Inf values.")

            print(test_data)
            return test_data
        except:
            raise Exception(f"Failed to load test anndata.")
        

    def load_synthetic_anndata(self):
        try: 
            syn_data_pth = os.path.join(self.synthetic_data_path, "onek1k_annotated_synthetic.h5ad")
            syn_data = sc.read_h5ad(syn_data_pth)
            #syn_data.obs[self.cell_label_col] = (
            #    syn_data.obs[self.cell_label_col]
            #    .astype(str)
            #    .str.replace(" ", "_", regex=True)
            #)

            X_dense = syn_data.X.toarray() if hasattr(syn_data.X, "toarray") else syn_data.X
            # Count NaN and Inf values
            nan_count = np.isnan(X_dense).sum()
            inf_count = np.isinf(X_dense).sum()

            if nan_count > 0 or inf_count > 0:
                raise ValueError(f"Test data contains {nan_count} NaN values and {inf_count} Inf values.")

            print(syn_data)
            return syn_data
        except:
            raise Exception(f"Failed to load synthetic anndata.")
        

    def initialize_datasets(self):
        test_anndata = self.load_test_anndata()
        synthetic_anndata = self.load_synthetic_anndata()

        print(f"Initial gene count - Real: {test_anndata.n_vars}, Synthetic: {synthetic_anndata.n_vars}")
        real_data = filter_low_quality_cells_and_genes(test_anndata)
        synthetic_data = filter_low_quality_cells_and_genes(synthetic_anndata)
        print(f"After filtering - Real: {real_data.n_vars}, Synthetic: {synthetic_data.n_vars}")

        # make sure both datasets have the same genes after filter
        common_genes = real_data.var_names.intersection(synthetic_data.var_names)
        real_data = real_data[:, common_genes]
        synthetic_data = synthetic_data[:, common_genes]

        print(f"After gene alignment - Real: {real_data.n_vars}, Synthetic: {synthetic_data.n_vars}")

        return real_data, synthetic_data
        

    def get_statistical_evals(self):
        real_data, synthetic_data = self.initialize_datasets()
        stats = Statistics(self.random_seed)
        scc, pcc = stats.compute_scc(real_data, synthetic_data)
        mmd = stats.compute_mmd_optimized(real_data, synthetic_data)
        lisi = stats.compute_lisi(real_data, synthetic_data)
        ari_real_syn, ari_gt_comb = stats.compute_ari(real_data, synthetic_data, self.cell_type_col)
        corr_of_corr = stats.compute_gene_correlation_similarity(real_data, synthetic_data) #n_hvgs=5000
        emd = stats.compute_emd(real_data, synthetic_data)


        return {
            'cc_spearman': scc,
            'cc_pearson': pcc,
            'mmd': mmd,
            'lisi': lisi,
            'ari_real_vs_syn': ari_real_syn,
            'ari_gt_vs_comb': ari_gt_comb,
            'emd': emd,
            'corr_of_corr': corr_of_corr
        }
    

    def get_umap_evals(self, n_hvgs:int):
        real_data, synthetic_data = self.initialize_datasets()
        visual = VisualizeClassify(self.res_figures_dir, self.random_seed)
        visual.plot_umap(real_data, synthetic_data, n_hvgs)



    def get_classification_evals(self):
        real_data, synthetic_data = self.initialize_datasets()
        classfier = VisualizeClassify(self.res_figures_dir, self.random_seed)
        ari_score, jaccard = classfier.celltypist_classification(real_data, 
                                                                 synthetic_data, 
                                                                 self.celltypist_model_path)
        roc_score, _ = classfier.random_forest_eval(real_data, synthetic_data)

        return {

            "celltypist_ari": ari_score,
            "celltypist_jaccard": jaccard,
            "randomforest_roc": roc_score,
        }

 
    
    @staticmethod
    def save_results_to_csv(results, output_file):
        df = pd.DataFrame([results])
        df.to_csv(output_file, index=False)


@click.group()
def cli():
    pass



@click.command()
def run_statistical_eval():
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    evaluator = SingleCellEvaluator(config=config)
    results = evaluator.get_statistical_evals()
    
    output_file = os.path.join(evaluator.res_files_dir, f"statistics_evals.csv")
    ##
    evaluator.save_results_to_csv(results, output_file)
    click.echo(f"Evaluation for classification is completed. Results saved to {output_file}")



@click.command()
#@click.argument("sc_figures_dir", type=str)
@click.argument("n_hvgs", type=int, default=2000)
def run_umap_eval(n_hvgs):
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    evaluator = SingleCellEvaluator(config=config)
    evaluator.get_umap_evals(n_hvgs)
    


@click.command()
@click.argument("cell_label", type=str, default="CD4 ET")
def run_qq_eval(cell_label:str):
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    evaluator = SingleCellEvaluator(config=config)
    evaluator.save_qq_evals(cell_label=cell_label)
    




### function runs for an individual split
### results are saved under
### results/files/{dataset_name}/{model_name}/{experiment_name}
@click.command()
def run_classification_eval():
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    evaluator = SingleCellEvaluator(config=config)
    results = evaluator.get_classification_evals()
    
    output_file = os.path.join(evaluator.res_files_dir, f"classification_evals.csv")
    ##
    evaluator.save_results_to_csv(results, output_file)
    click.echo(f"Evaluation for classification is completed. Results saved to {output_file}")




cli.add_command(run_classification_eval)
cli.add_command(run_umap_eval)
cli.add_command(run_statistical_eval)
cli.add_command(run_qq_eval)


if __name__ == '__main__':
    cli()


