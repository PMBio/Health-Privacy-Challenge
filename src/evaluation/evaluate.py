import os
import click
import yaml
import sys
import fnmatch
import logging
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, 
                             f1_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

from evaluation.utils.data_handler import EvaluationDataLoader
from evaluation.utils.stats import Statistics
from evaluation.utils.plots import Plotting
from evaluation.utils.prdc import PRDensityCoverage


def check_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class BaseEvaluator:
    def __init__(self, 
                 config, 
                 split_no, 
                 generator_name=None, 
                 experiment_name=None):
        self.config = config
        self.split_no = split_no
        self.generator_name = generator_name
        self.experiment_name = experiment_name
        home_dir = config["dir_list"]["home"]
        self.dataset_name = config["dataset_config"]["name"]
        self.save_dir = os.path.join(home_dir, "data_splits")
        self.random_seed = config["evaluator_config"]["random_seed"]

        ## experiment name
        #self.experiment_name = self.config['generator_config']['experiment_name']
        #self.generator_name = self.config['generator_config']['name']
        self.res_figures_dir = os.path.join(home_dir, 
                                            config["dir_list"]["figures"], 
                                            self.dataset_name, 
                                            self.generator_name, 
                                            self.experiment_name
                                            )
        self.res_files_dir = os.path.join(home_dir, 
                                          config["dir_list"]["res_files"], 
                                          self.dataset_name, 
                                          self.generator_name, 
                                          self.experiment_name)
        
        self.bio_files_dir = os.path.join(home_dir, 
                                          config["dir_list"]["bio_files"], 
                                          self.dataset_name, 
                                          self.generator_name, 
                                          self.experiment_name)

        check_dirs( self.res_figures_dir)
        check_dirs( self.res_files_dir)


        self.model = self.initialize_model()
        self.data_loader = EvaluationDataLoader(
                                    self.dataset_name, 
                                    self.generator_name,
                                    self.save_dir, 
                                    self.experiment_name, 
                                    split_no
                                    )
        self.results = {}

    def initialize_model(self):
        return OneVsRestClassifier(LogisticRegression(max_iter=1000))
    
    @staticmethod
    def save_split_results(results, output_file):
        df = pd.DataFrame([results])
        df.to_csv(output_file, index=False)

    @staticmethod
    def combine_csv_files(results_files, output_file):
        combined_df = pd.concat([pd.read_csv(f) for f in results_files], ignore_index=True)
        mmd_label_cols = [col for col in combined_df.columns if col.startswith('submmd_')]

        summary_dict = {
            'split_no': ['average'],
            'accuracy_synthetic': [combined_df['accuracy_synthetic'].mean()],
            'avg_pr_macro_synthetic': [combined_df['avg_pr_macro_synthetic'].mean()],
            'avg_pr_weight_synthetic': [combined_df['avg_pr_weight_synthetic'].mean()],
            'f1_synthetic': [combined_df['f1_synthetic'].mean()],
            'auroc_synthetic': [combined_df['auroc_synthetic'].mean()],
            'accuracy_real': [combined_df['accuracy_real'].mean()],
            'avg_pr_macro_real': [combined_df['avg_pr_macro_real'].mean()],
            'avg_pr_weight_real': [combined_df['avg_pr_weight_real'].mean()],
            'f1_real': [combined_df['f1_real'].mean()],
            'auroc_real': [combined_df['auroc_real'].mean()],
            'feature_overlap_count': [combined_df['feature_overlap_count'].mean()],
            'feature_overlap_proportion': [combined_df['feature_overlap_proportion'].mean()],
            'MMD_train': [combined_df['MMD_train'].mean()],
            'MMD_test': [combined_df['MMD_test'].mean()],
            'MMD_base': [combined_df['MMD_base'].mean()],
            'discriminative_score': [combined_df['discriminative_score'].mean()],
            'distance_to_closest': [combined_df['distance_to_closest'].mean()],
            'distance_to_closest_base': [combined_df['distance_to_closest_base'].mean()],
            'kl_mean_train': [combined_df['kl_mean_train'].mean()],
            'kl_mean_test': [combined_df['kl_mean_test'].mean()], 
            'kl_mean_base': [combined_df['kl_mean_base'].mean()],
            'prdc_precision': [combined_df['prdc_precision'].mean()],
            'prdc_recall': [combined_df['prdc_recall'].mean()],
            'prdc_density': [combined_df['prdc_density'].mean()],
            'prdc_coverage': [combined_df['prdc_coverage'].mean()]

        }
        # Add per-label MMD averages
        for col in mmd_label_cols:
            summary_dict[col] = [combined_df[col].mean()]

        summary_df = pd.DataFrame(summary_dict)
        combined_df = pd.concat([combined_df, summary_df], ignore_index=True)
        combined_df.to_csv(output_file, index=False)

        for f in results_files:
            if os.path.exists(f):
                os.remove(f)
        logging.info("Cleanup completed.")
    

class ModelEvaluator(BaseEvaluator):
    def get_top_n_feature_importance(self, N=10):
        feature_importances = {}

        if isinstance(self.model, OneVsRestClassifier):
            estimators = self.model.estimators_
            classes = self.model.classes_

            for i, model in enumerate(estimators):
                coef = model.coef_[0]
                top_n_indices = coef.argsort()[-N:][::-1]
                top_n_features = [f'Feature_{idx}' for idx in top_n_indices]

                feature_importances[classes[i]] = top_n_features

        return feature_importances

    #### compute discriminative score
    def discriminative_score(self, synthetic_data, X_train_real, X_test_real):
        X_train = np.vstack([X_train_real, synthetic_data])  
        # Labels: 1 for real, 0 for fake
        y_train = np.array([1] * X_train_real.shape[0] + [0] * synthetic_data.shape[0])  

        X_train, y_train = shuffle(X_train, y_train, random_state=self.random_seed) 
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
                                      test_size=0.3, random_state=self.random_seed)

        # Standard scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_test_real_scaled = scaler.transform(X_test_real)

        # Train the model with increased max_iter
        model = LogisticRegression(max_iter=1000)  
        model.fit(X_train, y_train)

        # combine test data with  unseen real data
        X_test_extended = np.vstack([X_test_real_scaled, X_test])
        y_test_extended = np.concatenate([np.array([1] * X_test_real.shape[0]), y_test])
        X_testf, y_testf = shuffle(X_test_extended, y_test_extended, random_state=self.random_seed)

        # Predictions
        y_pred = model.predict(X_testf)
        f1 = f1_score(y_testf, y_pred)

        return f1


    ## issue if there's a dimensional mismatch.. 
    def train_and_evaluate(self, 
                           X_train, 
                           y_train, 
                           X_test, 
                           y_test, 
                           y_test_encoded, 
                           X_train_real,
                           reset_model=False):
        if reset_model:
            self.model = self.initialize_model()

        # 1. Fit scaler on the *training* set 
        fitted_scaler = StandardScaler().fit(X_train_real)

        # 2. Transform both train and test using *that same scaler*
        X_train_scaled = fitted_scaler.transform(X_train)
        X_test_scaled = fitted_scaler.transform(X_test)
        
        # 3. Fit model
        self.model.fit(X_train_scaled, y_train)
        #self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)
  

        accuracy = np.round(accuracy_score(y_test, y_pred), 4)
        avg_pr_macro = np.round(average_precision_score(y_test_encoded, y_prob, average='macro'), 4)
        important_features = self.get_top_n_feature_importance()
        f1 = f1_score(y_test, y_pred, average="weighted")
        auc_roc = np.nan
        avg_pr_weighted = np.nan

          # safe aucroc & aupr
        logging.info("Safe AUROC and AUPR")
        logging.info(len(np.unique(y_test_encoded)))
        logging.info(y_prob.shape[1])
        if len(np.unique(y_test_encoded)) > 1 and len(np.unique(y_test_encoded)) == y_prob.shape[1]:
            auc_roc = roc_auc_score(
                y_test_encoded, y_prob, average="macro", multi_class="ovr"
            )
            avg_pr_weighted = average_precision_score(y_test, y_prob, average="weighted")
          


        return accuracy, avg_pr_macro, important_features, f1, auc_roc, avg_pr_weighted

   

    def run_train_and_evaluate(self, split_no=5):
        (synthetic_data, synthetic_labels, X_train_real, y_train_real, 
         X_test_real, y_test_real) = self.data_loader.load_data()

        class_encoder = LabelEncoder()
        y_test_encoded = class_encoder.fit_transform(y_test_real)
        ### Shuffling synthetic data just in case..
        synthetic_data, synthetic_labels = shuffle(synthetic_data, synthetic_labels, random_state=self.random_seed)
        
        accuracy_synthetic, avg_pr_macro_synthetic, ifeat_synthetic, f1_syn, auroc_syn, avg_pr_weight_syn = self.train_and_evaluate(
            synthetic_data, synthetic_labels, X_test_real, y_test_real, y_test_encoded,
             X_train_real, reset_model=True)

        accuracy_real, avg_pr_macro_real, ifeat_real, f1_real, auroc_real, avg_pr_weight_real = self.train_and_evaluate(
            X_train_real, y_train_real, X_test_real, y_test_real, y_test_encoded, 
            X_train_real, reset_model=True)

        overlap_count, overlap_proportion = Statistics.count_feature_overlap(ifeat_synthetic, ifeat_real)
        avg_distance = Statistics.distance_to_the_closest_neighbor(X_train_real, synthetic_data)
        avg_distance_base = Statistics.distance_to_the_closest_neighbor(X_train_real, X_test_real)
        disc_score = self.discriminative_score(synthetic_data, X_train_real, X_test_real)
        ### MMD
        train_mmd_score = Statistics.get_mmd_score(X_train_real, synthetic_data)
        test_mmd_score = Statistics.get_mmd_score(X_test_real, synthetic_data)
        base_mmd_score = Statistics.get_mmd_score(X_test_real, X_train_real)
        ## label-based MMD
        train_mmd_score_label = Statistics.label_based_mmd_scores(X_train_real, y_train_real, 
                                                synthetic_data, y_synthetic=synthetic_labels)
        # how to save this to csv nicely? right now it's a list


        ### KL 
        train_kl_mean, _ = Statistics.compute_kl_divergences(
            synthetic_data, X_train_real
        )
        test_kl_mean, _ = Statistics.compute_kl_divergences(
            synthetic_data, X_test_real
        )
        base_test_kl_mean, _ = Statistics.compute_kl_divergences( ## to compare against..
            X_train_real, X_test_real
        )
        prdc_metrics_train = PRDensityCoverage.compute_prdc(
            real_features=X_train_real,
            fake_features=synthetic_data,
            nearest_k=5
        )
        prdc_metrics_test = PRDensityCoverage.compute_prdc(
            real_features=X_test_real,
            fake_features=synthetic_data,
            nearest_k=5
        )


        #### add here....
        return {
            'split_no': split_no,
            'accuracy_synthetic': accuracy_synthetic,
            'avg_pr_macro_synthetic': avg_pr_macro_synthetic,
            'avg_pr_weight_synthetic': avg_pr_weight_syn,
            'f1_synthetic': f1_syn,
            'auroc_synthetic': auroc_syn,
            'accuracy_real': accuracy_real,
            'avg_pr_macro_real': avg_pr_macro_real,
            'avg_pr_weight_real': avg_pr_weight_real,
            'f1_real': f1_real,
            'auroc_real': auroc_real,
            'feature_overlap_count': overlap_count,
            'feature_overlap_proportion': overlap_proportion,
            'MMD_train': train_mmd_score,
            'MMD_test': test_mmd_score,
            'MMD_base': base_mmd_score,
            'discriminative_score': disc_score,
            'distance_to_closest': avg_distance,
            'distance_to_closest_base': avg_distance_base,
            'kl_mean_train': train_kl_mean,
            'kl_mean_test': test_kl_mean,
            'kl_mean_base': base_test_kl_mean,
            'prdc_precision': prdc_metrics_train['precision'],
            'prdc_recall': prdc_metrics_train['recall'],
            'prdc_density': prdc_metrics_train['density'],
            'prdc_coverage': prdc_metrics_train['coverage'],
            'prdc_density_test': prdc_metrics_test['density'],
            'prdc_coverage_test': prdc_metrics_test['coverage'],
            **train_mmd_score_label
        }





@click.group()
def cli():
    pass


### function runs for an individual split
### results are saved under
### results/files/{dataset_name}/{model_name}/{experiment_name}
@click.command()
@click.argument('split-no', type=int, default=1)
@click.argument('generator_name', type=str, default=None)
@click.argument('experiment_name', type=str, default=None)
@click.option('--configfile', type=str, default="config.yaml")
def run_evaluator(split_no: int, generator_name: str, experiment_name: str, configfile: str):
    with open(configfile, 'r') as file:
        config = yaml.safe_load(file)
    
    evaluator = ModelEvaluator(config=config, 
                               split_no=split_no, 
                               generator_name=generator_name, 
                               experiment_name=experiment_name)
    results = evaluator.run_train_and_evaluate(split_no=split_no)
    
    output_file = os.path.join(evaluator.res_files_dir, f"evaluation_split_{split_no}.csv")
    evaluator.save_split_results(results, output_file)
    click.echo(f"Evaluation for split {split_no} completed. Results saved to {output_file}")


@click.command()
@click.argument('generator_name', type=str, default=None)
@click.argument('experiment_name', type=str, default=None)
@click.option('--configfile', type=str, default="config.yaml")
def combine_results(generator_name: str, experiment_name: str, configfile: str):
    with open(configfile, 'r') as file:
        config = yaml.safe_load(file)

    evaluator = ModelEvaluator(config=config, 
                               split_no=0, 
                               generator_name=generator_name, 
                               experiment_name=experiment_name)
    #results_files = [os.path.join(evaluator.res_files_dir, f) 
    #                 for f in os.listdir(evaluator.res_files_dir,) if f.endswith('.csv')]
    results_files = [os.path.join(evaluator.res_files_dir, f) 
                     for f in os.listdir(evaluator.res_files_dir) 
                     if fnmatch.fnmatch(f, 'evaluation_split*.csv')]
    output_file = os.path.join(evaluator.res_files_dir, f"evaluation_results.csv")
    ModelEvaluator.combine_csv_files(results_files, output_file)
    click.echo(f"Combined results saved to {output_file}")


@click.command()
@click.argument('cutoff', type=float, default=0.0)
@click.argument('generator_name', type=str, default=None)
@click.argument('experiment_name', type=str, default=None)
@click.option('--configfile', type=str, default="config.yaml")
def combine_coexpress_results(cutoff, generator_name, experiment_name, configfile):
    with open(configfile, 'r') as file:
        config = yaml.safe_load(file)
    evaluator = ModelEvaluator(config=config, 
                               split_no=0, 
                               generator_name=generator_name, 
                               experiment_name=experiment_name)
    results_files = [
        os.path.join(evaluator.bio_files_dir, f)
        for f in os.listdir(evaluator.bio_files_dir)
        if fnmatch.fnmatch(f, f"coexpr_*{cutoff:g}_split*.csv")
    ]

    print(results_files)

    combined_df = pd.concat(
        [
            pd.read_csv(f).assign(
                fold=int(re.search(r"_split_(\d+)", f).group(1))
            )
            for f in results_files
        ],
        ignore_index=True
    )

    combined_df.to_csv(
        os.path.join(evaluator.bio_files_dir, f"coexpr_cutoff={cutoff}_results.csv"),
        index=False
    )

    for f in results_files:
        if os.path.exists(f):
            os.remove(f)
    logging.info("Cleanup completed.")


@click.command()
@click.argument('lfc_threshold', type=float, default=0)
@click.argument('generator_name', type=str, default=None)
@click.argument('experiment_name', type=str, default=None)
@click.option('--configfile', type=str, default="config.yaml")
def combine_diffexpress_results(lfc_threshold, generator_name, experiment_name, configfile):
    with open(configfile, 'r') as file:
        config = yaml.safe_load(file)
    evaluator = ModelEvaluator(config=config, 
                               split_no=0, 
                               generator_name=generator_name, 
                               experiment_name=experiment_name)
    for kyw in ['tpr', 'fpr']:
        results_files = [
            os.path.join(evaluator.bio_files_dir, f)
            for f in os.listdir(evaluator.bio_files_dir)
            if fnmatch.fnmatch(f, f"DE_*{lfc_threshold:.1f}_split*_{kyw}.csv")
        ]

        print(results_files)

        combined_df = pd.concat(
            [
                pd.read_csv(f).assign(
                    fold=int(re.search(r"_split_(\d+)", f).group(1))
                )
                for f in results_files
            ],
            ignore_index=True
        )

        combined_df.to_csv(
            os.path.join(evaluator.bio_files_dir, f"DE_lfc={lfc_threshold}_results_{kyw}.csv"),
            index=False
        )

        for f in results_files:
            if os.path.exists(f):
                os.remove(f)
        logging.info("Cleanup completed.")


@click.command()
@click.argument('generator_name', type=str, default=None)
@click.argument('experiment_name', type=str, default=None)
@click.option('--configfile', type=str, default="config.yaml")
def combine_pathway_results(generator_name, experiment_name, configfile):
    with open(configfile, 'r') as f:
        config = yaml.safe_load(f)

    evaluator = ModelEvaluator(
        config=config,
        split_no=0,
        generator_name=generator_name,
        experiment_name=experiment_name
    )

    results_files = [
        os.path.join(evaluator.bio_files_dir, f)
        for f in os.listdir(evaluator.bio_files_dir)
        if fnmatch.fnmatch(f, "pathway_metrics_split_*.csv")
    ]

    if not results_files:
        logging.warning("No pathway metrics files found.")
        return

    print(results_files)

    combined_df = pd.concat(
        [
            pd.read_csv(f).assign(
                fold=int(re.search(r"_split_(\d+)", f).group(1))
            )
            for f in results_files
        ],
        ignore_index=True
    )

    out_path = os.path.join(evaluator.bio_files_dir, "pathway_metrics_results.csv")
    combined_df.to_csv(out_path, index=False)
    logging.info(f"Saved combined pathway metrics to {out_path}")

    for f in results_files:
        if os.path.exists(f):
            os.remove(f)
    logging.info("Cleanup completed.")




### function runs for an individual split
### results are saved under
### results/figures/{dataset_name}/{model_name}/{experiment_name}
@click.command()
@click.argument('split_no', type=int, default=1)
@click.argument('generator_name', type=str, default=None)
@click.argument('experiment_name', type=str, default=None)
def plot_pca(split_no: int, generator_name: str, experiment_name: str):
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    base_eval = BaseEvaluator(config, split_no, generator_name, experiment_name)
    synthetic_data, _,  X_train_real, _, _, _ = base_eval.data_loader.load_data()
    
    # Combine real + synthetic or fit on realrun-evaluator  first
    real_pca_df, fitted_scaler, fitted_pca = Plotting.perform_pca(X_train_real)

    # Apply the same scaler and PCA to synthetic
    synthetic_pca_df, _, _ = Plotting.perform_pca(synthetic_data, scaler=fitted_scaler, pca_model=fitted_pca)
    
    plot_path = os.path.join(base_eval.res_figures_dir, f'pca_split_{split_no}.png')
    Plotting.plot_pca_and_save(real_pca_df, synthetic_pca_df, plot_path)
    click.echo(f"PCA plot saved to {plot_path}")




cli.add_command(run_evaluator)
cli.add_command(combine_results)
cli.add_command(plot_pca)
cli.add_command(combine_coexpress_results)
cli.add_command(combine_diffexpress_results)
cli.add_command(combine_pathway_results)

if __name__ == '__main__':
    cli()