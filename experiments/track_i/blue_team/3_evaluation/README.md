# :blueberries: BLUE TEAM: Evaluate synthetic bulk RNA-seq data

Evaluation workflow:

1. [**Prerequisites**](#prerequisites): Preparations evaluation workflow setup.
2. [**Generate split-specific evaluation results**](#step-1-generate-split-specific-results): Run an evaluation script for each data split.
3. [**Combine evaluation results**](#step-2-combine-split-specific-results): Merge all split-specific results into a single file.
4. [**Run PCA analysis**](#step-3-run-pca-analysis): Perform PCA for each data split and save the visualizations.
5. [**Biological analysis**](#step-4-biological-analysis): Perform differential gene expression and gene co-expression analyses.
6. [**Evaluation metrics**](#evaluation): Evaluation metrics provided are explained. 
7. [**Baseline evaluation results**](#baseline-results): Performances of the baseline generator methods are provided for comparison. 

---

## Prerequisites

### **Activate the python environment**: 

```bash
micromamba activate <environment>
```

### **Ensure the following directories exist**:
Remember to update `home_dir`. The other configurations can stay depending on your preferences. 
   - `res_files`: Directory for storing evaluation metric scores in a CSV file.
   - `figures`: Directory for saving PCA plots.
   - `bio_files`: Directory to store image and csv files for biological evalutions. 
   - `mia_files`: Directory to store membership inference attack scores. Please refer to [Red Team homepage](/experiments/track_i/red_team/README.md) if you want to test this out. 
 

### **Configuration Variables**:
You need to modify `config.yaml` according to your need for each experiment. Define the following variables according to your setup in `config.yaml`:
   - `dataset_config`: Update the name of the dataset you want to evaluate. Total number of splits remains same as in your generation configuration (e.g., `5`).
   - `generator_config`: Update the name of the method and experiment name you want to generate evaluation scores. 

Please be reminded that you need to put `config.yaml` in the same directory you are running your experiment. 


### Example config for evaluating multivariate

e.g. In the Generation step, we set the `--experiment_name` argument as `noise_0.5`, therefore we will use the same `experiment_name` in `generator_config`.  

```bash
dataset_config:
  name: "TCGA-COMBINED" 
  num_splits: 5


generator_config:
  name: "multivariate"
  experiment_name: "noise_0.5"
```


## Step 1: Generate Split-Specific Results
Refer to [Evaluation Metrics](#evaluation-metrics) for the list of evaluation metrics generated. 

Make sure to replace `<src_dir>`  with your corresponding path. For each data split (from 1 to `split_num`), run the following command:

```bash
python {src_dir}/evaluation/evaluate.py run-evaluator {split_no}
```

For each split, evaluation metrics are computed and saved under: ``{home_dir}/{res_files}/{dataset_name}/{generator_name}/{experiment_name}/split_{split}.csv``


## Step 2: Combine Split-Specific Results

`combine-results` function  assumes all generated split results (split_{split_no}.csv) are saved under the following directory: `{home_dir}/{res_files}/{dataset_name}/{generator_name}/{experiment_name}`. 

Run the below line to combine the results from each split file into single CSV file and to compute the average of the folds. 

```bash
python {src_dir}/evaluation/evaluate.py combine-results 
```

The output is a CSV file in the same directory above named `evaluation_results.csv`.


## Step 3: Run PCA Analysis

For each split, generate a PCA visualization using the following command:

```bash
python {src_dir}/evaluation/evaluate.py plot-pca {split}
```

This command generates a PCA plot for each split in the following format:
``{home_dir}/{figures}/{dataset_name}/{generator_name}/{experiment_name}/pca_compare_split_{split}.png``


## Step 4: Biological Analysis

We adopt [the analysis](https://github.com/MarieOestreich/PRO-GENE-GEN/tree/main/eval/bio_eval) generated for [(Chen, Oestreich, & Afonja et al. 2024)](https://arxiv.org/abs/2402.04912) and utilize `hcocena` for gene co-expression and `scran` for differential gene expression analysis on R. 

### Install R libraries

The scripts are executed in `R/4.3.2` using the below R packages, make sure to install them before running the provided scripts.  

```R
library(hcocena)
library(org.Hs.eg.db)
library(AnnotationDbi)
library(dplyr)
library(ggplot2)
library(scran)
```

### Differential expression

`{p_value_th}` is assigned as `0.05` for the significance threshold, and can be modified from `bio_params` key `config.yaml`. Essentially, pairwise Wilcoxon tests are carried out in this script. 

```bash
    module load R/4.3.2
    Rscript {src_dir}/evaluation/bio/diffexpression.R {home_dir} {split_no} {dataset_name} {generator_name} {experiment_name} {p_value_th} {lfc}
```

- While running this script, we explored setting log-fold-change (lfc) to  `0.0 ` and  `0.5 `. We encourage you to experiment with these values to assess how your model responds.


### Co-expression

 `hcocena` package requires some reference files from the [hcocena repo](https://github.com/MarieOestreich/hCoCena). Please clone the repo first, and save it under `{hcocena_dir}` in your local.

```bash     
    module load R/4.3.2
    Rscript {src_dir}/evaluation/bio/coexpression.R {home_dir} {split_no} {dataset_name} {generator_name} {experiment_name} {hcocena_dir} {cutoff}
```
- While running this script, we explored different co-expression correlation cut-offs such as  `0.0 ` and  `0.3 `. We encourage you to experiment with these values to assess how your model responds.


# Evaluation 

## Metric definitions 
- Here we describe the list of evaluation metrics used in [evaluate.py](/src/evaluation/evaluate.py). **We strongly encourage the participants to also use other evaluation metrics, or even, propose their own.**

- Within the Blue Team evaluation script, we provide **distance-to-closest** as a proxy metric for privacy. **Please follow the instructions on the [Red Team Homepage](/experiments/track_i/red_team/) in order to run the provided black-box MIA models** on your synthetic data to get a more comprehensive assesment of privacy risk. 
  - Here we report attack success under the GAN-Leaks method, measured as TPR at FPR = 0.1. Note that reproducing the below results for this metric and additional MIA-relevant metrics **require running the full Red Team setup**.

- Biological plausibility metrics are computed post hoc from the output CSV files generated by running the [**Bio Evaluation**](/src/evaluation/bio/) scripts. Participants are free to compute additional relevant metrics from these files. Refer to [**Biological analysis**](#step-4-biological-analysis) for details. 


- The term "synthetic datasets" here refers to the datasets generated for each training set in each split. The test split, which is never used in the training or synthetic data generation process, is reserved solely for evaluation purposes, such as training on synthetic data and testing on real data, among other evaluations.

- Downstream task performance may differ slightly from last year’s results due to the introduction of standard scaling in this step.


| Category | Method Name                | Method Details                       | Description                                         | Value (Better) |
|----------|----------------------------|--------------------------------------|-----------------------------------------------------|----------------|
| Utility  | accuracy_synthetic         | Accuracy    |  Train on Synthetic, Test on Real (for downstream task)                                                  | (High)         |
| Utility  | avg_pr_macro_synthetic     | AUPR        | Train on Synthetic, Test on Real (for downstream task)                                                    | (High)         |
| Utility  | accuracy_real         | Accuracy    |  Train on Real, Test on Real (for downstream task)                                                  | (High)         |
| Utility  | avg_pr_macro_real     | AUPR        | Train on Real Test on Real (for downstream task)                                                    | (High)         |
| Utility  | feature_overlap_count      | Number of Overlapping Important Features | 10 features * per class                                               | (High)         |
| Utility  | PCA Plot                   | Visualizing 2D clusters                                   | -                                                   |                |
| Fidelity | MMD_test                  | Maximum Mean Discrepancy             | Difference between synthetic and real test datasets' probability distributions    | (Low)          |
| Fidelity | MMD_train                  | Maximum Mean Discrepancy             | Difference between synthetic and real train datasets probability distributions    | (Low)          |
| Fidelity | KL (test)                 | Maximum Mean Discrepancy             | Difference between synthetic and real test datasets' probability distributions    | (Low)          |
| Fidelity | KL (train)                 | Maximum Mean Discrepancy             | Difference between synthetic and real train datasets' probability distributions    | (Low)          |
| Fidelity | discriminative_score             | Discriminative score           | F1 score for distinguishing  synthetic and real dataset  | (Low)          |
| Privacy  | distance_to_closest        | Distance to the Closest Neighbor     | Average distance of synthetic dataset to the nearest real data point                  | (High)         |
| Privacy  | distance_to_closest_base        | Distance to the Closest Neighbor     | Average distance within real dataset to the nearest data point                  | (High)         |
| Privacy  | tpr_at_fpr_01       | MIA performance under GAN-leaks    | TPR @ FPR = 0.1             | (High)         |
| Biological Plausibility  | co-expr_precision       | Co-expression Mean Precision (r >0.3)    | Correct / (Correct + False)                  | (High)         |
| Biological Plausibility  | co-expr_num_correct_edges       | Co-expression Mean Correctly Recovered Edges (r >0.3)    | Number of Correct Edges                | (High)         |
| Biological Plausibility  | DE_TPR_up      | Differential Expression Recovery True Positive Rate (Up-regulation)  |  TPR @ FPR <= 0.05                 | (High)         |
| Biological Plausibility  | DE_TPR_down      | Differential Expression Recovery True Positive  Rate (Up-regulation)   | TPR @ FPR <= 0.05                 | (High)         |


## Baseline results

- Here we include the performance of some of the provided baseline generative methods. Default values in [config.yaml](/experiments/track_i/blue_team/2_generation/config.yaml) are used, and the average scores across folds are reported.  


### TCGA-BRCA

| Metric / Method            | Multivariate | CVAE-GMM  | DP-CVAE | WGAN-GP | 
|----------------------------|--------------|-----------|---------|---------|
| accuracy_synthetic         | 0.8494       | 0.8296    | 0.6694  | 0.7924  | 
| accuracy_real              | 0.8485       | 0.8485    | 0.8485  | 0.8485  | 
| avg_pr_macro_synthetic     | 0.8335       | 0.7991    | 0.4374  | 0.7819  | 
| avg_pr_macro_real          | 0.8513       | 0.8513    | 0.8513  | 0.8513  | 
| feature_overlap_count      | 17.0.        | 15.8      | 1.8.    | 6.8     | 
| MMD_train                  | 0.0180       | 0.0526    | 0.1463  | 0.0323  | 
| kl_mean_train              | 0.1402       | 1.2003    | 0.4386  | 0.1196  | 
| discriminative_score       | 0.5280       | 0.6939    | 1.0000  | 0.7052  | 
| distance_to_closest        | 28.5348      | 16.7735   | 43.5594 | 19.3614 | 
| distance_to_closest_base   | 24.0435      | 24.0435   | 24.0435 | 24.0435 | 
| tpr_at_fpr_01              | 0.1152       | 0.1492    | 0.1003  | 0.1196  | 
| co-expr_num_correct_edges  | 29691.80     | 36350.80  | 89.25	  | 36369.2 | 
| co-expr_precision          | 0.8785       | 0.3346    | 0.1145  | 0.3763  | 
| DE_TPR_up                  | 0.7857       | 0.9413    | 0.0832  | 0.7515  | 
| DE_TPR_down                | 0.8341       | 0.9463    | 0.1443  | 0.7770  | 



### TCGA-COMBINED

| Metric                        | Multivariate | CVAE-GMM | DP-CVAE | WGAN-GP |
|-------------------------------|--------------|----------|---------|---------|
| accuracy_synthetic            | 0.9780       | 0.9711   | 0.8808  | 0.9623  | 
| accuracy_real                 | 0.9789       | 0.9789   | 0.9789  | 0.9789  | 
| avg_pr_macro_synthetic        | 0.9907       | 0.9825   | 0.6173  | 0.9793  | 
| avg_pr_macro_real             | 0.9913       | 0.9913   | 0.9913  | 0.9913  | 
| feature_overlap_count         | 52.6         | 39.4     | 16.4    | 24.8    | 
| MMD_train                     | 0.0093       | 0.0259   | 0.0953  | 0.0141  | 
| kl_mean_train                 | 0.0554       | 0.4914   | 0.3030  | 0.1327  | 
| discriminative_score          | 0.5775       | 0.7898   | 0.9996  | 0.9532  | 
| distance_to_closest           | 28.8775      | 16.1798  | 58.1306 | 20.1304 | 
| distance_to_closest_base      | 23.2721      | 23.2721  | 23.2721 | 23.2721 | 
| tpr_at_fpr_01                 | 0.1105       | 0.1205   | 0.0999  | 0.1016  | 
| co-expr_num_correct_edges     | 25567.0	     | 32634.6  | -       | 32810.6 | 
| co-expr_precision             | 0.9611       | 0.3796   | -       | 0.4309  | 
| DE_TPR_up                     | 0.9171       | 0.9789   | -       | 0.9498  | 
| DE_TPR_down                   | 0.9078       | 0.9752   | -       | 0.9509  | 




