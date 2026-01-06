# :blueberries: Track II: Single-cell RNA-seq (Blue Team)

Welcome to Track II home page! 

**Contents**
- [Prerequisites](#prerequisites)
- [Tasks](#tasks)
- [Guideline for running and evaluating baseline methods](#thread-guideline-for-running-and-evaluating-baseline-methods)
- [Guideline for developing your own method](#bookmark_tabs-guideline-for-developing-your-own-method)
- [Submission checklist](#white_check_mark-submission-checklist)
- [Citations](#pencil-citations)


## Prerequisites

### Python environment

We suggest installing the provided environment to reproduce the starter kit. We recommend using micromamba, though you may also replace micromamba with conda. 

```bash
micromamba create --file sc_environment.yaml --name <env_name>
```

or

```bash
conda env create -f sc_environment.yaml -n <env_name>
```

### scib package

In the sc_environment.yaml scib is installed from Github directly.
```bash
pip install git+https://github.com/theislab/scib.git
```

This should solve two potential errors you might face while running evaluation script `sc_evaluate.py`.

1. https://github.com/theislab/scib/issues/308 `FileNotFoundError: [Errno 2] No such file or directory: '~/lisi_002vg90t/graph_lisi_indices_0.txt'`
2. https://github.com/theislab/scib/issues/253 `Exec format error: '~~/scib/knn_graph/knn_graph.o' ` 



## Tasks 
- In Track II, the participants work towards **developing methods that improve the baseline methods and generating novel insights into privacy preservation in multi-sample per donor setting** on scRNA-seq dataset.
- Train and test splits of the [OneK1K single-cell RNA-seq counts](https://onek1k.org) are provided in `h5ad` format, with corresponding cell-type inside `annData` observation. 
- **Train dataset** should be used to **train the generator**, and the **evaluation** should be performed against the **test set**. 

## OneK1K Dataset

We re-distribute raw counts of OneK1K single-cell RNA-seq dataset (https://onek1k.org/), a cohort containing 1.26 million peripheral blood mononuclear cells (PBMCs) of 981 donors. 

After the following filtering, 
```python 
    sc.pp.filter_genes(adata, min_cells=3) 
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.total_counts > 10,:]
    adata = adata[adata.obs.total_counts < 40000,:]
```

- The dataset is split into donor-based train and test sets of relatively equal numbers of cells, and similar cell-type distributions. 

- **Train dataset:** < 633711 cells from 490 donors x 25834 genes>, **to be used in training** generative models
- **Test dataset:**  < 634022 cells from 491 donors x 25834 genes>, **to be used in evaluating** the generated synthetic dataset

We share these datasets in `annData` format with the following annotations: `individual`, `barcode_col`, `cell_type`, `cell_label`. 

### Download 

- Train and test datasets can be downloaded from the [ELSA Benchmark website](https://benchmarks.elsa-ai.eu/?ch=8) after registration and signing the data download agreement. 
- `config.yaml` files inside [generation](/experiments/track_ii/1_generation/) and  [evaluation](/experiments/track_ii/2_evaluation/) organize directory structure and configurations. Please ensure that the downloaded datasets are placed in the corresponding directories under `dataset_config`, or update the directory path according to your preference.

```bash
dataset_config:
  name: "onek1k"
  train_count_file: "data/processed/onek1k/onek1k_annotated_train.h5ad" 
  test_count_file: "data/processed/onek1k/onek1k_annotated_test.h5ad" 
```



## :thread: Guideline for running and evaluating baseline methods

Please follow suggested steps below:
1. **[Generate synthetic data](/experiments/track_ii/1_generation/)**: Generates synthetic data using baseline methods. 
2. **[Evaluate](/experiments/track_ii/2_evaluation/)**: Reports the evaluation metrics for the given synthetic single-cell dataset. 
3. **[Investigate privacy](/experiments/track_ii/3_privacy/)**: A demonstration of a membership inference attack on a small, subsampled synthetic single-cell RNA-seq dataset. 



## :bookmark_tabs: Guideline for developing your own method

We provide `BaseSingleCellDataGenerator` class inside [generators/models/sc_base.py](/src/generators/models/sc_base.py) for generator models to inherit. This class handles the configuration arguments passed through `config.yaml`. 

It also has three abstract functions:  
- `train()`, 
- `load_from_checkpoint()`, 
- `generate()`, returns synthetic anndata, with cell types given as integers 
-  and `save_synthetic_anndata()`, a fixed function that implements saving the synthetic datasets `h5ad` format. 

We expect you to adopt the following logic [blue_team.py](/src/generators/blue_team.py) while developing your own generator. This will help us to re-produce your code easily. 



```python
## your synthetic data will be saved accordingly to config.yaml
## e.g. data_splits/{dataset_name}/synthetic/{generator_name}/{experiment_name}
## change the corresponding keys in the config.yaml
@click.command()
@click.option('--experiment_name', type=str, default="")
def run_singlecell_generator(experiment_name: str = None):
    # Load the config file
    configfile = "config.yaml"
    config = yaml.safe_load(open(configfile))

    generator_name = config.get('generator_name')
    GeneratorClass = get_generator_class(generator_name)


    if not GeneratorClass:
        raise ValueError(f"Unknown generator name: {generator_name}")

    generator = GeneratorClass(config)


    if not config.get("load_from_checkpoint", False):
        if config.get("train", False):
            generator.train()
    else:
        generator.load_from_checkpoint()

    if config.get("generate", False):
        syn_data = generator.generate()
        generator.save_synthetic_anndata(syn_data, experiment_name)
```

Please test your code inside `run_singlecell_generator` function to ensure its reproducibility. 


## :white_check_mark: Submission checklist
The following files are required for submission, in zipped form, `trackii_{yourteamname}_onek1k.zip`. 

1. Modified [blue_team.py](/src/generators/blue_team.py) with your generator class included, and any additional `.py` file required to run your method. 
2. Updated `config.yaml` file containing `{your_generator}_config`. 
3. `environment.yaml` that is required to run your method.

Please note that method submissions are collected through [**ELSA Benchmark Platform**](https://benchmarks.elsa-ai.eu/?ch=8), and the accompanying CAMDA extended abstracts must be summited through [**ISMB submission system**](https://www.iscb.org/ismb2026/home).  


### :pencil: CITATIONS

Please make sure to cite the following papers if any of the baseline methods and evaluation metrics are mentioned/utilized **in your CAMDA extended abstracts.**

**Competition related**

1. CAMDA 2026 Health Privacy Challenge

**Dataset sources**

2. Yazar S., Alquicira-Hernández J., Wing K., Senabouth A., Gordon G., Andersen S., Lu Q., Rowson A., Taylor T., Clarke L., Maccora L., Chen C., Cook A., Ye J., Fairfax K., Hewitt A., Powell J. "Single cell eQTL mapping identified cell type specific control of autoimmune disease." Science. (2022) (https://onek1k.org) 


**Dataset preprocessing**

3. Lun ATL, McCarthy DJ, Marioni JC. “A step-by-step workflow for low-level analysis of single-cell RNA-seq data with Bioconductor.” F1000Res. (2016)

4. Wolf, F. Alexander, Philipp Angerer, and Fabian J. Theis. "SCANPY: large-scale single-cell gene expression data analysis." Genome biology. (2018)


**Evaluations**

5. Luo, E., Hao, M., Wei, L., & Zhang, X. "scDiffusion: conditional generation of high-quality single-cell data using diffusion model." arXiv preprint. (2024)

6. Gretton, A., Sejdinovic, D., Strathmann, H., Balakrishnan, S., Pontil, M., Fukumizu, K., & Sriperumbudur, B. K.  "Optimal kernel choice for large-scale two-sample tests." Advances in neural information processing systems. (2012)

7. Korsunsky, I., Millard, N., Fan, J., Slowikowski, K., Zhang, F., Wei, K., ... & Raychaudhuri, S. "Fast, sensitive and accurate integration of single-cell data with Harmony." Nature methods. (2019)

8. Luecken, M.D., Büttner, M., Chaichoompu, K. et al. "Benchmarking atlas-level data integration in single-cell genomics." Nat Methods. (2022)

9. McInnes, Leland, John Healy, and James Melville. "Umap: Uniform manifold approximation and projection for dimension reduction." arXiv preprint. (2018)

10. Domínguez Conde, C., Xu, C., Jarvis, L. B., Rainbow, D. B., Wells, S. B., Gomes, T., ... & Teichmann, S. A. "Cross-tissue immune cell analysis reveals tissue-specific features in humans." Science. (2022)
 

**Membership inference attack models**

11. Van Breugel, B., Sun, H., Qian, Z., & van der Schaar, M. "Membership inference attacks against synthetic data through overfitting detection." arXiv preprint. (2023)

12. Chen, D, Yu, N., Zhang, Y., and Fritz, M. "Gan-leaks: A taxonomy of membership inference attacks against generative models."  In Proceedings of the 2020 ACM SIGSAC conference on computer and communications security (2020)

13. Hilprecht, B., Härterich, M., & Bernau, D.  "Monte carlo and reconstruction membership inference attacks against generative models." Proceedings on Privacy Enhancing Technologies. (2019)

<!-- comment 16. Hayes, J., Melis, L., Danezis, G. & De Cristofaro, E. "Logan: Membership inference attacks against generative models." arXiv preprint. (2019) -->






**Best of luck!** :four_leaf_clover:





