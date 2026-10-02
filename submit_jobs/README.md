# Cluster pipeline instructions

Regenerate the benchmark results from scratch: data preperation, synthetic data generation and evaluation. 
Generation and evaluation run as **SLURM job arrays**; the R stages need **R 4.3.2**. 

Each dataset has its own config (`config_brca.yaml`, `config_combined.yaml`). Run the steps once per dataset by pointing at the relevant config.

> [!NOTE]
> Read [Environment setup](#environment-setup) first. Several scripts require updates
> to paths and module names. 


## Running pipeline 

```bash
# 0. install and activate the environment
micromamba create --file environment.yaml && micromamba activate health-privacy-env

# 1. data prep (per dataset)
python src/prepare/fetch_data.py                                       # pull + verify + extract from Zenodo
python src/prepare/split_data.py --config workflows/config_brca.yaml   # folds + MIA membership labels
python src/prepare/split_data.py --config workflows/config_combined.yaml

# 2. generate synthetic data (per dataset)
bash submit/submit_generators.sh workflows/config_brca.yaml
bash submit/submit_generators.sh workflows/config_combined.yaml

# 3. score — run AFTER generation completes (each stage walks the generated tree)
bash submit/submit_eval.sh workflows/config_brca.yaml    # evaluator + fan-in combine
bash submit/submit_mia.sh  workflows/config_brca.yaml    # membership-inference attacks
bash submit/submit_bio.sh  workflows/config_brca.yaml    # DE + pathway + coexpression (R)

# 4. compile results for the figure notebook
{{ python src/paper/compile_results.py }}
```

The dataset is selected by which config you pass; the scripts read all paths,
the dataset name, and column settings from it. Each scoring stage builds its task list
from the generated synthetic tree, so scoring must run after generation has finished.

## Environment setup

Several things in these scripts and configs are specific to the original setup and must
be pointed at your own before the pipeline will run:
 
1. **`home` path in the configs.** Every script resolves paths against `dir_list.home`
   in the config YAMLs (default `~/Health-Privacy-Challenge`). Set it to your path before running, in both [`config-brca.yaml`](config-brca.yaml) and [`config_combined.yaml`](config-combined.yaml).


2. **Environment activation.** The scripts source [`activate_env.sh`](activate_env.sh), which hardcodes a
   micromamba binary and env name. Point it at yours by editing `MAMBA_EXE` / `ENV_NAME`. 

3. **`module load` in the R scripts.** `run_de.sh`, `run_coexpr.sh`, and
   `run_pathway.sh` load R via `module load R/4.3.2-..`. Replace that line with however you provide R 4.3.2 (your own module, a conda env, or a direct path).

4. **R packages.** Install the CRAN / Bioconductor packages with `Rscript
   install_R_deps.R`. Coexpression additionally requires
   [hcocena](https://github.com/MarieOestreich/hCoCena), installed separately by
   following that repository's own instructions. If you can't get hcocena installed, comment
   out the coexpression block in `submit_bio.sh` to skip that stage, DE and pathway evaluations should 
   still run.


5. **SBATCH directives.** Partition names, `--gres=gpu`, memory, and walltime in the job
   scripts reflect the original cluster, please adjust them for your scheduler.


## Stage notes

- **TCGA-COMBINED, MIA step:** pass `--reference_file <..._reference.tsv>` to the MIA
  call (see the comment in `run_mia.sh`). TGCA-BRCA does not have a reference file, therefore, cannot use it. 

- **DP-CTGAN:** If you'd like to generate data using DP-CTGAN, please use [`dpctgan_environment.yaml`](../dpctgan_environment.yaml) during `submit_generators.sh` stage. 
