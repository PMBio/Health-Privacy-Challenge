# CAMDA / ELSA Health Privacy Challenge

This repository holds the code and documentation for the [Health Privacy Challenge](https://benchmarks.elsa-ai.eu/?ch=8), run within [CAMDA/ISMB Conference](https://bipress.boku.ac.at/camda2026/) in 2025 and 2026, and its accompanying paper. The  Health Privacy Challenge is organized in the context of the European Lighthouse on Safe and Secure AI (ELSA, https://elsa-ai.eu). 

**Active edition:** None. The 2026 challenge is concluded in July, 2026.

<!-- When a new edition opens, replace the line above with, e.g.:
     **Active edition:** running in 2027 → see the [`2027`](link) branch. -->
**Jump to:** [Challenge tracks](#challenge-tracks) · [Paper](#paper) ([Data](#datasets) · [Reproducing](#reproducing-the-paper)) · [Editions & branches](#editions) · [Citation](#citation)
 


## Challenge tracks 
The challenge ran two tracks (full details on the edition branches):
- **[Track I — Bulk RNA-seq](https://github.com/PMBio/Health-Privacy-Challenge/tree/camda-2026/experiments/track_i)**. A Blue Team 🫐 vs Red Team 🍅 scheme. Blue teams build privacy-preserving generative methods for bulk gene expression; red teams run membership-inference attacks against the baselines and blue-team solutions to test their privacy robustness.

- **[Track II — Single-cell RNA-seq](https://github.com/PMBio/Health-Privacy-Challenge/tree/camda-2026/experiments/track_ii)**. Privacy and utility of synthetic scRNA-seq data in a multi-sample donor setting: building generative methods, proposing suitable evaluation metrics, and revealing privacy risks. 


## Paper 
[**Towards Useful and Private Synthetic Omics: Community Benchmarking of Generative Models for Transcriptomics Data**](https://www.biorxiv.org/content/10.64898/2026.03.02.707794v1.abstract) (biorXiv, 2026)
 
The paper focuses on **Track I, Blue Team methods**  — the privacy-preserving generative
models — evaluated against membership-inference attacks provided in this repo. Track II and the Red-team tasks are documented on the challenge edition branches but are outside this paper's scope.

The **2025 Blue Team submissions** are the generative methods under analysis in addition to the baseline methods provided in the repo.

**Participant methods included** (code archived by the authors; cite their DOIs):
| Method | Authors | DOI |
|--------|---------|-----|
| DP-PGM | Pentyala et al. 2026 | [zenodo.23068244](https://doi.org/10.5281/zenodo.23068244) |
| Embedded-Diffusion | Kreuer. 2026 | [zenodo.22693680]( https://doi.org/10.5281/zenodo.22693680) |
| NMF / P-NMF | Wicks. 2026 | [zenodo.22818979](https://doi.org/10.5281/zenodo.22818979 ) |

Follow their instructions for model training and generation. 

The paper also introduces additional metrics beyond the challenge scoring. Reproduce the baseline generative methods and evaluation with [v1.0-paper]() (Track I, Blue team). 


## Datasets

- **Preprocessed data:** available at [zenodo.22996320](https://doi.org/10.5281/zenodo.22996320).
- Also available at [ELSA Benchmarks  Platform](https://benchmarks.elsa-ai.eu/?ch=8&com=introduction) after registration.
- All benchmark materials are collected in the [Zenodo challenge community](https://zenodo.org/communities/health-privacy-challenge-2025/records).
 
 We re-distribute pre-processed versions of two open-access bulk TCGA RNA-seq datasets, available through the  [GDC portal](https://gdc.cancer.gov); use follows the [NIH Data
Access Policy](https://gdc.cancer.gov/access-data/data-access-policies) and TCGA citation guidelines.

- **TCGA-BRCA.** suitable for cancer subtype prediction (5 subtypes); 1089 x 978 (individuals x [landmark genes](data/f1000_lm_ensembl.tsv)). 

- **TCGA-COMBINED.** suitable for cancer type prediction (10 tissues / 12 projects); 4323 x 978 (individuals x [landmark genes](data/f1000_lm_ensembl.tsv)).



## Reproducing the paper
 
**Figures only.** The paper's results are frozen in
`paper/results/` and the `paper_figures.ipynb` regenerates tables and figures from them:
 
```bash
#   open paper/paper_figures.ipynb and run all cells
```

 
**Regenerate from the scratch (cluster version).** Prepare the data, run the generative
methods and evaluation (including baseline MIA), then compile the results in the notebook. This assumes 
the code runs on SLURM cluster (Use GPU for training if available). 


**Activate the environment**

Create the Conda environment using the provided YAML file:

```bash
conda env create -f environment.yaml
```

Then activate it:

```bash
conda activate health-privacy-env
```


Each dataset has its own config (`config_brca.yaml`, `config_combined.yaml`); run the steps once per
dataset by pointing at the relevant config.

> > **Important:** Update the `home` path under `dir_list` in the configuration YAML file. By default, it is set to `~/Health-Privacy-Challenge`.



```bash
# 1. data prep (per dataset)
python src/prepare/fetch_data.py                            # pull + verify + extract from Zenodo
python src/prepare/split_data.py --config submit_jobs/config-brca.yaml  # folds + MIA membership labels
python src/prepare/split_data.py --config submit_jobs/config-combined.yaml
 

# 2. generate synthetic data (SLURM job arrays, per dataset)
bash submit_jobs/submit_generators.sh submit_jobs/config-brca.yaml
 
# 3. evaluate synthetic (SLURM job arrays, per dataset) — run after generation completes
bash submit_jobs/submit_eval.sh submit_jobs/config-brca.yaml   # fidelity + utility 
bash submit_jobs/submit_mia.sh  submit_jobs/config-brca.yaml    # MIA
bash submit_jobs/submit_bio.sh  submit_jobs/config-brca.yaml    # DE, co-expr, pathway

```
 
 > [!NOTE]
> - **TCGA-COMBINED, MIA step:** pass `--reference_file <..._reference.tsv>` to the MIA
>   call (see the comment in `submit_jobs/run_mia.sh`). BRCA does not use it.
> - **R stages (DE, coexpression, pathway)** run under **R 4.3.2**. Install the CRAN /
>   Bioconductor packages with `Rscript env_R.R`. Coexpression
>   requires [hcocena](https://github.com/MarieOestreich/hCoCena) — install it by
>   following repository's own instructions.

 
## Editions

This code-base evolved with the challenges and its accompaniying paper. Each edition is frozen as a release and  mirrored as a branch for browsing.
 
| Edition | Release (frozen) | Branch |
|--------|------------------|--------|
| 2025 | [`challenge-2025`](https://github.com/PMBio/Health-Privacy-Challenge/releases/tag/v2025) | [`camda-2025`](https://github.com/PMBio/Health-Privacy-Challenge/tree/camda-2025) |
| 2026 | [`challenge-2026`](https://github.com/PMBio/Health-Privacy-Challenge/releases/tag/v2026) | [`camda-2026`](https://github.com/PMBio/Health-Privacy-Challenge/tree/camda-2026) |
| Paper | [`v1.0-paper`]() | `main` |
 

 ## Citation
 
If you use this benchmark, please cite:
 
```bibtex
@article {{\"O}zt{\"u}rk2026,
	author = {{\"O}zt{\"u}rk, Hakime and Afonja, Tejumade and J{\"a}lk{\"o}, Joonas and Binkyte, Ruta and Rodriguez-Mier, Pablo and Lobentanzer, Sebastian and Wicks, Andrew and Kreuer, Jules and Ouaari, Sofiane and Pfeifer, Nico and Menzies, Shane and Pentyala, Sikha and Filienko, Daniil and Golob, Steven and McKeever, Patrick and Banerjee, Jineta and Foschini, Luca and De Cock, Martine and Saez-Rodriguez, Julio and Fritz, Mario and Stegle, Oliver and Honkela, Antti},
	title = {Towards Useful and Private Synthetic Omics: Community Benchmarking of Generative Models for Transcriptomics Data},
	year = {2026},
	doi = {10.64898/2026.03.02.707794},
	journal = {bioRxiv}
}

```


## License
Code is licensed under **BSD-3-Clause** (see [`LICENSE`](LICENSE)). Data terms are
stated on the Zenodo deposit (derived from public TCGA gene expression; see
[Data](#datasets)).
 





<!-- 
## :pushpin: Statement

Membership inference attacks (MIA) aim to re-identify the training data points used to generate synthetic datasets from the original dataset. This re-identification process pertains only to identifying the pseudo-identities within the dataset and **does not, in any way, attempt to re-identify the original donors.**
-->