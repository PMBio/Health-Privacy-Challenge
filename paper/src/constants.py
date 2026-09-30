from types import MappingProxyType


MODEL_PATHS = {
    #"real": ["real",  model_name, experiment_name],
    "MVN": [ "multivariate", "noise_0.5"],
    "CVAE": ["cvae", "pp_standard-bs_64-iters_10000-beta_0.001-type=embedding"],
    "DP-CVAE ε=4": ["dpcvae", "pp_standard-bs_64-iters_10000-beta_0.001-type=embedding-eps_4-clip_0.1"],
    #"CVAE-": ["cvae", "pp_standard-bs_64-iters_10000-beta_0.001-type=onehot"],


    "CTGAN": ["ctgan", "pp_standard-bs_64-iters_10000"],
    "DP-CTGAN ε=50": ["dpctgan", "pp_minmax-bs_64-iters_10000-eps_50-clip_0.1"],

    "WGAN-GP": ["wgan_gp", "pp_standard-bs_64-nc_5-nd_128-lgp_10-type=embedding"],
    #"WGAN-GP-": ["wgan_gp", "pp_standard-bs_64-nc_5-nd_128-lgp_10-type=onehot"],
    "CVAE-GMM": ["cvae_gmm", "pp_standard-bs_64-beta_0.001-type=embedding"],
    #"CVAE-GMM-": ["cvae_gmm", "pp_standard-bs_64-beta_0.001-type=onehot"],

    ## participants..
    "EMB-Diffusion": ["embedded_diffusion", "noise"],
    "NMF": ["nmf", "noise"],
    "P-NMF": ["dpnmf", "epsilon=4"],
    "DP-PGM ε=4": ["pgg_pgm", "epsilon=4"],
}


MVN_MODEL_PATHS = {
        "MVN n=0.5": [ "multivariate", "noise_0.5"],
        "MVN n=1": [ "multivariate", "noise_1"],
        "MVN n=5": [ "multivariate", "noise_5"],
}

PNMF_MODEL_PATHS ={
    "P-NMF ε=1": ["dpnmf", "epsilon=1"],
    "P-NMF ε=4": ["dpnmf", "epsilon=4"],
    "P-NMF ε=10": ["dpnmf", "epsilon=10"],
    "P-NMF ε=50": ["dpnmf", "epsilon=50"],

}

REPRESENTATION_PATHS = {

    "WGAN-GP": ["wgan_gp", "pp_standard-bs_64-nc_5-nd_128-lgp_10-type=embedding"],
    "WGAN-GP-": ["wgan_gp", "pp_standard-bs_64-nc_5-nd_128-lgp_10-type=onehot"],
    "CVAE-GMM": ["cvae_gmm", "pp_standard-bs_64-beta_0.001-type=embedding"],
    "CVAE-GMM-": ["cvae_gmm", "pp_standard-bs_64-beta_0.001-type=onehot"],
    "CVAE": ["cvae", "pp_standard-bs_64-iters_10000-beta_0.001-type=embedding"],
    "CVAE-": ["cvae", "pp_standard-bs_64-iters_10000-beta_0.001-type=onehot"],
 
}


DPCVAE_PATHS = {
    "DP-CVAE ε=1": ["dpcvae", "pp_standard-bs_64-iters_10000-beta_0.001-type=embedding-eps_1-clip_0.1"],
    "DP-CVAE ε=4": ["dpcvae", "pp_standard-bs_64-iters_10000-beta_0.001-type=embedding-eps_4-clip_0.1"],
    "DP-CVAE ε=10": ["dpcvae", "pp_standard-bs_64-iters_10000-beta_0.001-type=embedding-eps_10-clip_0.1"],
    "DP-CVAE ε=50": ["dpcvae", "pp_standard-bs_64-iters_10000-beta_0.001-type=embedding-eps_50-clip_0.1"],
}


DPMODEL_PATHS = {
    "DP-CVAE ε=1": ["dpcvae", "pp_standard-bs_64-iters_10000-beta_0.001-type=embedding-eps_1-clip_0.1"],
    "DP-CVAE ε=4": ["dpcvae", "pp_standard-bs_64-iters_10000-beta_0.001-type=embedding-eps_4-clip_0.1"],
    "DP-CVAE ε=10": ["dpcvae", "pp_standard-bs_64-iters_10000-beta_0.001-type=embedding-eps_10-clip_0.1"],
    "DP-CVAE ε=50": ["dpcvae", "pp_standard-bs_64-iters_10000-beta_0.001-type=embedding-eps_50-clip_0.1"],

    "DP-CTGAN ε=50": ["dpctgan", "pp_minmax-bs_64-iters_10000-eps_50-clip_0.1"],

    "DP-PGM ε=1": ["pgg_pgm", "epsilon=1"],
    "DP-PGM ε=4": ["pgg_pgm", "epsilon=4"],
    "DP-PGM ε=10": ["pgg_pgm", "epsilon=10"],
    "DP-PGM ε=50": ["pgg_pgm", "epsilon=50"]
}


DPCVAE_MODEL_COLORS = ({
    "DP-CVAE ε=1": "#f7b538", 
    "DP-CVAE ε=4": "#db7c26", 
    "DP-CVAE ε=10": "#c32f27", 
    "DP-CVAE ε=50":  "#780116",     
})

MVN_MODEL_COLORS = ({
    "MVN n=0.5": "#8ab17d", 
    "MVN n=1": "#e9c46a", 
    "MVN n=5": "#f4a261", 
})

PNMF_MODEL_COLORS = ({
    "P-NMF ε=1": "#8ab17d",
    "P-NMF ε=4": "#e9c46a",
    "P-NMF ε=10": "#f4a261",
    "P-NMF ε=50": "#e76f51",
})


DPMODEL_COLORS = {
    # DP-CVAE — red/orange ramp
    "DP-CVAE \u03b5=1":   "#fdae6b",
    "DP-CVAE \u03b5=4":   "#fd8d3c",
    "DP-CVAE \u03b5=10":  "#e6550d",
    "DP-CVAE \u03b5=50":  "#a63603",

    # DP-PGM — purple ramp
    "DP-PGM \u03b5=1":     "#bcbddc",
    "DP-PGM \u03b5=4":     "#9e9ac8",
    "DP-PGM \u03b5=10":    "#756bb1",
    "DP-PGM \u03b5=50":    "#54278f",
    # DP-CTGAN — single blue point (distinct from all ramps)
    "DP-CTGAN \u03b5=50": "#2171b5",
}

# Create a mapping from column names to nice labels
MIA_METHOD_LABELS = {
    'gan_leaks': 'GAN-leaks',
    'MC': 'MC',
    'loss_rf': 'Confidence RF',
    'loss_lr': 'Confidence LR',
    'gan_leaks_cal': 'GAN-leaks (calbr.)',
    'LOGAN_D1': 'LOGAN D1',
    'domias_kde': 'DOMIAS (KDE)'
}

FS_TOKEN_LABELS = {
    "pca": "PCA",
    "vDE": "DVG", #differentially variable genes 
    "sDE": "SDEG", #supervised differentially expressed genes
    "dDE": "DG", # discriminative genes 
    "none": "None"
}

FS_TOKENS = list(FS_TOKEN_LABELS)
 

EVAL_METRIC_LABELS = {

    "MMD_test": "MMD (test)",
    "MMD_train": "MMD (train)",
    "kl_mean_train": "KL (train)",
    "kl_mean_test": "KL (test)",
    "kl_mean_test_inverted": "Inverted KL (test)\n1 / (1+KL)",
    "MMD_test_inverted": "Inverted MMD (test)\n1 / (1+MMD)",
    "kl_mean_train_inverted": "Inverted KL (train)\n1 / (1+KL)",
    "MMD_train_inverted": "Inverted MMD (train)\n1 / (1+MMD)",
    "coexpression_precision": "Co-expr. recovery\nPrecision",
    "coexpression_recall":  "Co-expr. recovery\nTPR (Sensitivity)",
    "coexpression_specificity": "Co-expr. recovery\nSpecificity",
    "n_correct_edges": "Co-expr. recovery\nNum. correct edges",
    "coexpression_false_edge_rate": "Co-expr. recovery\nFalse Edge Rate",
    "correct_edges_norm": "Co-expr. recovery\nEdge recovery (frac. of max)",
    "f1_relative": "F1 Relative",
    "accuracy_relative": "Accuracy Relative",
    "avg_pr_macro_relative": "PR Macro Relative",
    "avg_pr_weight_relative": "PR Weight Relative",
    "auroc_relative": "AUROC Relative",
    "feature_overlap_proportion": "Feature Overlap %", 
    "discriminative_score":  "Discriminative Score",
    "indistinguishability_score": "Inverted Discriminative Score\n(1 - Discriminative Score)",
    "distance_to_closest": "Distance to closest",
    "distance_to_closest_relative": "Relative\nDistance to closest",
    "DE_preservation_up": "DE recovery\nUp-regulated",
    "DE_preservation_down": "DE recovery\nDown-regulated",
    "tpr_at_fpr_01": "MIA TPR@FPR=0.1",
    "tpr_at_fpr_001":  "MIA TPR@FPR=0.01",  
    "log_tpr_at_fpr_01": "Log(MIA TPR@FPR=0.1)",
    "mia_aucroc": "MIA AUC-ROC", 
    "mia_pr_auc": "MIA AUC-PR",
    "mia_precision@5pt": "MIA Precision 5%",
    "DE_preservation_avg": "DE recovery\nAvg. TPR up/down"            
}


METRIC_NAME_MAP_FOR_PLOT = {
    "Inverted MMD (test)\n1 / (1+MMD)": "MMD (test)",
    "Co-expr. recovery\nSpecificity": "Co-expr. recovery\nFalse Edge Rate",
    "DE recovery\nAvg. TPR up/down": "DE recovery\nUp-regulated",  # tested metric for trendline
    "F1 Relative": "F1 Relative",
    "Co-expr. recovery\nRecall": "Co-expr. recovery\nRecall",
    "MIA TPR@FPR=0.1": "MIA TPR@FPR=0.1",
    "Inverted Discriminative Score\n(1 - Discriminative Score)" : "Discriminative Score",
    # add others as needed
}



CUSTOM_MODEL_COLORS = ({
    #"real": "#6B4C9A", 
    "MVN": "#3B9BDC",  
    "CVAE": "#3A64A9", 
    "DP-CVAE ε=4": "#3A64A9", 
    "CTGAN": "#456C45", # #a50f15
    "DP-CTGAN ε=50":  "#456C45", 
    "CVAE-GMM": "#BE47B0",   
    "WGAN-GP":   "#088840",
    "EMB-Diffusion": "#5B26AA",
    "NMF":"#C48484", 
    "P-NMF":"#C48484",  #ε=4
    "DP-PGM ε=4": "#8C3A4C"     
})



# Define metric families
METRIC_FAMILY_MAP = {
    # Biological preservation
    'DE_preservation_down': 'Biological',
    'DE_preservation_up': 'Biological',
    'coexpression_precision': 'Biological',
    'n_correct_edges': 'Biological',
    'correct_edges_norm': 'Biological',

    # Privacy / Membership inference
    'mia_aucroc': 'Privacy',
    'average_precision': 'Privacy',
    'mia_pr_auc': 'Privacy',
    'mia_f1_median': 'Privacy',
    'tpr_at_fpr_01': 'Privacy',
    'tpr_at_fpr_001': 'Privacy',
    'mia_precision@5pt': 'Privacy',
    

    # Utility / downstream performance
    'accuracy_synthetic': 'Utility',
    'avg_pr_macro_synthetic': 'Utility',
    'avg_pr_weight_synthetic': 'Utility',
    'f1_synthetic': 'Utility',
    'auroc_synthetic': 'Utility',
    'feature_overlap_count': 'Utility',
    'feature_overlap_proportion': 'Utility',
    'avg_pr_macro_relative': 'Utility',
    'f1_relative': 'Utility',

    # Distributional fidelity
    'MMD_train': 'Fidelity',
    'MMD_test': 'Fidelity',
    'kl_mean_train': 'Fidelity',
    'kl_mean_test': 'Fidelity',
    'discriminative_score' : 'Fidelity',
    'distance_to_closest': 'Fidelity'
}



### real data 
REALDATA_COEXPR_CORRECTEDGES = {
    "TCGA-BRCA": {
        "0.0": {"1": 182245, "2": 182318, "3": 181391, "4": 181972, "5": 182440},
        "0.3": {"1": 36689, "2": 37275, "3": 35669, "4": 36465, "5": 36655},
        "0.5": {"1": 4437, "2": 4597, "3": 4215, "4": 4458, "5": 4394}
    },
    "TCGA-COMBINED": {
        "0.0": {"1": 211571, "2": 211426, "3": 211131, "4": 210784, "5": 211318},
        "0.3": {"1": 33256, "2": 33205, "3": 33233, "4": 32593, "5": 33167 },
        "0.5": {"1": 3864, "2": 3827, "3": 3846, "4": 3715, "5": 3876 }
    }
}


CATEGORIES = {
    "baseline":   ["MVN", "CVAE", "DP-CVAE ε=4", "CTGAN", "DP-CTGAN ε=50"],
    "submission": ["EMB-Diffusion", "NMF", "P-NMF", "DP-PGM ε=4"],
    "postmodel":  ["WGAN-GP", "CVAE-GMM"],
}

# derived once — single source for category lookup and row ordering
CATEGORY_MAP = {m: cat for cat, members in CATEGORIES.items() for m in members}
METHOD_ORDER = [m for members in CATEGORIES.values() for m in members]


# Metrics entering the global correlation matrix  and the
# trade-off analyses. Order controls row/column order in the heatmap.
CORR_METRICS = [
    "privacy_tpr",                    # worst-case attack AUCROC/TPR@FPR=0.1 (lower = more private)
    "coexpr_tpr",                     # co-expr recovery recall @ r>0.3 (higher better)
    "de_tpr",                         # DE recovery TPR, avg up/down (higher better)
    "f1_relative",                    # relative downstream F1 (higher better)
    "MMD_test",                       # MMD on held-out (lower better)
    "kl_mean_test",                   # KL on held-out (lower better)
    "discriminative_score",           # real-vs-synthetic discriminability (lower better)
    "prdc_coverage_test",                  # PRDC coverage (higher better)
    "prdc_density_test",                   # PRDC density (-> 1 target)
    "distance_to_closest_relative",   # relative DCR (-> 1 target)
    "pathway_ks",                     # mean ssGSEA KS statistic (lower better)
]
 
# Metrics with a ->1 ideal: entered into the correlation as |value - 1|
# (deviation from target) so the rank correlation is directionally meaningful.
CORR_TARGET_METRICS = ("prdc_density", "distance_to_closest_relative")
 
# Pretty labels for the correlation heatmap / trade-off axes.
CORR_METRIC_LABELS = {
    "privacy_tpr": "Privacy AUC-ROC",#TPR@FPR=0.1
    "coexpr_tpr": "Co-expression TPR",
    "de_tpr": "DE TPR (avg(up/down))",
    "f1_relative": "F1 relative",
    "auroc_relative": "AUROC Relative",
    "avg_pr_macro_relative": "PR Macro Relative",
    "feature_overlap_proportion": "Feature Overlap", 
    "discriminative_score": "Discriminative score",
    "prdc_coverage": "Coverage (train)",
    "prdc_density": "Density (train)",
    "prdc_coverage_test": "Coverage (test)",
    "prdc_density_test": "Density (test)",
    "distance_to_closest_relative": "Distance to closest\nRelative",
    "pathway_ks": "Pathway KS",
    "MMD_test": "MMD (test)",
    "MMD_train": "MMD (train)",
    "kl_mean_train": "KL (train)",
    "kl_mean_test": "KL (test)",
}
