import os 
import glob
import re 
import pandas as pd
import numpy as np
from plotnine import *
from scipy.stats import spearmanr
import statsmodels.api as sm
from scipy.cluster.hierarchy import (linkage, leaves_list, 
                                    fcluster)

from notebooks._logs.utils_legacy_delete_later import * 
from src.constants import MODEL_PATHS, CUSTOM_MODEL_COLORS

def get_de_sensitivity_values(home_dir,dataset_name, corr_threshold = 0.3):
    lfc_thresholds = [0.0,  0.5]
    FPR_MAX_values = [0.01, 0.05, 0.1]

    sensitivity_records = []

    ### Coexpression
    bio_complete_dfs = {}
    for label, path_parts in MODEL_PATHS.items():
        path = os.path.join(home_dir, "results/bio", dataset_name, *path_parts)
        bio_complete_dfs[label] = load_CoE_results(path, label, corr_threshold)


    coexpression_df = pd.concat(bio_complete_dfs.values(), ignore_index=True)
    coexp_agg = (
        coexpression_df
        .groupby(["origin", "fold", "dir"], as_index=False)["rec"]
        .sum()
        .pivot(index=["origin", "fold"], columns="dir", values="rec")
        .reset_index()
    )

    # compute preservation score
    coexp_agg["coexpression_preservation"] = (
        coexp_agg["correct"] / (coexp_agg["correct"] + coexp_agg["false"])
    )

    coexp_scores = coexp_agg[["origin", "fold", "coexpression_preservation"]]


    for lfc in lfc_thresholds:
        for fpr_max in FPR_MAX_values:
            # Load DE results for current LFC threshold
            bio_complete_dfs = {}
            for label, path_parts in MODEL_PATHS.items():
                path = os.path.join(home_dir, "results/bio", dataset_name, *path_parts)
                bio_complete_dfs[label] = load_DE_results(path, label, lfc)
            
            diffexp_df = pd.concat(bio_complete_dfs.values(), ignore_index=True)
            diffexp_wide_df_all = diffexp_df.pivot_table(
                index=['comparison', 'seed', 'fold', 'direction', 'origin', 'category'],
                columns='metric',
                values='correct'
            ).reset_index()
            
            # Keep only synthetic rows
            diffexp_wide_df = diffexp_wide_df_all.loc[diffexp_wide_df_all["seed"]=="synthetic data",]
            
            # Filter by FPR threshold
            df_de_fpr = diffexp_wide_df[diffexp_wide_df["fpr"] <= fpr_max]
            
            # Compute mean DE preservation per origin/fold/direction
            de_scores_dir = (
                df_de_fpr
                .groupby(["origin", "fold", "direction"], as_index=False)["tpr"]
                .mean()
                .rename(columns={"tpr": "DE_preservation_score_dir"})
            )
            
            # Merge with coexpression and other metrics (same as your workflow)
            bio_merged = (
                util_fidel_df
                .merge(coexp_scores, on=["origin", "fold"], how="left")
                .merge(de_scores_dir, on=["origin", "fold"], how="left")
            )
            
            # Pivot DE to wide
            de_wide = (
                bio_merged.pivot_table(
                    index=["origin", "fold"],
                    columns="direction",
                    values="DE_preservation_score_dir"
                )
                .reset_index()
                .rename(columns={"up": "DE_preservation_up", "down": "DE_preservation_down"})
            )
            
            # Merge wide DE back
            other_metrics = bio_merged.drop(
                columns=["direction", "DE_preservation_score_dir"]
            ).drop_duplicates(subset=["origin", "fold"])
            bio_merged_wide = de_wide.merge(other_metrics, on=["origin", "fold"], how="left")
            
            # Select metrics for correlation
            select_corr_cols = [
                "auroc_gap", "f1_gap", "MMD_test", "discriminative_score", 
                "distance_to_closest", "kl_mean_test",
                "coexpression_preservation", "DE_preservation_up", "DE_preservation_down",
                "tpr_at_fpr_01", "tpr_at_fpr_001", "mia_aucroc"
            ]
            
            bio_agg_df = bio_merged_wide.groupby(['origin'])[select_corr_cols].mean().reset_index()
            
            # Compute correlation matrix
            corr_mat = bio_agg_df[select_corr_cols].corr(method="pearson")
            
            # Flatten correlations into long format
            corr_long = corr_mat.reset_index().melt(id_vars="index", var_name="metric2", value_name="rho")
            corr_long = corr_long.rename(columns={"index":"metric1"})
            
            # Add threshold info
            corr_long["LFC_threshold"] = lfc
            corr_long["FPR_MAX"] = fpr_max
            
            # Append to records
            sensitivity_records.append(corr_long)

    sensitivity_df = pd.concat(sensitivity_records, ignore_index=True)
    sensitivity_df.head()

    return sensitivity_df
