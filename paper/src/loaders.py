"""Load and combine evaluation-result CSVs into tidy long DataFrames."""
import os
import glob
import re
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.constants import MODEL_PATHS, CATEGORY_MAP

logger = logging.getLogger(__name__)


def assign_category(origin: str) -> str:
    """Map a method name to its category ('baseline'/'submission'/'postmodel')."""
    return CATEGORY_MAP.get(origin, "unknown")


def _load_res_df(file_path: str, origin: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(file_path, "evaluation_results.csv"))
    df = df.rename(columns={"split_no": "fold"})
    df = df[df["fold"] != "average"]            # drop the summary row
    df["fold"] = df["fold"].astype(int)
    df["origin"] = origin
    return df


def combine_result_dfs(home_dir, dataset_name, res_dir="files", model_paths=None):
    model_paths = model_paths or MODEL_PATHS
    REL_VS_REAL = ["accuracy", "avg_pr_macro", "avg_pr_weight", "auroc", "f1"]
    frames = []
    for label, path_parts in model_paths.items():
        path = os.path.join(home_dir, f"results/{res_dir}", dataset_name, *path_parts)
        frames.append(_load_res_df(path, label))

    merged = pd.concat(frames, ignore_index=True)
   
    for m in REL_VS_REAL:
        merged[f"{m}_relative"] = (
            merged[f"{m}_synthetic"] / merged[f"{m}_real"]
        )
    
    merged["distance_to_closest_relative"] = (
        merged["distance_to_closest"] / merged["distance_to_closest_base"]
    )

    return merged


def _load_mia_df(lira_path: str, origin: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(lira_path, "split_*/evaluation_results.csv"))
    frames = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        fold = os.path.basename(os.path.dirname(f)).replace("split_", "")
        df["fold"] = int(fold)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["origin"] = origin
    out["category"] = out["origin"].apply(assign_category)
    return out


def combine_mia_dfs(home_dir, dataset_name, res_dir="mia", model_paths=None):
    model_paths = model_paths or MODEL_PATHS #MODEL_PATHS_FOR_MIA
    frames = []
    for label, path_parts in model_paths.items():
        path = os.path.join(home_dir, f"results/{res_dir}", dataset_name,
                            "domias_baselines", *path_parts)
        #print("loading MIA from %s", path)
        frames.append(_load_mia_df(path, label))
    combined = pd.concat(frames, ignore_index=True)
    return combined.rename(columns={
        "aucroc": "mia_aucroc", "pr_auc": "mia_pr_auc",
        "precision@5pt": "mia_precision@5pt", "f1_median": "mia_f1_median",
    })


def _empty_with_origin(label: str, what: str, cutoff) -> pd.DataFrame:
    logger.warning("No %s files found for %s at cutoff %s.", what, label, cutoff)
    return pd.DataFrame(columns=["origin"])


def load_DE_results(lira_path, origin, lfc_threshold):
    file_map = {
        "tpr": glob.glob(os.path.join(lira_path, f"DE_lfc={lfc_threshold}_results_tpr.csv")),
        "fpr": glob.glob(os.path.join(lira_path, f"DE_lfc={lfc_threshold}_results_fpr.csv")),
    }
    frames = []
    for metric, files in file_map.items():
        for f in files:
            df = pd.read_csv(f, index_col=0)
            df["metric"] = metric
            df["origin"] = origin
            frames.append(df)
    if not frames:
        return _empty_with_origin(origin, "DE", lfc_threshold)
    out = pd.concat(frames, ignore_index=True)
    out["category"] = out["origin"].apply(assign_category)
    return out


def load_CoE_results(path, origin, corr_th):
    files = glob.glob(os.path.join(path, f"coexpr_cutoff={corr_th}_results.csv"))
    if not files:
        return _empty_with_origin(origin, "co-expression", corr_th)
    frames = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        df["origin"] = origin
        frames.append(df)
    return pd.concat(frames, ignore_index=True)



def combine_bio_dfs(home_dir, dataset_name,lfc_threshold, res_dir="bio", model_paths=None,
):
    """Combine per-model DE-results CSVs into one long frame."""

    home_dir = Path(home_dir)
    model_paths = model_paths or MODEL_PATHS

    frames = []

    for label, path_parts in model_paths.items():
        path = (
            home_dir / "results" / res_dir / dataset_name / Path(*path_parts)
        )

        frames.append(
            load_DE_results(path, label, lfc_threshold)
        )

    return pd.concat(frames, ignore_index=True)


def prepare_de_points(combined_bio_df, fpr_max, all_models=None):
    """TPR/FPR points for the DE-recovery plot, with disappearance tracking.

    Adds a `status` column per point:
      "in_range"     – FPR within the cap
      "out_of_range" – produced data but FPR > cap (clamped to the edge; a result,
                       not noise — usually a collapsed/over-permissive model)
    Models that produced no data at all are absent from combined_bio_df entirely;
    pass `all_models` (e.g. METHOD_ORDER) to log which expected models are missing.

    Returns (overall_summary, fold_summary). overall_summary carries `status`
    ("in_range"/"out_of_range") and `any_out_of_range` per group.
    """
    wide = combined_bio_df.pivot_table(
        index=["comparison", "seed", "fold", "direction", "origin", "category"],
        columns="metric", values="correct",
    ).reset_index()

    syn = wide[wide["seed"] != "real data"].copy()
    syn["is_DP"] = syn["origin"].str.contains("ε=", case=False, na=False)

    # --- disappearance type 1: produced no data at all ---
    if all_models is not None:
        present = set(syn["origin"].unique())
        missing = [m for m in all_models if m not in present]
        if missing:
            logger.warning("DE plot: %d model(s) produced no usable data and are "
                           "absent: %s", len(missing), ", ".join(missing))

    # --- disappearance type 2: off-chart (clamp, don't drop) ---
    n_out = int((syn["fpr"] > fpr_max).sum())
    if n_out:
        offenders = sorted(syn.loc[syn["fpr"] > fpr_max, "origin"].unique())
        logger.warning("DE plot: %d point(s) exceed FPR cap %.3g and are clamped "
                       "to the edge: %s", n_out, fpr_max, ", ".join(offenders))
    syn["status"] = np.where(syn["fpr"] > fpr_max, "out_of_range", "in_range")
    syn["fpr"] = syn["fpr"].clip(upper=fpr_max)

    fold = (syn.groupby(["origin", "fold", "direction", "is_DP"])
               .agg(fpr=("fpr", "mean"), tpr=("tpr", "mean"),
                    any_out_of_range=("status", lambda s: (s == "out_of_range").any()))
               .reset_index())
    overall = (fold.groupby(["origin", "direction", "is_DP"])
                   .agg(fpr_mean=("fpr", "mean"), fpr_std=("fpr", "std"),
                        tpr_mean=("tpr", "mean"), tpr_std=("tpr", "std"),
                        any_out_of_range=("any_out_of_range", "any"))
                   .reset_index())
    overall["status"] = np.where(overall["any_out_of_range"], "out_of_range", "in_range")
    return overall, fold


def combine_coexpr_dfs(home_dir, dataset_name, thresholds, real_edges, res_dir="bio",
                       model_paths=None):
    """Combine co-expression results across correlation thresholds.

    real_edges : {str(threshold): {fold: real_correct_edge_count}} for this dataset
                 (e.g. REALDATA_COEXPR_CORRECTEDGES[dataset_name]).
    Logs any (model, threshold) combination that produced no data. 
    """
    model_paths = model_paths or MODEL_PATHS
    frames = []
    for th in thresholds:
        present, per_model = [], []
        for label, path_parts in model_paths.items():
            path = os.path.join(home_dir, f"results/{res_dir}", dataset_name, *path_parts)
            df = load_CoE_results(path, label, th)
            if df.empty or "rec" not in df.columns:
                continue
            per_model.append(df)
            present.append(label)
        missing = [m for m in model_paths if m not in present]
        if missing:
            logger.warning("Co-expr r>%.2g: %d model(s) produced no data: %s",
                           th, len(missing), ", ".join(missing))
        combined = pd.concat(per_model, ignore_index=True)
        combined["corr_threshold"] = th
        edges = {int(k): v for k, v in real_edges[str(th)].items()}
        combined["real_correct_edges"] = combined["fold"].map(edges)
        frames.append(combined)
    return pd.concat(frames, ignore_index=True)


def prepare_coexpr_points(combined_bio_df):
    """Recall vs false-edge-rate points for the co-expression plot.

    recall          = correct edges / real correct edges
    false_edge_rate = false   edges / real correct edges
    Returns (summary_stats, pivot) where summary has *_mean / *_std per
    (origin, is_DP, corr_threshold). Warns about zeros.
    """
    pivot = combined_bio_df.pivot_table(
        index=["origin", "fold", "corr_threshold", "real_correct_edges"],
        columns="dir", values="rec", aggfunc="first",
    ).reset_index()

    pivot["recall"] = pivot["correct"] / pivot["real_correct_edges"]
    pivot["false_edge_rate"] = pivot["false"] / pivot["real_correct_edges"]
    pivot["is_DP"] = pivot["origin"].str.contains("ε=", case=False, na=False)

    n_zero = int(((pivot["recall"] == 0) | (pivot["false_edge_rate"] == 0)).sum())
    if n_zero:
        logger.warning("Co-expr: %d point(s) have recall or false-edge-rate = 0; "
                       "these disappear on log axes.", n_zero)

    summary = (pivot.groupby(["origin", "is_DP", "corr_threshold"])
                    .agg(recall_mean=("recall", "mean"), recall_std=("recall", "std"),
                         fer_mean=("false_edge_rate", "mean"),
                         fer_std=("false_edge_rate", "std"))
                    .reset_index())
    return summary, pivot




def combine_pathway_dfs(
    home_dir,
    dataset_name,
    res_dir="bio",
    filename="pathway_metrics_results.csv",
    model_paths=None,
):
    """Combine the per-model ssGSEA pathway-fidelity CSVs across models.

    Each model writes one already-combined file (folds stacked inside).
    Logs models that produced none.
    """
    home_dir = Path(home_dir)
    model_paths = model_paths or MODEL_PATHS

    frames, present = [], []

    for label, path_parts in model_paths.items():
        path = (
            home_dir / "results"/ res_dir / dataset_name / Path(*path_parts) / filename
        )

        if not path.exists():
            continue

        df = pd.read_csv(path)
        df["origin"] = label
        frames.append(df)
        present.append(label)

    missing = [m for m in model_paths if m not in present]
    if missing:
        logger.warning(
            "Pathway fidelity: %d model(s) produced no data: %s",
            len(missing),
            ", ".join(missing),
        )

    return pd.concat(frames, ignore_index=True)


def prepare_comparison_difficulty(combined_bio_df, fpr_max=None):
    """Mean TPR per (comparison, direction, model) for the difficulty plot.

    Expects the long DE frame (columns: correct, direction, comparison, seed,
    fold, metric, origin, category). Keeps synthetic rows, extracts tpr from the
    long metric/correct pair, then averages over folds. Also returns the
    across-model average TPR per comparison (used to order easy -> hard).
    """
    df = combined_bio_df[combined_bio_df["seed"] != "real data"].copy()

    # long -> wide so each row has a tpr (and fpr, if present) value
    wide = df.pivot_table(
        index=["comparison", "direction", "origin", "fold"],
        columns="metric", values="correct", aggfunc="first",
    ).reset_index()

    if fpr_max is not None and "fpr" in wide.columns:
        wide = wide[wide["fpr"] <= fpr_max]

    by_model = (wide.groupby(["comparison", "direction", "origin"], as_index=False)["tpr"]
                    .mean())
    avg = (by_model.groupby(["comparison", "direction"], as_index=False)["tpr"]
                   .mean().rename(columns={"tpr": "avg_tpr"}))
    out = by_model.merge(avg, on=["comparison", "direction"])
    out["comparison"] = out["comparison"].str.replace("TCGA-", "", regex=False)
    out["is_DP"] = out["origin"].str.contains("ε=", case=False, na=False)
    return out


def _split_attack_fs(method, fs_tokens):
    """Split a fused method string into (attack, feature_selection).

    Matches a trailing feature-selection token from `fs_tokens` (so underscores
    inside the attack name don't break the split). Returns (attack, "") if none
    of the tokens match the suffix.
    """
    for fs in sorted(fs_tokens, key=len, reverse=True):
        suffix = f"_{fs}"
        if method.endswith(suffix):
            return method[: -len(suffix)], fs
    return method, ""


def prepare_mia_full(merged_syn_df, metrics=("tpr_at_fpr_01", "mia_aucroc"),
                     fs_tokens=None):
    """Per-(origin, attack, feature_selection) mean (std) over folds, every MIA metric.

    Full attack surface for the supplementary table. If `fs_tokens` is given, the
    fused `method` string is split into separate `attack` and `feature_selection`
    columns; otherwise the whole `method` is kept as `attack`.
    """
    df = merged_syn_df.copy()
    if fs_tokens:
        split = df["method"].map(lambda m: _split_attack_fs(m, fs_tokens))
        df["attack"] = split.map(lambda t: t[0])
        df["feature_selection"] = split.map(lambda t: t[1])
        keys = ["origin", "attack", "feature_selection"]
    else:
        df["attack"] = df["method"]
        keys = ["origin", "attack"]

    out = df.groupby(keys)[list(metrics)].agg(["mean", "std"])
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    return out.reset_index()


def prepare_mia_worstcase(merged_syn_df, metrics=("tpr_at_fpr_01", "mia_aucroc"),
                          fs_tokens=None):
    """Worst-case (most successful) attack per generator, PER METRIC.

    Two-stage reduction: average over folds (noise axis, keep std), then take the
    MAX across attack configs (adversary axis) — privacy is best-performing over
    adversaries. The winning config differs by metric. Returns long: origin,
    metric, value, std, method (raw winning config), attack, feature_selection
    (split if fs_tokens given, else attack=method/fs=""), is_DP.
    """
    perattack = (merged_syn_df.groupby(["origin", "method"])[list(metrics)]
                 .agg(["mean", "std"]))
    rows = []
    for metric in metrics:
        means = perattack[(metric, "mean")]
        for origin, sub in means.groupby(level="origin"):
            worst_method = sub.idxmax()[1]          # config with highest leakage
            if fs_tokens:
                attack, fs = _split_attack_fs(worst_method, fs_tokens)
            else:
                attack, fs = worst_method, ""
            rows.append({
                "origin": origin, "metric": metric,
                "value": perattack.loc[(origin, worst_method), (metric, "mean")],
                "std":   perattack.loc[(origin, worst_method), (metric, "std")],
                "method": worst_method, "attack": attack, "feature_selection": fs,
            })
    out = pd.DataFrame(rows)
    out["is_DP"] = out["origin"].str.contains("ε=", case=False, na=False)
    return out


def prepare_mia_worstcase_folds(merged_syn_df, metrics=("tpr_at_fpr_01", "mia_aucroc"),
                                fs_tokens=None):
    """Per-fold values of the worst-case (strongest) attack, per generator per metric.

    Like prepare_mia_worstcase, but instead of collapsing to mean/std it returns
    every fold value of the winning config — for a boxplot/dotstrip. The winner
    is selected by fold-MEAN (so the chosen attack matches the bar figure), then
    its individual fold rows are returned.
    Columns: origin, metric, fold, value, method, attack, feature_selection, is_DP.
    """
    perattack_mean = (merged_syn_df.groupby(["origin", "method"])[list(metrics)]
                      .mean())
    rows = []
    for metric in metrics:
        means = perattack_mean[metric]
        for origin, sub in means.groupby(level="origin"):
            worst_method = sub.idxmax()[1]
            folds = merged_syn_df[(merged_syn_df["origin"] == origin)
                                  & (merged_syn_df["method"] == worst_method)]
            if fs_tokens:
                attack, fs = _split_attack_fs(worst_method, fs_tokens)
            else:
                attack, fs = worst_method, ""
            for _, fr in folds.iterrows():
                rows.append({
                    "origin": origin, "metric": metric, "fold": fr["fold"],
                    "value": fr[metric], "method": worst_method,
                    "attack": attack, "feature_selection": fs,
                })
    out = pd.DataFrame(rows)
    out["is_DP"] = out["origin"].str.contains("ε=", case=False, na=False)
    return out


def prepare_mia_heatmap(merged_syn_df, metric, fs_tokens=None,
                        attack_labels=None, fs_labels=None):
    """Per-(attack, generator) mean leakage for one metric — for the MIA heatmap.

    Fold-averaged leakage per attack config x generator. Returns long frame:
    attack_disp (pretty attack [+FS]), origin, value. 
    """
    attack_labels = attack_labels or {}
    fs_labels = fs_labels or {}
    df = merged_syn_df.copy()

    if fs_tokens:
        split = df["method"].map(lambda m: _split_attack_fs(m, fs_tokens))
        df["attack"] = split.map(lambda t: t[0])
        df["feature_selection"] = split.map(lambda t: t[1])
    else:
        df["attack"] = df["method"]
        df["feature_selection"] = ""

    def _pretty(r):
        name = attack_labels.get(r["attack"], r["attack"])
        if r["feature_selection"]:
            name += f" ({fs_labels.get(r['feature_selection'], r['feature_selection'])})"
        return name
    df["attack_disp"] = df.apply(_pretty, axis=1)

    out = (df.groupby(["attack_disp", "origin"], as_index=False)[metric]
             .mean().rename(columns={metric: "value"}))
    return out


def assemble_per_method_metrics(merged_syn_df, combined_coexpr, combined_de_df, pw_df,
                                *, coexpr_threshold=0.3, fpr_metric="tpr_at_fpr_01",
                                model_metrics=("MMD_test", "discriminative_score",
                                               "kl_mean_test", "prdc_density_test",
                                               "prdc_coverage_test", "f1_relative",
                                               "distance_to_closest_relative")):
    """One row per generator with all trade-off / correlation metrics.

    Joins four sources at their native granularity, each reduced to one value per
    method (privacy = worst-case over attacks; everything else fold-averaged):
      - model_metrics : attack-invariant cols in merged_syn_df -> dedup attacks,
                         fold-mean
      - privacy       : fpr_metric -> fold-mean per attack, MAX over attacks
      - coexpr_tpr    : recall (correct/real_correct_edges) at one threshold, fold-mean
      - de_tpr        : tpr averaged over comparisons & folds, then over up/down
      - pathway_ks    : ks_stat averaged over pathways & folds

    Asserts the origin sets match across sources and reports any method dropped by
    the join (which would silently shrink n). Returns a per-method DataFrame.
    """
    # ---- model-level metrics (attack-invariant): dedup attacks, then fold mean+std
    base = merged_syn_df.drop_duplicates(["origin", "fold"])
    base_mean = base.groupby("origin", as_index=False)[list(model_metrics)].mean()
    base_std = (base.groupby("origin")[list(model_metrics)].std()
                .add_suffix("__std").reset_index())
    base = base_mean.merge(base_std, on="origin")

    # ---- privacy: worst-case = max over attacks of the fold-mean (keep winner+std)
    pa = (merged_syn_df.groupby(["origin", "method"])[fpr_metric]
          .agg(["mean", "std"]).reset_index())
    idx = pa.groupby("origin")["mean"].idxmax()
    priv = (pa.loc[idx, ["origin", "method", "mean", "std"]]
              .rename(columns={"mean": "privacy_tpr", "std": "privacy_tpr__std",
                               "method": "privacy_worst_attack"})
              .reset_index(drop=True))

    # ---- co-expr recovery TPR (recall) at one threshold (fold mean + std)
    ce = combined_coexpr[(combined_coexpr["corr_threshold"] == coexpr_threshold)
                         & (combined_coexpr["dir"] == "correct")].copy()
    ce["recall"] = ce["rec"] / ce["real_correct_edges"]
    ce_fold = ce.groupby(["origin", "fold"], as_index=False)["recall"].mean()
    coexpr = (ce_fold.groupby("origin")["recall"].agg(["mean", "std"]).reset_index()
              .rename(columns={"mean": "coexpr_tpr", "std": "coexpr_tpr__std"}))

    # ---- DE recovery TPR: tpr over comparisons & folds, avg up/down (fold std)
    de = combined_de_df[(combined_de_df["seed"] != "real data")
                        & (combined_de_df["metric"] == "tpr")].copy()
    de_fold = de.groupby(["origin", "fold"], as_index=False)["correct"].mean()
    de_tpr = (de_fold.groupby("origin")["correct"].agg(["mean", "std"]).reset_index()
              .rename(columns={"mean": "de_tpr", "std": "de_tpr__std"}))

    # ---- pathway mean KS: over pathways & folds (fold std of the per-fold mean)
    pw_fold = pw_df.groupby(["origin", "fold"], as_index=False)["ks_stat"].mean()
    pw = (pw_fold.groupby("origin")["ks_stat"].agg(["mean", "std"]).reset_index()
          .rename(columns={"mean": "pathway_ks", "std": "pathway_ks__std"}))

    # ---- join, guarding n
    sources = {"model": base, "privacy": priv, "coexpr": coexpr,
               "de": de_tpr, "pathway": pw}
    sets = {k: set(v["origin"]) for k, v in sources.items()}
    common = set.intersection(*sets.values())
    for k, s in sets.items():
        missing = s - common
        if missing:
            logger.warning("assemble: %s has %d method(s) not in all sources: %s",
                           k, len(missing), ", ".join(sorted(missing)))
    out = base
    for k in ["privacy", "coexpr", "de", "pathway"]:
        out = out.merge(sources[k], on="origin", how="inner")

    # surface std columns that are entirely NaN (e.g. a single fold present) —
    # their error bars would silently not draw, so flag rather than hide.
    std_cols = [c for c in out.columns if c.endswith("__std")]
    all_nan = [c for c in std_cols if out[c].isna().all()]
    if all_nan:
        logger.warning("assemble: %d std column(s) are all-NaN (error bars won't "
                       "draw): %s", len(all_nan), ", ".join(all_nan))
    logger.info("assemble: %d methods in final per-method table", len(out))
    return out


def prepare_tradeoff(per_method, corr_long, anchor="privacy_tpr",
                     against=("coexpr_tpr", "de_tpr", "f1_relative", "MMD_test"),
                     lower_is_better=("privacy_tpr", "MMD_test"),
                     metric_labels=None):
    """Long frame for anchored trade-off / concordance panels.

    `anchor` is the y-axis metric (privacy_tpr for the trade-off figure, or e.g.
    de_tpr / MMD_test for the concordance figures). For each metric in `against`:
    stacks (anchor on y, that metric on x) per method, flags Pareto-optimal points
    (polarity-aware — only meaningful for the privacy trade-off), and attaches the
    pair's Spearman rho and BH-FDR q from corr_long (the global matrix). One block
    per `against` metric (facet = `panel`).
    """
    from .stats import pareto_front
    metric_labels = metric_labels or {}
    lower = set(lower_is_better)

    def _lookup(m1, m2):
        row = corr_long[((corr_long.m1 == m1) & (corr_long.m2 == m2))
                        | ((corr_long.m1 == m2) & (corr_long.m2 == m1))]
        if len(row):
            r = row.iloc[0]
            return float(r["rho"]), float(r["q"]), str(r["stars"])
        return float("nan"), float("nan"), ""

    blocks = []
    for xm in against:
        cols = ["origin", anchor, xm]
        d = per_method[cols].copy()
        d = d.rename(columns={anchor: "y", xm: "x"})
        # carry fold std if present (named "<metric>__std")
        d["y_std"] = per_method.get(f"{anchor}__std", pd.Series(0.0, index=per_method.index)).values
        d["x_std"] = per_method.get(f"{xm}__std", pd.Series(0.0, index=per_method.index)).values
        d["panel"] = metric_labels.get(xm, xm)
        d["x_metric"] = xm
        d["pareto"] = pareto_front(d, "x", "y",
                                   x_lower_better=xm in lower,
                                   y_lower_better=anchor in lower)
        rho, q, stars = _lookup(anchor, xm)
        d["rho"], d["q"], d["stars"] = rho, q, stars
        d["is_DP"] = d["origin"].str.contains("ε=", case=False, na=False)
        blocks.append(d)
    return pd.concat(blocks, ignore_index=True)


def prepare_epsilon_ablation(source, x_metric="f1_relative",
                             y_metric="privacy_tpr", fpr_metric="tpr_at_fpr_01",
                             from_per_method=False, sweep_token="\u03b5"):
    """Privacy (or any y) vs a utility metric for a parameter sweep ablation.

    Generic over the swept parameter: `sweep_token` is the symbol in the model
    name carrying the swept value ("ε" for DP epsilon, "n" for MVN noise, etc.).
    The family is the name with that token stripped; the numeric value becomes
    `sweep`/`eps` (kept as `eps` too for backward compat).

    Two input modes:
      from_per_method=False: `source` is a merged_syn_df; computes worst-case
        fpr_metric (y) and x_metric fold mean+std directly (light path).
      from_per_method=True: `source` is an assemble_per_method_metrics table;
        slices its y_metric/x_metric columns (+ "__std") so the ablation uses the
        SAME reductions as the main figures (needed for bio axes like de_tpr).

    One row per model: origin, family, sweep (=eps), x, x_std, y, y_std, is_DP.
    """
    if from_per_method:
        out = source[["origin", y_metric, x_metric]].copy()
        out = out.rename(columns={y_metric: "y", x_metric: "x"})
        out["y_std"] = source.get(f"{y_metric}__std", 0.0)
        out["x_std"] = source.get(f"{x_metric}__std", 0.0)
    else:
        pa = (source.groupby(["origin", "method"])[fpr_metric]
              .agg(["mean", "std"]).reset_index())
        idx = pa.groupby("origin")["mean"].idxmax()
        priv = (pa.loc[idx, ["origin", "mean", "std"]]
                  .rename(columns={"mean": "y", "std": "y_std"}).reset_index(drop=True))
        base = source.drop_duplicates(["origin", "fold"])
        util = (base.groupby("origin")[x_metric].agg(["mean", "std"]).reset_index()
                .rename(columns={"mean": "x", "std": "x_std"}))
        out = priv.merge(util, on="origin")

    tok = re.escape(sweep_token)
    out["sweep"] = out["origin"].str.extract(rf"{tok}\s*=\s*([\d.]+)").astype(float)
    out["eps"] = out["sweep"]                                   # backward-compat alias
    out["family"] = (out["origin"]
                     .str.replace(rf"\s*{tok}\s*=\s*[\d.]+\s*", "", regex=True)
                     .str.strip())
    out["is_DP"] = True
    return out

 
def prepare_de_lfc_sweep(home_dir, dataset_name, lfc_thresholds, fpr_max,
                         combine_fn, prepare_fn, all_models=None,
                         average_direction=False):
    """Stack DE-recovery points across LFC thresholds for the LFC ablation.
 
    For each lfc in `lfc_thresholds`: loads via combine_fn(home, dataset, lfc),
    reduces via prepare_fn(combined, fpr_max, all_models) -> (overall, fold), and
    tags the overall-summary with `lfc`. Returns one stacked overall-summary frame
    with an `lfc` column for faceting. (combine_fn=combine_bio_dfs,
    prepare_fn=prepare_de_points — passed in to avoid import cycles.)
 
    average_direction=True collapses up/down into one point per (origin, lfc):
    fpr/tpr averaged over the two directions, std as the mean of the two — for a
    compact 1-row-per-metric sensitivity figure where direction detail isn't needed.
    """
    frames = []
    for lfc in lfc_thresholds:
        combined = combine_fn(home_dir, dataset_name, lfc)
        overall, _ = prepare_fn(combined, fpr_max, all_models=all_models)
        overall["lfc"] = lfc
        frames.append(overall)
    out = pd.concat(frames, ignore_index=True)
 
    if average_direction:
        out = (out.groupby(["origin", "is_DP", "lfc"], as_index=False)
                  .agg(fpr_mean=("fpr_mean", "mean"), fpr_std=("fpr_std", "mean"),
                       tpr_mean=("tpr_mean", "mean"), tpr_std=("tpr_std", "mean")))
    out["lfc_label"] = out["lfc"].map(lambda v: f"\u0394VST = {v}")
    return out
 
 



def prepare_fs_leakage(merged_syn_df, metric="tpr_at_fpr_01", fs_tokens=None,
                       main_methods=("gan_leaks_cal", "LOGAN_D1", "domias_kde"),
                       collapse="max"):
    """Leakage per (generator, feature_selection), for the FS-ablation figures.
 
    Splits the fused `method` into (attack, feature_selection); keeps only the
    factorial main methods; treats bare attacks (no FS suffix) as fs="none" so
    each method carries a no-FS baseline. Fold-averages per (generator, method,
    fs), then COLLAPSES over the main methods to one value per (generator, fs):
      collapse="max"  -> worst-case over methods (consistent with the worst-case
                         privacy framing elsewhere; recommended)
      collapse="mean" -> typical-case over methods
    Returns long: origin, feature_selection, value, is_DP. The kept main-method
    rows (pre-collapse) are returned too (second frame) for the faceted heatmap.
    """
    fs_tokens = fs_tokens or []
    df = merged_syn_df.copy()
    split = df["method"].map(lambda m: _split_attack_fs(m, fs_tokens))
    df["attack"] = split.map(lambda t: t[0])
    df["feature_selection"] = split.map(lambda t: t[1] or "none")
 
    df = df[df["attack"].isin(main_methods)].copy()        # drop non-factorial extras
    # fold-mean per (generator, method, fs)
    per = (df.groupby(["origin", "attack", "feature_selection"], as_index=False)[metric]
             .mean().rename(columns={metric: "value"}))
    per["is_DP"] = per["origin"].str.contains("ε=", case=False, na=False)
 
    # collapse over main methods -> one value per (generator, fs)
    agg = "max" if collapse == "max" else "mean"
    pooled = (per.groupby(["origin", "feature_selection"], as_index=False)["value"]
                .agg(agg))
    pooled["is_DP"] = pooled["origin"].str.contains("ε=", case=False, na=False)
    return pooled, per
 
 
def fs_rank_agreement(pooled, fs_order=None):
    """Spearman rho of generator leakage rankings between every FS pair.
 
    `pooled` is the per-(generator, fs) frame from prepare_fs_leakage. Returns an
    FS x FS DataFrame of rank correlations (how consistently FS variants order the
    generators by leakage). High off-diagonal -> FS is a nuisance lens (does not
    reorder generators); low -> FS interacts with the generator (privacy ranking
    depends on the adversary's feature selection).
    """
    from scipy.stats import spearmanr
    wide = pooled.pivot(index="origin", columns="feature_selection", values="value")
    if fs_order:
        wide = wide[[f for f in fs_order if f in wide.columns]]
    fs = list(wide.columns)
    mat = pd.DataFrame(np.eye(len(fs)), index=fs, columns=fs, dtype=float)
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            r, _ = spearmanr(wide[fs[i]], wide[fs[j]])
            mat.iloc[i, j] = mat.iloc[j, i] = r
    return mat


def fs_rank_agreement_multi(merged_syn_df, metrics, *, fs_tokens=None,
                            main_methods=("gan_leaks_cal", "LOGAN_D1", "domias_kde"),
                            collapse="max", fs_order=None, metric_labels=None):
    """FS x FS rank-agreement for several MIA metrics, as a long frame for facets.
 
    Runs prepare_fs_leakage + fs_rank_agreement once per metric in `metrics`
    (e.g. ("mia_aucroc", "tpr_at_fpr_01")) and stacks the resulting matrices into
    a long frame with a `metric` column, so plot_fs_rank_agreement can facet by
    metric. `metric_labels` maps raw metric names to display names for the facet
    strips. Returns long: f1, f2, rho, metric.
    """
    metric_labels = metric_labels or {}
    blocks = []
    for m in metrics:
        pooled, _ = prepare_fs_leakage(merged_syn_df, metric=m, fs_tokens=fs_tokens,
                                       main_methods=main_methods, collapse=collapse)
        mat = fs_rank_agreement(pooled, fs_order=fs_order)
        long = (mat.reset_index().melt(id_vars="index", var_name="f2",
                                       value_name="rho").rename(columns={"index": "f1"}))
        long["metric"] = metric_labels.get(m, m)
        blocks.append(long)
    return pd.concat(blocks, ignore_index=True)
 
 


def prepare_de_threshold_grid(home_dir, dataset_name, lfc_thresholds, fpr_caps,
                              combine_fn, prepare_fn, all_models=None):
    """DE recovery TPR across a (ΔVST × FPR-cap) grid, up/down-averaged per model.
 
    For the joint sensitivity figure answering "do DE-recovery conclusions depend
    on the effect-size threshold or the FPR cap?". Loads DE once per ΔVST
    (combine_fn), then applies each FPR cap (prepare_fn clamps at fpr_max) — so
    n_lfc loads, n_lfc*n_fpr reductions. Within each (model, lfc, cap) the up- and
    down-regulation TPR are AVERAGED into one value (direction collapsed; the
    ablation is about threshold stability, not direction).
 
    Returns long: origin, is_DP, lfc, lfc_label, fpr_cap, fpr_label,
    tpr (up/down mean), tpr_std (mean of the two directions' fold std).
    """
    frames = []
    for lfc in lfc_thresholds:
        combined = combine_fn(home_dir, dataset_name, lfc)   # load once per ΔVST
        for cap in fpr_caps:
            overall, _ = prepare_fn(combined, cap, all_models=all_models)
            # overall has one row per (origin, direction, is_DP): tpr_mean, tpr_std
            g = (overall.groupby(["origin", "is_DP"], as_index=False)
                        .agg(tpr=("tpr_mean", "mean"), tpr_std=("tpr_std", "mean")))
            g["lfc"] = lfc
            g["fpr_cap"] = cap
            frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out["lfc_label"] = out["lfc"].map(lambda v: f"\u0394VST = {v}")
    out["fpr_label"] = out["fpr_cap"].map(lambda v: f"FPR \u2264 {v}")
    return out
 

 
def prepare_de_threshold_grid(home_dir, dataset_name, lfc_thresholds, fpr_caps,
                              combine_fn, prepare_fn, all_models=None):
    """DE recovery TPR across a (ΔVST × FPR-cap) grid, up/down-averaged per model.
 
    For the joint sensitivity figure answering "do DE-recovery conclusions depend
    on the effect-size threshold or the FPR cap?". Loads DE once per ΔVST
    (combine_fn), then applies each FPR cap (prepare_fn clamps at fpr_max) — so
    n_lfc loads, n_lfc*n_fpr reductions. Within each (model, lfc, cap) the up- and
    down-regulation TPR are AVERAGED into one value (direction collapsed; the
    ablation is about threshold stability, not direction).
 
    Returns long: origin, is_DP, lfc, lfc_label, fpr_cap, fpr_label,
    tpr (up/down mean), tpr_std (mean of the two directions' fold std).
    """
    frames = []
    for lfc in lfc_thresholds:
        combined = combine_fn(home_dir, dataset_name, lfc)   # load once per ΔVST
        for cap in fpr_caps:
            overall, _ = prepare_fn(combined, cap, all_models=all_models)
            # overall has one row per (origin, direction, is_DP): tpr_mean, tpr_std
            g = (overall.groupby(["origin", "is_DP"], as_index=False)
                        .agg(tpr=("tpr_mean", "mean"), tpr_std=("tpr_std", "mean")))
            g["lfc"] = lfc
            g["fpr_cap"] = cap
            frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out["lfc_label"] = out["lfc"].map(lambda v: f"\u0394VST = {v}")
    out["fpr_label"] = out["fpr_cap"].map(lambda v: f"FPR \u2264 {v}")
    return out