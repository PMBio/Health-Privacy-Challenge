import pandas as pd
from pathlib import Path

def summarize_mean_std(
    df,
    group_col="origin",
    metrics=None,
    exclude=("fold", "category", "is_DP"),
    decimals=3,
    method_order=None,
    metric_labels=None,
):
    """One row per method, one column per metric, cells = 'mean (std)' across folds.

    Parameters
    ----------
    metrics      : list of metric columns. If None, auto-picks numeric columns
                   that aren't in `exclude` or the group column.
    method_order : optional list to fix row order (subset is fine).
    metric_labels: optional {old_name: pretty_name} dict for the column headers.
    """
    if metrics is None:
        skip = set(exclude) | {group_col}
        metrics = [c for c in df.columns
                   if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

    agg = df.groupby(group_col)[metrics].agg(["mean", "std"])

    out = pd.DataFrame(index=agg.index)
    for m in metrics:
        out[m] = [f"{mu:.{decimals}f} ({sd:.{decimals}f})"
                  for mu, sd in zip(agg[(m, "mean")], agg[(m, "std")])]

    if method_order is not None:
        out = out.reindex([m for m in method_order if m in out.index])
    if metric_labels is not None:
        out = out.rename(columns=metric_labels)

    out.index.name = group_col
    return out


def save_table_html(table, path, title=None, escape=True):
    """Write a summary table to HTML that pastes cleanly into Google Docs.

    escape=False lets <b>/<u> markup in cells render (needed for bold-best tables);
    keep escape=True for plain-text tables.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    html = table.to_html(border=1, justify="center", escape=escape)
    if title:
        html = f"<h3>{title}</h3>\n{html}"
    with open(path, "w") as f:
        f.write(html)
    return path


def summarize_subgroup_mmd(
    df,
    group_col="origin",
    prefix="submmd_",
    subgroup_sizes=None,
    decimals=4,
):
    """Per-method subgroup-MMD summary: mean, size-weighted mean, worst-case.

    Averages each subgroup's MMD across folds, then per method reports:
      - mmd_mean          : unweighted mean over subgroups
      - mmd_weighted_mean : prevalence-weighted mean (if subgroup_sizes given)
      - mmd_max           : worst (largest) subgroup MMD
      - worst_subgroup    : which subgroup that was

    subgroup_sizes : optional {short_subgroup_name: n} for the weighted mean.
                     Keys should match the names *after* stripping `prefix`.
    """
    sub_cols = [c for c in df.columns if c.startswith(prefix)]
    if not sub_cols:
        raise ValueError(f"no columns start with prefix {prefix!r}")

    per = df.groupby(group_col)[sub_cols].mean()          # method x subgroup
    per.columns = [c[len(prefix):] for c in per.columns]  # strip prefix

    out = pd.DataFrame(index=per.index)
    out["mmd_mean"] = per.mean(axis=1)
    out["mmd_std"] = per.std(axis=1)               # spread ACROSS subgroups (heterogeneity)
    if subgroup_sizes is not None:
        w = pd.Series(subgroup_sizes).reindex(per.columns)
        out["mmd_weighted_mean"] = (per * w).sum(axis=1) / w.sum()
    out["mmd_max"] = per.max(axis=1)
    out["mmd_min"] = per.min(axis=1)
    out["worst_subgroup"] = per.idxmax(axis=1)
    out["best_subgroup"] = per.idxmin(axis=1)
    return out.round(decimals)


# ---------------------------------------------------------------------------
# Best-method highlighting (copy-paste safe: emits <b>/<u>, not Styler CSS)
# ---------------------------------------------------------------------------
import numpy as np
from scipy.stats import wilcoxon


def not_worse_than_best_wilcoxon(
    df, metrics, group_col="origin", fold_col="fold",
    lower_is_better=(), alpha=0.05,
):
    """Boolean (method x metric): True = best, or not significantly worse than best.

    Per metric: pick the best method by mean, then paired Wilcoxon signed-rank
    (across folds) of every other method vs the best. p >= alpha -> not
    distinguishable -> True. Underpowered/degenerate tests fall back to True
    """
    lower_is_better = set(lower_is_better)
    mask = pd.DataFrame(False, index=df[group_col].unique(), columns=metrics)

    for m in metrics:
        pivot = df.pivot_table(index=fold_col, columns=group_col, values=m)
        means = pivot.mean()
        best = means.idxmin() if m in lower_is_better else means.idxmax()
        mask.loc[best, m] = True
        for method in pivot.columns:
            if method == best:
                continue
            paired = pivot[[best, method]].dropna()
            try:
                _, p = wilcoxon(paired[best], paired[method])
                mask.loc[method, m] = p >= alpha
            except ValueError:
                mask.loc[method, m] = True   # all-equal / too few pairs -> can't distinguish
    return mask.reindex(df[group_col].unique())


def summarize_with_best(
    df, metrics, group_col="origin", method_order=None, metric_labels=None,
    lower_is_better=(), skip_highlight=(), decimals=3, rule="one_std", not_worse_mask=None,
):
    """`mean (std)` table with the best method(s) bolded and the runner-up underlined.

    rule="one_std" : bold every method within one std of the best mean (direction-aware).
    rule="mask"    : bold per a precomputed boolean not_worse_mask (e.g. from Wilcoxon).
    skip_highlight : columns shown plain (no bold/underline) — for target-seeking
                     metrics (e.g. "best ≈ 1") where "best" is ill-defined.
    Runner-up is underlined only when a single method is bolded (a tie already shows
    the contenders). Cells carry <b>/<u> -> save with save_table_html(..., escape=False).
    """
    lower_is_better = set(lower_is_better)
    skip_highlight = set(skip_highlight)
    means = df.groupby(group_col)[metrics].mean()
    stds = df.groupby(group_col)[metrics].std()

    out = pd.DataFrame(index=means.index, columns=metrics, dtype=object)
    for c in metrics:
        plain = c in skip_highlight
        if not plain:
            asc = c in lower_is_better
            order = means[c].sort_values(ascending=asc)
            if rule == "mask":
                bold = set(not_worse_mask.index[not_worse_mask[c].fillna(False)])
            else:  # one_std
                b_mean, b_std = order.iloc[0], stds.loc[order.index[0], c]
                if asc:
                    bold = set(means.index[means[c] <= b_mean + b_std])
                else:
                    bold = set(means.index[means[c] >= b_mean - b_std])
            underline = order.index[1] if (len(bold) == 1 and len(order) > 1) else None
        else:
            bold, underline = set(), None

        for method in means.index:
            cell = f"{means.loc[method, c]:.{decimals}f} ({stds.loc[method, c]:.{decimals}f})"
            if method in bold:
                cell = f"<b>{cell}</b>"
            elif method == underline:
                cell = f"<u>{cell}</u>"
            out.loc[method, c] = cell

    if method_order is not None:
        out = out.reindex([m for m in method_order if m in out.index])
    if metric_labels is not None:
        out = out.rename(columns=metric_labels)
    out.index.name = group_col
    return out


def format_mia_full_table(full, metrics=("tpr_at_fpr_01", "mia_aucroc"),
                          metric_labels=None, method_order=None, decimals=3):
    """Lay out the MIA attack surface as a copy-paste table.
 
    Rows = (origin[, feature_selection], attack); columns = metric, cells =
    'mean (std)'. Expects the frame from prepare_mia_full (with <metric>_mean /
    <metric>_std and an `attack` column, optionally `feature_selection`).
    """
    metric_labels = metric_labels or {"tpr_at_fpr_01": "TPR@FPR=0.1",
                                       "mia_aucroc": "AUC-ROC"}
    d = full.copy()
    for m in metrics:
        d[metric_labels[m]] = (d[f"{m}_mean"].map(f"{{:.{decimals}f}}".format)
                               + " (" + d[f"{m}_std"].map(f"{{:.{decimals}f}}".format) + ")")
 
    has_fs = "feature_selection" in d.columns
    index_cols = ["origin", "feature_selection", "attack"] if has_fs else ["origin", "attack"]
    if method_order is not None:
        d["origin"] = pd.Categorical(d["origin"], categories=method_order, ordered=True)
 
    out = (d.sort_values(index_cols)
             .set_index(index_cols)[[metric_labels[m] for m in metrics]])
    return out
 


def wins_summary(df, metrics, group_col="origin", method_order=None,
                 lower_is_better=(), skip_highlight=(), rule="one_std",
                 not_worse_mask=None, decimals=0):
    """Per-model tally of metric 'wins', as an overall-signal companion to the
    detailed tables — answers "which model is best across metrics" without
    normalising values onto a common scale.
 
    Uses the SAME best/runner-up logic as summarize_with_best so the counts match
    the bolding: `n_best` = metrics where the model is bolded (best, or within one
    std / mask of best); `n_top2` = n_best plus metrics where it's the underlined
    runner-up. Target-seeking metrics in `skip_highlight` (e.g. DCR, density) are
    excluded from the tally entirely, since "best" is ill-defined for them —
    `n_metrics_scored` reports how many metrics contributed.
 
    Returns a DataFrame indexed by model: n_best, n_top2, n_metrics_scored,
    sorted by n_best (then n_top2) descending.
    """
    lower_is_better = set(lower_is_better)
    skip_highlight = set(skip_highlight)
    means = df.groupby(group_col)[metrics].mean()
    stds = df.groupby(group_col)[metrics].std()
 
    scored = [c for c in metrics if c not in skip_highlight]
    best = {m: 0 for m in means.index}
    top2 = {m: 0 for m in means.index}
 
    for c in scored:
        asc = c in lower_is_better
        order = means[c].sort_values(ascending=asc)
        if rule == "mask" and not_worse_mask is not None:
            bold = set(not_worse_mask.index[not_worse_mask[c].fillna(False)])
        else:
            b_mean, b_std = order.iloc[0], stds.loc[order.index[0], c]
            bold = (set(means.index[means[c] <= b_mean + b_std]) if asc
                    else set(means.index[means[c] >= b_mean - b_std]))
        runner = order.index[1] if (len(bold) == 1 and len(order) > 1) else None
        for m in bold:
            best[m] += 1
            top2[m] += 1
        if runner is not None:
            top2[runner] += 1
 
    out = pd.DataFrame({"n_best": best, "n_top2": top2})
    out["n_metrics_scored"] = len(scored)
    out = out.sort_values(["n_best", "n_top2"], ascending=False)
    if method_order is not None:
        out = out.reindex([m for m in method_order if m in out.index])
    out.index.name = group_col
    return out
 

def summarize_metrics_transposed(dfs_by_dataset, metrics, *, group_col="origin",
                                 method_order=None, metric_labels=None,
                                 dataset_labels=None, decimals=3,
                                 highlight=False, lower_is_better=(),
                                 skip_highlight=(), rule="one_std", flat=False):
    """Transposed (multi-)dataset table: metrics as ROWS, models as COLUMNS.

    dfs_by_dataset : {dataset_name: dataframe} — each df has group_col + metrics,
                     fold-level rows (mean/std computed here). One dataset is fine.
    Returns a DataFrame with a 2-level row index (Dataset, Metric) and one column
    per model; cells are "mean (std)". With flat=True the Dataset level is dropped
    (plain Metric index) — use for a single dataset with no Dataset column.

    highlight=True bolds the best model(s) and underlines the runner-up WITHIN
    each metric row (best-per-metric; per-dataset automatically since each row is
    one dataset). Direction-aware via `lower_is_better`; `skip_highlight` metrics
    (e.g. target-seeking, ideal≈1) shown plain. Cells carry <b>/<u> -> save with
    save_table_html(..., escape=False).
    """
    metric_labels = metric_labels or {}
    dataset_labels = dataset_labels or {}
    lower_is_better = set(lower_is_better)
    skip_highlight = set(skip_highlight)

    def _fmt_row(means_row, stds_row, metric, models):
        cells = {mod: f"{means_row[mod]:.{decimals}f} ({stds_row[mod]:.{decimals}f})"
                 for mod in models}
        if not highlight or metric in skip_highlight:
            return cells
        asc = metric in lower_is_better
        order = means_row[models].sort_values(ascending=asc)
        b_mean, b_std = order.iloc[0], stds_row[order.index[0]]
        if asc:
            bold = set(order.index[means_row[order.index] <= b_mean + b_std])
        else:
            bold = set(order.index[means_row[order.index] >= b_mean - b_std])
        underline = order.index[1] if (len(bold) == 1 and len(order) > 1) else None
        for mod in models:
            if mod in bold:
                cells[mod] = f"<b>{cells[mod]}</b>"
            elif mod == underline:
                cells[mod] = f"<u>{cells[mod]}</u>"
        return cells

    blocks = []
    for ds_name, df in dfs_by_dataset.items():
        agg = df.groupby(group_col)[metrics].agg(["mean", "std"])
        models = ([m for m in method_order if m in agg.index] if method_order
                  else list(agg.index))
        rows = {}
        for m in metrics:
            rows[metric_labels.get(m, m)] = _fmt_row(agg[(m, "mean")],
                                                     agg[(m, "std")], m, models)
        block = pd.DataFrame(rows).T[models]
        if not flat:
            block.index = pd.MultiIndex.from_product(
                [[dataset_labels.get(ds_name, ds_name)], block.index],
                names=["Dataset", "Metric"])
        else:
            block.index.name = "Metric"
        blocks.append(block)
    return pd.concat(blocks)
