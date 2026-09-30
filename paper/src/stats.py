"""Correlation and hierarchical-clustering helpers for metric matrices."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform


def _matrix_to_long(mat, value_name, order=None):
    """Melt a square matrix to long form with MetricA/MetricB columns."""
    long = (mat.reset_index()
               .melt(id_vars="index", var_name="MetricB", value_name=value_name)
               .rename(columns={"index": "MetricA"}))
    if order is not None:
        for col in ("MetricA", "MetricB"):
            long[col] = pd.Categorical(long[col], categories=order, ordered=True)
    return long


def get_cluster_order(rho_matrix, method="average"):
    """Leaf order from hierarchical clustering of a correlation matrix."""
    dist = (1 - rho_matrix).to_numpy(dtype=float, copy=True)
    dist = (dist + dist.T) / 2                 # enforce symmetry
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method=method)
    return list(rho_matrix.index[leaves_list(Z)])


def get_clustered_metrics(corr_mat, k=4, method="average"):
    """Cluster a correlation matrix; return (cluster_table, long-form corr)."""
    Z = linkage(corr_mat, method=method)
    order_idx = leaves_list(Z)
    corr_reordered = corr_mat.iloc[order_idx, order_idx]
    clusters = fcluster(Z, t=k, criterion="maxclust")

    cluster_table = (pd.DataFrame({"metric": corr_reordered.index,
                                   "cluster": clusters})
                     .sort_values("cluster"))
    order = list(corr_reordered.index)
    corr_long = _matrix_to_long(corr_reordered, "Corr", order=order)
    return cluster_table, corr_long


def spearman_corr_with_p(df, metrics, fdr_correction=True):
    """Spearman correlation + p-value matrices (FDR-BH corrected by default)."""
    rho = pd.DataFrame(index=metrics, columns=metrics, dtype=float)
    pval = pd.DataFrame(index=metrics, columns=metrics, dtype=float)
    for i in metrics:
        for j in metrics:
            rho.loc[i, j], pval.loc[i, j] = spearmanr(df[i], df[j])

    if not fdr_correction:
        return rho, pval

    arr = pval.to_numpy(dtype=float, copy=True)         # writable
    triu = np.triu_indices(len(metrics), k=1)
    _, corrected, _, _ = multipletests(arr[triu], alpha=0.05, method="fdr_bh")
    arr[triu] = corrected
    arr[(triu[1], triu[0])] = corrected                 # mirror lower triangle
    return rho, pd.DataFrame(arr, index=metrics, columns=metrics)


def _sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def prepare_corr_long(rho_matrix, pval_matrix, order, use_stars=True):
    """Long-form correlation table with significance labels in a fixed order."""
    rho_long = _matrix_to_long(rho_matrix.loc[order, order], "Corr")
    pval_long = _matrix_to_long(pval_matrix.loc[order, order], "pval")
    corr_long = rho_long.merge(pval_long, on=["MetricA", "MetricB"])

    if use_stars:
        stars = corr_long["pval"].astype(float).apply(_sig_stars)
        corr_long["Label"] = corr_long["Corr"].round(2).astype(str) + stars
    else:
        corr_long["Label"] = (corr_long["Corr"].round(2).astype(str)
                              + "\n(p=" + corr_long["pval"].round(3).astype(str) + ")")

    for col in ("MetricA", "MetricB"):
        corr_long[col] = pd.Categorical(corr_long[col], categories=order, ordered=True)
    return corr_long


def spearman_corr_test(df, x="util_values", y="privacy_value_mean"):
    """Spearman r/p for a (sub)group; NaN if too few distinct points."""
    if len(df) < 3 or df[x].nunique() < 3:
        return pd.Series({"r": np.nan, "p": 1.0})
    r, p = spearmanr(df[x], df[y])
    return pd.Series({"r": r, "p": p})


def hclust_order(matrix, method="average"):
    """Hierarchical-clustering leaf order of the ROWS of a numeric matrix."""
    filled = matrix.fillna(0.0)
    Z = linkage(filled.values, method=method)
    return list(matrix.index[leaves_list(Z)])


def prepare_pathway_summary(df, alpha=0.05):
    """Aggregate ssGSEA pathway metrics into a per-method table + heatmap frame.

    Returns
    -------
    perfold : per (origin, fold) — feed to summarize_with_best for mean (std):
              ks_stat (down), abs_effect (down), var_ratio_geom (->1),
              frac_sig_fdr (down; fraction of pathways with FDR-significant KS).
    per_pathway : per (origin, pathway) signed effect_size + ks_stat (folds
              averaged) — the long frame for the heatmap.

    Notes: var ratio is summarised as a GEOMETRIC mean (ratios are symmetric in
    log space; 0.5 and 2.0 are equal-and-opposite). KS significance is FDR-BH
    corrected across pathways *within each fold* before counting.
    """
    df = df.copy()
    df["abs_effect"] = df["effect_size"].abs()

    def _agg(g):
        vr = g["var_ratio"]
        vr = vr[vr > 0]
        p = g["ks_pval"].dropna()
        if len(p):
            reject, _, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
            frac_sig = float(reject.mean())
        else:
            frac_sig = np.nan
        return pd.Series({
            "ks_stat": g["ks_stat"].mean(),
            "abs_effect": g["abs_effect"].mean(),
            "var_ratio_geom": np.exp(np.log(vr).mean()) if len(vr) else np.nan,
            "frac_sig_fdr": frac_sig,
        })

    perfold = (df.groupby(["origin", "fold"], group_keys=False)
                 .apply(_agg).reset_index())
    per_pathway = (df.groupby(["origin", "pathway"])
                     .agg(effect_size=("effect_size", "mean"),
                          ks_stat=("ks_stat", "mean")).reset_index())
    return perfold, per_pathway


def compute_metric_correlations(per_method, metrics, target_metrics=(),
                                alpha=0.05, method="fdr_bh"):
    """Spearman correlation matrix across metrics, with global BH-FDR.

    per_method : one row per generative method, one column per metric.
    metrics    : ordered list of metric columns to correlate.
    target_metrics : metrics with a ->1 ideal (e.g. density, relative DCR);
        these are transformed to |value - 1| so the rank correlation is
        directionally meaningful (deviation from target; lower = better).
    Correction is applied ACROSS ALL unique pairs (n*(n-1)/2) jointly.

    Returns dict with:
      rho   : DataFrame (metrics x metrics) Spearman rho
      q     : DataFrame (metrics x metrics) BH-FDR adjusted p (NaN on diagonal)
      long  : tidy DataFrame [m1, m2, rho, p, q, stars] for the unique pairs
      n     : number of methods (rows)
    """
    X = per_method[list(metrics)].copy()
    for t in target_metrics:
        if t in X.columns:
            X[t] = (X[t] - 1.0).abs()

    cols = list(metrics)
    k = len(cols)
    rho = pd.DataFrame(np.eye(k), index=cols, columns=cols, dtype=float)
    pmat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            r, p = spearmanr(X[cols[i]], X[cols[j]])
            rho.iloc[i, j] = rho.iloc[j, i] = r
            pairs.append((cols[i], cols[j], r, p))

    praw = [p for *_, p in pairs]
    reject, qvals, _, _ = multipletests(praw, alpha=alpha, method=method)

    def _stars(q):
        return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""

    long_rows, qmat = [], pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for (m1, m2, r, p), q in zip(pairs, qvals):
        qmat.loc[m1, m2] = qmat.loc[m2, m1] = q
        long_rows.append(dict(m1=m1, m2=m2, rho=r, p=p, q=q, stars=_stars(q)))

    return {"rho": rho, "q": qmat, "long": pd.DataFrame(long_rows),
            "n": len(X)}


def pareto_front(df, x, y, x_lower_better, y_lower_better):
    """Boolean mask of Pareto-optimal (non-dominated) rows on axes (x, y).

    A point is dominated if another point is at least as good on both axes and
    strictly better on one. "Good" direction per axis is set by *_lower_better.
    """
    import numpy as np
    xv = df[x].to_numpy(); yv = df[y].to_numpy()
    xb = -xv if x_lower_better else xv          # transform so larger = better
    yb = -yv if y_lower_better else yv
    n = len(df)
    nondom = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (xb[j] >= xb[i] and yb[j] >= yb[i]
                    and (xb[j] > xb[i] or yb[j] > yb[i])):
                nondom[i] = False
                break
    return nondom


def correlations_per_attack(merged_syn_df, metric_pairs, fs_tokens=None,
                            target_metrics=(), min_methods=4):
    """Within-attack Spearman rho for each metric pair, across generators.

    For each attack (the `method` column, optionally split off its FS token) and
    each (m1, m2) in `metric_pairs`: fold-average per generator, then Spearman
    across generators. Target-seeking metrics in `target_metrics` enter as
    |value-1| (same monotonic transform as the global matrix). Attacks with fewer
    than `min_methods` generators are skipped (rho undefined).

    Returns long DataFrame: attack, pair (label "m1 vs m2"), m1, m2, rho, p, n.
    """
    df = merged_syn_df.copy()
    if fs_tokens:
        from .loaders import _split_attack_fs
        df["attack"] = df["method"].map(lambda m: _split_attack_fs(m, fs_tokens)[0])
    else:
        df["attack"] = df["method"]

    metrics = sorted({m for pair in metric_pairs for m in pair})
    # fold-average per (attack, origin)
    g = df.groupby(["attack", "origin"])[metrics].mean().reset_index()
    for t in target_metrics:
        if t in g.columns:
            g[t] = (g[t] - 1.0).abs()

    rows = []
    for attack, sub in g.groupby("attack"):
        if sub["origin"].nunique() < min_methods:
            continue
        for m1, m2 in metric_pairs:
            x, y = sub[m1].to_numpy(), sub[m2].to_numpy()
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            r, p = spearmanr(x, y)
            rows.append(dict(attack=attack, pair=f"{m1} vs {m2}",
                             m1=m1, m2=m2, rho=r, p=p, n=len(sub)))
    return pd.DataFrame(rows)