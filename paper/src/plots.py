"""Shared plotnine theme and figure builders for the evaluation pipeline.

Style is centralized in the constants + house_theme below so every figure shares
one type hierarchy, legend vocabulary, and colour/shape convention. Tune once
here rather than per builder.
"""
import os
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list

from plotnine import *
from plotnine import theme as theme_fn


# ---------------------------------------------------------------------------
# Publication style constants — single source of truth
# ---------------------------------------------------------------------------
BASE_SIZE = 10                 # base font size (pt); bump for slides/posters
POINT_SIZE = 10                # default scatter marker size
LEGEND_GEN = "Generative\nModel"
LEGEND_PRIV = "Privacy"
LEGEND_NCOL = 4                # columns in the generator legend

ANNOT_SIZE = 8.5                 # in-plot annotation text (rho/q, codes)
CELL_SIZE = 7.5                # heatmap cell value text
AXIS_Y_SMALL = 9.5               # y tick text on dense (many-row) heatmaps

GREY = "#666666"               # anchor / reference lines, mid grey
LIGHT_GREY = "#999999"         # floor / frontier lines
GRID_GREY = "#e6e6e6"          # panel grid

# Canonical canvas size (width, height) per figure type — deliberate aspect
# ratios, not arbitrary. bbox_inches="tight" trims to content, so these set the
# starting scale / aspect (how large text reads relative to the panel), not exact
# output dimensions. Centralized so figures stay consistent across the paper.
FIG_SIZES = {
    "de":          (10, 6),     # ROC-style, 2 direction facets
    "coexpr":      (12, 6.5),   # threshold facets / DP grid
    "difficulty":  (11, 8),     # stacked up/down spaghetti
    "pathway":     (8, 9),      # tall: ~50 pathway rows
    "mia_bar":     (9, 8),      # 2 stacked metric facets
    "mia_box":     (9, 8),
    "mia_heatmap": (9, 7),      # attacks x generators
    "corr":        (9, 8),      # square-ish triangle
    "tradeoff":    (16, 5),     # wide 1x4 strip
}

# Paper-wide privacy convention, THREE categories:
#   non-DP      -> circle    (no privacy mechanism)
#   DP (ε)      -> triangle  (formal, data-level (ε,δ)-DP: DP-CVAE, DP-PGM, DP-CTGAN)
#   Privatized  -> square    (noise added but NOT a formal data-level DP guarantee:
#                             P-NMF, whose ε covers only the released per-cluster means)
PRIV_NONDP = "non-DP"
PRIV_DP = "DP (\u03b5)"
PRIV_PRIVATIZED = "Privatized"
PRIVACY_LABEL = {False: PRIV_NONDP, True: PRIV_DP}   # kept for back-compat (binary is_DP)
# Fixed legend order for the Privacy shape scale, applied everywhere so the key
# order never changes between figures.
PRIVACY_ORDER = [PRIV_NONDP, PRIV_DP, PRIV_PRIVATIZED]
DP_SHAPES = {PRIV_NONDP: "o", PRIV_DP: "^", PRIV_PRIVATIZED: "s"}

# Models that are privatized but NOT formally DP at the data level. Matched by
# base name so "P-NMF", "P-NMF ε=4" etc. all classify the same way.
PRIVATIZED_MODELS = ("P-NMF", "PNMF", "Privatized-NMF")


def privacy_label(df, name_col="origin", is_dp_col="is_DP"):
    """Three-level privacy category per row: Privatized (P-NMF) takes precedence,
    else DP (ε) when is_DP, else non-DP. Returns a Series aligned to df.

    P-NMF is deliberately NOT counted as formally DP: its calibrated noise applies
    only to the released per-cluster gene means, not to the synthetic data, so it
    gets its own 'Privatized' category (square) rather than the DP triangle.
    """
    def _is_privatized(n):
        s = str(n).replace("_", "-").lower()
        return any(s.startswith(m.lower()) for m in PRIVATIZED_MODELS)
    priv = df[name_col].map(_is_privatized)
    isdp = df[is_dp_col] if is_dp_col in df.columns else False
    out = df[name_col].map(lambda _: PRIV_NONDP)
    out = out.mask(isdp if isdp is not False else False, PRIV_DP)
    out = out.mask(priv, PRIV_PRIVATIZED)   # privatized overrides (P-NMF never DP)
    # ordered categorical -> stable legend order across all figures
    return pd.Categorical(out, categories=PRIVACY_ORDER, ordered=True)
# Ordered marker palette for categorical shape encodings (e.g. MIA attacks).
# ~12 is the practical ceiling for distinguishable markers.
SHAPE_VALUES = ["o", "s", "^", "v", "D", "P", "X", "*", "p", "h", "<", ">"]


def _paired_dp_mask(df, name_col="origin"):
    """Boolean Series: True for DP variants that HAVE a non-DP counterpart in the
    set (DP-CTGAN when CTGAN is present), False for intrinsically-DP methods with
    no twin (e.g. PGG-PGM). Data-driven — strips the DP marker and checks whether
    the base name appears among the models. Used to decide the see-through fill:
    paired DP -> transparent (helps read overlap with its twin); standalone DP ->
    solid (no twin to disambiguate, keep full colour). Both keep size + dark edge.
    """
    import re as _re
    is_dp = df[name_col].str.contains("\u03b5=", case=False, na=False)
    def _base(n):
        b = _re.sub(r"\s*\u03b5\s*=\s*[\d.]+", "", n)          # drop "ε=.."
        b = _re.sub(r"^DP[-_ ]?", "", b, flags=_re.IGNORECASE)      # drop leading DP-
        return b.strip()
    bases = df[name_col].map(_base)
    present = set(df[name_col])
    # paired if DP AND its stripped base is present as another (non-DP) model
    return is_dp & bases.isin(present) & (bases != df[name_col])


def _priv_emphasis_layers(df, fill="origin", *, point_size, x=None, y=None):
    """Return the ordered geom_point layers for the three-category privacy
    encoding, sharing one recipe across de/coexpr/tradeoff figures:
      - non-DP     -> circle, base size, white edge, solid
      - DP (ε)     -> triangle, larger, black edge; paired DP see-through, solo solid
      - Privatized -> square, larger, black edge, see-through (P-NMF; twin of NMF)
    `df` must have a 'Privacy' column (from privacy_label) and 'is_DP'. Pass x/y to
    set aes explicitly, else the caller's ggplot aes is inherited.
    """
    df = df.copy()
    df["_paired_dp"] = _paired_dp_mask(df)
    def _aes():
        return aes(x=x, y=y, fill=fill, shape="Privacy") if x else \
               aes(fill=fill, shape="Privacy")
    nondp = df[df["Privacy"] == PRIV_NONDP]
    dp_pair = df[(df["Privacy"] == PRIV_DP) & df["_paired_dp"]]
    dp_solo = df[(df["Privacy"] == PRIV_DP) & ~df["_paired_dp"]]
    priv = df[df["Privacy"] == PRIV_PRIVATIZED]
    layers = [
        geom_point(_aes(), data=nondp, color="white",
                   size=point_size, stroke=0.6, alpha=0.95),
        geom_point(_aes(), data=dp_solo, color="black",
                   size=point_size + 1.4, stroke=1.0, alpha=0.95),
        geom_point(_aes(), data=dp_pair, color="black",
                   size=point_size + 1.4, stroke=1.0, alpha=0.55),
        geom_point(_aes(), data=priv, color="black",
                   size=point_size , stroke=1.0, alpha=0.55),
    ]
    return layers

# MIA label maps live in constants; fall back to empty so plots still build.
try:
    from src.constants import MIA_METHOD_LABELS, FS_TOKEN_LABELS
except Exception:  # pragma: no cover - constants optional at import time
    MIA_METHOD_LABELS, FS_TOKEN_LABELS = {}, {}



def house_theme(base_size=BASE_SIZE, title_scale=1.3, axis_scale=1.1, strip_scale=1.0):
    """House style derived from one base font size.

    theme_bw(base_size=...) scales axis/legend text proportionally; this adds the
    non-proportional bits (bold titles/strips) plus shared legend, caption and
    subtitle styling so individual builders don't re-declare them. Pass a larger
    base_size for talks/posters.
    """
    return (
        theme_bw(base_size=base_size)
        + theme(
            plot_title=element_text(size=base_size * title_scale, face="bold", ha="center"),
            plot_subtitle=element_text(size=base_size * 0.8, color=GREY, ha="center"),
            plot_caption=element_text(size=base_size * 0.9, color=GREY, ha="right"),
            axis_title=element_text(size=base_size * axis_scale * 1.1, face="bold"),
            strip_text=element_text(size=base_size * strip_scale, face="bold"),
            legend_title=element_text(size=base_size * 0.9, face="bold"),
            legend_text=element_text(size=base_size * 0.95),
            axis_text_x=element_text(size=base_size * axis_scale),
            axis_text_y=element_text(size=base_size * axis_scale),
            legend_position="bottom",
            legend_direction="horizontal",
            panel_grid_minor=element_blank(),
            panel_grid_major=element_line(color=GRID_GREY, size=0.4),
            panel_spacing=0.02,
        )
    )



THEME = house_theme()        # default; e.g. house_theme(14) for a presentation


def _gen_priv_guides(channel="color", ncol=LEGEND_NCOL):
    """Standard generator + Privacy legends, identical across every figure.

    `channel` is whichever aesthetic carries the generator colour ("color" or
    "fill"); the Privacy shape legend keys are forced dark so they're visible.
    Returns a list [guides(...), theme(...)] — the trailing theme re-asserts the
    bottom legend, because adding multiple guides can otherwise let plotnine
    revert legend placement to the right.
    """
    override = ({"fill": "black", "color": "black"} if channel == "fill"
                else {"color": "black"})
    return [
        guides(**{
            channel: guide_legend(title=LEGEND_GEN, ncol=ncol),
            "shape": guide_legend(title=LEGEND_PRIV, ncol=1, override_aes=override),
        }),
        theme_fn(legend_position="bottom", legend_direction="horizontal",
                 legend_box="horizontal"),
    ]


def save_fig(fig, path, *, size=None, width=11, height=6.5, dpi=300):
    """Save a plotnine figure, creating the dir and never clipping legends/titles.

    Pass `size=FIG_SIZES["..."]` (a (width, height) tuple) to use a canonical
    canvas, or set width/height directly. bbox_inches='tight' recomputes the
    bounding box to include the bottom legend and title, so cut-off elements
    stop happening regardless of the starting size.
    """
    if size is not None:
        width, height = size
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.save(path, width=width, height=height, dpi=dpi,
             bbox_inches="tight", pad_inches=0.1, verbose=False)
    return path


def fig_path(figures_dir, dataset_name, name, **params):
    """Templated figure path so it can't drift from the hyperparameters.

    fig_path(FIG_DIR, "BRCA", "DE", lfc=0.0, fpr=0.05)
      -> "{FIG_DIR}/BRCA/BRCA_DE_lfc=0.0_fpr=0.05.png"
    """
    tag = "_".join(f"{k}={v}" for k, v in params.items())
    fname = f"{dataset_name}_{name}" + (f"_{tag}" if tag else "") + ".png"
    return os.path.join(figures_dir, dataset_name, fname)


def plot_de_recovery(points, *, dataset_name, lfc_threshold, fpr_max, colors,
                     dp_split=False, dp_emphasis=True, lfc_facet=False,
                     point_size=POINT_SIZE):
    """ROC-style DE gene-recovery scatter: mean +/- std TPR/FPR per generator.

    points   : overall-summary df (fpr_mean/std, tpr_mean/std, origin,
               direction, is_DP).
    dp_emphasis : (default) all generators on shared axes; DP points drawn larger
               with a dark outline and a see-through fill so overlapping points
               stay readable. The clearest DP/non-DP contrast without faceting.
    dp_split : if True, instead facet DP vs non-DP into columns (direction rows).
    lfc_facet: if True, facet direction (rows) x LFC threshold (cols).
    """
    pts = points.copy()
    pts["Privacy"] = privacy_label(pts)

    if lfc_facet:
        facet = facet_grid("direction ~ lfc_label")
        title = f"{dataset_name} \u2014 DE gene recovery across \u0394VST"
    elif dp_split:
        facet = facet_grid("direction ~ Privacy")
        title = (f"{dataset_name} \u2014 DE gene recovery "
                 f"(\u0394VST={lfc_threshold}") #, FPR\u2264{fpr_max})
    else:
        facet = facet_wrap("~direction")
        title = (f"{dataset_name} \u2014 DE gene recovery "
                 f"(\u0394VST={lfc_threshold})") #, FPR\u2264{fpr_max}

    p = (
        ggplot(pts, aes("fpr_mean", "tpr_mean"))
        + geom_hline(yintercept=1, linetype="dashed", color="gray", alpha=0.5)
        + geom_vline(xintercept=0, linetype="dashed", color="gray", alpha=0.5)
        + geom_errorbar(aes(ymin="tpr_mean - tpr_std", ymax="tpr_mean + tpr_std",
                            color="origin"), width=0.002, alpha=0.6, size=0.8)
        + geom_errorbarh(aes(xmin="fpr_mean - fpr_std", xmax="fpr_mean + fpr_std",
                             color="origin"), height=0.02, alpha=0.6, size=0.8)
    )

    if dp_emphasis and not dp_split:
        # three-category recipe (non-DP circle / DP triangle / Privatized square);
        # DP and Privatized larger with dark edge, paired ones see-through so
        # overlap with their non-DP twin stays readable.
        for _layer in _priv_emphasis_layers(pts, fill="origin",
                                            point_size=(point_size - 2)):
            p += _layer
    else:
        p += geom_point(aes(fill="origin", shape="Privacy"), color="white",
                        size=point_size, stroke=0.6, alpha=0.95)

    p += facet
    p += scale_fill_manual(values=colors)
    p += scale_color_manual(values=colors)
    p += scale_shape_manual(values=DP_SHAPES)
    p += coord_cartesian(xlim=(-0.001, fpr_max + 0.001), ylim=(0, 1.05))
    p += labs(title=title, x="False positive rate", y="True positive rate")
    p += THEME
    p += _gen_priv_guides("fill")
    p += guides(color=False)
    return p


def plot_coexpr_recovery(summary, *, dataset_name, colors, dp_split=False,
                         dp_emphasis=True, log=True, point_size=POINT_SIZE, ncol=3):
    """Co-expression edge recovery: recall vs false-edge-rate per generator,
    faceted by correlation threshold.

    dp_emphasis : (default) all generators together; DP points drawn larger with a
               dark edge and see-through fill so overlaps stay readable.
    dp_split : if True, instead facet is_DP x threshold (grid, DP as a facet row).
    """
    s = summary.copy()
    s["Privacy"] = privacy_label(s)
    s["threshold"] = s["corr_threshold"].map(lambda x: f"r > {x}")

    common = [
        geom_abline(intercept=0, slope=1, linetype="dashed", color="gray",
                    size=0.8, alpha=0.6),
        labs(title=f"{dataset_name} \u2014 Co-expression edge recovery",
             x="False edge rate (spurious / real edges)",
             y="Recall (correct / real edges)"),
        THEME,
    ]
    if log:
        common += [scale_x_log10(), scale_y_log10()]

    if dp_split:
        p = ggplot(s, aes("fer_mean", "recall_mean", color="origin"))
        p += geom_errorbar(aes(ymin="recall_mean - recall_std",
                               ymax="recall_mean + recall_std"),
                           width=0, size=1, alpha=0.8)
        p += geom_errorbarh(aes(xmin="fer_mean - fer_std", xmax="fer_mean + fer_std"),
                            height=0, size=1, alpha=0.8)
        p += geom_point(aes(shape="Privacy"), size=point_size, alpha=0.95)
        p += facet_grid("Privacy ~ threshold", scales="free_x")
        p += scale_color_manual(values=colors)
        p += scale_shape_manual(values=DP_SHAPES)
        p += _gen_priv_guides("color")
    else:
        p = ggplot(s, aes("fer_mean", "recall_mean"))
        p += geom_errorbar(aes(ymin="recall_mean - recall_std",
                               ymax="recall_mean + recall_std", color="origin"),
                           width=0, size=1, alpha=0.8)
        p += geom_errorbarh(aes(xmin="fer_mean - fer_std", xmax="fer_mean + fer_std",
                                color="origin"), height=0, size=1, alpha=0.8)
        if dp_emphasis:
            s["_paired_dp"] = _paired_dp_mask(s)
            nondp = s[s["Privacy"] == PRIV_NONDP]
            dp_pair = s[(s["Privacy"] == PRIV_DP) & s["_paired_dp"]]
            dp_solo = s[(s["Privacy"] == PRIV_DP) & ~s["_paired_dp"]]
            priv = s[s["Privacy"] == PRIV_PRIVATIZED]
            p += geom_point(aes(fill="origin", shape="Privacy"), data=nondp,
                            color="white", size=point_size, stroke=0.6, alpha=0.95)
            p += geom_point(aes(fill="origin", shape="Privacy"), data=dp_solo,
                            color="black", size=point_size + 1.4, stroke=1.0, alpha=0.95)
            p += geom_point(aes(fill="origin", shape="Privacy"), data=dp_pair,
                            color="black", size=point_size + 1.4, stroke=1.0, alpha=0.55)
            p += geom_point(aes(fill="origin", shape="Privacy"), data=priv,
                            color="black", size=point_size, stroke=1.0, alpha=0.55)
        else:
            p += geom_point(aes(fill="origin", shape="Privacy"), color="white",
                            size=point_size, stroke=0.6, alpha=0.95)
        p += facet_wrap("~threshold", scales="free_x", ncol=ncol)
        p += scale_fill_manual(values=colors)
        p += scale_color_manual(values=colors)
        p += scale_shape_manual(values=DP_SHAPES)
        p += _gen_priv_guides("fill")
        p += guides(color=False)

    for layer in common:
        p += layer
    return p


def plot_comparison_difficulty(diff, *, dataset_name, lfc_threshold, fpr_max,
                               colors, target=0.8, point_size=2.6):
    """Mean TPR per subtype comparison, one line per model, ordered easy->hard.

    A thick grey 'average difficulty' line runs through the per-model lines so
    comparisons that are hard for everyone are visible under the spaghetti.
    """
    d = diff.copy()
    d["Privacy"] = privacy_label(d)
    return (
        ggplot(d, aes("reorder(comparison, avg_tpr)", "tpr",
                      color="origin", group="origin"))
        + geom_hline(yintercept=target, linetype="dashed",
                     color="darkgreen", size=0.8, alpha=0.5)
        + geom_line(aes(y="avg_tpr", group="1"), color=GREY, size=1.5, alpha=0.9)
        + geom_line(size=0.9, alpha=0.7)
        + geom_point(aes(shape="Privacy"), size=point_size, alpha=0.85)
        + facet_wrap("~direction", ncol=1, scales="free_x")
        + scale_color_manual(values=colors)
        + scale_shape_manual(values=DP_SHAPES)
        + coord_cartesian(ylim=(0, 1.05))
        + labs(title=f"{dataset_name} \u2014 DE recovery across subtype comparisons "
                     f"(\u0394VST={lfc_threshold}, FPR\u2264{fpr_max})",
               x="Pairwise comparison (ordered easy \u2192 difficult)",
               y="True positive rate")
        + THEME
        + theme_fn(axis_text_x=element_text(rotation=45, ha="right"),
                   legend_position="bottom", legend_direction="horizontal")
        + _gen_priv_guides("color", ncol=5)
    )


def plot_pathway_heatmap(per_pathway, *, dataset_name, value_col="effect_size",
                         method_order=None, cluster_rows=True,
                         diverging=None, limits=None, strip_prefix="HALLMARK_"):
    """Pathway (rows) x method (cols) heatmap of ssGSEA fidelity.

    value_col="effect_size" -> signed, diverging palette centered at 0
                               (blue = synthetic under-scores, red = over-scores).
    value_col="ks_stat"     -> unsigned magnitude, sequential palette.
    Rows are hierarchically clustered so co-distorted pathways sit together.
    """
    d = per_pathway.copy()
    d["pathway"] = d["pathway"].str.replace(strip_prefix, "", regex=False)

    mat = d.pivot(index="pathway", columns="origin", values=value_col)
    if method_order is not None:
        mat = mat[[m for m in method_order if m in mat.columns]]

    if cluster_rows and mat.shape[0] > 2:
        Z = linkage(mat.fillna(0.0).values, method="average")
        row_order = list(mat.index[leaves_list(Z)])
    else:
        row_order = list(mat.index)

    long = mat.reset_index().melt(id_vars="pathway", var_name="origin",
                                  value_name=value_col)
    long["pathway"] = pd.Categorical(long["pathway"], categories=row_order, ordered=True)
    if method_order is not None:
        long["origin"] = pd.Categorical(
            long["origin"], categories=[m for m in method_order if m in mat.columns],
            ordered=True)

    if diverging is None:
        diverging = (value_col == "effect_size")

    p = (
        ggplot(long, aes("origin", "pathway", fill=value_col))
        + geom_tile(color="white", size=0.3)
    )
    if diverging:
        m = float(limits) if limits else float(long[value_col].abs().max())
        p += scale_fill_gradient2(low="#2166ac", mid="white", high="#b2182b",
                                  midpoint=0, limits=(-m, m))
    else:
        p += scale_fill_gradient(low="#f7fbff", high="#08306b")

    p += labs(title=f"{dataset_name} \u2014 pathway fidelity ({value_col})",
              x="", y="", fill=value_col)
    p += THEME
    p += theme_fn(axis_text_x=element_text(rotation=45, ha="right"),
                  axis_text_y=element_text(size=AXIS_Y_SMALL),
                  panel_grid_major=element_blank(),
                  legend_position="right", legend_direction="vertical")
    return p


def plot_mia_worstcase(worst, *, dataset_name, colors=None,
                       metric_labels=None, floors=None,
                       annotate="code", attack_codes=None, max_shapes=12,
                       attack_labels=None, fs_labels=None):
    """Worst-case MIA leakage per generator (strongest attack), faceted by metric.

    Bars sorted by leakage (position = generator robustness; lower/left = more
    private). The winning attack is shown via `annotate`:
      "label" — full attack name as rotated text on each bar
      "code"  — short code on the bar + a subtitle key expanding the codes
                (default; neutral bars, keeps the generator-colour convention free)
      "shape" — a marker at each bar top keyed to the attack (max `max_shapes`
                distinct; falls back to "code" beyond that)
    attack_labels / fs_labels : {raw: pretty} maps for nice attack and FS names.
    """
    metric_labels = metric_labels or {"tpr_at_fpr_01": "TPR@FPR=0.1",
                                       "mia_aucroc": "AUC-ROC"}
    floors = floors or {"TPR@FPR=0.1": 0.1, "AUC-ROC": 0.5}
    attack_labels = attack_labels if attack_labels is not None else MIA_METHOD_LABELS
    fs_labels = fs_labels if fs_labels is not None else FS_TOKEN_LABELS

    d = worst.copy()
    d["metric"] = d["metric"].map(metric_labels)
    floor_df = pd.DataFrame({"metric": list(floors), "y": list(floors.values())})

    def _pretty(row):
        name = attack_labels.get(row["attack"], row["attack"])
        fs = row.get("feature_selection", "")
        if fs:
            name += f" ({fs_labels.get(fs, fs)})"
        return name
    d["attack_disp"] = d.apply(_pretty, axis=1)

    winners = sorted(d["attack_disp"].unique())
    if annotate == "shape" and len(winners) > max_shapes:
        annotate = "code"

    if attack_codes is None:
        def _code(a):
            toks = [t for t in a.replace("-", "_").replace(" ", "_")
                     .replace("(", "").replace(")", "").split("_") if t]
            return "".join(t[0] for t in toks).upper()[:4]
        attack_codes = {a: _code(a) for a in winners}
    d["code"] = d["attack_disp"].map(attack_codes)
    code_legend = "   ".join(f"{c} = {a}" for a, c in attack_codes.items())

    p = ggplot(d, aes("reorder(origin, value)", "value"))
    p += geom_col(fill="#9ecae1", alpha=0.9, width=0.8)
    p += geom_errorbar(aes(ymin="value - std", ymax="value + std"),
                       width=0.25, alpha=0.7)

    legend_pos = "none"
    if annotate == "label":
        p += geom_text(aes(label="attack_disp"), va="bottom", ha="left",
                       size=ANNOT_SIZE, nudge_y=0.005, angle=90)
        sub = ""
    elif annotate == "shape":
        p += geom_point(aes(shape="attack_disp", y="value + std"), size=3)
        p += scale_shape_manual(values=dict(zip(winners, SHAPE_VALUES[:len(winners)])),
                                name="Strongest attack")
        legend_pos = "bottom"
        sub = ""
    else:  # code
        p += geom_text(aes(label="code"), va="bottom", ha="center",
                       size=ANNOT_SIZE + 1, nudge_y=0.006)
        sub = code_legend

    p += geom_hline(aes(yintercept="y"), data=floor_df, inherit_aes=False,
                    linetype="dashed", color=GREY, size=0.6)
    p += facet_wrap("~metric", scales="free_y", ncol=1)
    p += labs(title=f"{dataset_name} \u2014 worst-case MIA risk (strongest attack)",
              subtitle=sub or None,
              x="Generative model", y="Leakage (lower = more private)")
    p += THEME
    p += theme_fn(axis_text_x=element_text(rotation=45, ha="right"),
                  legend_position=legend_pos, legend_direction="horizontal")
    return p



def plot_mia_worstcase_box(worst_folds, *, dataset_name, metric_labels=None,
                           floors=None, show_points=True,
                           annotate="shape", attack_labels=None, fs_labels=None):
    """Boxplot of the worst-case attack's per-fold leakage, per generator.

    One box per generator over the folds of its strongest attack (selected by
    fold-mean, matching the bar figure), faceted by metric, sorted by median.
    The winning attack is marked by a single large shape ABOVE each box
    (annotate="shape"), or as a short code / rotated label; this keeps the attack
    identity without cluttering every fold point.
    """
    metric_labels = metric_labels or {"tpr_at_fpr_01": "TPR@FPR=0.1",
                                       "mia_aucroc": "AUC-ROC"}
    floors = floors or {"TPR@FPR=0.1": 0.1, "AUC-ROC": 0.5}
    attack_labels = attack_labels if attack_labels is not None else MIA_METHOD_LABELS
    fs_labels = fs_labels if fs_labels is not None else FS_TOKEN_LABELS

    d = worst_folds.copy()
    d["metric"] = d["metric"].map(metric_labels)
    floor_df = pd.DataFrame({"metric": list(floors), "y": list(floors.values())})

    def _pretty(r):
        name = attack_labels.get(r["attack"], r["attack"])
        fs = r.get("feature_selection", "")
        if fs:
            name += f" ({fs_labels.get(fs, fs)})"
        return name
    d["attack_disp"] = d.apply(_pretty, axis=1)

    winners = sorted(d["attack_disp"].unique())
    def _code(a):
        toks = [t for t in a.replace("-", "_").replace(" ", "_")
                 .replace("(", "").replace(")", "").split("_") if t]
        return "".join(t[0] for t in toks).upper()[:4]
    codes = {a: _code(a) for a in winners}
    d["code"] = d["attack_disp"].map(codes)
    code_legend = "   ".join(f"{c} = {a}" for a, c in codes.items())

    # one row per box: winning attack + a y just ABOVE the box top for the marker
    tops = (d.groupby(["origin", "metric"])
              .agg(value=("value", "max"), code=("code", "first"),
                   attack_disp=("attack_disp", "first")).reset_index())
    span = d.groupby("metric")["value"].agg(lambda s: s.max() - s.min()).to_dict()
    tops["ypos"] = tops.apply(
        lambda r: r["value"] + 0.06 * (span.get(r["metric"], r["value"]) or 1), axis=1)

    p = (
        ggplot(d, aes("reorder(origin, value)", "value"))
        + geom_boxplot(fill="#d9d9d9", color=GREY, alpha=0.6,
                       outlier_alpha=0, width=0.6)
    )
    # plain fold points (no shape mapping) — show spread without clutter
    if show_points:
        p += geom_jitter(width=0.10, size=1.3, alpha=0.6, color="#08306b")

    sub = ""
    if annotate == "shape":
        if len(winners) > len(SHAPE_VALUES):
            annotate = "code"                      # too many attacks for shapes
        else:
            # ONE large marker per box, ABOVE it, keyed to the winning attack
            p += geom_point(aes(x="reorder(origin, value)", y="ypos",
                                shape="attack_disp"), data=tops, inherit_aes=False,
                            size=4, color="#08306b", fill="#08306b")
            p += scale_shape_manual(values=dict(zip(winners,
                                    SHAPE_VALUES[:len(winners)])),
                                    name="Strongest attack")
            p += guides(shape=guide_legend(title="Strongest attack", ncol=3))
    if annotate == "label":
        p += geom_text(aes(x="reorder(origin, value)", y="ypos", label="attack_disp"),
                       data=tops, inherit_aes=False, va="bottom", ha="left",
                       size=ANNOT_SIZE - 1, angle=90)
    elif annotate == "code":
        p += geom_text(aes(x="reorder(origin, value)", y="ypos", label="code"),
                       data=tops, inherit_aes=False, va="bottom", ha="center",
                       size=ANNOT_SIZE + 1)
        sub = code_legend

    p += geom_hline(aes(yintercept="y"), data=floor_df, inherit_aes=False,
                    linetype="dashed", color=GREY, size=0.6)
    p += facet_wrap("~metric", scales="free_y", ncol=1)
    p += labs(title=f"{dataset_name} \u2014 Strongest-attack MIA risk",
              subtitle=sub or None,
              x="Generative model", y="MIA risk (lower = more private)")
    p += THEME
    p += theme_fn(axis_text_x=element_text(rotation=45, ha="right"),
                  legend_position=("bottom" if annotate == "shape" else "none"),
                  legend_direction="horizontal")
    return p


def plot_mia_worstcase_box2(worst_folds, *, dataset_name, metric_labels=None,
                           floors=None, show_points=True,
                           annotate="code", attack_labels=None, fs_labels=None):
    """Boxplot of the worst-case attack's per-fold leakage, per generator.

    One box per generator over the folds of its strongest attack (selected by
    fold-mean, matching the bar figure), faceted by metric, sorted by median.
    Each box annotated with the winning attack (code/label text, or shaped fold
    points) so the attack identity isn't lost vs the bar figure.
    """
    metric_labels = metric_labels or {"tpr_at_fpr_01": "TPR@FPR=0.1",
                                       "mia_aucroc": "AUC-ROC"}
    floors = floors or {"TPR@FPR=0.1": 0.1, "AUC-ROC": 0.5}
    attack_labels = attack_labels if attack_labels is not None else MIA_METHOD_LABELS
    fs_labels = fs_labels if fs_labels is not None else FS_TOKEN_LABELS

    d = worst_folds.copy()
    d["metric"] = d["metric"].map(metric_labels)
    floor_df = pd.DataFrame({"metric": list(floors), "y": list(floors.values())})

    def _pretty(r):
        name = attack_labels.get(r["attack"], r["attack"])
        fs = r.get("feature_selection", "")
        if fs:
            name += f" ({fs_labels.get(fs, fs)})"
        return name
    d["attack_disp"] = d.apply(_pretty, axis=1)

    winners = sorted(d["attack_disp"].unique())
    def _code(a):
        toks = [t for t in a.replace("-", "_").replace(" ", "_")
                 .replace("(", "").replace(")", "").split("_") if t]
        return "".join(t[0] for t in toks).upper()[:4]
    codes = {a: _code(a) for a in winners}
    d["code"] = d["attack_disp"].map(codes)
    code_legend = "   ".join(f"{c} = {a}" for a, c in codes.items())

    tops = (d.groupby(["origin", "metric"])
              .agg(value=("value", "max"), code=("code", "first"),
                   attack_disp=("attack_disp", "first")).reset_index())

    p = (
        ggplot(d, aes("reorder(origin, value)", "value"))
        + geom_boxplot(fill="#9ecae1", color="#3182bd", alpha=0.6,
                       outlier_alpha=0, width=0.6)
    )

    if annotate == "shape":
        if len(winners) > len(SHAPE_VALUES):
            annotate = "code"
        else:
            p += geom_jitter(aes(shape="attack_disp"), width=0.12, size=2.2,
                             alpha=0.85, color="#08306b")
            p += scale_shape_manual(values=dict(zip(winners, SHAPE_VALUES[:len(winners)])),
                                    name="Strongest attack")
            sub = ""
    elif show_points:
        p += geom_jitter(width=0.12, size=1.4, alpha=0.7, color="#08306b")

    if annotate == "label":
        p += geom_text(aes(x="reorder(origin, value)", y="value", label="attack_disp"),
                       data=tops, inherit_aes=False, va="bottom", ha="left",
                       size=ANNOT_SIZE - 1, angle=90, nudge_y=0.01)
        sub = ""
    elif annotate == "code":
        p += geom_text(aes(x="reorder(origin, value)", y="value", label="code"),
                       data=tops, inherit_aes=False, va="bottom", ha="center",
                       size=ANNOT_SIZE + 1, nudge_y=0.012)
        sub = code_legend

    p += geom_hline(aes(yintercept="y"), data=floor_df, inherit_aes=False,
                    linetype="dashed", color=GREY, size=0.6)
    p += facet_wrap("~metric", scales="free_y", ncol=1)
    p += labs(title=f"{dataset_name} \u2014 Strongest-attack MIA risk (per-fold)",
              subtitle=sub or None,
              x="Generative model", y="MIA Risk (lower = more private)")
    p += THEME
    p += theme_fn(axis_text_x=element_text(rotation=45, ha="right"),
                  legend_position=("bottom" if annotate == "shape" else "none"),
                  legend_direction="horizontal")
    return p


def plot_mia_heatmap(heat, *, dataset_name, metric_label, method_order=None,
                     palette=("#f7fbff", "#08306b"), floor=None,
                     show_values=True, decimals=2):
    """MIA leakage heatmap: attacks (rows) x generators (columns), one metric.

    Sequential fill (pale = safe, dark = leaky). Rows ordered by mean leakage
    across generators (strongest attacks at top). `floor` (e.g. 0.5 for AUC-ROC,
    0.1 for TPR@FPR) is noted in the legend title, not subtracted.
    `show_values` overlays each cell's score (white on dark, dark on pale cells).
    """
    d = heat.copy()
    row_order = (d.groupby("attack_disp")["value"].mean()
                  .sort_values(ascending=True).index.tolist())  # leakiest at top
    d["attack_disp"] = pd.Categorical(d["attack_disp"], categories=row_order, ordered=True)
    if method_order is not None:
        cols = [m for m in method_order if m in d["origin"].unique()]
        d["origin"] = pd.Categorical(d["origin"], categories=cols, ordered=True)

    fill_title = metric_label + (f"\n(floor {floor})" if floor is not None else "")
    p = (
        ggplot(d, aes("origin", "attack_disp", fill="value"))
        + geom_tile(color="white", size=0.3)
    )
    if show_values:
        vmin, vmax = d["value"].min(), d["value"].max()
        thresh = vmin + 0.6 * (vmax - vmin)            # darker than this -> white text
        d["lab"] = d["value"].map(lambda v: f"{v:.{decimals}f}")
        d["txt"] = d["value"].map(lambda v: "white" if v >= thresh else "#222222")
        p += geom_text(aes(label="lab", color="txt"), size=CELL_SIZE)
        p += scale_color_identity(guide=None)
    p += scale_fill_gradient(low=palette[0], high=palette[1], name=fill_title)
    p += labs(title=f"{dataset_name} \u2014 MIA Risk: {metric_label}", x="", y="")
    p += THEME
    p += theme_fn(axis_text_x=element_text(rotation=45, ha="right"),
                  axis_text_y=element_text(size=AXIS_Y_SMALL),
                  panel_grid_major=element_blank(),
                  legend_position="right", legend_direction="vertical")
    return p


def plot_correlation_heatmap(corr, *, dataset_name, metric_labels=None,
                             caption=None, cluster=False):
    """Metric-metric Spearman correlation heatmap (lower triangle).

    `corr` is the dict from compute_metric_correlations (rho, q, long, n). Each
    cell shows rho and significance stars from the BH-FDR q. Diverging blue->red
    centered at 0. `caption` is rendered small/grey at the bottom.
    """
    metric_labels = metric_labels or {}
    rho, q = corr["rho"], corr["q"]
    order = list(rho.columns)
    if cluster and len(order) > 2:
        Z = linkage(rho.fillna(0).values, method="average")
        order = [rho.columns[i] for i in leaves_list(Z)]
        rho, q = rho.loc[order, order], q.loc[order, order]

    long = (rho.reset_index().melt(id_vars="index", var_name="m2", value_name="rho")
               .rename(columns={"index": "m1"}))
    qlong = (q.reset_index().melt(id_vars="index", var_name="m2", value_name="q")
              .rename(columns={"index": "m1"}))
    d = long.merge(qlong, on=["m1", "m2"])

    pos = {m: i for i, m in enumerate(order)}
    d["i"], d["j"] = d["m1"].map(pos), d["m2"].map(pos)
    # clustered -> full square (dendrogram order reads on both axes);
    # unclustered -> lower triangle + diagonal (compact).
    if not cluster:
        d = d[d["i"] >= d["j"]].copy()

    def _stars(qv):
        if pd.isna(qv):
            return ""
        return "***" if qv < 0.001 else "**" if qv < 0.01 else "*" if qv < 0.05 else ""
    d["lab"] = d.apply(lambda r: ("" if r["i"] == r["j"]
                                  else f"{r['rho']:.2f}\n{_stars(r['q'])}"), axis=1)
    title_extra = " (clustered)" if cluster else ""

    labs_map = {m: metric_labels.get(m, m) for m in order}
    d["m1"] = pd.Categorical(d["m1"].map(labs_map),
                             categories=[labs_map[m] for m in order], ordered=True)
    d["m2"] = pd.Categorical(d["m2"].map(labs_map),
                             categories=[labs_map[m] for m in order], ordered=True)

    return (
        ggplot(d, aes("m2", "m1", fill="rho"))
        + geom_tile(color="white", size=0.4)
        + geom_text(aes(label="lab"), size=ANNOT_SIZE, lineheight=0.8)
        + scale_fill_gradient2(low="#2166ac", mid="white", high="#b2182b",
                               midpoint=0, limits=(-1, 1), name="Spearman \u03c1")
        + labs(title=f"{dataset_name} \u2014 metric correlations{title_extra}",
               x="", y="", caption=caption)
        + THEME
        + theme_fn(axis_text_x=element_text(rotation=45, ha="right"),
                   panel_grid_major=element_blank(),
                   legend_position="right", legend_direction="vertical")
    )


def plot_tradeoff(tradeoff, *, dataset_name, colors, q_threshold=0.05,
                  line_q_threshold=0.048,
                  error_bars=True, ncol=2, privacy_floor=None, utility_floor=None,
                  anchor_label="Privacy (worst-case TPR@FPR)",
                  title=None, caption=None, point_size=3, dp_emphasis=True):
    """Anchored scatter panels: y-anchor vs each x-metric.

    `tradeoff` is the long frame from prepare_tradeoff. Points coloured by
    generator, SHAPED by DP status (triangle = DP). `dp_emphasis` makes DP points
    larger with a dark outline so they're easy to pick out among the colours.
    `error_bars` draws light 2D fold-std crosses. A grey LOESS trend is drawn only
    on panels with BH-FDR q < q_threshold. `privacy_floor` (e.g. 0.1) draws the
    no-leakage line on the y-anchor. `utility_floor` shades/lines an excluded
    low-utility region on x (only meaningful when x is a utility metric) — use to
    show a deployability cutoff without silently dropping points.
    """
    d = tradeoff.copy()
    d["Privacy"] = privacy_label(d)
    has_std = "x_std" in d.columns and "y_std" in d.columns

    rng = (d.groupby("panel")
             .agg(xmin=("x", "min"), xmax=("x", "max"),
                  ymin=("y", "min"), ymax=("y", "max"),
                  rho=("rho", "first"), q=("q", "first"), stars=("stars", "first"))
             .reset_index())
    rng["ax"] = rng["xmin"] + 0.02 * (rng["xmax"] - rng["xmin"])
    rng["ay"] = rng["ymax"] - 0.02 * (rng["ymax"] - rng["ymin"])
    rng["lab"] = rng.apply(
        lambda r: (f"\u03c1={r['rho']:.2f} {r['stars']}"
                   + (f"\nq={r['q']:.3f}"
                      if (pd.notna(r['q']) and r['q'] < q_threshold) else "")),
        axis=1)

    p = ggplot(d, aes("x", "y"))
    if utility_floor is not None:
        # shade the excluded low-utility region + a line at the floor
        p += geom_vline(xintercept=utility_floor, linetype="dotted",
                        color=GREY, size=0.6)
    if privacy_floor is not None:
        p += geom_hline(yintercept=privacy_floor, linetype="dashed",
                        color=LIGHT_GREY, size=0.6)

    if error_bars and has_std:
        p += geom_errorbar(aes(ymin="y - y_std", ymax="y + y_std", color="origin"),
                           width=0, size=0.5, alpha=0.4)
        p += geom_errorbarh(aes(xmin="x - x_std", xmax="x + x_std", color="origin"),
                            height=0, size=0.5, alpha=0.4)

    sig = d[d["q"] < line_q_threshold]   # buffer below 0.05: skip borderline
    if len(sig):
        p += geom_smooth(data=sig, method="loess", span=1.0, se=False,
                         color=GREY, size=0.7, linetype="solid")

    if dp_emphasis:
        d["_paired_dp"] = _paired_dp_mask(d)
        dp_pair = d[(d["Privacy"] == PRIV_DP) & d["_paired_dp"]]
        dp_solo = d[(d["Privacy"] == PRIV_DP) & ~d["_paired_dp"]]
        priv = d[d["Privacy"] == PRIV_PRIVATIZED]
        nondp = d[d["Privacy"] == PRIV_NONDP]
        p += geom_point(aes(fill="origin", shape="Privacy"), data=nondp,
                        color="white", size=point_size, stroke=0.4, alpha=0.9)
        p += geom_point(aes(fill="origin", shape="Privacy"), data=dp_solo,
                        color="black", size=point_size + 1.2, stroke=0.8, alpha=0.95)
        p += geom_point(aes(fill="origin", shape="Privacy"), data=dp_pair,
                        color="black", size=point_size + 1.2, stroke=0.8, alpha=0.55)
        p += geom_point(aes(fill="origin", shape="Privacy"), data=priv,
                        color="black", size=point_size + 1.2, stroke=0.8, alpha=0.55)
    else:
        p += geom_point(aes(fill="origin", shape="Privacy"),
                        color="white", size=point_size, stroke=0.4, alpha=0.9)

    p += geom_label(aes(x="ax", y="ay", label="lab"), data=rng, inherit_aes=False,
                    ha="left", va="top", size=ANNOT_SIZE, color="#333333",
                    fill="white", label_size=0, alpha=0.75)
    p += facet_wrap("~panel", scales="free_x", ncol=ncol)
    p += scale_fill_manual(values=colors)     # points read fill
    p += scale_color_manual(values=colors)    # error bars still read color
    p += scale_shape_manual(values=DP_SHAPES)
    p += labs(title=title or f"{dataset_name} \u2014 trade-offs",
              x="Comparison metric", y=anchor_label, caption=caption)
    p += THEME
    p += theme_fn(legend_position="bottom",
                  panel_grid_major=element_line(color=GRID_GREY, size=0.5),
                  panel_grid_minor=element_line(color=GRID_GREY, size=0.25))
    
    p += _gen_priv_guides("fill")
    p += guides(color=False)   # suppress duplicate legend from error bars' color
    return p


def plot_epsilon_ablation(abl, *, dataset_name, colors,
                          x_label="F1 relative (utility)",
                          y_label="Privacy (worst-case TPR@FPR)",
                          privacy_floor=0.1, point_size=4, connect=False,
                          error_bars=True, title=None, legend_ncol=LEGEND_NCOL,
                          facet_col=None, ncol=2):
    """DP privacy-utility across an epsilon sweep.

    `abl` is the frame from prepare_epsilon_ablation (x, y, x_std, y_std, family,
    eps, origin). `colors` is a {origin: hex} dict (e.g. DP_EPSILON_COLORS from
    constants) — one hue per family, shaded by epsilon. Points are triangles
    (DP convention); epsilon is read from the colour shade + legend. `connect=True`
    optionally joins each family's points in epsilon order; off by default.

    `facet_col`: if given (e.g. "panel"), facet across its values with free x
    (shared privacy y-axis) — pass a frame stacking several quality metrics vs
    privacy (utility-vs-privacy | DE-vs-privacy) with that column set per block.
    When faceting, x_label is used as the shared x-title (or set to "" and let
    strip labels name each panel's metric).
    """
    d = abl.copy().sort_values(["family", "eps"])
    order = list(dict.fromkeys(d["origin"]))
    d["origin"] = pd.Categorical(d["origin"], categories=order, ordered=True)
    has_std = error_bars and "x_std" in d.columns and "y_std" in d.columns

    p = ggplot(d, aes("x", "y", color="origin"))
    if privacy_floor is not None:
        p += geom_hline(yintercept=privacy_floor, linetype="dashed",
                        color=LIGHT_GREY, size=0.6)
    if connect:
        grp = "family" if facet_col is None else "family"
        p += geom_path(aes(group=grp), size=0.6, alpha=0.5)
    if has_std:
        p += geom_errorbar(aes(ymin="y - y_std", ymax="y + y_std"),
                           width=0, size=0.5, alpha=0.4)
        p += geom_errorbarh(aes(xmin="x - x_std", xmax="x + x_std"),
                            height=0, size=0.5, alpha=0.4)
    p += geom_point(shape="^", size=point_size, alpha=0.95)
    p += scale_color_manual(values=colors)
    if facet_col is not None:
        p += facet_wrap("~" + facet_col, scales="free_x", ncol=ncol)
    p += labs(title=title or f"{dataset_name} \u2014 DP privacy\u2013utility across \u03b5",
              x=x_label, y=y_label)
    p += THEME
    p += guides(color=guide_legend(title="DP model (\u03b5)", ncol=legend_ncol))
    return p


def plot_rho_per_attack(rho_df, *, dataset_name, pair_labels=None,
                        attack_labels=None, sort_by=None, title=None, caption=None):
    """Dotplot of within-attack Spearman rho per metric pair.

    `rho_df` is from correlations_per_attack (attack, pair, rho, n). Each metric
    pair is a colour; attacks on y (sorted by `sort_by` pair's rho, or by mean
    rho across pairs). `attack_labels` ({raw: pretty}, defaults to
    MIA_METHOD_LABELS) prettifies the attack names on the y-axis. A reference
    line at rho=0 separates positive/negative association.
    """
    d = rho_df.copy()
    pair_labels = pair_labels or {}
    attack_labels = attack_labels if attack_labels is not None else MIA_METHOD_LABELS
    d["pair_disp"] = d["pair"].map(lambda p: pair_labels.get(p, p))
    d["attack_disp"] = d["attack"].map(lambda a: attack_labels.get(a, a))

    # attack order: by chosen pair's rho, else mean rho across pairs (keep the
    # ordering on the raw attack, then map to display names)
    if sort_by is not None and sort_by in set(d["pair"]):
        raw_order = (d[d["pair"] == sort_by].sort_values("rho")["attack"].tolist())
        raw_order += [a for a in d["attack"].unique() if a not in raw_order]
    else:
        raw_order = (d.groupby("attack")["rho"].mean().sort_values().index.tolist())
    raw_to_disp = dict(zip(d["attack"], d["attack_disp"]))
    disp_order = list(dict.fromkeys(raw_to_disp[a] for a in raw_order))
    d["attack_disp"] = pd.Categorical(d["attack_disp"], categories=disp_order, ordered=True)

    return (
        ggplot(d, aes("rho", "attack_disp", color="pair_disp"))
        + geom_vline(xintercept=0, linetype="dashed", color=GREY, size=0.6)
        + geom_point(size=3, alpha=0.9)
        + scale_x_continuous(limits=(-1, 1))
        + THEME
        + labs(title=title or f"{dataset_name} \u2014 Metric agreement per MIA method",
               x="Spearman \u03c1 (across generators, within attack)", y="",
               caption=caption)
        + guides(color=guide_legend(title="Metric pair", ncol=(LEGEND_NCOL - 2)))
        + theme_fn(legend_position="bottom", legend_direction="horizontal",
                   legend_box="horizontal")
    )


def plot_fs_leakage_heatmap(per_method_fs, *, dataset_name, metric_label="TPR@FPR=0.1",
                            method_order=None, fs_order=None, fs_labels=None,
                            method_labels=None, palette=("#f7fbff", "#08306b"),
                            floor=None, show_values=True, decimals=2):
    """Leakage heatmap: generators (rows) x feature selection (cols), faceted by
    main MIA method. Shows whether a generator's leakage is flat across FS (FS is
    a nuisance lens) or changes (FS interacts with the generator).

    `per_method_fs` is the pre-collapse frame from prepare_fs_leakage (origin,
    attack, feature_selection, value).
    """
    fs_labels = fs_labels or {}
    method_labels = method_labels or MIA_METHOD_LABELS
    d = per_method_fs.copy()
    d["fs_disp"] = d["feature_selection"].map(lambda f: fs_labels.get(f, f))
    d["method_disp"] = d["attack"].map(lambda a: method_labels.get(a, a))

    if fs_order:
        cats = [fs_labels.get(f, f) for f in fs_order if f in set(d["feature_selection"])]
        d["fs_disp"] = pd.Categorical(d["fs_disp"], categories=cats, ordered=True)
    if method_order is not None:
        rows = [m for m in method_order if m in set(d["origin"])]
        d["origin"] = pd.Categorical(d["origin"], categories=rows, ordered=True)

    p = (ggplot(d, aes("fs_disp", "origin", fill="value"))
         + geom_tile(color="white", size=0.3))
    if show_values:
        vmin, vmax = d["value"].min(), d["value"].max()
        thresh = vmin + 0.6 * (vmax - vmin)
        d["lab"] = d["value"].map(lambda v: f"{v:.{decimals}f}")
        d["txt"] = d["value"].map(lambda v: "white" if v >= thresh else "#222222")
        p += geom_text(aes(label="lab", color="txt"), size=CELL_SIZE)
        p += scale_color_identity()
    fill_title = metric_label + (f"\n(floor {floor})" if floor is not None else "")
    p += scale_fill_gradient(low=palette[0], high=palette[1], name=fill_title)
    p += facet_wrap("~method_disp")
    p += labs(title=f"{dataset_name} \u2014 MIA risk by feature selection ({metric_label})",
              x="Feature selection", y="")
    p += THEME
    p += theme_fn(axis_text_x=element_text(rotation=45, ha="right"),
                  axis_text_y=element_text(size=AXIS_Y_SMALL),
                  panel_grid_major=element_blank(),
                  legend_position="right", legend_direction="vertical")
    return p

 

def plot_fs_rank_agreement(mat, *, dataset_name, fs_labels=None, caption=None,
                           fs_order=None):
    """FS x FS heatmap of generator-ranking agreement (Spearman rho).

    `mat` is either an FS x FS DataFrame from fs_rank_agreement (single metric),
    or a long frame from fs_rank_agreement_multi (columns f1, f2, rho, metric) —
    in which case the plot is faceted by metric. High rho = FS does not reorder
    generators (nuisance lens); low rho = FS reorders them. Diverging palette on
    rho in [-1, 1]; lower triangle shown with coefficients.
    """
    fs_labels = fs_labels or {}
    faceted = isinstance(mat, pd.DataFrame) and "metric" in mat.columns

    if faceted:
        long = mat.copy()
        order = fs_order or list(dict.fromkeys(long["f1"]))
    else:
        order = list(mat.columns)
        long = (mat.reset_index().melt(id_vars="index", var_name="f2",
                                       value_name="rho").rename(columns={"index": "f1"}))
        long["metric"] = ""

    order = [f for f in order if f in set(long["f1"])]
    pos = {f: i for i, f in enumerate(order)}
    long["i"], long["j"] = long["f1"].map(pos), long["f2"].map(pos)
    long = long[long["i"] >= long["j"]].copy()             # lower triangle + diagonal
    long["lab"] = long.apply(lambda r: "" if r["i"] == r["j"] else f"{r['rho']:.2f}", axis=1)

    disp = {f: fs_labels.get(f, f) for f in order}
    cats = [disp[f] for f in order]
    long["f1"] = pd.Categorical(long["f1"].map(disp), categories=cats, ordered=True)
    long["f2"] = pd.Categorical(long["f2"].map(disp), categories=cats, ordered=True)

    p = (
        ggplot(long, aes("f2", "f1", fill="rho"))
        + geom_tile(color="white", size=0.4)
        + geom_text(aes(label="lab"), size=ANNOT_SIZE)
        + scale_fill_gradient2(low="#2166ac", mid="white", high="#b2182b",
                               midpoint=0, limits=(-1, 1), name="Rank \u03c1")
        + labs(title=f"{dataset_name} \u2014 Generator ranking-agreement across feature selection",
               x="", y="", caption=caption)
        + THEME
    )
    if faceted:
        p += facet_wrap("~metric")
    p += theme_fn(axis_text_x=element_text(rotation=45, ha="right"),
                  panel_grid_major=element_blank(),
                  legend_position="right", legend_direction="vertical")
    return p


def plot_de_threshold_grid(grid, *, dataset_name, colors, point_size=POINT_SIZE):
    """DE recovery TPR across the ΔVST × FPR-cap grid (rank-stability check).

    `grid` is from prepare_de_threshold_grid. Models on y (horizontal), TPR on x,
    faceted ΔVST (rows) × FPR-cap (cols). up/down-averaged TPR with fold-std bars.
    DP generators get the house DP-emphasis (larger, dark-edged, see-through fill)
    so they read at a glance, consistent with the DE / co-expr / trade-off figures.
    Rank stability = each model holds its position/colour across all panels.
    """
    d = grid.copy()
    d["Privacy"] = privacy_label(d)
    d["_paired_dp"] = _paired_dp_mask(d)
    # order models by mean TPR across the grid (stable y-order across panels)
    order = (d.groupby("origin")["tpr"].mean().sort_values().index.tolist())
    d["origin"] = pd.Categorical(d["origin"], categories=order, ordered=True)

    nondp = d[d["Privacy"] == PRIV_NONDP]
    dp_pair = d[(d["Privacy"] == PRIV_DP) & d["_paired_dp"]]      # DP with a non-DP twin -> see-through
    dp_solo = d[(d["Privacy"] == PRIV_DP) & ~d["_paired_dp"]]     # intrinsically DP (PGG-PGM) -> solid
    priv = d[d["Privacy"] == PRIV_PRIVATIZED]                     # P-NMF -> square, see-through
    p = (
        ggplot(d, aes("tpr", "origin"))
        + geom_errorbarh(aes(xmin="tpr - tpr_std", xmax="tpr + tpr_std",
                             color="origin"), height=0, size=0.6, alpha=0.5)
        + geom_point(aes(fill="origin", shape="Privacy"), data=nondp,
                     color="white", size=point_size, stroke=0.6, alpha=0.95)
        + geom_point(aes(fill="origin", shape="Privacy"), data=dp_solo,
                     color="black", size=point_size + 1.4, stroke=1.0, alpha=0.95)
        + geom_point(aes(fill="origin", shape="Privacy"), data=dp_pair,
                     color="black", size=point_size + 1.4, stroke=1.0, alpha=0.55)
        + geom_point(aes(fill="origin", shape="Privacy"), data=priv,
                     color="black", size=point_size, stroke=1.0, alpha=0.55)
        + facet_grid("lfc_label ~ fpr_label")
        + scale_fill_manual(values=colors)
        + scale_color_manual(values=colors)
        + scale_shape_manual(values=DP_SHAPES)
        + coord_cartesian(xlim=(0, 1.02))
        + labs(title=f"{dataset_name} \u2014 DE recovery sensitivity "
                     f"(\u0394VST \u00d7 FPR cap)",
               x="True positive rate (up/down averaged)", y="")
        + THEME
        + _gen_priv_guides("fill")
        + guides(color=False)
    )
    return p


def plot_de_lfc_sweep(sweep, *, dataset_name, fpr_max, colors, point_size=POINT_SIZE):
    """DE recovery across ΔVST thresholds — one ROC panel per threshold (1 row).

    `sweep` is from prepare_de_lfc_sweep(..., average_direction=True): one point
    per (origin, lfc), up/down averaged. Panels left→right are increasing effect-
    size stringency; rank stability across panels = conclusions robust to ΔVST.
    DP-emphasis (paired see-through / standalone solid) matches the other DE
    figures. Realized FPR is shown on the x-axis (the reviewer's concern), while
    ΔVST — the genuine stringency knob — is the facet.
    """
    d = sweep.copy()
    d["Privacy"] = privacy_label(d)
    d["_paired_dp"] = _paired_dp_mask(d)
    nondp = d[d["Privacy"] == PRIV_NONDP]
    dp_pair = d[(d["Privacy"] == PRIV_DP) & d["_paired_dp"]]
    dp_solo = d[(d["Privacy"] == PRIV_DP) & ~d["_paired_dp"]]
    priv = d[d["Privacy"] == PRIV_PRIVATIZED]

    return (
        ggplot(d, aes("fpr_mean", "tpr_mean"))
        + geom_hline(yintercept=1, linetype="dashed", color="gray", alpha=0.5)
        + geom_vline(xintercept=0, linetype="dashed", color="gray", alpha=0.5)
        + geom_errorbar(aes(ymin="tpr_mean - tpr_std", ymax="tpr_mean + tpr_std",
                            color="origin"), width=0.002, alpha=0.6, size=0.8)
        + geom_errorbarh(aes(xmin="fpr_mean - fpr_std", xmax="fpr_mean + fpr_std",
                             color="origin"), height=0.02, alpha=0.6, size=0.8)
        + geom_point(aes(fill="origin", shape="Privacy"), data=nondp,
                     color="white", size=point_size, stroke=0.6, alpha=0.95)
        + geom_point(aes(fill="origin", shape="Privacy"), data=dp_solo,
                     color="black", size=point_size + 1.4, stroke=1.0, alpha=0.95)
        + geom_point(aes(fill="origin", shape="Privacy"), data=dp_pair,
                     color="black", size=point_size + 1.4, stroke=1.0, alpha=0.55)
        + geom_point(aes(fill="origin", shape="Privacy"), data=priv,
                     color="black", size=point_size, stroke=1.0, alpha=0.55)
        + facet_wrap("~lfc_label", nrow=1)
        + scale_fill_manual(values=colors)
        + scale_color_manual(values=colors)
        + scale_shape_manual(values=DP_SHAPES)
        + coord_cartesian(xlim=(-0.001, fpr_max + 0.001), ylim=(0, 1.05))
        + labs(title=f"{dataset_name} \u2014 DE recovery across \u0394VST "
                     f"(up/down averaged, FPR\u2264{fpr_max})",
               x="False positive rate", y="True positive rate")
        + THEME
        + _gen_priv_guides("fill")
        + guides(color=False)
    )


def plot_tradeoff_grouped(tradeoff, *, dataset_name, group_colors,
                          q_threshold=0.05, error_bars=True, ncol=3,
                          privacy_floor=None,
                          anchor_label="Privacy (worst-case AUC-ROC)",
                          title=None, caption=None, point_size=3,
                          annotate=True, annot_size=None, label_models=(),
                          label_angle=0, label_nudge=None):
    """Anchored trade-off panels coloured by model GROUP (baseline / submitted-X /
    submitted-Y), shaped by DP status (triangle = DP), with per-point text labels
    for model identity.

    Unlike plot_tradeoff (which colours by individual model), this encodes the
    group in colour so baseline vs submitted cohorts are visually separated, keeps
    shape = DP status, and annotates each point with its model name so specific
    models (e.g. the best performer) remain identifiable. Expects `tradeoff` to be
    the prepare_tradeoff frame with an added `group` column. `group_colors` is a
    {group: hex} dict.
    """
    d = tradeoff.copy()
    if "group" not in d.columns:
        raise ValueError("tradeoff frame needs a 'group' column "
                         "(e.g. 'Baseline' / 'Submitted').")
    d["Privacy"] = privacy_label(d)
    has_std = "x_std" in d.columns and "y_std" in d.columns
    annot_size = annot_size if annot_size is not None else ANNOT_SIZE

    rng = (d.groupby("panel")
             .agg(xmin=("x", "min"), xmax=("x", "max"),
                  ymin=("y", "min"), ymax=("y", "max"),
                  rho=("rho", "first"), q=("q", "first"), stars=("stars", "first"))
             .reset_index())
    rng["ax"] = rng["xmin"] + 0.02 * (rng["xmax"] - rng["xmin"])
    rng["ay"] = rng["ymax"] - 0.02 * (rng["ymax"] - rng["ymin"])
    rng["lab"] = rng.apply(
        lambda r: (f"\u03c1={r['rho']:.2f} {r['stars']}"
                   + (f"\nq={r['q']:.3f}" if pd.notna(r['q']) else "")), axis=1)

    p = ggplot(d, aes("x", "y"))
    if privacy_floor is not None:
        p += geom_hline(yintercept=privacy_floor, linetype="dashed",
                        color=LIGHT_GREY, size=0.6)
    if error_bars and has_std:
        p += geom_errorbar(aes(ymin="y - y_std", ymax="y + y_std", color="group"),
                           width=0, size=0.5, alpha=0.35)
        p += geom_errorbarh(aes(xmin="x - x_std", xmax="x + x_std", color="group"),
                            height=0, size=0.5, alpha=0.35)

    sig = d[d["q"] < q_threshold]
    if len(sig):
        p += geom_smooth(data=sig, method="loess", span=1.0, se=False,
                         color=GREY, size=0.7, linetype="solid")

    # DP points slightly larger with dark edge so DP is readable on top of group colour
    dp = d[d["is_DP"]]
    p += geom_point(aes(shape="Privacy"), data=dp, color="black",
                    size=point_size + 2.0, stroke=0.9)
    p += geom_point(aes(color="group", shape="Privacy"), data=d[~d["is_DP"]],
                    size=point_size, alpha=0.9)
    p += geom_point(aes(color="group", shape="Privacy"), data=dp,
                    size=point_size + 1.0, alpha=0.95)

    if annotate and len(label_models):
        # label only the models explicitly listed in label_models (manual
        # selection), keeping their full names.
        lab_df = d[d["origin"].isin(set(label_models))].copy()
        if len(lab_df):
            # manual repel: shift each label by a per-model (dx, dy) offset from
            # label_nudge (data units). No adjustText dependency. Models absent
            # from the dict get a small default rightward nudge.
            nudge = label_nudge or {}
            xr = d["x"].max() - d["x"].min()
            yr = d["y"].max() - d["y"].min()
            dfltx, dflty = 0.015 * xr, 0.02 * yr
            lab_df["lx"] = lab_df.apply(
                lambda r: r["x"] + nudge.get(r["origin"], (dfltx, dflty))[0], axis=1)
            lab_df["ly"] = lab_df.apply(
                lambda r: r["y"] + nudge.get(r["origin"], (dfltx, dflty))[1], axis=1)
            p += geom_text(aes(x="lx", y="ly", label="origin", color="group"),
                           data=lab_df, size=annot_size, va="bottom", ha="left",
                           angle=label_angle, show_legend=False)

    p += geom_label(aes(x="ax", y="ay", label="lab"), data=rng, inherit_aes=False,
                    ha="left", va="top", size=ANNOT_SIZE, color="#333333",
                    fill="white", label_size=0, alpha=0.75)
    p += facet_wrap("~panel", scales="free_x", ncol=ncol)
    p += scale_color_manual(values=group_colors)
    p += scale_shape_manual(values=DP_SHAPES)
    p += labs(title=title or f"{dataset_name} \u2014 trade-offs by model group",
              x="Comparison metric", y=anchor_label, caption=caption)
    p += THEME
    p += theme_fn(legend_position="bottom",
                  panel_grid_major=element_line(color=GRID_GREY, size=0.5),
                  panel_grid_minor=element_line(color=GRID_GREY, size=0.25))
    p += guides(color=guide_legend(title="Model group"),
                shape=guide_legend(title="Privacy"))
    return p