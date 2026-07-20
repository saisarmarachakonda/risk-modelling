"""
reporting/figure_generator.py
All Matplotlib figure generators for the Risk Modelling Pipeline.

Rules
-----
* Never call plt.show() — always return a base64 PNG string.
* Always close figures after encoding (avoids memory leaks).
* Use tight_layout() and the professional palette defined below.
* DPI = 150 for report-quality output.
"""
import base64
import io
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

warnings.filterwarnings("ignore")

# ── Professional colour palette ──────────────────────────────────────────────
PALETTE = [
    "#1a3a5c",  # navy
    "#0ea5e9",  # sky blue
    "#059669",  # emerald
    "#d97706",  # amber
    "#7c3aed",  # violet
    "#dc2626",  # rose
    "#0891b2",  # teal
    "#ea580c",  # orange
    "#4338ca",  # indigo
    "#be185d",  # pink
    "#065f46",  # dark green
    "#92400e",  # brown
]

TREE_COLOR = "#d97706"   # amber  — tree-based methods
LR_COLOR   = "#0ea5e9"   # sky    — logistic regression methods
FILTER_COLOR = "#7c3aed" # violet — filter methods

_STYLE = {
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.color":         "#cbd5e1",
    "axes.labelcolor":    "#0f172a",
    "xtick.color":        "#475569",
    "ytick.color":        "#475569",
    "font.family":        "sans-serif",
    "figure.facecolor":   "white",
    "axes.facecolor":     "#f8fafc",
}


# ── Core utilities ────────────────────────────────────────────────────────────

def _apply_style():
    plt.rcParams.update(_STYLE)


def fig_to_base64(fig: plt.Figure, dpi: int = 150) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


# ── Distribution plots ────────────────────────────────────────────────────────

def histogram(
    values: np.ndarray,
    col_name: str,
    bins: int = 40,
    color: str = PALETTE[0],
) -> str:
    """Single-column histogram with KDE overlay."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = values[~np.isnan(values)]
    ax.hist(vals, bins=bins, color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(col_name, fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"Distribution: {col_name}", fontsize=13, fontweight="bold", color="#0f172a")
    plt.tight_layout()
    return fig_to_base64(fig)


def distribution_grid(
    data: dict,
    cols: List[str],
    title: str = "Numeric Feature Distributions",
    max_plots: int = 12,
) -> str:
    """Grid of histograms for up to max_plots numeric columns."""
    _apply_style()
    cols    = cols[:max_plots]
    n       = len(cols)
    ncols   = min(4, n)
    nrows   = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes    = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(cols):
        ax  = axes[i]
        arr = np.array(data.get(col, []), dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr):
            ax.hist(arr, bins=30, color=PALETTE[i % len(PALETTE)],
                    alpha=0.75, edgecolor="white", linewidth=0.3)
        ax.set_title(col[:22], fontsize=9, fontweight="bold")
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Freq", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig_to_base64(fig)


def boxplot_grid(
    data: dict,
    cols: List[str],
    title: str = "Box Plots",
    max_plots: int = 12,
) -> str:
    """Grid of box plots."""
    _apply_style()
    cols  = cols[:max_plots]
    n     = len(cols)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
    axes  = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(cols):
        ax  = axes[i]
        arr = np.array(data.get(col, []), dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr):
            bp = ax.boxplot(arr, patch_artist=True, widths=0.5,
                            medianprops=dict(color="white", linewidth=2))
            bp["boxes"][0].set_facecolor(PALETTE[i % len(PALETTE)])
            bp["boxes"][0].set_alpha(0.75)
        ax.set_title(col[:22], fontsize=9, fontweight="bold")
        ax.set_xlabel("", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig_to_base64(fig)


# ── Correlation / heatmap ─────────────────────────────────────────────────────

def correlation_heatmap(
    corr_matrix: np.ndarray,
    labels: List[str],
    title: str = "Correlation Heatmap",
    max_cols: int = 40,
) -> str:
    """Heatmap of a correlation matrix (truncated to max_cols)."""
    _apply_style()
    n = min(len(labels), max_cols)
    corr  = corr_matrix[:n, :n]
    lbls  = [l[:18] for l in labels[:n]]

    cmap  = LinearSegmentedColormap.from_list(
        "risk_div", ["#1d4ed8", "#f8fafc", "#dc2626"]
    )
    size  = max(8, n * 0.4)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(lbls, rotation=45, ha="right", fontsize=max(5, 10 - n // 10))
    ax.set_yticklabels(lbls, fontsize=max(5, 10 - n // 10))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    return fig_to_base64(fig)


def missing_bar(
    columns: List[str],
    missing_pct: List[float],
    title: str = "Missing Value % by Column",
    top_n: int = 40,
) -> str:
    """Horizontal bar chart of missing value percentages."""
    _apply_style()
    pairs = sorted(zip(missing_pct, columns), reverse=True)[:top_n]
    pcts, cols = zip(*pairs) if pairs else ([], [])
    pcts = list(pcts)
    cols = [c[:35] for c in cols]

    fig, ax = plt.subplots(figsize=(10, max(4, len(cols) * 0.32)))
    colors = ["#dc2626" if p > 30 else "#d97706" if p > 10 else "#059669" for p in pcts]
    bars = ax.barh(cols, pcts, color=colors, alpha=0.82, edgecolor="white")
    ax.axvline(5,  color="#d97706", linestyle="--", linewidth=1.2, alpha=0.7, label="5% threshold")
    ax.axvline(30, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.7, label="30% threshold")
    ax.set_xlabel("Missing %", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    for bar, pct in zip(bars, pcts):
        ax.text(pct + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    return fig_to_base64(fig)


# ── Target distribution ───────────────────────────────────────────────────────

def target_bar(
    labels: List[str],
    counts: List[int],
    pcts: List[float],
    title: str = "Target Distribution",
) -> str:
    """Bar chart showing class distribution."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Bar chart
    ax = axes[0]
    bars = ax.bar([str(l) for l in labels], counts,
                  color=[PALETTE[i % len(PALETTE)] for i in range(len(labels))],
                  alpha=0.82, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Class Counts", fontsize=12, fontweight="bold")
    for bar, cnt, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                f"{cnt:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Pie chart
    ax2 = axes[1]
    colors_pie = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    wedges, texts, autotexts = ax2.pie(
        counts, labels=[str(l) for l in labels], autopct="%1.1f%%",
        colors=colors_pie, startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color("white")
    ax2.set_title("Class Proportion", fontsize=12, fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig_to_base64(fig)


# ── Feature importance / selection ────────────────────────────────────────────

def feature_importance_bar(
    features: List[str],
    importances: List[float],
    title: str = "Feature Importance",
    color: str = PALETTE[0],
    top_n: int = 30,
    xlabel: str = "Importance Score",
) -> str:
    """Horizontal bar chart of feature importances."""
    _apply_style()
    pairs = sorted(zip(importances, features), reverse=True)[:top_n]
    if not pairs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig_to_base64(fig)

    vals, feats = zip(*pairs)
    vals  = list(vals)
    feats = [f[:40] for f in feats]

    fig, ax = plt.subplots(figsize=(10, max(5, len(feats) * 0.32)))
    norm    = plt.Normalize(min(vals), max(vals))
    cmap    = plt.get_cmap("Blues")
    colors  = [cmap(norm(v)) for v in vals]
    bars    = ax.barh(feats[::-1], vals[::-1], color=colors[::-1],
                      alpha=0.88, edgecolor="white")
    ax.axvline(0, color="#475569", linewidth=0.8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    for bar, val in zip(bars, vals[::-1]):
        ax.text(val + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    return fig_to_base64(fig)


def method_comparison_heatmap(
    features: List[str],
    methods: List[str],
    ranks: np.ndarray,
    title: str = "Feature Selection: Method Comparison Heatmap",
) -> str:
    """
    Heatmap showing rank of each feature (rows) per method (cols).
    Lower rank (darker) = more important.
    """
    _apply_style()
    n_feat = min(len(features), 50)
    n_meth = len(methods)
    feats  = [f[:35] for f in features[:n_feat]]
    data   = ranks[:n_feat, :]

    cmap = LinearSegmentedColormap.from_list(
        "rank_map", ["#1a3a5c", "#0ea5e9", "#f8fafc"]
    )
    fig, ax = plt.subplots(figsize=(max(10, n_meth * 2), max(8, n_feat * 0.35)))
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label="Rank (lower = more important)", fraction=0.02, pad=0.02)

    ax.set_xticks(range(n_meth))
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9, fontweight="bold")
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    return fig_to_base64(fig)


def tree_vs_lr_scatter(
    features: List[str],
    tree_scores: List[float],
    lr_scores: List[float],
    title: str = "Tree-Based vs Logistic Regression Importance",
    top_n_label: int = 20,
) -> str:
    """
    Scatter plot comparing tree-based vs LR-based importance scores.
    Features in top-N of both methods are labelled.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    tree = np.array(tree_scores)
    lr   = np.array(lr_scores)

    # Normalise to [0,1]
    def norm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else x

    tree_n = norm(tree)
    lr_n   = norm(lr)

    # Classify each feature
    consensus   = (tree_n + lr_n) / 2
    top_both    = (tree_n > 0.5) & (lr_n > 0.5)
    top_tree    = (tree_n > 0.5) & ~top_both
    top_lr      = (lr_n   > 0.5) & ~top_both
    neutral     = ~top_both & ~top_tree & ~top_lr

    ax.scatter(tree_n[neutral],   lr_n[neutral],   s=40,  alpha=0.3, color="#94a3b8",    label="Neutral")
    ax.scatter(tree_n[top_tree],  lr_n[top_tree],  s=80,  alpha=0.7, color=TREE_COLOR,   label="Tree-dominant")
    ax.scatter(tree_n[top_lr],    lr_n[top_lr],    s=80,  alpha=0.7, color=LR_COLOR,     label="LR-dominant")
    ax.scatter(tree_n[top_both],  lr_n[top_both],  s=120, alpha=0.9, color="#059669",    label="Consensus (top both)")

    # Label top-N consensus features
    feat_arr = np.array(features)
    top_idx  = np.argsort(-consensus)[:top_n_label]
    for idx in top_idx:
        ax.annotate(
            feat_arr[idx][:20],
            (tree_n[idx], lr_n[idx]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
            color="#0f172a",
        )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "--", color="#475569", linewidth=1, alpha=0.5, label="Agreement diagonal")

    ax.set_xlabel("Tree-Based Importance (normalised)", fontsize=11)
    ax.set_ylabel("LR-Based Importance (normalised)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    return fig_to_base64(fig)


def consensus_rank_bar(
    features: List[str],
    consensus_scores: List[float],
    method_counts: List[int],
    title: str = "Consensus Feature Ranking",
    top_n: int = 40,
) -> str:
    """
    Bar chart of consensus scores with bar colour encoding how many
    methods agreed on this feature.
    """
    _apply_style()
    pairs = sorted(zip(consensus_scores, features, method_counts), reverse=True)[:top_n]
    scores, feats, mcounts = zip(*pairs) if pairs else ([], [], [])
    feats  = [f[:40] for f in feats]
    max_m  = max(mcounts) if mcounts else 1

    cmap   = plt.get_cmap("RdYlGn")
    colors = [cmap(c / max_m) for c in mcounts]

    fig, ax = plt.subplots(figsize=(11, max(5, len(feats) * 0.32)))
    bars    = ax.barh(feats[::-1], list(scores)[::-1], color=colors[::-1],
                      alpha=0.85, edgecolor="white")

    # Colour legend
    patches = [
        mpatches.Patch(color=cmap(i / max_m), label=f"Selected by {i} method(s)")
        for i in range(1, max_m + 1)
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    ax.set_xlabel("Consensus Score (normalised)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig_to_base64(fig)


def iv_chart(
    features: List[str],
    iv_values: List[float],
    title: str = "Information Value (IV) by Feature",
    top_n: int = 40,
) -> str:
    """IV bar chart with predictive power threshold lines."""
    _apply_style()
    pairs = sorted(zip(iv_values, features), reverse=True)[:top_n]
    vals, feats = zip(*pairs) if pairs else ([], [])
    feats = [f[:40] for f in feats]

    fig, ax = plt.subplots(figsize=(10, max(5, len(feats) * 0.32)))
    colors  = [
        "#059669" if v > 0.3
        else "#0ea5e9" if v > 0.1
        else "#d97706" if v > 0.02
        else "#dc2626"
        for v in vals
    ]
    ax.barh(list(feats)[::-1], list(vals)[::-1], color=colors[::-1],
            alpha=0.83, edgecolor="white")

    for threshold, label, color in [
        (0.02, "Weak (0.02)", "#d97706"),
        (0.1,  "Medium (0.10)", "#0ea5e9"),
        (0.3,  "Strong (0.30)", "#059669"),
    ]:
        ax.axvline(threshold, color=color, linestyle="--", linewidth=1.4,
                   alpha=0.8, label=label)

    ax.set_xlabel("Information Value", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig_to_base64(fig)


def stability_heatmap(
    features: List[str],
    method_selected: Dict[str, List[bool]],
    title: str = "Feature Selection Stability Across Methods",
) -> str:
    """
    Binary heatmap: was feature selected (1) or not (0) per method.
    """
    _apply_style()
    methods = list(method_selected.keys())
    feats   = features[:60]  # show at most 60
    matrix  = np.array([
        [1 if method_selected[m][i] else 0 for m in methods]
        for i in range(len(feats))
    ])

    cmap = LinearSegmentedColormap.from_list("sel", ["#f8fafc", "#1a3a5c"])
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.5), max(8, len(feats) * 0.3)))
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([f[:35] for f in feats], fontsize=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    # Annotate cells
    for r in range(len(feats)):
        for c in range(len(methods)):
            ax.text(c, r, "✓" if matrix[r, c] else "", ha="center", va="center",
                    fontsize=8, color="white" if matrix[r, c] else "#94a3b8")

    plt.tight_layout()
    return fig_to_base64(fig)


# ── Model evaluation ──────────────────────────────────────────────────────────

def roc_curves(
    models_data: List[Dict],
    title: str = "ROC Curves — Model Comparison",
) -> str:
    """
    Multi-model ROC curve plot.
    models_data: list of {name, fpr, tpr, auc}
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1.2, label="Random (AUC=0.50)")

    for i, md in enumerate(models_data):
        ax.plot(
            md["fpr"], md["tpr"],
            color=PALETTE[i % len(PALETTE)],
            linewidth=2.0,
            label=f"{md['name']} (AUC={md['auc']:.4f})",
        )

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    plt.tight_layout()
    return fig_to_base64(fig)


def pr_curves(
    models_data: List[Dict],
    title: str = "Precision-Recall Curves — Model Comparison",
) -> str:
    """
    models_data: list of {name, precision, recall, auc}
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, md in enumerate(models_data):
        ax.plot(
            md["recall"], md["precision"],
            color=PALETTE[i % len(PALETTE)],
            linewidth=2.0,
            label=f"{md['name']} (PR-AUC={md['auc']:.4f})",
        )

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    plt.tight_layout()
    return fig_to_base64(fig)


def confusion_matrix_plot(
    cm: np.ndarray,
    labels: List[str],
    title: str = "Confusion Matrix",
) -> str:
    """Annotated confusion matrix heatmap."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = LinearSegmentedColormap.from_list("cm_map", ["#f8fafc", "#1a3a5c"])
    im = ax.imshow(cm, cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    thresh = cm.max() / 2
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, f"{cm[r, c]:,}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if cm[r, c] > thresh else "#0f172a")
    plt.tight_layout()
    return fig_to_base64(fig)


def model_metric_bar(
    model_names: List[str],
    metric_values: List[float],
    metric_name: str = "ROC AUC",
    title: str = "Model Comparison",
) -> str:
    """Bar chart comparing models on a single metric."""
    _apply_style()
    pairs  = sorted(zip(metric_values, model_names), reverse=True)
    vals, names = zip(*pairs) if pairs else ([], [])

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    bars    = ax.bar(list(names), list(vals), color=colors, alpha=0.83, edgecolor="white",
                     width=0.55)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(max(0, min(vals) - 0.05), min(1.05, max(vals) + 0.08))
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig_to_base64(fig)


def ks_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    title: str = "KS Statistic Curve",
) -> str:
    """KS statistic curve (max separation between TPR and FPR curves)."""
    _apply_style()
    threshold = np.linspace(0, 1, len(fpr))
    ks_vals   = tpr - fpr
    ks_max    = ks_vals.max()
    ks_idx    = ks_vals.argmax()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(threshold, tpr, color=PALETTE[0], linewidth=2, label="TPR (Sensitivity)")
    ax.plot(threshold, fpr, color=PALETTE[3], linewidth=2, label="FPR (1-Specificity)")
    ax.fill_between(threshold, fpr, tpr, alpha=0.12, color=PALETTE[1])
    ax.axvline(threshold[ks_idx], color="#dc2626", linestyle="--", linewidth=1.5,
               label=f"KS = {ks_max:.4f}")
    ax.set_xlabel("Threshold / Percentile", fontsize=11)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig_to_base64(fig)


def learning_curve_plot(
    train_sizes: np.ndarray,
    train_scores_mean: np.ndarray,
    train_scores_std: np.ndarray,
    val_scores_mean: np.ndarray,
    val_scores_std: np.ndarray,
    metric: str = "ROC AUC",
    title: str = "Learning Curve",
) -> str:
    """Learning curve with confidence bands."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(train_sizes,
                    train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std,
                    alpha=0.12, color=PALETTE[0])
    ax.fill_between(train_sizes,
                    val_scores_mean - val_scores_std,
                    val_scores_mean + val_scores_std,
                    alpha=0.12, color=PALETTE[1])

    ax.plot(train_sizes, train_scores_mean, "o-", color=PALETTE[0],
            linewidth=2, markersize=5, label="Training score")
    ax.plot(train_sizes, val_scores_mean,   "o-", color=PALETTE[1],
            linewidth=2, markersize=5, label="Validation score")

    ax.set_xlabel("Training examples", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig_to_base64(fig)


def calibration_curve_plot(
    models_data: List[Dict],
    title: str = "Calibration Curves (Reliability Diagram)",
) -> str:
    """
    models_data: list of {name, fraction_of_positives, mean_predicted_value}
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1.2, label="Perfect calibration")

    for i, md in enumerate(models_data):
        ax.plot(
            md["mean_predicted_value"],
            md["fraction_of_positives"],
            "s-",
            color=PALETTE[i % len(PALETTE)],
            linewidth=2,
            markersize=6,
            label=md["name"],
        )

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    return fig_to_base64(fig)


def gain_lift_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Gain & Lift Curves",
) -> str:
    """Cumulative gain and lift charts."""
    _apply_style()
    order    = np.argsort(-y_prob)
    y_sorted = y_true[order]
    total    = y_true.sum()
    n        = len(y_true)

    pct_pop  = np.arange(1, n + 1) / n
    cum_gain = np.cumsum(y_sorted) / total
    lift     = cum_gain / pct_pop

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(pct_pop, cum_gain, color=PALETTE[0], linewidth=2.5, label="Model")
    ax1.plot([0, 1], [0, 1],   "--", color="#94a3b8", linewidth=1.2, label="Random")
    ax1.set_xlabel("% Population", fontsize=11)
    ax1.set_ylabel("% Positive Captured", fontsize=11)
    ax1.set_title("Cumulative Gain Curve", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)

    ax2.plot(pct_pop, lift, color=PALETTE[1], linewidth=2.5, label="Model Lift")
    ax2.axhline(1, color="#94a3b8", linestyle="--", linewidth=1.2, label="Baseline (1.0)")
    ax2.set_xlabel("% Population", fontsize=11)
    ax2.set_ylabel("Lift", fontsize=11)
    ax2.set_title("Lift Curve", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig_to_base64(fig)


def shap_summary_bar(
    feature_names: List[str],
    shap_mean_abs: List[float],
    title: str = "SHAP Feature Importance (Mean |SHAP value|)",
    top_n: int = 30,
) -> str:
    """Horizontal bar chart of mean absolute SHAP values."""
    return feature_importance_bar(
        feature_names, shap_mean_abs,
        title=title, color="#7c3aed", top_n=top_n,
        xlabel="Mean |SHAP value|",
    )
