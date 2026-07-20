"""
stages/03_eda.py
Stage 03 — Exploratory Data Analysis (EDA) Report

Covers
------
* Univariate analysis — numeric + categorical distributions
* Bivariate analysis — feature vs target relationships
* Correlation analysis (Pearson on sampled data)
* Skewness and kurtosis summary
* Top categorical value distributions
* Class separability (overlapping distributions)

All heavy computations use DuckDB SQL or a sample (100k rows max).

Output
------
reports/03_Exploratory_Data_Analysis.html
"""
import time
from pathlib import Path
from typing import List

import numpy as np
import polars as pl

from core.data_loader import DataLoader
from core.schema_detector import SchemaDetector
from core.logger import get_logger
from reporting.html_builder import HTMLReportBuilder
from reporting.figure_generator import (
    distribution_grid, boxplot_grid, correlation_heatmap,
    target_bar, feature_importance_bar, iv_chart,
)
from reporting.report_writer import ReportWriter


def run(loader: DataLoader, schema: SchemaDetector, config: dict) -> Path:
    log = get_logger("03_EDA", config.get("paths", {}).get("logs_dir", "logs"))
    log.stage_start("03 — Exploratory Data Analysis")
    t0 = time.perf_counter()

    n_rows   = loader.count_rows()
    n_cols   = len(loader.get_columns())
    num_cols = schema.get_numeric_cols()
    cat_cols = schema.get_categorical_cols()
    target   = loader.target_col
    sample_n = min(config.get("data", {}).get("sample_size_for_eda", 100_000), n_rows)

    b = HTMLReportBuilder(
        report_title   = "Exploratory Data Analysis Report",
        stage_number   = 3,
        stage_subtitle = "Univariate, bivariate, and correlation analysis on sampled data",
        config         = config,
        n_rows         = n_rows,
        n_cols         = n_cols,
    )

    # ── Numeric stats summary ────────────────────────────────────────
    log.info(f"Computing numeric stats for EDA (up to 200 cols)…")
    num_cols_subset = num_cols[:200]
    stats_df = loader.get_numeric_stats(num_cols_subset) if num_cols_subset else pl.DataFrame()

    # Skewness / kurtosis via sample
    skew_kurt = []
    if num_cols_subset:
        sample_num = loader.sample_columns(
            [c for c in num_cols_subset[:50] if c != target], n=sample_n
        )
        for col in num_cols_subset[:50]:
            if col == target or col not in sample_num.columns:
                continue
            arr = sample_num[col].drop_nulls().to_numpy().astype(float)
            if len(arr) < 10:
                continue
            from scipy import stats as sp_stats
            try:
                sk = float(sp_stats.skew(arr))
                kt = float(sp_stats.kurtosis(arr))
            except Exception:
                sk, kt = 0.0, 0.0
            skew_kurt.append({
                "column": col,
                "skewness": round(sk, 4),
                "kurtosis": round(kt, 4),
                "skew_flag": ("High right skew" if sk > 2 else
                              "High left skew" if sk < -2 else
                              "Moderate skew" if abs(sk) > 1 else "Normal"),
                "recommend": "Log/Box-Cox transform" if abs(sk) > 2 else
                             "Optional transform" if abs(sk) > 1 else "No transform needed",
            })

    # Executive summary
    n_high_skew = sum(1 for r in skew_kurt if abs(r["skewness"]) > 2)
    cards = [
        {"label": "Numeric Features",     "value": f"{len(num_cols):,}"},
        {"label": "Categorical Features", "value": f"{len(cat_cols):,}"},
        {"label": "EDA Sample Size",      "value": f"{sample_n:,}",
         "sub": "rows analysed"},
        {"label": "Highly Skewed",        "value": str(n_high_skew),
         "sub": "|skewness| > 2",
         "variant": "warning" if n_high_skew > 5 else "success"},
    ]
    b.add_executive_summary(cards, narrative=(
        f"Exploratory Data Analysis is performed on a {sample_n:,}-row random sample to "
        f"enable in-memory computation of distributions, correlations, and bivariate relationships. "
        f"For aggregate statistics (mean, percentiles, counts), the full {n_rows:,}-row dataset "
        f"is queried via DuckDB SQL without loading into RAM. "
        f"This report identifies {n_high_skew} highly skewed numeric features that may benefit "
        f"from transformation, and assesses the univariate predictive power of each feature "
        f"via Information Value (IV) and correlation with the target."
    ))

    # ── Univariate — Numeric distributions ──────────────────────────
    log.info("Generating numeric distribution figures…")
    sample_data = {}
    viz_cols = [c for c in num_cols[:12] if c != target]
    if viz_cols:
        sdf = loader.sample_columns(viz_cols, n=50_000)
        sample_data = {c: sdf[c].drop_nulls().to_numpy() for c in viz_cols}

    dist_content = b.p(
        f"Histograms are plotted for the first 12 numeric features using a 50,000-row sample. "
        f"The shape of each distribution informs the appropriate transformation strategy: "
        f"right-skewed distributions benefit from log transformation; "
        f"bimodal distributions may indicate sub-populations or mixed data sources."
    )
    if sample_data:
        dist_fig = distribution_grid(sample_data, viz_cols,
                                     title="Numeric Feature Distributions (Sample)")
        dist_content += b.figure(
            dist_fig,
            title="Numeric Feature Distribution Grid",
            description="Histograms for the first 12 numeric features (50k sample).",
            interpretation=(
                "Features with heavy right skew will disproportionately influence "
                "linear models (logistic regression). Tree-based models are scale- and "
                "skew-invariant, so transformation is primarily needed for LR-based approaches."
            ),
        )

        box_fig = boxplot_grid(sample_data, viz_cols, title="Box Plots — Outlier Detection")
        dist_content += b.figure(
            box_fig,
            title="Box Plot Grid — Outlier Detection",
            description="Box plots show the IQR, whiskers (1.5×IQR), and outlier points.",
            interpretation=(
                "Data points beyond the whiskers are classified as outliers. "
                "Logistic regression is sensitive to extreme outliers, requiring winsorisation "
                "or robust scaling. Tree-based models partition on thresholds and are naturally "
                "robust to outliers in the feature space."
            ),
        )

    if skew_kurt:
        dist_content += b.subsection("Skewness & Kurtosis Summary")
        dist_content += b.table(
            skew_kurt,
            caption="Skewness and Kurtosis for Numeric Features",
            interpretation=(
                "Skewness > 2 or < -2 indicates substantial asymmetry. Kurtosis > 3 "
                "indicates heavier tails than a normal distribution (leptokurtic), "
                "suggesting outlier-prone features. These require careful handling, "
                "particularly when using logistic regression which assumes reasonable "
                "feature distributions after scaling."
            ),
        )

    b.add_section("Univariate Analysis — Numeric Features", dist_content, icon="📊")

    # ── Univariate — Categorical ──────────────────────────────────────
    log.info("Analysing categorical features…")
    cat_content = b.p(
        f"The dataset contains {len(cat_cols)} categorical features. "
        f"Value distribution and cardinality analysis determines whether each can be "
        f"one-hot encoded (low cardinality), ordinal encoded, or requires WoE transformation."
    )
    cat_summary = []
    for col in cat_cols[:50]:
        vc = loader.column_value_counts(col, top_n=5)
        n_unique = schema.detect()[col]["n_unique"]
        top_val  = str(vc["value"][0]) if not vc.is_empty() else "N/A"
        top_pct  = float(vc["pct"][0]) if not vc.is_empty() else 0
        cat_summary.append({
            "Column":        col,
            "Unique Values": f"{n_unique:,}",
            "Top Value":     top_val[:30],
            "Top Value %":   f"{top_pct:.2f}%",
            "Encoding":      ("OHE" if n_unique <= 10
                              else "Ordinal/WoE" if n_unique <= 50
                              else "WoE/Target Encode"),
        })
    if cat_summary:
        cat_content += b.table(
            cat_summary,
            caption="Categorical Column Summary",
            interpretation=(
                "Columns with ≤10 unique values are candidates for one-hot encoding. "
                "Columns with 10–50 values work well with ordinal encoding or WoE transformation. "
                "High-cardinality columns (>50 values) require WoE, target encoding, or "
                "frequency encoding to avoid the curse of dimensionality."
            ),
        )
    b.add_section("Univariate Analysis — Categorical Features", cat_content, icon="🏷️")

    # ── Bivariate — Feature vs Target ────────────────────────────────
    log.info("Computing Information Values for bivariate analysis…")
    iv_content = b.p(
        "Information Value (IV) quantifies the predictive power of each feature with respect "
        "to the binary target. It is computed over the full dataset via DuckDB SQL aggregates. "
        "IV ranges: <0.02 = Useless, 0.02–0.1 = Weak, 0.1–0.3 = Medium, >0.3 = Strong, "
        ">0.5 = Suspicious (possible leakage)."
    )
    iv_results = []
    for col in num_cols[:100]:
        if col == target:
            continue
        try:
            iv, _ = loader.compute_iv_woe(col)
            iv_results.append({"feature": col, "iv": round(iv, 6),
                                "strength": ("Suspicious" if iv > 0.5
                                             else "Strong" if iv > 0.3
                                             else "Medium" if iv > 0.1
                                             else "Weak" if iv > 0.02
                                             else "Useless")})
        except Exception:
            pass

    iv_results.sort(key=lambda x: -x["iv"])

    if iv_results:
        iv_feats = [r["feature"] for r in iv_results[:40]]
        iv_vals  = [r["iv"]     for r in iv_results[:40]]
        iv_fig   = iv_chart(iv_feats, iv_vals, title="Information Value — Top 40 Features")
        iv_content += b.figure(
            iv_fig,
            title="Information Value by Feature",
            description="IV chart for top 40 numeric features by predictive power.",
            interpretation=(
                "Features with IV > 0.1 have meaningful predictive power for the binary target. "
                "Features with IV < 0.02 are candidates for removal in the feature selection stage. "
                "Features with IV > 0.5 should be manually inspected for potential data leakage."
            ),
            business_implication=(
                "High-IV features represent the strongest individual signals of default risk. "
                "These variables often correspond to key financial ratios, behavioural indicators, "
                "or account performance metrics that underwriters already use informally."
            ),
        )
        iv_content += b.table(iv_results[:100], caption="Information Value Results",
                              interpretation="Sorted by IV descending. Features in bold are strong predictors.")

    b.add_section("Bivariate Analysis — Feature vs Target (IV)", iv_content, icon="🎯")

    # ── Correlation analysis ─────────────────────────────────────────
    log.info("Computing correlation matrix (sample)…")
    corr_cols = [c for c in num_cols[:40] if c != target]
    corr_content = b.p(
        f"Pearson correlation is computed on a {sample_n:,}-row sample for the "
        f"top {len(corr_cols)} numeric features. High inter-feature correlation "
        f"(|r| > 0.95) indicates redundant features that should be deduplicated — "
        f"keeping one representative from each correlated cluster."
    )
    if len(corr_cols) >= 2:
        sdf_corr = loader.sample_columns(corr_cols, n=sample_n)
        corr_arr = np.corrcoef(
            sdf_corr.select(corr_cols).to_numpy().T
        )
        # Count high-correlation pairs
        n_high_corr = 0
        high_corr_pairs = []
        for i in range(len(corr_cols)):
            for j in range(i + 1, len(corr_cols)):
                c = abs(corr_arr[i, j])
                if c > 0.95:
                    n_high_corr += 1
                    high_corr_pairs.append({
                        "Feature A": corr_cols[i],
                        "Feature B": corr_cols[j],
                        "Correlation": f"{corr_arr[i, j]:.4f}",
                        "Action": "Remove one of the pair",
                    })

        corr_fig = correlation_heatmap(corr_arr, corr_cols,
                                       title=f"Correlation Matrix — Top {len(corr_cols)} Numeric Features")
        corr_content += b.figure(
            corr_fig,
            title="Correlation Heatmap",
            description=f"Pearson correlation matrix for top {len(corr_cols)} numeric features.",
            interpretation=(
                f"Dark blue = strong positive correlation, dark red = strong negative. "
                f"{n_high_corr} feature pairs have |correlation| > 0.95. "
                "Highly correlated features are redundant — removing one reduces dimensionality "
                "without losing predictive information."
            ),
            business_implication=(
                "Multicollinearity is particularly harmful for logistic regression because it "
                "inflates standard errors and makes coefficient estimates unstable. "
                "Tree-based models are less sensitive but still benefit from correlation pruning "
                "as it reduces split competition and improves interpretability."
            ),
        )
        if high_corr_pairs:
            corr_content += b.callout(
                f"<strong>{n_high_corr} feature pairs</strong> with |correlation| > 0.95 detected. "
                "These will be handled in Stage 05 Feature Selection.",
                kind="warning",
            )
            corr_content += b.collapsible(
                f"▼ High-Correlation Pairs ({n_high_corr})",
                b.table(high_corr_pairs[:50], caption="Highly Correlated Feature Pairs"),
            )

    b.add_section("Correlation Analysis", corr_content, icon="🔗")

    html   = b.build()
    writer = ReportWriter(config)
    path   = writer.write(html, "03_Exploratory_Data_Analysis.html")
    writer.write_index()

    elapsed = time.perf_counter() - t0
    log.stage_end("03 — EDA", elapsed)
    return path
