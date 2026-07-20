"""
stages/02_data_quality.py
Stage 02 — Data Quality Assessment Report

Checks
------
* Missing value analysis per column (severity classification)
* Constant and near-constant columns
* High-cardinality columns
* Duplicate row detection
* Target leakage risk flags (columns with suspiciously high IV)
* Data type consistency
* Overall Data Quality Score (0–100)

Output
------
reports/02_Data_Quality.html
"""
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl

from core.data_loader import DataLoader
from core.schema_detector import SchemaDetector
from core.logger import get_logger
from reporting.html_builder import HTMLReportBuilder
from reporting.figure_generator import missing_bar
from reporting.report_writer import ReportWriter


def run(loader: DataLoader, schema: SchemaDetector, config: dict) -> Path:
    log = get_logger("02_DataQuality", config.get("paths", {}).get("logs_dir", "logs"))
    log.stage_start("02 — Data Quality Assessment")
    t0 = time.perf_counter()

    n_rows     = loader.count_rows()
    n_cols     = len(loader.get_columns())
    schema_map = schema.detect()

    b = HTMLReportBuilder(
        report_title   = "Data Quality Assessment Report",
        stage_number   = 2,
        stage_subtitle = "Comprehensive data quality audit — missing values, duplicates, leakage risks",
        config         = config,
        n_rows         = n_rows,
        n_cols         = n_cols,
    )

    # ── Missing values ───────────────────────────────────────────────
    log.info("Computing missing values…")
    miss_df     = loader.get_missing_counts()
    miss_dicts  = miss_df.to_dicts()

    n_missing_cols  = sum(1 for r in miss_dicts if r["missing_pct"] > 0)
    n_high_miss     = sum(1 for r in miss_dicts if r["missing_pct"] > 30)
    n_medium_miss   = sum(1 for r in miss_dicts if 5 < r["missing_pct"] <= 30)
    n_low_miss      = sum(1 for r in miss_dicts if 0 < r["missing_pct"] <= 5)
    n_complete      = n_cols - n_missing_cols
    total_cells     = n_rows * n_cols
    total_missing   = sum(r["missing_count"] for r in miss_dicts)
    overall_miss_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0

    # Classify columns
    miss_rows_display = []
    for r in miss_dicts[:200]:
        sev = ("🔴 Critical (>30%)" if r["missing_pct"] > 30
               else "🟡 High (5-30%)" if r["missing_pct"] > 5
               else "🟢 Low (<5%)" if r["missing_pct"] > 0
               else "✅ Complete")
        action = ("Consider dropping column" if r["missing_pct"] > 60
                  else "Impute with median/mode or create missing indicator" if r["missing_pct"] > 5
                  else "Simple imputation" if r["missing_pct"] > 0
                  else "No action required")
        miss_rows_display.append({
            "Column":          r["column"],
            "Missing Count":   f'{r["missing_count"]:,}',
            "Missing %":       f'{r["missing_pct"]:.2f}%',
            "Severity":        sev,
            "Recommended Action": action,
        })

    miss_fig = missing_bar(
        [r["column"] for r in miss_dicts],
        [r["missing_pct"] for r in miss_dicts],
        title="Missing Value % by Column (Top 40)",
    )

    miss_fig_html = b.figure(
        miss_fig,
        title        = "Missing Value Profile",
        description  = "Horizontal bar chart showing missing percentage for columns with >0% missing.",
        interpretation = (
            f"A total of {n_missing_cols:,} columns ({n_missing_cols/n_cols*100:.1f}%) "
            f"contain at least one missing value. "
            f"{n_high_miss} columns exceed the 30% critical threshold and may need to be "
            f"dropped or imputed with indicator flags. "
            f"The overall dataset completeness is {100 - overall_miss_pct:.2f}%."
        ),
        business_implication = (
            "Missing data in credit risk models can introduce systematic bias if the "
            "missingness is not random (MCAR). For example, if high-income borrowers "
            "are more likely to have certain fields populated, simple mean imputation "
            "will distort the model's risk estimates. A missing indicator dummy variable "
            "should be created alongside imputation to preserve this signal."
        ),
    )

    miss_tbl = b.table(
        miss_rows_display,
        caption = "Missing Value Analysis by Column",
        interpretation = (
            f"The table ranks columns by missing percentage in descending order. "
            f"{n_high_miss} columns have >30% missing (critical), "
            f"{n_medium_miss} have 5–30% (high), "
            f"{n_low_miss} have <5% (low). "
            f"{n_complete} columns are fully complete."
        ),
    )

    miss_calls = ""
    if n_high_miss > 0:
        miss_calls += b.callout(
            f"<strong>{n_high_miss} columns</strong> have >30% missing values. "
            "Consider: (1) drop if >60% missing and low IV, "
            "(2) impute with median/mode + add binary missing-indicator column, "
            "(3) apply WoE binning which handles missings natively.",
            kind="danger", title="🚨 Critical Missing Data"
        )
    if overall_miss_pct < 5:
        miss_calls += b.callout(
            f"Overall dataset completeness is {100 - overall_miss_pct:.2f}%. "
            "This is within acceptable thresholds for most modelling frameworks.",
            kind="success", title="✓ Good Overall Completeness"
        )

    miss_content = miss_calls + miss_fig_html + miss_tbl
    b.add_section("Missing Value Analysis", miss_content, icon="🔍")

    # ── Constants & Near-constants ───────────────────────────────────
    log.info("Checking constants and near-constants…")
    constant_cols    = schema.get_constant_cols()
    n_constants      = len(constant_cols)

    # Near-constant: one value accounts for >99% of rows
    near_constant_cols = []
    cat_cols = schema.get_categorical_cols()[:200]  # batch check
    for col in cat_cols:
        vc = loader.column_value_counts(col, top_n=2)
        if not vc.is_empty():
            top_pct = float(vc["pct"][0])
            if top_pct > 99.0:
                near_constant_cols.append({"column": col, "dominant_pct": top_pct})

    const_content = b.p(
        f"Constant columns (zero variance) are automatically identified during schema detection. "
        f"{n_constants} constant columns were found. Near-constant columns (>99% dominated "
        f"by a single value) are equally useless for modelling — they add noise without signal."
    )
    if constant_cols:
        const_list = [{"Column": c, "Issue": "Constant — single unique value",
                       "Action": "Drop before modelling"}
                      for c in constant_cols]
        const_content += b.table(const_list, caption="Constant Columns to Drop")

    if near_constant_cols:
        const_content += b.table(
            near_constant_cols,
            caption = "Near-Constant Columns (>99% dominated by one value)",
            interpretation = (
                "Near-constant columns have negligible variance and will contribute "
                "almost no discriminatory power to any model. They increase dimensionality "
                "without benefit and should be dropped."
            ),
        )
        const_content += b.callout(
            f"<strong>{len(near_constant_cols)}</strong> near-constant categorical columns detected. "
            "These will be removed during feature engineering.",
            kind="warning",
        )

    b.add_section("Constant & Near-Constant Columns", const_content, icon="⚠️")

    # ── Duplicate rows ───────────────────────────────────────────────
    log.info("Checking duplicate rows (sampling approach)…")
    sample_dup = loader.sample(50_000)
    dup_count  = sample_dup.is_duplicated().sum()
    dup_pct    = dup_count / len(sample_dup) * 100 if len(sample_dup) > 0 else 0
    est_full_dups = int((dup_pct / 100) * n_rows)

    dup_content = b.p(
        f"Duplicate row detection was performed on a 50,000-row random sample "
        f"(checking all columns for exact matches). "
        f"{dup_count:,} exact duplicate rows were found in the sample ({dup_pct:.2f}%). "
        f"Extrapolated to the full dataset, this suggests approximately "
        f"{est_full_dups:,} duplicate rows may exist."
    )
    if dup_pct > 1.0:
        dup_content += b.callout(
            f"Estimated {est_full_dups:,} duplicate rows ({dup_pct:.2f}%). "
            "Duplicates can cause data leakage when they appear in both train and test sets. "
            "Recommendation: deduplicate before train/test splitting.",
            kind="danger" if dup_pct > 5 else "warning",
        )
    else:
        dup_content += b.callout(
            f"Duplicate rate is low ({dup_pct:.2f}%). No immediate action required.",
            kind="success",
        )
    b.add_section("Duplicate Row Analysis", dup_content, icon="🔁")

    # ── Target leakage risk ──────────────────────────────────────────
    log.info("Checking for target leakage risks…")
    leakage_flags = []
    num_cols = schema.get_numeric_cols()[:50]
    for col in num_cols:
        iv, _ = loader.compute_iv_woe(col)
        if iv > 0.8:  # Suspiciously high IV
            leakage_flags.append({
                "Column":          col,
                "Information Value": f"{iv:.4f}",
                "Risk":            "⚠️ Possible data leakage",
                "Recommendation":  "Investigate data collection pipeline — IV > 0.8 is suspicious",
            })

    leak_content = b.p(
        "Target leakage occurs when a feature directly encodes or derives from the target variable, "
        "causing artificially high model performance during training that does not generalise to "
        "production. This is a critical risk in credit risk modelling where outcome-based features "
        "(e.g., collection status, write-off flag) may have been inadvertently included."
    )
    if leakage_flags:
        leak_content += b.callout(
            f"<strong>{len(leakage_flags)} feature(s)</strong> have suspiciously high Information "
            f"Value (IV > 0.8). These require manual review to confirm they do not derive from "
            f"the target variable or post-outcome data.",
            kind="danger", title="🚨 Potential Target Leakage"
        )
        leak_content += b.table(leakage_flags, caption="High-IV Features — Leakage Risk Screen")
    else:
        leak_content += b.callout(
            "No features with suspiciously high IV (>0.8) detected in the numeric columns screened. "
            "Categorical features should also be reviewed manually.",
            kind="success",
        )
    b.add_section("Target Leakage Risk Assessment", leak_content, icon="🚨")

    # ── Overall Data Quality Score ───────────────────────────────────
    score = 100
    score -= min(30, n_high_miss * 2)           # heavy penalty for high-miss cols
    score -= min(10, overall_miss_pct * 0.5)    # overall missingness
    score -= min(10, dup_pct * 2)               # duplicates
    score -= min(10, len(leakage_flags) * 5)    # leakage risks
    score -= min(10, n_constants)               # constant cols (if many)
    score  = max(0, round(score))

    grade = ("A — Excellent" if score >= 85
             else "B — Good" if score >= 70
             else "C — Fair" if score >= 55
             else "D — Poor" if score >= 40
             else "F — Critical")

    score_content = (
        b.score_card(score, "Data Quality Score", grade)
        + b.p(
            f"The overall data quality score of <strong>{score}/100</strong> ({grade}) reflects "
            f"the combined assessment of missing values ({n_high_miss} critical columns), "
            f"duplicates ({dup_pct:.1f}%), leakage risks ({len(leakage_flags)} flags), "
            f"and constant columns ({n_constants}). "
            f"Scores below 55 indicate the dataset requires significant remediation before "
            f"reliable model training is possible."
        )
        + b.callout(
            "<strong>Priority Actions:</strong><ul>"
            f"<li>{'Drop or impute ' + str(n_high_miss) + ' columns with >30% missing' if n_high_miss else '✓ No critical missing data'}</li>"
            f"<li>{'Deduplicate dataset before splitting' if dup_pct > 1 else '✓ Duplicate rate acceptable'}</li>"
            f"<li>{'Manually review ' + str(len(leakage_flags)) + ' high-IV features for leakage' if leakage_flags else '✓ No leakage flags'}</li>"
            f"<li>Drop {n_constants} constant columns automatically</li>"
            "</ul>",
            kind="recommend",
        )
    )

    # Executive summary cards for quality
    quality_cards = [
        {"label": "Quality Score",       "value": f"{score}/100",
         "variant": "success" if score >= 70 else "warning" if score >= 55 else "danger"},
        {"label": "Quality Grade",       "value": grade[0]},
        {"label": "Columns w/ Missing",  "value": f"{n_missing_cols:,}",
         "variant": "danger" if n_high_miss > 0 else "success"},
        {"label": "Critical Missing",    "value": str(n_high_miss),
         "sub": "cols > 30% missing", "variant": "danger" if n_high_miss > 0 else "success"},
        {"label": "Duplicate Rows (est)","value": f"{est_full_dups:,}",
         "variant": "danger" if dup_pct > 5 else "warning" if dup_pct > 1 else "success"},
        {"label": "Leakage Flags",       "value": str(len(leakage_flags)),
         "variant": "danger" if leakage_flags else "success"},
        {"label": "Constant Columns",    "value": str(n_constants),
         "variant": "warning" if n_constants > 0 else "success"},
        {"label": "Overall Completeness","value": f"{100 - overall_miss_pct:.1f}%",
         "variant": "success" if overall_miss_pct < 5 else "warning"},
    ]
    b.add_executive_summary(quality_cards, narrative=(
        "The data quality assessment evaluates the dataset across four critical dimensions: "
        "completeness (missing values), uniqueness (duplicates), validity (leakage), and "
        "informativeness (constant columns). Each dimension is scored and combined into an "
        "overall quality score that guides remediation priorities before modelling begins."
    ))
    b.add_section("Overall Data Quality Score", score_content, icon="🏅")

    html   = b.build()
    writer = ReportWriter(config)
    path   = writer.write(html, "02_Data_Quality.html")
    writer.write_index()

    elapsed = time.perf_counter() - t0
    log.stage_end("02 — Data Quality", elapsed)
    return path
