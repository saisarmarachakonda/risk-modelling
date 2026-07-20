"""
stages/01_data_overview.py
Stage 01 — Data Overview Report

Performs
--------
* Dataset registration and row/column count
* DuckDB schema description
* Column type classification (numeric / categorical / boolean / etc.)
* Target column distribution
* Memory and file-size estimation
* Statistical summary for numeric columns (via DuckDB aggregates)
* Data dictionary generation

Output
------
reports/01_Data_Overview.html
"""
import time
from pathlib import Path
from typing import Dict

import polars as pl

from core.data_loader import DataLoader
from core.schema_detector import SchemaDetector
from core.memory_manager import MemoryManager
from core.logger import get_logger
from reporting.html_builder import HTMLReportBuilder
from reporting.figure_generator import (
    target_bar, distribution_grid, missing_bar
)
from reporting.report_writer import ReportWriter


def run(loader: DataLoader, schema: SchemaDetector, config: dict) -> Path:
    """
    Execute Stage 01 — Data Overview.

    Parameters
    ----------
    loader : DataLoader   — registered data loader
    schema : SchemaDetector — detected column schema
    config : dict         — pipeline configuration
    """
    log = get_logger("01_DataOverview", config.get("paths", {}).get("logs_dir", "logs"))
    log.stage_start("01 — Data Overview")
    t0 = time.perf_counter()

    n_rows     = loader.count_rows()
    columns    = loader.get_columns()
    n_cols     = len(columns)
    schema_map = schema.detect()
    schema_sum = schema.get_schema_summary()
    mem_mgr    = MemoryManager(config)
    file_mb    = loader.get_file_size_mb()
    est_gb     = mem_mgr.estimate_df_memory_gb(n_rows, n_cols)

    b = HTMLReportBuilder(
        report_title  = "Data Overview Report",
        stage_number  = 1,
        stage_subtitle= "Complete dataset profiling, schema analysis, and statistical baseline",
        config        = config,
        dataset_name  = str(loader.input_path),
        n_rows        = n_rows,
        n_cols        = n_cols,
    )

    # ── 1. Executive Summary ─────────────────────────────────────────
    pos_label = config.get("domain", {}).get("positive_class_label", "Positive")
    neg_label = config.get("domain", {}).get("negative_class_label", "Negative")
    target_dist = loader.get_target_distribution()

    pos_pct = neg_pct = 0.0
    if not target_dist.is_empty():
        vals = target_dist.to_dicts()
        for row in vals:
            if str(row["target_value"]) in ("1", "1.0", pos_label):
                pos_pct = float(row["pct"])
            else:
                neg_pct = float(row["pct"])

    imbalance_ratio = round(neg_pct / pos_pct, 1) if pos_pct > 0 else 0

    cards = [
        {"label": "Total Records",     "value": f"{n_rows:,}",      "sub": "observations"},
        {"label": "Total Features",    "value": f"{n_cols:,}",      "sub": "columns"},
        {"label": "Numeric Features",  "value": f"{schema_sum['numeric']:,}",
         "sub": "continuous variables"},
        {"label": "Categorical",       "value": f"{schema_sum['categorical']:,}",
         "sub": "categorical variables"},
        {"label": "File Size",         "value": f"{file_mb:.1f} MB",  "sub": "on disk"},
        {"label": "Est. RAM (dense)",  "value": f"{est_gb:.1f} GB",
         "sub": "float64 in memory", "variant": "warning" if est_gb > 4 else "success"},
        {"label": "Class Imbalance",   "value": f"{imbalance_ratio:.1f}:1",
         "sub": f"{neg_label} : {pos_label}",
         "variant": "danger" if imbalance_ratio > 10 else "warning" if imbalance_ratio > 4 else "success"},
        {"label": "Target Column",     "value": loader.target_col,   "sub": "binary classification"},
    ]


    imbalance_warning = (
        "This significant imbalance will require careful handling during modelling — "
        "techniques such as class weighting, oversampling (SMOTE), or threshold adjustment "
        "should be considered."
        if imbalance_ratio > 4 else ""
    )

    narrative = (
        f"This report provides a comprehensive overview of the input dataset used for the "
        f"{config.get('domain', {}).get('name', 'Risk')} binary classification model. "
        f"The dataset contains {n_rows:,} observations and {n_cols:,} variables, "
        f"of which {schema_sum['numeric']:,} are numeric, {schema_sum['categorical']:,} are categorical, "
        f"{schema_sum['boolean']:,} are boolean, and {schema_sum['constant']:,} are constant (zero-variance). "
        f"\n\n"
        f"The target variable '{loader.target_col}' exhibits a class imbalance ratio of approximately "
        f"{imbalance_ratio:.1f}:1 ({neg_label} to {pos_label}). "
        f"{imbalance_warning}"
        f"\n\n"
        f"The raw file occupies {file_mb:.1f} MB on disk. If loaded as a dense float64 matrix, "
        f"it would require approximately {est_gb:.1f} GB of RAM. The pipeline uses DuckDB and Polars "
        f"lazy evaluation to avoid materialising the full dataset — all aggregates are computed "
        f"via SQL push-down and only summary results are brought into Python memory."
    )
    b.add_executive_summary(cards=cards, narrative=narrative)

    # ── 2. Dataset Overview ──────────────────────────────────────────
    overview_rows = [
        {"Property": "Total Observations",    "Value": f"{n_rows:,}"},
        {"Property": "Total Variables",        "Value": f"{n_cols:,}"},
        {"Property": "Numeric Variables",      "Value": f"{schema_sum['numeric']:,}"},
        {"Property": "Categorical Variables",  "Value": f"{schema_sum['categorical']:,}"},
        {"Property": "Boolean Variables",      "Value": f"{schema_sum['boolean']:,}"},
        {"Property": "Datetime Variables",     "Value": f"{schema_sum['datetime']:,}"},
        {"Property": "Identifier Columns",     "Value": f"{schema_sum['identifier']:,}"},
        {"Property": "Constant Columns",       "Value": f"{schema_sum['constant']:,}"},
        {"Property": "Feature Columns",        "Value": f"{schema_sum['feature_columns']:,}"},
        {"Property": "Target Column",          "Value": loader.target_col},
        {"Property": "Problem Type",           "Value": "Binary Classification"},
        {"Property": "File Size (disk)",       "Value": f"{file_mb:.2f} MB"},
        {"Property": "Est. Dense RAM (float64)","Value": f"{est_gb:.2f} GB"},
        {"Property": "Input Format",           "Value": loader.input_format.upper()},
        {"Property": "Input Path",             "Value": str(loader.input_path)},
    ]

    overview_tbl = b.table(
        overview_rows,
        caption="Dataset Properties",
        interpretation=(
            "The table above summarises the key structural properties of the dataset. "
            f"With {n_cols:,} columns, this is a high-dimensional dataset. "
            f"The {schema_sum['constant']:,} constant columns will be dropped before modelling "
            f"as they carry zero predictive information. "
            f"The {schema_sum['identifier']:,} identifier columns will also be excluded "
            f"as they are surrogate keys with no generalizable signal."
        ),
    )

    # Target distribution figure
    if not target_dist.is_empty():
        tgt_rows  = target_dist.to_dicts()
        tgt_lbls  = [str(r["target_value"]) for r in tgt_rows]
        tgt_cnts  = [int(r["count"]) for r in tgt_rows]
        tgt_pcts  = [float(r["pct"]) for r in tgt_rows]
        tgt_fig   = target_bar(
            tgt_lbls, tgt_cnts, tgt_pcts,
            title=f"Target Distribution — '{loader.target_col}'"
        )
        tgt_fig_html = b.figure(
            tgt_fig,
            title    = "Target Class Distribution",
            description = f"Bar chart and pie chart showing the distribution of '{loader.target_col}'.",
            interpretation = (
                f"The target variable has {len(tgt_lbls)} classes. "
                f"The minority class represents {min(tgt_pcts):.2f}% of all records. "
                f"A ratio greater than 5:1 is typically considered imbalanced and may "
                f"require resampling or cost-sensitive learning strategies."
            ),
            business_implication = (
                f"Class imbalance directly affects model training. Without correction, "
                f"the model will be biased toward predicting the majority class. "
                f"The business cost of missed defaults (false negatives) typically "
                f"greatly exceeds the cost of false alarms (false positives) in credit risk, "
                f"making recall for the minority class the primary operational concern."
            ),
        )
    else:
        tgt_fig_html = b.callout("Target distribution could not be computed.", kind="warning")

    overview_content = overview_tbl + tgt_fig_html
    b.add_section("Dataset Overview", overview_content, icon="📦")

    # ── 3. Schema Summary ────────────────────────────────────────────
    schema_rows = []
    for col, info in schema_map.items():
        type_badge = b.badge(info["col_type_label"], info["col_type"])
        miss_var   = "danger" if info["missing_pct"] > 30 else \
                     "warning" if info["missing_pct"] > 5 else "low"
        miss_str   = f'{info["missing_pct"]:.2f}%'
        schema_rows.append({
            "Column":       col,
            "DB Type":      info["db_type"],
            "Category":     info["col_type_label"],
            "Unique Values":f'{info["n_unique"]:,}',
            "Missing %":    miss_str,
            "Preprocessing":info["preprocessing_suggestion"][:60],
        })

    schema_tbl = b.table(
        schema_rows,
        caption        = "Complete Column Schema",
        interpretation = (
            "Each column is classified based on its DuckDB data type and statistical properties "
            "(uniqueness ratio, cardinality). The preprocessing suggestion column indicates "
            "the recommended transformation for each column type. Columns classified as "
            "'Constant' or 'Identifier' will be automatically excluded from modelling."
        ),
        max_rows       = 500,
    )
    schema_collapse = b.collapsible(
        f"▼ Full Data Dictionary ({n_cols:,} columns)", schema_tbl, open_=False
    )
    schema_content = (
        b.p(
            f"The schema detector classified all {n_cols:,} columns by analysing DuckDB data types "
            f"combined with statistical heuristics (cardinality ratio, uniqueness ratio, column "
            f"naming patterns). This classification drives all subsequent preprocessing decisions."
        )
        + schema_collapse
    )
    b.add_section("Data Dictionary & Schema", schema_content, icon="📖")

    # ── 4. Numeric Statistical Summary ──────────────────────────────
    num_cols = schema.get_numeric_cols()[:100]  # cap at 100 for speed
    if num_cols:
        log.info(f"Computing numeric stats for {len(num_cols)} columns…")
        stats_df = loader.get_numeric_stats(num_cols)
        stats_tbl = b.table(
            stats_df,
            caption = "Descriptive Statistics — Numeric Columns",
            interpretation = (
                "Descriptive statistics are computed via DuckDB aggregate SQL queries — "
                "no data is loaded into Python memory. Columns with extremely high coefficient "
                "of variation (CV > 1.0) or skewness may benefit from log transformation. "
                "Large IQR values relative to the mean suggest the presence of outliers."
            ),
        )
        stats_collapse = b.collapsible(
            f"▼ Numeric Statistics ({len(num_cols)} columns)", stats_tbl, open_=True
        )

        # Visualise sample distributions
        sample_df = loader.sample_columns(num_cols[:12], n=50_000)
        sample_data = {col: sample_df[col].drop_nulls().to_numpy() for col in num_cols[:12]}
        dist_fig = distribution_grid(
            sample_data, num_cols[:12],
            title="Numeric Feature Distributions (50k row sample)"
        )
        dist_fig_html = b.figure(
            dist_fig,
            title = "Numeric Feature Distribution Grid",
            description = "Histograms for the first 12 numeric features using a 50,000-row random sample.",
            interpretation = (
                "Skewed distributions (long right or left tails) are visible in several features. "
                "Heavy right skew suggests log transformation may be appropriate. "
                "Near-uniform or near-bimodal distributions may indicate mixing of sub-populations "
                "and could carry strong predictive signal."
            ),
        )
        numeric_content = b.p(
            f"Numeric statistics are computed for {len(num_cols)} continuous columns using "
            f"DuckDB PERCENTILE_CONT and STDDEV_POP aggregates — processing the full "
            f"{n_rows:,} rows without loading into RAM."
        ) + stats_collapse + dist_fig_html

    else:
        numeric_content = b.callout("No numeric columns detected in the dataset.", kind="warning")

    b.add_section("Numeric Statistical Summary", numeric_content, icon="📐")

    # ── 5. System & Hardware Information ────────────────────────────
    sys_info = mem_mgr.get_system_info()
    sys_rows = [{"Property": k, "Value": str(v)} for k, v in sys_info.items()]
    sys_tbl  = b.table(sys_rows, caption="Execution Environment")
    sys_content = (
        b.p(
            f"The pipeline executed on the following hardware configuration. "
            f"Available RAM at report generation time: {sys_info.get('available_ram_gb', 'N/A')} GB. "
            f"DuckDB is configured to use a maximum of "
            f"{config.get('memory', {}).get('duckdb_memory_limit', '6GB')} memory and "
            f"{config.get('memory', {}).get('duckdb_threads', 4)} threads."
        )
        + sys_tbl
    )
    b.add_section("Execution Environment", sys_content, icon="🖥️")

    # ── Build & Write ────────────────────────────────────────────────
    html   = b.build()
    writer = ReportWriter(config)
    path   = writer.write(html, "01_Data_Overview.html")
    writer.write_index()

    elapsed = time.perf_counter() - t0
    log.stage_end("01 — Data Overview", elapsed)
    return path
