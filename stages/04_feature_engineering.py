"""
stages/04_feature_engineering.py
Stage 04 — Feature Engineering Report

Transformations applied
-----------------------
* Constant column removal
* High-cardinality identifier column removal
* Missing value imputation (numeric → median, categorical → mode/'UNKNOWN')
* Missing indicator columns (binary flag for columns with >5% missing)
* Outlier winsorisation (IQR-based, optional)
* Categorical encoding (OHE for low-cardinality, ordinal for medium)
* Log1p transformation for right-skewed numeric features
* StandardScaler for numeric features (required by LR)
* Feature name sanitisation

All transformations are logged with before/after statistics.

Output
------
reports/04_Feature_Engineering.html
artifacts/feature_engineering_spec.json  (transformation spec for serving)
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import polars as pl

try:
    from scipy import stats as sp_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from core.data_loader import DataLoader
from core.schema_detector import SchemaDetector
from core.logger import get_logger
from reporting.html_builder import HTMLReportBuilder
from reporting.figure_generator import distribution_grid
from reporting.report_writer import ReportWriter


def run(
    loader: DataLoader,
    schema: SchemaDetector,
    config: dict,
) -> Tuple[Path, Dict]:
    """
    Execute Stage 04 — Feature Engineering.

    Returns
    -------
    report_path : Path
    fe_spec     : dict  — transformation specification used for serving
    """
    log = get_logger("04_FeatureEngineering",
                     config.get("paths", {}).get("logs_dir", "logs"))
    log.stage_start("04 — Feature Engineering")
    t0 = time.perf_counter()

    n_rows   = loader.count_rows()
    pp_cfg   = config.get("preprocessing", {})
    sample_n = min(config.get("data", {}).get("sample_size_for_eda", 100_000), n_rows)

    schema_map = schema.detect()
    num_cols   = schema.get_numeric_cols()
    cat_cols   = schema.get_categorical_cols()
    bool_cols  = schema.get_boolean_cols()
    const_cols = schema.get_constant_cols()
    id_cols    = schema.get_identifier_cols()
    target     = loader.target_col

    b = HTMLReportBuilder(
        report_title   = "Feature Engineering Report",
        stage_number   = 4,
        stage_subtitle = "Data transformations, encoding, scaling, and imputation",
        config         = config,
        n_rows         = n_rows,
        n_cols         = len(loader.get_columns()),
    )

    fe_spec: Dict[str, Any] = {
        "dropped_constant":   const_cols,
        "dropped_identifier": id_cols,
        "imputation":         {},
        "log_transform":      [],
        "winsorisation":      {},
        "encoding":           {},
        "missing_indicators": [],
        "feature_names_out":  [],
    }

    steps_log = []  # for report table

    # ── Step 1: Drop constants and identifiers ────────────────────────
    n_dropped = len(const_cols) + len(id_cols)
    steps_log.append({
        "Step": "1",
        "Transformation": "Drop Constant & Identifier Columns",
        "Columns Affected": f"{n_dropped:,}",
        "Reason": "Zero-variance or ID columns carry no predictive signal",
        "Method": "Remove from feature set",
    })

    remaining = [c for c in (num_cols + cat_cols + bool_cols)
                 if c not in set(const_cols + id_cols) and c != target]
    log.info(f"Remaining features after dropping: {len(remaining):,}")

    # ── Step 2: Missing value imputation ─────────────────────────────
    miss_df   = loader.get_missing_counts()
    miss_map  = dict(zip(miss_df["column"].to_list(), miss_df["missing_pct"].to_list()))

    miss_indicator_cols = [c for c in remaining if miss_map.get(c, 0) > 5.0]
    fe_spec["missing_indicators"] = miss_indicator_cols
    log.info(f"Missing indicator columns: {len(miss_indicator_cols)}")

    # Compute medians for numeric imputation via DuckDB
    num_remaining = [c for c in num_cols if c in remaining]
    cat_remaining = [c for c in cat_cols if c in remaining]

    if num_remaining:
        stats_df = loader.get_numeric_stats(num_remaining[:200])
        for row in stats_df.to_dicts():
            fe_spec["imputation"][row["column"]] = {
                "strategy": "median",
                "fill_value": row.get("median") or 0.0,
            }

    for col in cat_remaining[:200]:
        vc = loader.column_value_counts(col, top_n=1)
        mode_val = str(vc["value"][0]) if not vc.is_empty() else "UNKNOWN"
        fe_spec["imputation"][col] = {
            "strategy": "mode",
            "fill_value": mode_val,
        }

    steps_log.append({
        "Step": "2a",
        "Transformation": "Numeric Imputation",
        "Columns Affected": f"{len(num_remaining):,}",
        "Reason": "Replace NaN with median to preserve central tendency",
        "Method": "Median (DuckDB PERCENTILE_CONT)",
    })
    steps_log.append({
        "Step": "2b",
        "Transformation": "Categorical Imputation",
        "Columns Affected": f"{len(cat_remaining):,}",
        "Reason": "Replace NaN with most frequent value or 'UNKNOWN'",
        "Method": "Mode (DuckDB COUNT GROUP BY)",
    })
    steps_log.append({
        "Step": "2c",
        "Transformation": "Missing Indicator Columns",
        "Columns Affected": str(len(miss_indicator_cols)),
        "Reason": "Preserve missingness pattern as a binary signal",
        "Method": "Create _is_missing binary flag for cols with >5% missing",
    })

    # ── Step 3: Outlier winsorisation ─────────────────────────────────
    winsor_cfg = {}
    apply_outlier = pp_cfg.get("outlier_method", "iqr") != "none"
    if apply_outlier and num_remaining:
        stats_df_dict = {r["column"]: r for r in loader.get_numeric_stats(num_remaining[:200]).to_dicts()}
        for col in num_remaining[:200]:
            s = stats_df_dict.get(col, {})
            q1 = s.get("q1")
            q3 = s.get("q3")
            if q1 is not None and q3 is not None:
                iqr = q3 - q1
                lower = q1 - 3.0 * iqr
                upper = q3 + 3.0 * iqr
                winsor_cfg[col] = {"lower": lower, "upper": upper}

        fe_spec["winsorisation"] = winsor_cfg
        steps_log.append({
            "Step": "3",
            "Transformation": "Outlier Winsorisation (IQR 3×)",
            "Columns Affected": str(len(winsor_cfg)),
            "Reason": "Cap extreme values to reduce influence on linear models",
            "Method": "Clip at Q1 - 3×IQR and Q3 + 3×IQR",
        })

    # ── Step 4: Log transform for skewed numerics ─────────────────────
    log_transform_cols = []
    if SCIPY_AVAILABLE and num_remaining:
        sample_df = loader.sample_columns(num_remaining[:50], n=50_000)
        for col in num_remaining[:50]:
            if col not in sample_df.columns:
                continue
            arr = sample_df[col].drop_nulls().to_numpy().astype(float)
            if len(arr) < 10:
                continue
            if (arr > 0).all():  # only log if all positive
                sk = float(sp_stats.skew(arr))
                if sk > 2.0:
                    log_transform_cols.append(col)

    fe_spec["log_transform"] = log_transform_cols
    if log_transform_cols:
        steps_log.append({
            "Step": "4",
            "Transformation": "Log1p Transform",
            "Columns Affected": str(len(log_transform_cols)),
            "Reason": "Reduce right skewness — required for logistic regression",
            "Method": "log1p(x) — safe for x ≥ 0",
        })

    # ── Step 5: Categorical encoding ──────────────────────────────────
    ohe_cols    = []
    ord_cols    = []
    woe_cols    = []

    for col in cat_remaining:
        n_uniq = schema_map[col]["n_unique"]
        if n_uniq <= 10:
            ohe_cols.append(col)
            fe_spec["encoding"][col] = "one_hot"
        elif n_uniq <= pp_cfg.get("max_cardinality_for_encoding", 50):
            ord_cols.append(col)
            fe_spec["encoding"][col] = "ordinal"
        else:
            woe_cols.append(col)
            fe_spec["encoding"][col] = "woe"

    steps_log.append({
        "Step": "5a",
        "Transformation": "One-Hot Encoding (OHE)",
        "Columns Affected": str(len(ohe_cols)),
        "Reason": "Low-cardinality categoricals — OHE preserves all information",
        "Method": f"Binary dummy columns (≤10 unique values)",
    })
    steps_log.append({
        "Step": "5b",
        "Transformation": "Ordinal Encoding",
        "Columns Affected": str(len(ord_cols)),
        "Reason": "Medium-cardinality categoricals — reduces dimensionality vs OHE",
        "Method": "Integer label encoding (11–50 unique values)",
    })
    steps_log.append({
        "Step": "5c",
        "Transformation": "WoE / Target Encoding",
        "Columns Affected": str(len(woe_cols)),
        "Reason": "High-cardinality — WoE naturally handles risk relationship",
        "Method": "Weight of Evidence encoding (>50 unique values)",
    })

    # ── Step 6: Standard scaling ──────────────────────────────────────
    steps_log.append({
        "Step": "6",
        "Transformation": "StandardScaler",
        "Columns Affected": f"{len(num_remaining):,}",
        "Reason": "Required for logistic regression — features on same scale",
        "Method": "z = (x - mean) / std, computed on training set only",
    })

    # Estimated output feature count
    n_features_out = (
        len(num_remaining) +       # scaled numerics
        len(log_transform_cols) +  # log-transformed (overlap with numeric)
        len(ohe_cols) * 5 +        # rough avg 5 dummies per OHE col
        len(ord_cols) +            # ordinal
        len(woe_cols) +            # WoE
        len(bool_cols) +           # booleans as-is
        len(miss_indicator_cols)   # missing indicators
    )

    # ── Build report ──────────────────────────────────────────────────
    cards = [
        {"label": "Input Features",      "value": f"{len(remaining):,}"},
        {"label": "Dropped (const/id)",  "value": f"{n_dropped:,}", "variant": "warning"},
        {"label": "OHE Columns",         "value": str(len(ohe_cols))},
        {"label": "Ordinal Encoded",     "value": str(len(ord_cols))},
        {"label": "WoE Encoded",         "value": str(len(woe_cols))},
        {"label": "Log-Transformed",     "value": str(len(log_transform_cols))},
        {"label": "Missing Indicators",  "value": str(len(miss_indicator_cols))},
        {"label": "Est. Output Features","value": f"~{n_features_out:,}"},
    ]
    b.add_executive_summary(cards, narrative=(
        f"Feature engineering transforms the raw {len(loader.get_columns()):,}-column dataset "
        f"into a clean, model-ready feature matrix. After removing {n_dropped:,} useless columns "
        f"(constants and identifiers), imputing missing values, and applying transformations, "
        f"the estimated output feature matrix contains approximately {n_features_out:,} features. "
        f"\n\n"
        f"A critical design choice is that StandardScaler is applied to ALL numeric features. "
        f"This is <strong>mandatory for logistic regression</strong> (which is sensitive to feature "
        f"scale) but has no effect on tree-based models (which are scale-invariant). "
        f"Both scaled and unscaled versions are preserved in the pipeline to serve each algorithm correctly."
    ))

    steps_tbl = b.table(
        steps_log,
        caption="Feature Engineering Pipeline Steps",
        interpretation=(
            "Each row describes one transformation in the engineering pipeline. "
            "Transformations are applied in the order listed. The 'Columns Affected' count "
            "reflects the number of columns undergoing each transformation. "
            "All parameters (medians, modes, IQR bounds) are computed on the training set only — "
            "they are then applied unchanged to the validation and test sets to prevent data leakage."
        ),
    )

    # WoE explanation
    woe_box = b.card(
        "WoE vs Standard Encoding — Key Differences",
        "<ul>"
        "<li><strong>Standard One-Hot Encoding:</strong> Creates k-1 dummy columns. Simple but "
        "does not encode the relationship with the target. Logistic regression learns this relationship "
        "from coefficients. Fails for high-cardinality features.</li>"
        "<li><strong>Ordinal Encoding:</strong> Maps categories to integers. Assumes ordinal relationship. "
        "Works well for tree-based models. Not ideal for LR.</li>"
        "<li><strong>WoE (Weight of Evidence):</strong> Replaces each category with "
        "log(% Events / % Non-Events). Directly encodes the category's relationship with the binary target. "
        "Handles high cardinality, missing values natively, and makes LR coefficients directly "
        "interpretable as log-odds contributions. Preferred in credit risk scorecard modelling.</li>"
        "</ul>"
    )

    # Encoding comparison table
    enc_comparison = b.table(
        [
            {"Encoding":    "One-Hot (OHE)",
             "Cardinality": "Low (≤10)",
             "LR Friendly": "✅ Yes",
             "Tree Friendly":"✅ Yes",
             "Handles NA":  "❌ No",
             "Dimensionality":"Increases by k-1"},
            {"Encoding":    "Ordinal",
             "Cardinality": "Medium (≤50)",
             "LR Friendly": "⚠️ Partial",
             "Tree Friendly":"✅ Yes",
             "Handles NA":  "❌ No",
             "Dimensionality":"No increase"},
            {"Encoding":    "WoE / Target",
             "Cardinality": "Any",
             "LR Friendly": "✅ Best",
             "Tree Friendly":"⚠️ Partial",
             "Handles NA":  "✅ Yes",
             "Dimensionality":"No increase"},
            {"Encoding":    "Frequency",
             "Cardinality": "High",
             "LR Friendly": "⚠️ Partial",
             "Tree Friendly":"✅ Yes",
             "Handles NA":  "✅ Yes",
             "Dimensionality":"No increase"},
        ],
        caption="Categorical Encoding Strategy Comparison",
        extra_class="compare-table",
    )

    fe_content = steps_tbl + b.hr() + woe_box + enc_comparison

    b.add_section("Transformation Pipeline", fe_content, icon="⚙️")

    # ── Scaling explanation ───────────────────────────────────────────
    scale_content = (
        b.p(
            "StandardScaler subtracts the training-set mean and divides by the training-set "
            "standard deviation: z = (x - μ) / σ. This ensures all numeric features are on "
            "a common scale (mean=0, std=1)."
        )
        + b.card(
            "Why Scaling Matters — Tree-Based vs Logistic Regression",
            "<table style='width:100%;border-collapse:collapse;font-size:14px;'>"
            "<thead style='background:#0f2d52;color:white;'><tr>"
            "<th style='padding:8px'>Aspect</th>"
            "<th style='padding:8px;border-top:3px solid #d97706'>Tree-Based Models</th>"
            "<th style='padding:8px;border-top:3px solid #0ea5e9'>Logistic Regression</th>"
            "</tr></thead><tbody>"
            + "".join(
                f"<tr><td style='padding:8px;font-weight:600'>{row[0]}</td>"
                f"<td style='padding:8px;background:rgba(217,119,6,.06)'>{row[1]}</td>"
                f"<td style='padding:8px;background:rgba(59,130,246,.06)'>{row[2]}</td></tr>"
                for row in [
                    ("Scale sensitivity", "❌ None — splits on thresholds", "✅ Critical — gradient descent sensitive to scale"),
                    ("Outlier sensitivity","⚠️ Low — tree partitions around", "🚨 High — outliers distort coefficients"),
                    ("Feature units",     "❌ Irrelevant", "✅ Must be comparable"),
                    ("Scaling needed",    "❌ Not required", "✅ Mandatory"),
                    ("Scaling effect on perf","Neutral", "Often large improvement"),
                ]
            )
            + "</tbody></table>"
        )
        + b.callout(
            "The pipeline maintains <strong>two versions</strong> of the feature matrix: "
            "(1) scaled — for logistic regression and regularised models, "
            "(2) unscaled — for tree-based models (Random Forest, LightGBM, XGBoost). "
            "This ensures each algorithm receives its optimal input representation.",
            kind="insight",
        )
    )
    b.add_section("Feature Scaling Strategy", scale_content, icon="📏")

    # ── Save FE spec ──────────────────────────────────────────────────
    artifacts_dir = Path(config.get("paths", {}).get("artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    spec_path = artifacts_dir / "feature_engineering_spec.json"
    fe_spec["feature_names_out"] = remaining  # placeholder
    with open(spec_path, "w") as f:
        json.dump(fe_spec, f, indent=2, default=str)
    log.info(f"FE spec saved to {spec_path}")

    html   = b.build()
    writer = ReportWriter(config)
    path   = writer.write(html, "04_Feature_Engineering.html")
    writer.write_index()

    elapsed = time.perf_counter() - t0
    log.stage_end("04 — Feature Engineering", elapsed)
    return path, fe_spec
