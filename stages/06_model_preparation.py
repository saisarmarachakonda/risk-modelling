"""
stages/06_model_preparation.py
Stage 06 — Model Preparation Report

Covers
------
* Train / validation / test split (stratified)
* Stratified K-Fold cross-validation setup
* Class imbalance handling options
* Pipeline architecture design
* Feature matrix preparation

Output
------
reports/06_Model_Preparation.html
artifacts/train_test_split_info.json
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from core.data_loader import DataLoader
from core.schema_detector import SchemaDetector
from core.logger import get_logger
from reporting.html_builder import HTMLReportBuilder
from reporting.report_writer import ReportWriter


def run(
    loader: DataLoader,
    schema: SchemaDetector,
    config: dict,
    selected_features: List[str],
) -> Tuple[Path, Dict]:
    log = get_logger("06_ModelPrep", config.get("paths", {}).get("logs_dir", "logs"))
    log.stage_start("06 — Model Preparation")
    t0 = time.perf_counter()
    mod_cfg  = config.get("modeling", {})
    seed     = mod_cfg.get("random_seed", 42)
    cv_folds = mod_cfg.get("cv_folds", 5)
    balance  = mod_cfg.get("class_balance_method", "none")

    # Read sample group distribution from DuckDB
    grp_col = config.get("data", {}).get("group_column", "sample_group")
    tgt_col = config.get("data", {}).get("target_column", "funded")
    
    group_stats = loader._con.execute(
        f"SELECT {grp_col}, COUNT(*), SUM(CAST({tgt_col} AS INTEGER)) "
        f"FROM input_data GROUP BY {grp_col}"
    ).fetchall()
    
    n_train = n_test = n_oot = 0
    pos_tr = pos_te = pos_oot = 0
    neg_tr = neg_te = neg_oot = 0
    
    for row in group_stats:
        grp, cnt, pos = row[0], int(row[1]), int(row[2])
        neg = cnt - pos
        if grp == "Train":
            n_train, pos_tr, neg_tr = cnt, pos, neg
        elif grp == "Test":
            n_test, pos_te, neg_te = cnt, pos, neg
        elif grp == "OOT":
            n_oot, pos_oot, neg_oot = cnt, pos, neg

    n_rows = n_train + n_test + n_oot
    pos_count = pos_tr + pos_te + pos_oot
    neg_count = neg_tr + neg_te + neg_oot
    imbalance_ratio = neg_tr / max(pos_tr, 1)

    b = HTMLReportBuilder(
        report_title   = "Model Preparation Report",
        stage_number   = 6,
        stage_subtitle = "Contest Train/Test/OOT partitioning and CV strategy",
        config         = config,
        n_rows         = n_rows,
        n_cols         = len(selected_features),
    )

    cards = [
        {"label": "Training Rows",    "value": f"{n_train:,}", "sub": f"({n_train/n_rows*100:.1f}%)"},
        {"label": "Test Rows (Val)",  "value": f"{n_test:,}",  "sub": f"({n_test/n_rows*100:.1f}%)"},
        {"label": "OOT Rows (Hold)",  "value": f"{n_oot:,}",   "sub": f"({n_oot/n_rows*100:.1f}%)"},
        {"label": "CV Folds",          "value": str(cv_folds),  "sub": "stratified k-fold"},
        {"label": "Selected Features", "value": f"{len(selected_features):,}"},
        {"label": "Random Seed",       "value": str(seed)},
        {"label": "Class Imbalance",   "value": f"{imbalance_ratio:.1f}:1",
         "variant": "danger" if imbalance_ratio > 10 else "warning" if imbalance_ratio > 4 else "success"},
        {"label": "Balance Method",    "value": balance.upper() if balance != "none" else "None"},
    ]
    b.add_executive_summary(cards, narrative=(
        f"Model preparation configures the experimental framework for the Machine Learning Contest. "
        f"The dataset is pre-partitioned into Training ({n_train:,} rows), Test ({n_test:,} rows), "
        f"and Out-of-Time (OOT, {n_oot:,} rows) groups. "
        f"Models are trained strictly on the Train set, tuned on the Test set, "
        f"and evaluated on the OOT set for final validation. "
        f"A {cv_folds}-fold stratified cross-validation on the Train set is used for robust local validation."
    ))

    # Split strategy
    split_content = (
        b.p(
            "Data partitioning is predefined in the competition source dataset to guarantee "
            "temporal validation using the Out-of-Time (OOT) hold-out."
        )
        + b.table(
            [
                {"Split": "Train", "Rows": f"{n_train:,}",
                 "Pct": f"{n_train/n_rows*100:.1f}%",
                 "Positive (funded=1)": f"{pos_tr:,}",
                 "Negative (funded=0)": f"{neg_tr:,}",
                 "Purpose": "Model parameter fitting & local training"},
                {"Split": "Test", "Rows": f"{n_test:,}",
                 "Pct": f"{n_test/n_rows*100:.1f}%",
                 "Positive (funded=1)": f"{pos_te:,}",
                 "Negative (funded=0)": f"{neg_te:,}",
                 "Purpose": "Model selection, tuning, and local validation"},
                {"Split": "OOT (Holdout)", "Rows": f"{n_oot:,}",
                 "Pct": f"{n_oot/n_rows*100:.1f}%",
                 "Positive (funded=1)": f"{pos_oot:,}",
                 "Negative (funded=0)": f"{neg_oot:,}",
                 "Purpose": "Final scoring by contest organizer (strictly hold-out)"},
            ],
            caption="Pre-defined Competition Group Partitioning",
            interpretation=(
                "The OOT set is strictly held out and never used for model parameter fitting or selection. "
                "The Test set is used to evaluate candidate models and select the best submission."
            ),
        )
        + b.callout(
            "<strong>Leakage Prevention:</strong> All preprocessing parameters (medians, OHE dummies, etc.) "
            "are computed ONLY on the Train set and applied unchanged to the Test and OOT sets. "
            "This ensures zero information from Test or OOT leaks into training.",
            kind="insight",
        )
    )
    b.add_section("Train/Validation/Test Split", split_content, icon="✂️")


    # CV strategy
    cv_content = (
        b.p(
            f"Stratified {cv_folds}-Fold Cross-Validation ensures each fold has the same "
            f"class imbalance ratio as the full training set. This is critical for imbalanced "
            f"datasets where random splits could produce folds with very few positive samples."
        )
        + b.table(
            [{"Property": k, "Value": v} for k, v in {
                "CV Strategy":         f"Stratified {cv_folds}-Fold",
                "Primary Metric":      mod_cfg.get("primary_metric", "roc_auc").upper(),
                "Secondary Metrics":   "F1, Precision, Recall, PR-AUC, Brier Score",
                "Fold Shuffle":        "Yes (random_state=" + str(seed) + ")",
                "Positive per fold":   f"~{pos_count//cv_folds:,}",
                "Negative per fold":   f"~{neg_count//cv_folds:,}",
            }.items()],
            caption="Cross-Validation Configuration",
        )
    )
    b.add_section("Cross-Validation Strategy", cv_content, icon="🔄")

    # Class balance
    balance_options = b.table(
        [
            {"Method":     "None (baseline)",
             "Description":"Use raw imbalanced data. Apply class_weight='balanced' to model.",
             "Pros":        "Simple, no data modification, fastest",
             "Cons":        "Model may underfit minority class"},
            {"Method":     "Class Weights",
             "Description":"Increase loss weight for minority class proportionally.",
             "Pros":        "No sampling, works with any model",
             "Cons":        "May not be sufficient for extreme imbalance"},
            {"Method":     "Undersampling",
             "Description":"Randomly remove majority class samples.",
             "Pros":        "Reduces training time",
             "Cons":        "Loses potentially informative majority data"},
            {"Method":     "SMOTE",
             "Description":"Synthetic Minority Oversampling Technique.",
             "Pros":        "Preserves minority class distribution",
             "Cons":        "Can introduce noise; computationally expensive for 2M rows"},
            {"Method":     "Threshold Tuning",
             "Description":"Keep default training, adjust decision threshold post-hoc.",
             "Pros":        "Non-invasive, optimal for production",
             "Cons":        "Requires calibration set"},
        ],
        caption="Class Imbalance Handling Options",
        interpretation=(
            f"Given a {imbalance_ratio:.1f}:1 imbalance ratio, the pipeline uses "
            f"class_weight='balanced' for all sklearn models by default. "
            f"LightGBM and XGBoost use scale_pos_weight={int(imbalance_ratio)}."
        ),
    )
    b.add_section("Class Imbalance Handling", balance_options, icon="⚖️")

    # Pipeline architecture
    pipeline_content = b.card(
        "ML Pipeline Architecture",
        "<ol style='font-size:14px;line-height:2;'>"
        "<li><strong>Data Loading</strong> — DuckDB lazy scan (never full data in RAM)</li>"
        "<li><strong>Schema Detection</strong> — Auto-classify 4k+ columns</li>"
        "<li><strong>Feature Engineering</strong> — Imputation → Encoding → Scaling</li>"
        "<li><strong>Feature Selection</strong> — Filter + LR + Tree consensus ranking</li>"
        "<li><strong>Train/Test Split</strong> — Stratified, seed-fixed</li>"
        "<li><strong>Model Training</strong> — All candidates with class weighting</li>"
        "<li><strong>CV Evaluation</strong> — Stratified 5-fold, multiple metrics</li>"
        "<li><strong>Model Selection</strong> — Ranked by primary metric (ROC AUC)</li>"
        "<li><strong>Best Model Deep-Dive</strong> — ROC, PR, KS, Calibration, SHAP</li>"
        "<li><strong>Report Generation</strong> — 10 static HTML reports</li>"
        "</ol>"
    )
    b.add_section("Pipeline Architecture", pipeline_content, icon="🏗️")

    prep_info = {
        "n_train": n_train,
        "n_val": n_test,
        "n_test": n_oot,
        "n_features": len(selected_features),
        "seed": seed,
        "cv_folds": cv_folds,
        "imbalance_ratio": imbalance_ratio,
        "selected_features": selected_features,
    }
    artifacts_dir = Path(config.get("paths", {}).get("artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / "train_test_split_info.json", "w") as f:
        json.dump(prep_info, f, indent=2, default=str)

    html   = b.build()
    writer = ReportWriter(config)
    path   = writer.write(html, "06_Model_Preparation.html")
    writer.write_index()

    elapsed = time.perf_counter() - t0
    log.stage_end("06 — Model Preparation", elapsed)
    return path, prep_info
