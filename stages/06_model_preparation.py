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

    n_rows   = loader.count_rows()
    mod_cfg  = config.get("modeling", {})
    seed     = mod_cfg.get("random_seed", 42)
    test_sz  = mod_cfg.get("test_size", 0.2)
    val_sz   = mod_cfg.get("validation_size", 0.1)
    cv_folds = mod_cfg.get("cv_folds", 5)
    target   = loader.target_col
    balance  = mod_cfg.get("class_balance_method", "none")

    # Compute split sizes
    n_test  = int(n_rows * test_sz)
    n_val   = int(n_rows * val_sz)
    n_train = n_rows - n_test - n_val

    # Target distribution
    tgt_dist = loader.get_target_distribution().to_dicts()
    pos_count = neg_count = 0
    for r in tgt_dist:
        if str(r["target_value"]) in ("1", "1.0"):
            pos_count = int(r["count"])
        else:
            neg_count = int(r["count"])
    imbalance_ratio = neg_count / max(pos_count, 1)

    b = HTMLReportBuilder(
        report_title   = "Model Preparation Report",
        stage_number   = 6,
        stage_subtitle = "Train/test split, CV strategy, class balancing, pipeline architecture",
        config         = config,
        n_rows         = n_rows,
        n_cols         = len(selected_features),
    )

    cards = [
        {"label": "Training Rows",    "value": f"{n_train:,}", "sub": f"({100*(1-test_sz-val_sz):.0f}%)"},
        {"label": "Validation Rows",  "value": f"{n_val:,}",   "sub": f"({val_sz*100:.0f}%)"},
        {"label": "Test Rows",         "value": f"{n_test:,}",  "sub": f"({test_sz*100:.0f}%)"},
        {"label": "CV Folds",          "value": str(cv_folds),  "sub": "stratified k-fold"},
        {"label": "Selected Features", "value": f"{len(selected_features):,}"},
        {"label": "Random Seed",       "value": str(seed)},
        {"label": "Class Imbalance",   "value": f"{imbalance_ratio:.1f}:1",
         "variant": "danger" if imbalance_ratio > 10 else "warning" if imbalance_ratio > 4 else "success"},
        {"label": "Balance Method",    "value": balance.upper() if balance != "none" else "None"},
    ]
    b.add_executive_summary(cards, narrative=(
        f"Model preparation configures the experimental framework for training and evaluating "
        f"{len(mod_cfg.get('models_to_train', []))} candidate models. "
        f"The dataset ({n_rows:,} rows) is split into training ({n_train:,}), "
        f"validation ({n_val:,}), and test ({n_test:,}) sets using stratified sampling "
        f"to maintain class proportions across all splits. "
        f"A {cv_folds}-fold stratified cross-validation is used for hyperparameter-independent "
        f"performance estimation. The test set is held out until final model evaluation "
        f"and is NEVER used during training or model selection."
    ))

    # Split strategy
    split_content = (
        b.p(
            f"Data splitting uses stratified random sampling with seed={seed} for reproducibility. "
            f"Stratification ensures that the {imbalance_ratio:.1f}:1 class imbalance ratio is "
            f"preserved identically in each split."
        )
        + b.table(
            [
                {"Split":     "Training",   "Rows": f"{n_train:,}",
                 "Pct": f"{(1-test_sz-val_sz)*100:.0f}%",
                 "Positive":  f"{int(pos_count*(1-test_sz-val_sz)):,}",
                 "Negative":  f"{int(neg_count*(1-test_sz-val_sz)):,}",
                 "Purpose":   "Model parameter fitting"},
                {"Split":     "Validation", "Rows": f"{n_val:,}",
                 "Pct": f"{val_sz*100:.0f}%",
                 "Positive":  f"{int(pos_count*val_sz):,}",
                 "Negative":  f"{int(neg_count*val_sz):,}",
                 "Purpose":   "Early stopping & hyperparameter tuning"},
                {"Split":     "Test",       "Rows": f"{n_test:,}",
                 "Pct": f"{test_sz*100:.0f}%",
                 "Positive":  f"{int(pos_count*test_sz):,}",
                 "Negative":  f"{int(neg_count*test_sz):,}",
                 "Purpose":   "Final unbiased performance evaluation"},
            ],
            caption="Train/Validation/Test Split",
            interpretation=(
                "The test set is strictly held out and never used for model selection. "
                "Validation is used for early stopping in LightGBM/XGBoost and as a "
                "criterion for selecting the best hyperparameter configuration."
            ),
        )
        + b.callout(
            f"<strong>Leakage Prevention:</strong> All preprocessing parameters "
            f"(medians, StandardScaler statistics, WoE mappings) are computed ONLY on the "
            f"training set and then applied to validation and test sets. "
            f"Feature selection was performed using cross-validation on training data only.",
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
        "n_val": n_val,
        "n_test": n_test,
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
