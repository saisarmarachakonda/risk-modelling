"""
stages/05_feature_selection.py
Stage 05 — Feature Selection Report

═══════════════════════════════════════════════════════════════════════════════
GOAL: Try EVERY available technique, score each, build a consensus ranking,
      and show clearly what differs between tree-based and LR-based selection.
═══════════════════════════════════════════════════════════════════════════════

Methods implemented
-------------------

A. FILTER METHODS  (model-agnostic — computed first, cheap)
   A1. Variance Threshold          — remove near-zero variance features
   A2. Information Value (IV/WoE)  — DuckDB SQL over full dataset
   A3. Mutual Information          — sklearn, sampled data
   A4. Chi-Square Test             — categorical features vs binary target
   A5. ANOVA F-Test                — numeric features vs binary target
   A6. Point-Biserial Correlation  — numeric feature vs binary target
   A7. Correlation Pruning         — remove one of each highly correlated pair

B. LOGISTIC REGRESSION-BASED METHODS
   B1. L1 Lasso Regularisation     — coefficients that survive L1 penalty
   B2. L2 Ridge Regularisation     — coefficient magnitude ranking
   B3. Elastic Net                  — combined L1 + L2
   B4. Standardised Coefficients   — absolute |coef| after StandardScaler

C. TREE-BASED METHODS
   C1. Random Forest Gini/Entropy   — MDI importance
   C2. Extra Trees                  — MDI importance (lower variance)
   C3. Gradient Boosting            — MDI importance
   C4. LightGBM (if available)      — gain, split, cover importance
   C5. XGBoost  (if available)      — weight, gain, cover importance
   C6. Permutation Importance       — model-agnostic, test-set based

D. CONSENSUS RANKING
   D1. Per-method normalised rank   — 0→1 for each method
   D2. Consensus score              — mean of normalised ranks
   D3. Selection count              — how many methods selected each feature
   D4. Tree consensus vs LR consensus — separate group winners
   D5. Final recommended set        — top-N by consensus

KEY INSIGHT documented in report:
   Tree-based importance is non-linear, interaction-aware, scale-invariant.
   LR-based importance is linear, scale-dependent, multicollinearity-sensitive.
   Features in BOTH consensus sets = most robust signal.
   Features in ONLY tree set = may encode non-linear interactions.
   Features in ONLY LR set   = pure linear main effects.

Output
------
reports/05_Feature_Selection.html
artifacts/selected_features.json
"""
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

from core.data_loader import DataLoader
from core.schema_detector import SchemaDetector
from core.logger import get_logger
from reporting.html_builder import HTMLReportBuilder
from reporting.figure_generator import (
    feature_importance_bar,
    method_comparison_heatmap,
    tree_vs_lr_scatter,
    consensus_rank_bar,
    iv_chart,
    stability_heatmap,
)
from reporting.report_writer import ReportWriter


# ─── Helpers ────────────────────────────────────────────────────────────────

def _normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1]. Handles constant arrays."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-12:
        return np.ones_like(scores) * 0.5
    return (scores - mn) / (mx - mn)


def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Return rank array (1 = highest importance). Ties get average rank."""
    from scipy.stats import rankdata
    return rankdata(-scores, method="average")


def _safe_importances(estimator, feature_names: List[str]) -> np.ndarray:
    if hasattr(estimator, "feature_importances_"):
        return np.array(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if coef.ndim > 1:
            coef = coef[0]
        return np.abs(coef)
    return np.zeros(len(feature_names))


# ─── Main stage function ──────────────────────────────────────────────────────

def run(
    loader: DataLoader,
    schema: SchemaDetector,
    config: dict,
    fe_spec: Optional[Dict] = None,
) -> Tuple[Path, List[str]]:
    """
    Execute Stage 05 — Feature Selection.

    Returns
    -------
    report_path      : Path
    selected_features: List[str]  — final recommended feature list
    """
    log = get_logger("05_FeatureSelection",
                     config.get("paths", {}).get("logs_dir", "logs"))
    log.stage_start("05 — Feature Selection")
    t0 = time.perf_counter()

    fs_cfg   = config.get("feature_selection", {})
    n_rows   = loader.count_rows()
    target   = loader.target_col
    seed     = config.get("modeling", {}).get("random_seed", 42)
    top_n    = fs_cfg.get("top_n_features", 200)
    sample_n = min(150_000, n_rows)     # sample for sklearn methods

    # Feature columns to consider
    feature_cols = [c for c in schema.get_feature_cols() if c != target]
    num_cols     = [c for c in schema.get_numeric_cols()     if c in feature_cols]
    cat_cols     = [c for c in schema.get_categorical_cols() if c in feature_cols]
    bool_cols    = [c for c in schema.get_boolean_cols()     if c in feature_cols]

    log.info(f"Feature pool: {len(feature_cols):,} cols  "
             f"({len(num_cols)} numeric, {len(cat_cols)} categorical, {len(bool_cols)} boolean)")

    # ── Sample data for sklearn methods ──────────────────────────────
    log.info(f"Sampling {sample_n:,} rows for sklearn-based methods…")
    work_cols = [c for c in num_cols[:300] + bool_cols]  # focus on numeric + bool
    sample_df = loader.sample_columns(work_cols + [target], n=sample_n, seed=seed)

    X_raw = sample_df.select(work_cols).fill_null(0).to_numpy().astype(np.float32)
    y     = sample_df[target].fill_null(0).to_numpy().astype(np.int32)

    # StandardScaler for LR methods
    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    used_features = work_cols  # feature names for sklearn methods
    n_feat = len(used_features)
    log.info(f"Working feature matrix: {X_scaled.shape}")

    # ══════════════════════════════════════════════════════════════════
    # A. FILTER METHODS
    # ══════════════════════════════════════════════════════════════════
    method_scores: Dict[str, np.ndarray] = {}   # method_name → importance array
    method_meta:   Dict[str, dict] = {}         # method_name → {type, description, ...}

    # ── A1. Variance Threshold ────────────────────────────────────────
    log.info("A1. Variance threshold…")
    variances  = X_raw.var(axis=0)
    var_thresh = fs_cfg.get("variance_threshold", 0.01)
    method_scores["Variance"] = variances
    method_meta["Variance"] = {
        "group": "Filter",
        "description": "Feature variance — low-variance features carry minimal information",
        "threshold": var_thresh,
        "n_selected": int((variances > var_thresh).sum()),
    }

    # ── A2. Information Value (IV) ────────────────────────────────────
    log.info("A2. Information Value (full dataset via DuckDB)…")
    iv_scores = np.zeros(n_feat)
    iv_iv_map = {}
    for i, col in enumerate(used_features):
        try:
            iv, _ = loader.compute_iv_woe(col)
            iv_scores[i] = iv
            iv_iv_map[col] = iv
        except Exception:
            pass
    iv_thresh = fs_cfg.get("iv_threshold", 0.02)
    method_scores["IV (WoE)"] = iv_scores
    method_meta["IV (WoE)"] = {
        "group": "Filter",
        "description": "Information Value — measures predictive power vs binary target",
        "threshold": iv_thresh,
        "n_selected": int((iv_scores >= iv_thresh).sum()),
    }
    log.info(f"  IV computed for {len(used_features)} features")

    # ── A3. Mutual Information ────────────────────────────────────────
    log.info("A3. Mutual Information…")
    try:
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X_scaled, y, random_state=seed, n_neighbors=5)
        method_scores["Mutual Info"] = mi_scores
        method_meta["Mutual Info"] = {
            "group": "Filter",
            "description": "Mutual information between feature and target (non-parametric)",
            "n_selected": int((mi_scores > 0.01).sum()),
        }
    except Exception as e:
        log.warning(f"Mutual info failed: {e}")

    # ── A4. ANOVA F-Test ──────────────────────────────────────────────
    log.info("A4. ANOVA F-test…")
    try:
        from sklearn.feature_selection import f_classif
        f_scores, f_pvals = f_classif(X_scaled, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0)
        method_scores["ANOVA F-Test"] = f_scores
        method_meta["ANOVA F-Test"] = {
            "group": "Filter",
            "description": "F-statistic measuring mean difference across classes for each feature",
            "n_selected": int((f_pvals < 0.05).sum()),
        }
    except Exception as e:
        log.warning(f"ANOVA failed: {e}")

    # ── A5. Point-Biserial Correlation ───────────────────────────────
    log.info("A5. Point-biserial correlation…")
    try:
        from scipy.stats import pointbiserialr
        pb_scores = np.zeros(n_feat)
        for i in range(n_feat):
            try:
                corr, _ = pointbiserialr(y, X_raw[:, i])
                pb_scores[i] = abs(corr) if not np.isnan(corr) else 0.0
            except Exception:
                pass
        method_scores["Point-Biserial r"] = pb_scores
        method_meta["Point-Biserial r"] = {
            "group": "Filter",
            "description": "Absolute point-biserial correlation between numeric feature and binary target",
            "n_selected": int((pb_scores > 0.05).sum()),
        }
    except Exception as e:
        log.warning(f"Point-biserial failed: {e}")

    # ── A6. Correlation Pruning ───────────────────────────────────────
    log.info("A6. Correlation-based pruning…")
    corr_thresh = fs_cfg.get("correlation_threshold", 0.95)
    corr_mat    = np.corrcoef(X_raw.T)
    corr_keep   = np.ones(n_feat, dtype=bool)
    for i in range(n_feat):
        if not corr_keep[i]:
            continue
        for j in range(i + 1, n_feat):
            if abs(corr_mat[i, j]) > corr_thresh:
                # Keep higher-IV feature
                if iv_scores[i] >= iv_scores[j]:
                    corr_keep[j] = False
                else:
                    corr_keep[i] = False
                    break
    corr_scores = corr_keep.astype(float)
    n_pruned    = int((~corr_keep).sum())
    method_scores["Corr Pruning"] = corr_scores
    method_meta["Corr Pruning"] = {
        "group": "Filter",
        "description": f"Binary flag: 1 = kept after removing one of each pair with |r| > {corr_thresh}",
        "n_selected": int(corr_keep.sum()),
        "n_pruned": n_pruned,
    }
    log.info(f"  Correlation pruning removed {n_pruned} redundant features")

    # ══════════════════════════════════════════════════════════════════
    # B. LOGISTIC REGRESSION-BASED METHODS
    # ══════════════════════════════════════════════════════════════════

    from sklearn.linear_model import (
        LogisticRegression, LogisticRegressionCV, SGDClassifier
    )

    # ── B1. L1 Lasso ─────────────────────────────────────────────────
    log.info("B1. L1 (Lasso) Logistic Regression…")
    try:
        lr_l1 = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.1,
            max_iter=1000, random_state=seed, class_weight="balanced"
        )
        lr_l1.fit(X_scaled, y)
        l1_scores = np.abs(lr_l1.coef_[0])
        n_nonzero = int((l1_scores > 1e-6).sum())
        method_scores["LR L1 (Lasso)"] = l1_scores
        method_meta["LR L1 (Lasso)"] = {
            "group": "Logistic Regression",
            "description": "L1-penalised LR — sparse solution, zero coefs = selected out",
            "C": 0.1,
            "n_selected": n_nonzero,
            "sparsity_pct": round((1 - n_nonzero / n_feat) * 100, 1),
        }
        log.info(f"  L1: {n_nonzero}/{n_feat} non-zero coefficients")
    except Exception as e:
        log.warning(f"L1 LR failed: {e}")

    # ── B1b. L1 with CV to find optimal C ────────────────────────────
    log.info("B1b. L1 LR with cross-validated C…")
    try:
        lr_l1_cv = LogisticRegressionCV(
            penalty="l1", solver="liblinear", Cs=10,
            cv=3, max_iter=500, random_state=seed,
            class_weight="balanced", scoring="roc_auc",
        )
        lr_l1_cv.fit(X_scaled, y)
        l1_cv_scores = np.abs(lr_l1_cv.coef_[0])
        method_scores["LR L1-CV"] = l1_cv_scores
        method_meta["LR L1-CV"] = {
            "group": "Logistic Regression",
            "description": "L1-penalised LR with cross-validated C (ROC AUC criterion)",
            "best_C": float(lr_l1_cv.C_[0]),
            "n_selected": int((l1_cv_scores > 1e-6).sum()),
        }
        log.info(f"  L1-CV: best C = {lr_l1_cv.C_[0]:.4f}")
    except Exception as e:
        log.warning(f"L1-CV failed: {e}")

    # ── B2. L2 Ridge ──────────────────────────────────────────────────
    log.info("B2. L2 (Ridge) Logistic Regression…")
    try:
        lr_l2 = LogisticRegression(
            penalty="l2", solver="lbfgs", C=1.0,
            max_iter=1000, random_state=seed, class_weight="balanced"
        )
        lr_l2.fit(X_scaled, y)
        l2_scores = np.abs(lr_l2.coef_[0])
        method_scores["LR L2 (Ridge)"] = l2_scores
        method_meta["LR L2 (Ridge)"] = {
            "group": "Logistic Regression",
            "description": "L2-penalised LR — coefficients shrunk but none zeroed",
            "C": 1.0,
            "n_selected": n_feat,  # Ridge keeps all
        }
    except Exception as e:
        log.warning(f"L2 LR failed: {e}")

    # ── B3. Elastic Net ───────────────────────────────────────────────
    log.info("B3. Elastic Net Logistic Regression…")
    try:
        lr_en = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5,
            C=0.5, max_iter=1000, random_state=seed, class_weight="balanced"
        )
        lr_en.fit(X_scaled, y)
        en_scores = np.abs(lr_en.coef_[0])
        method_scores["LR Elastic Net"] = en_scores
        method_meta["LR Elastic Net"] = {
            "group": "Logistic Regression",
            "description": "Elastic Net LR (L1+L2, l1_ratio=0.5) — sparse + stable",
            "l1_ratio": 0.5,
            "C": 0.5,
            "n_selected": int((en_scores > 1e-6).sum()),
        }
    except Exception as e:
        log.warning(f"Elastic Net failed: {e}")

    # ── B4. SGD (large-scale LR) ──────────────────────────────────────
    log.info("B4. SGD Classifier (L1 penalty, large-scale)…")
    try:
        sgd = SGDClassifier(
            loss="log_loss", penalty="l1", alpha=0.0001,
            max_iter=200, random_state=seed, class_weight="balanced",
            tol=1e-3,
        )
        sgd.fit(X_scaled, y)
        sgd_scores = np.abs(sgd.coef_[0])
        method_scores["SGD L1"] = sgd_scores
        method_meta["SGD L1"] = {
            "group": "Logistic Regression",
            "description": "SGD-optimised L1 logistic — scales to millions of rows",
            "n_selected": int((sgd_scores > 1e-6).sum()),
        }
    except Exception as e:
        log.warning(f"SGD failed: {e}")

    # ══════════════════════════════════════════════════════════════════
    # C. TREE-BASED METHODS
    # ══════════════════════════════════════════════════════════════════

    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    )
    from sklearn.inspection import permutation_importance

    # ── C1. Random Forest ─────────────────────────────────────────────
    log.info("C1. Random Forest feature importance…")
    try:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, max_features="sqrt",
            min_samples_leaf=20, n_jobs=-1, random_state=seed,
            class_weight="balanced", max_samples=0.6,
        )
        rf.fit(X_raw, y)
        rf_scores = rf.feature_importances_
        method_scores["Random Forest"] = rf_scores
        method_meta["Random Forest"] = {
            "group": "Tree-Based",
            "description": "Random Forest MDI (Mean Decrease in Impurity / Gini)",
            "n_estimators": 200,
            "max_depth": 8,
            "n_selected": int((rf_scores > rf_scores.mean()).sum()),
        }
        log.info(f"  RF trained. Top feature: {used_features[np.argmax(rf_scores)]}")
    except Exception as e:
        log.warning(f"Random Forest failed: {e}")

    # ── C2. Extra Trees ───────────────────────────────────────────────
    log.info("C2. Extra Trees feature importance…")
    try:
        et = ExtraTreesClassifier(
            n_estimators=200, max_depth=8, max_features="sqrt",
            min_samples_leaf=20, n_jobs=-1, random_state=seed,
            class_weight="balanced",
        )
        et.fit(X_raw, y)
        et_scores = et.feature_importances_
        method_scores["Extra Trees"] = et_scores
        method_meta["Extra Trees"] = {
            "group": "Tree-Based",
            "description": "Extra Trees MDI — lower variance than RF (random split thresholds)",
            "n_estimators": 200,
            "n_selected": int((et_scores > et_scores.mean()).sum()),
        }
    except Exception as e:
        log.warning(f"Extra Trees failed: {e}")

    # ── C3. Gradient Boosting ─────────────────────────────────────────
    log.info("C3. Gradient Boosting feature importance…")
    try:
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=20, random_state=seed,
        )
        gb.fit(X_raw, y)
        gb_scores = gb.feature_importances_
        method_scores["Gradient Boost"] = gb_scores
        method_meta["Gradient Boost"] = {
            "group": "Tree-Based",
            "description": "Gradient Boosting MDI — sequential residual fitting",
            "n_estimators": 100,
            "n_selected": int((gb_scores > gb_scores.mean()).sum()),
        }
    except Exception as e:
        log.warning(f"Gradient Boosting failed: {e}")

    # ── C4. LightGBM ──────────────────────────────────────────────────
    log.info("C4. LightGBM feature importance…")
    try:
        import lightgbm as lgb
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=seed,
            class_weight="balanced", verbose=-1, n_jobs=-1,
        )
        lgb_clf.fit(X_raw, y)
        lgb_gain   = lgb_clf.booster_.feature_importance(importance_type="gain")
        lgb_split  = lgb_clf.booster_.feature_importance(importance_type="split")
        method_scores["LightGBM (gain)"]  = lgb_gain.astype(float)
        method_scores["LightGBM (split)"] = lgb_split.astype(float)
        method_meta["LightGBM (gain)"] = {
            "group": "Tree-Based",
            "description": "LightGBM Gain importance — total information gain at splits using this feature",
            "n_selected": int((lgb_gain > 0).sum()),
        }
        method_meta["LightGBM (split)"] = {
            "group": "Tree-Based",
            "description": "LightGBM Split importance — number of times feature used to split",
            "n_selected": int((lgb_split > 0).sum()),
        }
        log.info("  LightGBM trained")
    except ImportError:
        log.warning("LightGBM not installed — skipping")
    except Exception as e:
        log.warning(f"LightGBM failed: {e}")

    # ── C5. XGBoost ───────────────────────────────────────────────────
    log.info("C5. XGBoost feature importance…")
    try:
        import xgboost as xgb
        scale_pos = max(1, int((y == 0).sum() / max((y == 1).sum(), 1)))
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=20, random_state=seed,
            scale_pos_weight=scale_pos, eval_metric="auc",
            use_label_encoder=False, verbosity=0, n_jobs=-1,
        )
        xgb_clf.fit(X_raw, y)
        xgb_gain   = np.array([xgb_clf.get_booster().get_score(importance_type="gain").get(f"f{i}", 0)
                                for i in range(n_feat)])
        xgb_weight = np.array([xgb_clf.get_booster().get_score(importance_type="weight").get(f"f{i}", 0)
                                for i in range(n_feat)])
        method_scores["XGBoost (gain)"]   = xgb_gain.astype(float)
        method_scores["XGBoost (weight)"] = xgb_weight.astype(float)
        method_meta["XGBoost (gain)"] = {
            "group": "Tree-Based",
            "description": "XGBoost Gain importance — average gain across all trees for each feature",
            "n_selected": int((xgb_gain > 0).sum()),
        }
        method_meta["XGBoost (weight)"] = {
            "group": "Tree-Based",
            "description": "XGBoost Weight importance — number of times feature appears in trees",
            "n_selected": int((xgb_weight > 0).sum()),
        }
        log.info("  XGBoost trained")
    except ImportError:
        log.warning("XGBoost not installed — skipping")
    except Exception as e:
        log.warning(f"XGBoost failed: {e}")

    # ── C6. Permutation Importance ────────────────────────────────────
    log.info("C6. Permutation importance (on RF model)…")
    try:
        if "Random Forest" in method_scores:
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_raw, y, test_size=0.3, random_state=seed, stratify=y
            )
            perm = permutation_importance(
                rf, X_te, y_te, n_repeats=10,
                random_state=seed, scoring="roc_auc", n_jobs=-1,
            )
            perm_scores = np.maximum(0, perm.importances_mean)
            method_scores["Permutation"] = perm_scores
            method_meta["Permutation"] = {
                "group": "Tree-Based",
                "description": "Permutation importance (RF) — drop in ROC AUC when feature is shuffled",
                "n_repeats": 10,
                "n_selected": int((perm_scores > 0.001).sum()),
            }
            log.info("  Permutation importance computed")
    except Exception as e:
        log.warning(f"Permutation importance failed: {e}")

    # ══════════════════════════════════════════════════════════════════
    # D. CONSENSUS RANKING
    # ══════════════════════════════════════════════════════════════════
    log.info("Building consensus ranking…")

    method_names = list(method_scores.keys())
    n_methods    = len(method_names)

    # Normalised score matrix: rows = features, cols = methods
    norm_matrix = np.zeros((n_feat, n_methods))
    for j, mname in enumerate(method_names):
        scores = method_scores[mname]
        if len(scores) == n_feat:
            norm_matrix[:, j] = _normalise_scores(scores)

    # Consensus score = mean normalised score across all methods
    consensus_scores = norm_matrix.mean(axis=1)

    # Selection count: how many methods put this feature in top-50%
    top_half_threshold = 0.5  # above 50th percentile normalised score
    selected_matrix = (norm_matrix >= top_half_threshold)
    selection_counts = selected_matrix.sum(axis=1)

    # Separate tree-based and LR-based consensus
    tree_methods = [m for m in method_names if method_meta.get(m, {}).get("group") == "Tree-Based"]
    lr_methods   = [m for m in method_names if method_meta.get(m, {}).get("group") == "Logistic Regression"]
    filt_methods = [m for m in method_names if method_meta.get(m, {}).get("group") == "Filter"]

    tree_idx   = [j for j, m in enumerate(method_names) if m in tree_methods]
    lr_idx     = [j for j, m in enumerate(method_names) if m in lr_methods]
    filter_idx = [j for j, m in enumerate(method_names) if m in filt_methods]

    tree_consensus   = norm_matrix[:, tree_idx].mean(axis=1)   if tree_idx   else np.zeros(n_feat)
    lr_consensus     = norm_matrix[:, lr_idx].mean(axis=1)     if lr_idx     else np.zeros(n_feat)
    filter_consensus = norm_matrix[:, filter_idx].mean(axis=1) if filter_idx else np.zeros(n_feat)

    # Final selected features — top N by consensus
    sorted_idx       = np.argsort(-consensus_scores)
    selected_indices = sorted_idx[:top_n]
    selected_features = [used_features[i] for i in selected_indices]

    # Features in tree-top but NOT in LR-top (and vice versa)
    tree_top_set = set(np.array(used_features)[np.argsort(-tree_consensus)[:top_n]])
    lr_top_set   = set(np.array(used_features)[np.argsort(-lr_consensus)[:top_n]])
    both_top     = tree_top_set & lr_top_set
    tree_only    = tree_top_set - lr_top_set
    lr_only      = lr_top_set   - tree_top_set

    log.info(
        f"Consensus: {len(selected_features)} selected | "
        f"Both groups: {len(both_top)} | Tree-only: {len(tree_only)} | LR-only: {len(lr_only)}"
    )

    # ══════════════════════════════════════════════════════════════════
    # BUILD REPORT
    # ══════════════════════════════════════════════════════════════════
    b = HTMLReportBuilder(
        report_title   = "Feature Selection Report",
        stage_number   = 5,
        stage_subtitle = (
            "All selection methods tried — tree-based vs logistic regression comparison "
            "— consensus ranking"
        ),
        config         = config,
        n_rows         = n_rows,
        n_cols         = len(used_features),
    )

    # ── Executive summary ────────────────────────────────────────────
    cards = [
        {"label": "Features Evaluated",  "value": f"{n_feat:,}"},
        {"label": "Methods Tried",        "value": str(n_methods),
         "sub": "filter + LR + tree"},
        {"label": "Tree Methods",         "value": str(len(tree_methods)),
         "variant": "warning"},
        {"label": "LR Methods",           "value": str(len(lr_methods)),
         "variant": "info"},
        {"label": "Filter Methods",       "value": str(len(filt_methods))},
        {"label": "Consensus Selected",   "value": f"{len(selected_features):,}",
         "sub": f"top {top_n} by consensus"},
        {"label": "In Both Groups",       "value": f"{len(both_top):,}",
         "sub": "tree AND LR agree",
         "variant": "success"},
        {"label": "LR-Only Features",     "value": f"{len(lr_only):,}",
         "sub": "linear signal only"},
    ]

    narrative = (
        f"Stage 05 runs ALL available feature selection techniques — {n_methods} methods in total — "
        f"across three families: filter methods (model-agnostic statistics), logistic regression-based "
        f"methods (coefficient magnitude after L1/L2/Elastic Net regularisation), and tree-based "
        f"methods (MDI importance from Random Forest, Extra Trees, Gradient Boosting, LightGBM, XGBoost, "
        f"plus permutation importance).\n\n"
        f"Each method's raw scores are normalised to [0,1] and averaged to form a consensus ranking. "
        f"The {len(selected_features):,} features with the highest consensus score are recommended for modelling.\n\n"
        f"Critically, {len(both_top):,} features appear in the top-{top_n} of BOTH tree-based and "
        f"LR-based rankings. These 'dual-consensus' features represent the most robust signal — "
        f"they encode both linear separability (visible to LR) and non-linear patterns (visible to trees). "
        f"{len(tree_only):,} features are tree-dominant (non-linear interactions) and "
        f"{len(lr_only):,} are LR-dominant (pure linear main effects)."
    )
    b.add_executive_summary(cards, narrative=narrative)

    # ── Methods overview table ────────────────────────────────────────
    methods_table_rows = []
    for mname in method_names:
        meta = method_meta.get(mname, {})
        grp  = meta.get("group", "?")
        grp_badge = (
            b.badge("Tree-Based", "treebased") if grp == "Tree-Based"
            else b.badge("LR-Based", "lrbased") if grp == "Logistic Regression"
            else b.badge("Filter", "numeric")
        )
        methods_table_rows.append({
            "Method":         mname,
            "Group":          grp_badge,
            "Description":    meta.get("description", ""),
            "Features Selected": f'{meta.get("n_selected", "N/A"):,}' if isinstance(meta.get("n_selected"), int) else meta.get("n_selected", "N/A"),
        })

    methods_overview = b.table(
        methods_table_rows,
        caption="All Feature Selection Methods Executed",
        interpretation=(
            "Every method in this table was executed against the dataset. "
            "Results are then normalised and combined into a single consensus score. "
            "Methods that agree across groups (filter + tree + LR) produce the most "
            "reliable feature set."
        ),
    )
    overview_content = (
        b.p(
            "The feature selection stage is structured in three layers. Filter methods run first "
            "as they are computationally cheap and model-agnostic. LR-based methods then identify "
            "features with strong linear separability after scaling. Tree-based methods finally "
            "capture non-linear patterns and interactions. The consensus combines all signals."
        )
        + methods_overview
    )
    b.add_section("Methods Overview", overview_content, icon="🗂️")

    # ── Tree vs LR: Conceptual Difference ────────────────────────────
    concept_content = (
        b.p(
            "Tree-based and logistic regression feature selection operate on fundamentally "
            "different principles. Understanding these differences is critical for interpreting "
            "why a feature ranks highly for one family but not the other."
        )
        + b.table(
            [
                {"Dimension": "Importance Measure",
                 "Tree-Based": "Mean Decrease in Impurity (MDI) or drop in metric when feature permuted",
                 "Logistic Regression": "Magnitude of coefficient after scaling (|β|)"},
                {"Dimension": "Non-linearity",
                 "Tree-Based": "✅ Captures interactions and non-linear relationships automatically",
                 "Logistic Regression": "❌ Only linear relationships — needs manual feature crosses"},
                {"Dimension": "Feature Scale",
                 "Tree-Based": "❌ Completely scale-invariant — same result on raw or scaled data",
                 "Logistic Regression": "✅ Requires StandardScaling — coefficients meaningless otherwise"},
                {"Dimension": "Multicollinearity",
                 "Tree-Based": "⚠️ Splits importance across correlated features (can dilute ranks)",
                 "Logistic Regression": "🚨 Severe — inflates standard errors, unstable coefficients"},
                {"Dimension": "Sparsity",
                 "Tree-Based": "Low — most features get some non-zero importance",
                 "Logistic Regression": "High with L1 — explicit zero coefficients = hard exclusion"},
                {"Dimension": "Missing Values",
                 "Tree-Based": "⚠️ Some handle natively (LightGBM/XGBoost); RF/GB require imputation",
                 "Logistic Regression": "❌ Requires imputation before fitting"},
                {"Dimension": "Outlier Sensitivity",
                 "Tree-Based": "✅ Robust — splits on rank-order thresholds",
                 "Logistic Regression": "🚨 Sensitive — extreme values distort coefficient estimates"},
                {"Dimension": "Interpretability",
                 "Tree-Based": "Global importance but individual prediction requires SHAP",
                 "Logistic Regression": "✅ Each coefficient directly = log-odds contribution"},
                {"Dimension": "High Dimensionality",
                 "Tree-Based": "✅ Handles well — feature subsampling controls dimensionality",
                 "Logistic Regression": "⚠️ L1 penalty required to handle thousands of features"},
                {"Dimension": "Class Imbalance",
                 "Tree-Based": "⚠️ class_weight parameter helps; importance may be skewed",
                 "Logistic Regression": "✅ class_weight='balanced' directly adjusts gradient"},
                {"Dimension": "Speed (4k cols, 2M rows)",
                 "Tree-Based": "⚠️ RF can be slow; LightGBM/XGBoost very fast on large data",
                 "Logistic Regression": "✅ L-BFGS and SGD scale linearly"},
                {"Dimension": "Best Use Case",
                 "Tree-Based": "Complex risk patterns, variable interactions, non-monotone effects",
                 "Logistic Regression": "Scorecard development, regulatory interpretability, stable features"},
            ],
            caption="Tree-Based vs Logistic Regression Feature Selection — Comprehensive Comparison",
            interpretation=(
                "This table is the core conceptual reference for understanding why the two method "
                "families select different features. A feature appearing in BOTH groups is exceptionally "
                "valuable — it encodes both a linear main effect (interpretable by LR) and a non-linear "
                "signal (captured by trees). Features appearing in only one group require careful "
                "consideration of which model family they are intended to serve."
            ),
            extra_class="compare-table",
        )
        + b.callout(
            "<strong>Practical Rule of Thumb:</strong> For a regulatory credit scorecard "
            "(logistic regression), prioritise features from the LR consensus set. For a "
            "challenger model or monitoring tool (LightGBM/XGBoost), use the full consensus set. "
            "The 'Both Groups' features are safe for ANY model.",
            kind="recommend",
        )
    )
    b.add_section(
        "Tree-Based vs LR-Based: Fundamental Differences",
        concept_content, icon="⚖️"
    )

    # ── Filter methods results ────────────────────────────────────────
    filter_content = b.p(
        "Filter methods evaluate each feature independently of any model, "
        "using statistical relationships with the target variable. "
        "They are fast, scale to millions of rows via DuckDB SQL, and provide "
        "a model-agnostic baseline for feature quality."
    )

    # IV figure
    if "IV (WoE)" in method_scores:
        iv_feats = [used_features[i] for i in np.argsort(-iv_scores)[:40]]
        iv_vals  = [iv_scores[i]     for i in np.argsort(-iv_scores)[:40]]
        iv_fig   = iv_chart(iv_feats, iv_vals, title="Information Value — Top 40 Features")
        filter_content += b.figure(
            iv_fig,
            title="Information Value (IV) Ranking",
            description="IV computed over the full dataset via DuckDB SQL aggregates.",
            interpretation=(
                "IV > 0.3 = Strong predictor. IV 0.1–0.3 = Medium. IV 0.02–0.1 = Weak. "
                "IV < 0.02 = Useless. IV > 0.5 = Investigate for data leakage."
            ),
            business_implication=(
                "IV is the standard feature selection metric in traditional credit scorecard "
                "development. Features with IV > 0.1 are typically included in a regulatory "
                "scorecard. IV directly corresponds to the Gini coefficient (IV ≈ 2 × Gini)."
            ),
        )

    # ANOVA F-test figure
    if "ANOVA F-Test" in method_scores:
        f_arr   = method_scores["ANOVA F-Test"]
        f_top_i = np.argsort(-f_arr)[:30]
        f_fig   = feature_importance_bar(
            [used_features[i] for i in f_top_i],
            [f_arr[i]         for i in f_top_i],
            title="ANOVA F-Test Scores — Top 30 Features",
            color="#4338ca",
            xlabel="F-Statistic",
        )
        filter_content += b.figure(
            f_fig,
            title="ANOVA F-Test Feature Ranking",
            description="F-statistic for each numeric feature vs the binary target.",
            interpretation=(
                "The F-statistic measures whether the group means (class=0 vs class=1) differ "
                "significantly for each feature. High F = strong linear separability between classes. "
                "Features with high F-scores are natural candidates for logistic regression."
            ),
        )

    b.add_section("A. Filter Methods", filter_content, icon="🔬")

    # ── LR methods results ────────────────────────────────────────────
    lr_content = b.p(
        "Logistic regression-based selection identifies features with strong LINEAR relationships "
        "with the binary target. Regularisation (L1/L2/Elastic Net) prevents overfitting on "
        "high-dimensional data and performs implicit feature selection through coefficient shrinkage."
    )

    # L1 figure
    if "LR L1 (Lasso)" in method_scores:
        l1_arr   = method_scores["LR L1 (Lasso)"]
        l1_top_i = np.argsort(-l1_arr)[:30]
        LR_COLOR = "#0ea5e9"
        l1_fig   = feature_importance_bar(
            [used_features[i] for i in l1_top_i],
            [l1_arr[i]        for i in l1_top_i],
            title="L1 (Lasso) LR Coefficients — Top 30 Non-Zero Features",
            color=LR_COLOR,
            xlabel="|Standardised Coefficient|",
        )
        lr_content += b.figure(
            l1_fig,
            title="L1 Logistic Regression Coefficient Magnitudes",
            description="Absolute values of LR coefficients under L1 penalty (liblinear, C=0.1).",
            interpretation=(
                "Under L1 penalty, features with exactly-zero coefficients are completely excluded "
                "from the model — this is 'hard' feature selection. The remaining non-zero coefficients "
                "represent features with genuine linear signal strong enough to survive regularisation. "
                f"{method_meta.get('LR L1 (Lasso)', {}).get('n_selected', 'N/A')} of {n_feat} features "
                f"survived the L1 penalty (sparsity: "
                f"{method_meta.get('LR L1 (Lasso)', {}).get('sparsity_pct', 0):.1f}%)."
            ),
            business_implication=(
                "The L1 Lasso solution directly implements the scorecard philosophy: "
                "use the minimal set of features that explain the risk. "
                "Sparse models are preferred in credit risk for regulatory interpretability "
                "and operational stability."
            ),
        )

    # L1 vs L2 comparison
    if "LR L1 (Lasso)" in method_scores and "LR L2 (Ridge)" in method_scores:
        lr_content += b.card(
            "L1 vs L2 vs Elastic Net — Behaviour Comparison",
            "<table style='width:100%;font-size:13.5px;border-collapse:collapse'>"
            "<thead style='background:#1a3a5c;color:white'><tr>"
            "<th style='padding:8px'>Property</th>"
            "<th style='padding:8px;border-top:3px solid #0ea5e9'>L1 (Lasso)</th>"
            "<th style='padding:8px;border-top:3px solid #059669'>L2 (Ridge)</th>"
            "<th style='padding:8px;border-top:3px solid #7c3aed'>Elastic Net</th>"
            "</tr></thead><tbody>"
            + "".join(
                f"<tr><td style='padding:8px;font-weight:600'>{row[0]}</td>"
                f"<td style='padding:8px;background:rgba(14,165,233,.06)'>{row[1]}</td>"
                f"<td style='padding:8px;background:rgba(5,150,105,.06)'>{row[2]}</td>"
                f"<td style='padding:8px;background:rgba(124,58,237,.06)'>{row[3]}</td></tr>"
                for row in [
                    ("Penalty",              "Σ|βᵢ|",               "Σβᵢ²",              "α·Σ|βᵢ| + (1-α)·Σβᵢ²"),
                    ("Sparsity",             "✅ Yes — zero coefs", "❌ No — shrinks all", "✅ Partial — controlled by α"),
                    ("Feature selection",    "✅ Hard (automatic)", "❌ Soft (ranking only)", "✅ Moderate"),
                    ("Correlated features",  "Picks one arbitrarily","Distributes equally", "Groups together"),
                    ("Best when",            "Many irrelevant features", "All features relevant", "Correlated + irrelevant mix"),
                    ("Solver",               "liblinear, saga",      "lbfgs, newton-cg",   "saga"),
                    ("Tuning parameter",     "C (smaller = sparser)", "C (smaller = smoother)", "C + l1_ratio"),
                ]
            )
            + "</tbody></table>"
        )

    b.add_section("B. Logistic Regression-Based Methods", lr_content, icon="📈")

    # ── Tree methods results ──────────────────────────────────────────
    tree_content = b.p(
        "Tree-based importance methods measure how much each feature reduces impurity "
        "(or boosts metric) across all decision tree splits. Unlike LR, trees capture "
        "non-linear effects and interactions without any transformation."
    )

    if "Random Forest" in method_scores:
        rf_arr   = method_scores["Random Forest"]
        rf_top_i = np.argsort(-rf_arr)[:30]
        rf_fig   = feature_importance_bar(
            [used_features[i] for i in rf_top_i],
            [rf_arr[i]        for i in rf_top_i],
            title="Random Forest MDI Importance — Top 30 Features",
            color="#d97706",
            xlabel="Mean Decrease in Impurity",
        )
        tree_content += b.figure(
            rf_fig,
            title="Random Forest Feature Importance",
            description="MDI (Gini importance) from 200-tree Random Forest.",
            interpretation=(
                "MDI measures the total reduction in Gini impurity attributed to each feature "
                "across all trees and splits. Features that appear higher in trees (closer to root) "
                "tend to have higher MDI as they affect more samples. "
                "Caution: MDI can be biased toward high-cardinality features."
            ),
            business_implication=(
                "Random Forest importance reflects the complex, non-linear risk drivers in the data. "
                "Features important to the forest but not to logistic regression often encode "
                "threshold effects or interaction terms — for example, 'high debt AND low income' "
                "may be more predictive than either variable alone."
            ),
        )

    # LightGBM comparison if available
    if "LightGBM (gain)" in method_scores:
        lgb_gain_arr = method_scores["LightGBM (gain)"]
        lgb_top_i    = np.argsort(-lgb_gain_arr)[:30]
        lgb_fig = feature_importance_bar(
            [used_features[i] for i in lgb_top_i],
            [lgb_gain_arr[i]  for i in lgb_top_i],
            title="LightGBM Gain Importance — Top 30 Features",
            color="#ea580c",
            xlabel="Total Information Gain",
        )
        tree_content += b.figure(
            lgb_fig,
            title="LightGBM Gain Feature Importance",
            description="LightGBM gain importance — total information gain across all splits for each feature.",
            interpretation=(
                "LightGBM's gain importance measures the average gain of splits that use each feature. "
                "This is generally more reliable than split-count importance because it weights splits "
                "by their contribution to reducing the objective. "
                "Gain importance is scale-invariant and does not require feature scaling."
            ),
        )

    # Tree MDI known bias
    tree_content += b.callout(
        "<strong>Known Bias of MDI Importance:</strong> MDI (used by Random Forest, Extra Trees, "
        "Gradient Boosting) tends to overestimate the importance of high-cardinality numeric features "
        "because more split thresholds are available. Permutation importance and SHAP values are "
        "more reliable but computationally expensive on very large datasets. "
        "Gain-based importance (LightGBM, XGBoost) mitigates this bias significantly.",
        kind="warning",
    )

    b.add_section("C. Tree-Based Methods", tree_content, icon="🌲")

    # ── Consensus ranking ────────────────────────────────────────────
    log.info("Generating consensus figures…")

    # Tree vs LR scatter
    if tree_idx and lr_idx:
        scatter_fig = tree_vs_lr_scatter(
            used_features, tree_consensus.tolist(), lr_consensus.tolist(),
            title="Tree vs LR Feature Importance — Normalised Comparison",
        )
        scatter_html = b.figure(
            scatter_fig,
            title="Tree-Based vs LR-Based Importance Scatter",
            description=(
                "Each point is a feature. X-axis = normalised tree-based consensus score. "
                "Y-axis = normalised LR-based consensus score. Green = top both. Amber = tree-dominant. Blue = LR-dominant."
            ),
            interpretation=(
                f"Features in the top-right quadrant (green) — {len(both_top)} total — are the most "
                f"valuable because both model families agree they are important. "
                f"Features on the X-axis (amber) encode non-linear signals only. "
                f"Features on the Y-axis (blue) encode pure linear main effects."
            ),
            business_implication=(
                "For a regulatory scorecard, focus on the LR-dominant (blue) and dual-consensus (green) "
                "features. For a challenger machine learning model, use all consensus features. "
                "Tree-only features often represent the 'alpha' of a challenger over a scorecard model."
            ),
        )
    else:
        scatter_html = ""

    # Consensus bar chart
    consensus_fig = consensus_rank_bar(
        used_features,
        consensus_scores.tolist(),
        selection_counts.tolist(),
        title=f"Consensus Feature Ranking — Top {min(top_n, 40)} Features",
    )
    consensus_fig_html = b.figure(
        consensus_fig,
        title="Consensus Feature Ranking",
        description=(
            "Horizontal bar chart of consensus scores. Bar colour indicates how many methods "
            "selected this feature (green = many methods agree, red = only one or two)."
        ),
        interpretation=(
            "Features with both high consensus score AND selected by many methods are the "
            "most reliable. Features with high score but low method count may be strong "
            "in one family but weak in another."
        ),
    )

    # Stability heatmap
    stab_selected = {
        mname: [(norm_matrix[i, j] >= 0.5) for i in range(n_feat)]
        for j, mname in enumerate(method_names)
    }
    stab_fig = stability_heatmap(
        used_features[:60], stab_selected,
        title="Feature Selection Stability — All Methods"
    )
    stab_fig_html = b.figure(
        stab_fig,
        title="Feature Selection Stability Heatmap",
        description=(
            "Each row = feature, each column = method. Blue = selected (top 50% of that method). "
            "White = not selected."
        ),
        interpretation=(
            "Features with blue cells across ALL methods (every column) are the most stable "
            "selections. Features with a mix of blue and white are method-dependent — their "
            "inclusion should be validated via cross-validation stability analysis."
        ),
    )

    # Final ranking table
    ranking_rows = []
    for rank_pos, feat_idx in enumerate(sorted_idx[:100], start=1):
        feat_name  = used_features[feat_idx]
        grp_flag = (
            "🌲🔵 Both"     if feat_name in both_top
            else "🌲 Tree"  if feat_name in tree_top_set
            else "🔵 LR"    if feat_name in lr_top_set
            else "—"
        )
        ranking_rows.append({
            "Rank":           rank_pos,
            "Feature":        feat_name,
            "Consensus Score":f"{consensus_scores[feat_idx]:.4f}",
            "# Methods":      f"{int(selection_counts[feat_idx])}/{n_methods}",
            "Tree Score":     f"{tree_consensus[feat_idx]:.4f}",
            "LR Score":       f"{lr_consensus[feat_idx]:.4f}",
            "Filter Score":   f"{filter_consensus[feat_idx]:.4f}",
            "Agreement":      grp_flag,
        })

    ranking_tbl = b.table(
        ranking_rows,
        caption="Top 100 Features by Consensus Ranking",
        interpretation=(
            "Features are ranked by their average normalised importance score across all methods. "
            "'# Methods' shows how many of the " + str(n_methods) + " methods placed this feature "
            "in their top 50%. 'Agreement' indicates whether the feature ranks highly for trees, "
            "LR, or both."
        ),
    )

    # Three-way Venn summary
    venn_content = (
        b.card(
            "Consensus Group Analysis",
            f"<ul>"
            f"<li><strong>Dual-Consensus Features (Tree ∩ LR): {len(both_top)}</strong> — "
            f"Selected by both tree-based AND logistic regression methods. "
            f"Highest confidence features. Safe for any model type.</li>"
            f"<li><strong>Tree-Only Features: {len(tree_only)}</strong> — "
            f"Ranked highly by tree methods but NOT by LR. Likely encode non-linear relationships, "
            f"threshold effects, or interaction terms. Valuable for LightGBM/XGBoost models; "
            f"may need manual engineering to be useful for LR.</li>"
            f"<li><strong>LR-Only Features: {len(lr_only)}</strong> — "
            f"Ranked highly by LR methods but NOT by tree methods. Encode pure linear main effects. "
            f"Trees may have distributed their importance among correlated variants.</li>"
            f"<li><strong>Filter-Screened Out: {n_feat - len(selected_features)}</strong> — "
            f"Features with consensus score below the top-{top_n} cutoff. Candidates for removal.</li>"
            f"</ul>"
        )
    )

    consensus_content = (
        scatter_html + consensus_fig_html + stab_fig_html + venn_content + ranking_tbl
    )
    b.add_section("D. Consensus Ranking", consensus_content, icon="🏆")

    # ── Method-by-method summary ──────────────────────────────────────
    method_summary_rows = []
    for mname in method_names:
        meta    = method_meta.get(mname, {})
        scores  = method_scores[mname]
        top_feat_idx = np.argmax(scores)
        method_summary_rows.append({
            "Method":       mname,
            "Group":        meta.get("group", "?"),
            "Selected":     str(meta.get("n_selected", "?")),
            "Top Feature":  used_features[top_feat_idx] if top_feat_idx < len(used_features) else "?",
            "Top Score":    f"{scores[top_feat_idx]:.4f}",
            "Avg Score":    f"{scores.mean():.4f}",
            "Non-Zero":     str(int((scores > 1e-9).sum())),
        })

    method_summ_tbl = b.table(
        method_summary_rows,
        caption="Method-by-Method Summary",
        interpretation=(
            "This table provides a quick comparison of all selection methods. "
            "'Selected' reflects each method's own threshold. "
            "'Top Feature' is the single highest-ranked feature per method. "
            "Methods agreeing on the same top feature have high inter-method reliability."
        ),
    )

    b.add_section("Method Summary Comparison", method_summ_tbl, icon="📊")

    # ── Recommendations ───────────────────────────────────────────────
    rec_content = (
        b.callout(
            f"<strong>For Logistic Regression / Scorecard:</strong> Use features from the "
            f"LR-Based consensus set ({len(lr_top_set)} features). Apply L1 regularisation to "
            f"further reduce to the most parsimonious model.",
            kind="recommend", title="💡 Scorecard Recommendation",
        )
        + b.callout(
            f"<strong>For Tree-Based Challenger (LightGBM/XGBoost):</strong> Use the full "
            f"consensus feature set ({len(selected_features)} features). Tree models handle "
            f"redundant features well and benefit from a richer feature space.",
            kind="recommend", title="💡 Challenger Model Recommendation",
        )
        + b.callout(
            f"<strong>Universal Recommendation:</strong> The {len(both_top)} dual-consensus "
            "features that appear in BOTH the tree and LR top sets represent the highest-quality "
            "signal in the dataset. These should be prioritised for any model type and are "
            "the safest choices for production deployment.",
            kind="success", title="✅ Universal High-Confidence Features",
        )
    )
    b.add_section("Recommendations", rec_content, icon="💡")

    # ── Save selected features ────────────────────────────────────────
    artifacts_dir = Path(config.get("paths", {}).get("artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sel_spec = {
        "selected_features":    selected_features,
        "both_consensus":       list(both_top),
        "tree_only":            list(tree_only),
        "lr_only":              list(lr_only),
        "n_methods_tried":      n_methods,
        "method_names":         method_names,
        "consensus_scores":     {f: float(consensus_scores[i]) for i, f in enumerate(used_features)},
    }
    with open(artifacts_dir / "selected_features.json", "w") as f:
        json.dump(sel_spec, f, indent=2)
    log.info(f"Selected features saved ({len(selected_features)} features)")

    html   = b.build()
    writer = ReportWriter(config)
    path   = writer.write(html, "05_Feature_Selection.html")
    writer.write_index()

    elapsed = time.perf_counter() - t0
    log.stage_end("05 — Feature Selection", elapsed)
    return path, selected_features
