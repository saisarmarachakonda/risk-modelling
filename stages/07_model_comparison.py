"""
stages/07_model_comparison.py
Stage 07 — Full Combinatorial Experiment Grid

Tries EVERY combination of:
  - Feature Set    (7: baseline, IV, MI, L1, RF, XGB, Consensus)
  - Model Config   (17: LR variants, RF, ET, GB, XGB, DT, GNB)
  - Scaler         (3 for LR: Standard/Robust/MinMax | None for trees)
  - Imbalance      (3: balanced weights, raw, undersampling)

All experiments are ranked by 3-fold CV ROC AUC (descending).
Results are written to artifacts/experiment_results.csv and the HTML report.

Runtime estimate (50k sample, 3-fold CV, parallelised): ~25-40 min
"""
import json
import time
import warnings
import itertools
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
    roc_curves, pr_curves, model_metric_bar, feature_importance_bar,
)
from reporting.report_writer import ReportWriter


# ── Experiment configuration ──────────────────────────────────────────────────

SAMPLE_N  = 30_000   # rows for experiments (balance speed vs accuracy)
FS_SAMPLE = 10_000   # smaller subsample just for building feature sets
CV_FOLDS  = 3        # stratified k-fold
TEST_FRAC = 0.20     # hold-out test fraction
TOP_N_FEATURES = 25  # features to keep for non-baseline sets


def _make_feature_sets(X_all: np.ndarray, y: np.ndarray,
                       feat_cols: List[str], seed: int,
                       imbalance_ratio: float) -> Dict[str, List[str]]:
    """Build all 7 feature sets quickly using a 10k subsample."""
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    # Use small subsample just for ranking features (fast)
    rng_fs = np.random.default_rng(seed)
    n_fs   = min(FS_SAMPLE, len(X_all))
    idx_fs = rng_fs.choice(len(X_all), size=n_fs, replace=False)
    Xf, yf = X_all[idx_fs], y[idx_fs]

    n_feats = Xf.shape[1]
    top_n   = min(TOP_N_FEATURES, n_feats)
    scaler  = StandardScaler()
    Xf_sc   = scaler.fit_transform(Xf)

    feature_sets = {}

    # FS1 — Baseline (all features)
    feature_sets["FS1_All"] = feat_cols

    # FS2 — IV proxy via point-biserial correlation > threshold
    corrs   = np.abs(np.corrcoef(Xf.T, yf)[-1, :-1])
    iv_mask = corrs > 0.02
    feature_sets["FS2_IV_Filter"] = [feat_cols[i] for i in range(n_feats) if iv_mask[i]] or feat_cols[:top_n]

    # FS3 — Mutual Information top-N
    mi_scores = mutual_info_classif(Xf, yf, random_state=seed)
    mi_top    = np.argsort(-mi_scores)[:top_n]
    feature_sets["FS3_MutualInfo"] = [feat_cols[i] for i in mi_top]

    # FS4 — L1 Lasso non-zero (fast: liblinear, 200 iter, small data)
    lr_l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                max_iter=200, random_state=seed, class_weight="balanced")
    lr_l1.fit(Xf_sc, yf)
    l1_mask  = np.abs(lr_l1.coef_[0]) > 1e-6
    selected = [feat_cols[i] for i in range(n_feats) if l1_mask[i]]
    feature_sets["FS4_L1_Lasso"] = selected if len(selected) >= 5 else feat_cols[:top_n]

    # FS5 — Random Forest MDI top-N (50 shallow trees on small subsample)
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, max_features="sqrt",
                                 n_jobs=-1, random_state=seed, class_weight="balanced")
    rf.fit(Xf, yf)
    rf_top = np.argsort(-rf.feature_importances_)[:top_n]
    feature_sets["FS5_RandomForest"] = [feat_cols[i] for i in rf_top]

    # FS6 — XGBoost top-N (50 shallow trees on small subsample)
    if HAS_XGB:
        import xgboost as xgb
        xgb_m = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.2,
                                    scale_pos_weight=int(imbalance_ratio),
                                    random_state=seed, verbosity=0, n_jobs=-1)
        xgb_m.fit(Xf, yf)
        xgb_top = np.argsort(-xgb_m.feature_importances_)[:top_n]
        feature_sets["FS6_XGBoost"] = [feat_cols[i] for i in xgb_top]
    else:
        from sklearn.ensemble import ExtraTreesClassifier
        et = ExtraTreesClassifier(n_estimators=50, max_depth=5, n_jobs=-1,
                                  random_state=seed, class_weight="balanced")
        et.fit(Xf, yf)
        et_top = np.argsort(-et.feature_importances_)[:top_n]
        feature_sets["FS6_ExtraTrees"] = [feat_cols[i] for i in et_top]

    # FS7 — Consensus: features appearing in at least 3 of the above sets
    counts: Dict[str, int] = {}
    for fs_name, fs_cols in list(feature_sets.items())[1:]:  # skip FS1
        for c in fs_cols:
            counts[c] = counts.get(c, 0) + 1
    consensus = [c for c, n in sorted(counts.items(), key=lambda x: -x[1]) if n >= 2]
    feature_sets["FS7_Consensus"] = consensus[:top_n] if len(consensus) >= 5 else feature_sets["FS5_RandomForest"]

    return feature_sets


def _make_models(seed: int, imbalance_ratio: float) -> List[Dict]:
    """Define optimised model configurations with metadata to speed up grid search."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    )
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB

    scale_pos = int(imbalance_ratio)
    configs   = []

    # -- Logistic Regression (L1, L2, Elastic Net) --
    configs.append({
        "id": "LR_L1",
        "label": "LR L1 C=0.1",
        "family": "Logistic Regression",
        "model": LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                     max_iter=150, random_state=seed),
        "needs_scale": True,
    })
    configs.append({
        "id": "LR_L2",
        "label": "LR L2 C=1.0",
        "family": "Logistic Regression",
        "model": LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                     max_iter=150, random_state=seed),
        "needs_scale": True,
    })
    configs.append({
        "id": "LR_EN",
        "label": "LR ElasticNet l1=0.5",
        "family": "Logistic Regression",
        "model": LogisticRegression(penalty="elasticnet", solver="saga",
                                     l1_ratio=0.5, C=0.5, max_iter=150,
                                     random_state=seed),
        "needs_scale": True,
    })

    # -- Decision Tree --
    configs.append({
        "id": "DT_d5",
        "label": "Decision Tree depth=5",
        "family": "Decision Tree",
        "model": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "needs_scale": False,
    })

    # -- Naive Bayes --
    configs.append({
        "id": "GNB",
        "label": "Gaussian Naive Bayes",
        "family": "Naive Bayes",
        "model": GaussianNB(),
        "needs_scale": False,
    })

    # -- Random Forest --
    configs.append({
        "id": "RF_n60_d6",
        "label": "Random Forest n=60 depth=6",
        "family": "Random Forest",
        "model": RandomForestClassifier(n_estimators=60, max_depth=6,
                                         max_features="sqrt", min_samples_leaf=20,
                                         n_jobs=1, random_state=seed,
                                         max_samples=0.7),
        "needs_scale": False,
    })

    # -- Extra Trees --
    configs.append({
        "id": "ET_n60_d8",
        "label": "Extra Trees n=60 depth=8",
        "family": "Extra Trees",
        "model": ExtraTreesClassifier(n_estimators=60, max_depth=8, min_samples_leaf=20,
                                       n_jobs=1, random_state=seed),
        "needs_scale": False,
    })

    # -- Gradient Boosting --
    configs.append({
        "id": "GB_lr0.1_d4",
        "label": "Gradient Boosting n=60 depth=4",
        "family": "Gradient Boosting",
        "model": GradientBoostingClassifier(n_estimators=60, learning_rate=0.1,
                                             max_depth=4, subsample=0.8,
                                             min_samples_leaf=20, random_state=seed),
        "needs_scale": False,
    })

    # -- XGBoost --
    try:
        import xgboost as xgb
        configs.append({
            "id": "XGB_lr0.1_d4",
            "label": "XGBoost n=60 depth=4",
            "family": "XGBoost",
            "model": xgb.XGBClassifier(n_estimators=60, learning_rate=0.1,
                                        max_depth=4, subsample=0.8,
                                        colsample_bytree=0.8, min_child_weight=20,
                                        scale_pos_weight=scale_pos,
                                        eval_metric="auc", verbosity=0,
                                        random_state=seed, n_jobs=1,
                                        use_label_encoder=False),
            "needs_scale": False,
        })
    except ImportError:
        pass

    # -- LightGBM --
    try:
        import lightgbm as lgb
        configs.append({
            "id": "LGB_lr0.1_d4",
            "label": "LightGBM n=80 depth=4",
            "family": "LightGBM",
            "model": lgb.LGBMClassifier(n_estimators=80, learning_rate=0.1,
                                        max_depth=4, num_leaves=15, subsample=0.8,
                                        colsample_bytree=0.8, scale_pos_weight=scale_pos,
                                        random_state=seed, n_jobs=1, verbose=-1),
            "needs_scale": False,
        })
    except ImportError:
        pass

    return configs



def _make_scalers() -> List[Dict]:
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    return [
        {"id": "StandardScaler", "label": "StandardScaler", "obj": StandardScaler()},
        {"id": "RobustScaler",   "label": "RobustScaler",   "obj": RobustScaler()},
        {"id": "MinMaxScaler",   "label": "MinMaxScaler",   "obj": MinMaxScaler()},
    ]


def _make_imbalance_strategies() -> List[Dict]:
    return [
        {"id": "balanced_weight",  "label": "class_weight=balanced",
         "class_weight": "balanced", "undersample": False},
        {"id": "no_weight",        "label": "No weighting",
         "class_weight": None,     "undersample": False},
        {"id": "undersample",      "label": "RandomUnderSampler",
         "class_weight": None,     "undersample": True},
    ]


def _apply_class_weight(model_cfg: Dict, imb_cfg: Dict) -> Any:
    """Clone model with appropriate class_weight setting."""
    import copy
    model = copy.deepcopy(model_cfg["model"])
    cw    = imb_cfg.get("class_weight")
    params = model.get_params()
    if "class_weight" in params:
        model.set_params(class_weight=cw)
    return model


def _run_experiment(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    X_oot: np.ndarray, y_oot: np.ndarray,
    model, cv, seed: int,
) -> Dict[str, Any]:
    """Run one experiment: CV + test evaluation + OOT evaluation."""
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, recall_score, precision_score,
        brier_score_loss, matthews_corrcoef,
        confusion_matrix, roc_curve,
    )
    import copy

    result = {}
    try:
        t0 = time.perf_counter()
        # 3-fold CV on Train set
        cv_res = cross_validate(
            copy.deepcopy(model), X_tr, y_tr,
            cv=cv, scoring="roc_auc", n_jobs=1,
        )
        result["cv_roc_auc_mean"] = round(float(cv_res["test_score"].mean()), 4)
        result["cv_roc_auc_std"]  = round(float(cv_res["test_score"].std()),  4)

        # Fit on full Train set
        model.fit(X_tr, y_tr)
        
        # Test evaluation
        y_prob_te = model.predict_proba(X_te)[:, 1]
        y_pred_te = (y_prob_te >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred_te).ravel()
        spec_te = tn / max(tn + fp, 1)
        ks_te   = float(np.max(np.abs(
            np.cumsum(y_te[np.argsort(-y_prob_te)])   / max(y_te.sum(), 1) -
            np.cumsum(1 - y_te[np.argsort(-y_prob_te)]) / max((1-y_te).sum(), 1)
        )))

        # OOT evaluation
        y_prob_oot = model.predict_proba(X_oot)[:, 1]
        y_pred_oot = (y_prob_oot >= 0.5).astype(int)
        tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_oot, y_pred_oot).ravel()
        spec_oot = tn_o / max(tn_o + fp_o, 1)
        ks_oot   = float(np.max(np.abs(
            np.cumsum(y_oot[np.argsort(-y_prob_oot)])   / max(y_oot.sum(), 1) -
            np.cumsum(1 - y_oot[np.argsort(-y_prob_oot)]) / max((1-y_oot).sum(), 1)
        )))

        result.update({
            "test_roc_auc":  round(float(roc_auc_score(y_te, y_prob_te)), 4),
            "test_pr_auc":   round(float(average_precision_score(y_te, y_prob_te)), 4),
            "test_f1":       round(float(f1_score(y_te, y_pred_te, zero_division=0)), 4),
            "test_recall":   round(float(recall_score(y_te, y_pred_te, zero_division=0)), 4),
            "test_precision":round(float(precision_score(y_te, y_pred_te, zero_division=0)), 4),
            "test_specificity": round(float(spec_te), 4),
            "test_ks":       round(float(ks_te), 4),
            
            "oot_roc_auc":   round(float(roc_auc_score(y_oot, y_prob_oot)), 4),
            "oot_pr_auc":    round(float(average_precision_score(y_oot, y_prob_oot)), 4),
            "oot_f1":        round(float(f1_score(y_oot, y_pred_oot, zero_division=0)), 4),
            "oot_recall":    round(float(recall_score(y_oot, y_pred_oot, zero_division=0)), 4),
            "oot_precision": round(float(precision_score(y_oot, y_pred_oot, zero_division=0)), 4),
            "oot_specificity": round(float(spec_oot), 4),
            "oot_ks":        round(float(ks_oot), 4),

            "train_time_s":  round(time.perf_counter() - t0, 2),
            "error": "",
        })
        
        # Store for ROC curves
        fpr, tpr, _ = roc_curve(y_oot, y_prob_oot)
        result["_fpr"] = fpr.tolist()
        result["_tpr"] = tpr.tolist()

    except Exception as e:
        result.update({
            "cv_roc_auc_mean": 0.0, "cv_roc_auc_std": 0.0,
            "test_roc_auc": 0.0, "test_pr_auc": 0.0,
            "test_f1": 0.0, "test_recall": 0.0, "test_precision": 0.0,
            "test_specificity": 0.0, "test_ks": 0.0,
            "oot_roc_auc": 0.0, "oot_pr_auc": 0.0,
            "oot_f1": 0.0, "oot_recall": 0.0, "oot_precision": 0.0,
            "oot_specificity": 0.0, "oot_ks": 0.0,
            "train_time_s": 0.0,
            "error": str(e)[:100],
        })
    return result


def _run_single_task(
    exp_id: int,
    fs_name: str,
    fs_cols: List[str],
    model_cfg: Dict,
    scaler_cfg: Dict,
    imb_cfg: Dict,
    X_tr_raw: np.ndarray,
    y_tr: np.ndarray,
    X_te_raw: np.ndarray,
    y_te: np.ndarray,
    X_oot_raw: np.ndarray,
    y_oot: np.ndarray,
    feat_idx: Dict[str, int],
    cv_folds: int,
    seed: int,
    has_imblearn: bool,
) -> Dict:
    """Helper executed on a worker process to run a single experiment combination."""
    import copy
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # 1. Slice features
    fs_idx = [feat_idx[c] for c in fs_cols if c in feat_idx]
    X_tr_fs = X_tr_raw[:, fs_idx]
    X_te_fs = X_te_raw[:, fs_idx]
    X_oot_fs = X_oot_raw[:, fs_idx]

    # 2. Scale features
    if scaler_cfg["obj"] is not None:
        sc = copy.deepcopy(scaler_cfg["obj"])
        X_tr_sc = sc.fit_transform(X_tr_fs)
        X_te_sc = sc.transform(X_te_fs)
        X_oot_sc = sc.transform(X_oot_fs)
    else:
        X_tr_sc = X_tr_fs
        X_te_sc = X_te_fs
        X_oot_sc = X_oot_fs

    # 3. Apply class weight
    model = _apply_class_weight(model_cfg, imb_cfg)

    # 4. Apply imbalance strategy
    X_tr_use = X_tr_sc
    y_tr_use = y_tr
    if imb_cfg.get("undersample") and has_imblearn:
        try:
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=seed)
            X_tr_use, y_tr_use = rus.fit_resample(X_tr_sc, y_tr)
        except Exception:
            pass
    elif imb_cfg.get("undersample") and not has_imblearn:
        # Manual undersampling fallback
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        n_pos = len(pos_idx)
        neg_sampled = np.random.default_rng(seed).choice(
            neg_idx, size=min(n_pos * 3, len(neg_idx)), replace=False
        )
        keep = np.concatenate([pos_idx, neg_sampled])
        np.random.default_rng(seed).shuffle(keep)
        X_tr_use = X_tr_sc[keep]
        y_tr_use = y_tr[keep]

    # 5. Run experiment
    res = _run_experiment(X_tr_use, y_tr_use, X_te_sc, y_te, X_oot_sc, y_oot, model, cv, seed)

    return {
        "rank":             0,
        "experiment_id":    exp_id,
        "feature_set":      fs_name,
        "n_features":       len(fs_cols),
        "model_id":         model_cfg["id"],
        "model_label":      model_cfg["label"],
        "model_family":     model_cfg["family"],
        "scaler":           scaler_cfg["label"],
        "imbalance":        imb_cfg["label"],
        "cv_roc_auc_mean":  res.get("cv_roc_auc_mean", 0.0),
        "cv_roc_auc_std":   res.get("cv_roc_auc_std",  0.0),
        "test_roc_auc":     res.get("test_roc_auc",    0.0),
        "test_pr_auc":      res.get("test_pr_auc",     0.0),
        "test_f1":          res.get("test_f1",         0.0),
        "test_recall":      res.get("test_recall",     0.0),
        "test_precision":   res.get("test_precision",  0.0),
        "test_specificity": res.get("test_specificity",0.0),
        "test_ks":          res.get("test_ks",         0.0),
        
        "oot_roc_auc":      res.get("oot_roc_auc",     0.0),
        "oot_pr_auc":       res.get("oot_pr_auc",      0.0),
        "oot_f1":           res.get("oot_f1",          0.0),
        "oot_recall":       res.get("oot_recall",      0.0),
        "oot_precision":    res.get("oot_precision",   0.0),
        "oot_specificity":  res.get("oot_specificity", 0.0),
        "oot_ks":           res.get("oot_ks",          0.0),
        
        "train_time_s":     res.get("train_time_s",    0.0),
        "error":            res.get("error",           ""),
        "_fpr":             res.get("_fpr",            []),
        "_tpr":             res.get("_tpr",            []),
    }



def run(
    loader: DataLoader,
    schema: SchemaDetector,
    config: dict,
    selected_features: List[str],
    prep_info: Dict,
) -> Tuple[Path, Dict]:
    log = get_logger("07_ModelComparison",
                     config.get("paths", {}).get("logs_dir", "logs"))
    log.stage_start("07 — Full Combinatorial Experiment Grid")
    t0_stage = time.perf_counter()

    from sklearn.model_selection import StratifiedKFold, train_test_split
    try:
        from imblearn.under_sampling import RandomUnderSampler
        HAS_IMBLEARN = True
    except ImportError:
        HAS_IMBLEARN = False

    mod_cfg       = config.get("modeling", {})
    seed          = mod_cfg.get("random_seed", 42)
    target        = loader.target_col
    n_rows        = loader.count_rows()
    imb_ratio     = prep_info.get("imbalance_ratio", 1.0)
    primary       = mod_cfg.get("primary_metric", "roc_auc")

    # ── Resolve selected features (handle resume from --from-stage 7) ──
    artifacts_dir = Path(config.get("paths", {}).get("artifacts_dir", "artifacts"))
    _feats = list(selected_features)  # copy

    if not _feats:
        # Try loading from stage 05/06 artifact
        split_info_path = artifacts_dir / "train_test_split_info.json"
        if split_info_path.exists():
            with open(split_info_path) as _f:
                _info = json.load(_f)
                _feats = _info.get("selected_features", [])
                imb_ratio = _info.get("imbalance_ratio", imb_ratio)
                log.info(f"Loaded {len(_feats)} features from train_test_split_info.json")

    if not _feats:
        # Final fallback: use all numeric columns from schema
        log.info("No saved features found — using all numeric schema columns")
        _feats = schema.get_numeric_cols()

    # ── Load sample ────────────────────────────────────────────────────
    sample_n  = min(SAMPLE_N, n_rows)
    all_cols  = loader.get_columns()
    db_types  = loader.get_db_types()

    # Keep only numeric-compatible columns that exist in the dataset
    _NUMERIC_TYPES = {
        "BIGINT","HUGEINT","INTEGER","INT","SMALLINT","TINYINT",
        "UBIGINT","UINTEGER","USMALLINT","UTINYINT",
        "DOUBLE","FLOAT","REAL","DECIMAL","NUMERIC",
    }
    valid_feats = [
        c for c in _feats
        if c in all_cols
        and c != target
        and db_types.get(c, "VARCHAR").upper().split("(")[0] in _NUMERIC_TYPES
    ]
    if not valid_feats:
        # Absolute last resort — grab first 50 numeric cols
        valid_feats = [
            c for c in all_cols
            if c != target
            and db_types.get(c, "VARCHAR").upper().split("(")[0] in _NUMERIC_TYPES
        ][:50]

    log.info(f"Loading {sample_n:,}-row sample with {len(valid_feats)} numeric features…")
    grp_col = config.get("data", {}).get("group_column", "sample_group")
    cols_to_load = valid_feats + [target, grp_col]
    df       = loader.sample_columns(cols_to_load, n=sample_n, seed=seed)
    feat_cols = [c for c in valid_feats if c in df.columns]

    # Pre-split using the pre-defined group column!
    df_tr = df.filter(pl.col(grp_col) == "Train")
    df_te = df.filter(pl.col(grp_col) == "Test")
    df_oot = df.filter(pl.col(grp_col) == "OOT")

    X_tr_raw = df_tr.select(feat_cols).fill_null(0).to_numpy().astype(np.float32)
    y_tr = df_tr[target].fill_null(0).to_numpy().astype(np.int32)

    X_te_raw = df_te.select(feat_cols).fill_null(0).to_numpy().astype(np.float32)
    y_te = df_te[target].fill_null(0).to_numpy().astype(np.int32)

    X_oot_raw = df_oot.select(feat_cols).fill_null(0).to_numpy().astype(np.float32)
    y_oot = df_oot[target].fill_null(0).to_numpy().astype(np.int32)

    log.info(f"Train set: {X_tr_raw.shape[0]:,} rows, Test set: {X_te_raw.shape[0]:,} rows, OOT set: {X_oot_raw.shape[0]:,} rows")
    log.info(f"Features: {X_tr_raw.shape[1]}")

    # Recompute imbalance ratio from data if not in prep_info
    if imb_ratio <= 1.0:
        _pos = int(y_tr.sum())
        _neg = len(y_tr) - _pos
        imb_ratio = _neg / max(_pos, 1)
    log.info(f"Class imbalance ratio: {imb_ratio:.1f}:1")


    # ── Build feature sets ─────────────────────────────────────────────
    use_selected_only = mod_cfg.get("use_selected_features_only", False)
    if use_selected_only:
        log.info("use_selected_features_only is TRUE. Loading pre-selected features from Stage 05…")
        sel_features_file = artifacts_dir / "selected_features.json"
        if sel_features_file.exists():
            try:
                with open(sel_features_file, "r") as f:
                    sel_data = json.load(f)
                
                fs_sel = sel_data.get("selected_features", [])
                fs_both = sel_data.get("both_consensus", [])
                fs_tree = sel_data.get("tree_only", [])
                fs_lr = sel_data.get("lr_only", [])

                feature_sets = {}
                feature_sets["Stage5_Selected"] = [c for c in fs_sel if c in feat_cols]
                if fs_both:
                    feature_sets["Stage5_BothConsensus"] = [c for c in fs_both if c in feat_cols]
                if fs_tree:
                    feature_sets["Stage5_TreeOnly"] = [c for c in fs_tree if c in feat_cols]
                if fs_lr:
                    feature_sets["Stage5_LROnly"] = [c for c in fs_lr if c in feat_cols]
            except Exception as e:
                log.warning(f"Error loading selected_features.json: {e}. Falling back to default feature sets.")
                feature_sets = _make_feature_sets(X_tr_raw, y_tr, feat_cols, seed, imb_ratio)
        else:
            log.warning("selected_features.json not found in artifacts. Falling back to default feature sets.")
            feature_sets = _make_feature_sets(X_tr_raw, y_tr, feat_cols, seed, imb_ratio)
    else:
        log.info("Building feature sets (7 methods)…")
        feature_sets = _make_feature_sets(X_tr_raw, y_tr, feat_cols, seed, imb_ratio)

    for fs_name, fs_cols in feature_sets.items():
        log.info(f"  {fs_name}: {len(fs_cols)} features")

    # ── Build experiment catalogue ──────────────────────────────────────
    model_configs   = _make_models(seed, imb_ratio)
    scaler_configs  = _make_scalers()
    imbalance_cfgs  = _make_imbalance_strategies()

    # Pre-index features so we can slice X by feature set
    feat_idx = {f: i for i, f in enumerate(feat_cols)}

    log.info(f"Experiment dimensions:")
    log.info(f"  Feature sets:  {len(feature_sets)}")
    log.info(f"  Model configs: {len(model_configs)}")
    log.info(f"  Scalers:       {len(scaler_configs)} (LR) | 1 (Trees)")
    log.info(f"  Imbalance:     {len(imbalance_cfgs)}")

    # ── Prepare job list for parallel execution ────────────────────────
    tasks = []
    exp_count = 0
    for fs_name, fs_cols in feature_sets.items():
        for model_cfg in model_configs:
            needs_scale = model_cfg["needs_scale"]
            scalers_to_try = scaler_configs if needs_scale else [
                {"id": "None", "label": "No Scaling", "obj": None}
            ]
            for scaler_cfg in scalers_to_try:
                for imb_cfg in imbalance_cfgs:
                    exp_count += 1
                    tasks.append((exp_count, fs_name, fs_cols, model_cfg, scaler_cfg, imb_cfg))

    log.info(f"Launching {len(tasks):,} combinations in parallel using all available cores...")
    
    from joblib import Parallel, delayed
    try:
        raw_results = Parallel(n_jobs=-1, verbose=1, backend="threading")(
            delayed(_run_single_task)(
                exp_id, fs_name, fs_cols, model_cfg, scaler_cfg, imb_cfg,
                X_tr_raw, y_tr, X_te_raw, y_te, X_oot_raw, y_oot, feat_idx, CV_FOLDS, seed, HAS_IMBLEARN
            )
            for exp_id, fs_name, fs_cols, model_cfg, scaler_cfg, imb_cfg in tasks
        )

    except Exception as ex:
        log.warning(f"Parallel job failed ({ex}). Falling back to serial run...")
        raw_results = []
        for exp_id, fs_name, fs_cols, model_cfg, scaler_cfg, imb_cfg in tasks:
            res = _run_single_task(
                exp_id, fs_name, fs_cols, model_cfg, scaler_cfg, imb_cfg,
                X_tr_raw, y_tr, X_te_raw, y_te, X_oot_raw, y_oot, feat_idx, CV_FOLDS, seed, HAS_IMBLEARN
            )
            raw_results.append(res)

    all_results = list(raw_results)

    # ── Sort and rank with tie-breaker ────────────────────────────────
    log.info("Ranking all experiments with contest tie-breaking rules...")
    # Sort primarily by OOT ROC AUC descending, and secondary by n_features ascending
    all_results.sort(key=lambda x: (-x["oot_roc_auc"], x["n_features"], x["experiment_id"]))
    
    # Apply 0.01 ROC AUC tie-breaking tolerance
    n_res = len(all_results)
    for i in range(n_res):
        for j in range(i + 1, n_res):
            diff = abs(all_results[i]["oot_roc_auc"] - all_results[j]["oot_roc_auc"])
            if diff <= 0.01:
                # If model j uses fewer features than model i, it wins the tie break
                if all_results[j]["n_features"] < all_results[i]["n_features"]:
                    all_results[i], all_results[j] = all_results[j], all_results[i]
                    
    # Assign ranks
    for i, r in enumerate(all_results):
        r["rank"] = i + 1



    # ── Top 3 per dimension ───────────────────────────────────────────
    def top3_by(key: str) -> Dict[str, List[Dict]]:
        """Return top 3 results for each unique value of key."""
        groups: Dict[str, List[Dict]] = {}
        for r in all_results:          # already sorted by cv_roc_auc_mean desc
            k = r[key]
            if k not in groups:
                groups[k] = []
            if len(groups[k]) < 3:
                groups[k].append(r)
        return groups

    top3_per_fs     = top3_by("feature_set")
    top3_per_family = top3_by("model_family")
    top3_per_scaler = top3_by("scaler")
    top3_per_imb    = top3_by("imbalance")

    # ── ROC curves for top-5 unique models ────────────────────────────
    seen_families = set()
    roc_data = []
    for r in all_results:
        if r["model_family"] not in seen_families and r["_fpr"]:
            seen_families.add(r["model_family"])
            roc_data.append({
                "name": f'{r["model_family"]} ({r["feature_set"]})',
                "fpr":  r["_fpr"],
                "tpr":  r["_tpr"],
                "auc":  r["test_roc_auc"],
            })
        if len(roc_data) >= 8:
            break

    # ── Save CSV artifact ──────────────────────────────────────────────
    artifacts_dir = Path(config.get("paths", {}).get("artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = [{k: v for k, v in r.items() if not k.startswith("_")}
                for r in all_results]
    pl.DataFrame(csv_rows).write_csv(str(artifacts_dir / "experiment_results.csv"))
    log.info(f"Saved experiment_results.csv ({len(csv_rows)} rows)")

    # Save best model info
    best = all_results[0]
    with open(artifacts_dir / "model_results.json", "w") as f:
        json.dump({
            "best_experiment":  {k: v for k, v in best.items() if not k.startswith("_")},
            "best_model":       best["model_label"],
            "best_feature_set": best["feature_set"],
            "best_scaler":      best["scaler"],
            "best_imbalance":   best["imbalance"],
            "total_experiments":exp_count,
            "ranking": [
                {k: v for k, v in r.items() if not k.startswith("_")}
                for r in all_results[:50]
            ],
        }, f, indent=2, default=str)

    # ── Build HTML report ──────────────────────────────────────────────
    b = HTMLReportBuilder(
        report_title   = "Full Combinatorial Experiment Report",
        stage_number   = 7,
        stage_subtitle = (
            f"{exp_count} experiments | {len(feature_sets)} feature sets × "
            f"{len(model_configs)} model configs × scalers × imbalance strategies"
        ),
        config         = config,
        n_rows         = n_rows,
        n_cols         = len(feat_cols),
    )

    # Executive summary cards
    cards = [
        {"label": "Total Experiments",  "value": f"{exp_count:,}",   "variant": "success"},
        {"label": "Best Model",         "value": best["model_label"][:25], "variant": "success"},
        {"label": "Best Feature Set",   "value": best["feature_set"]},
        {"label": "Best Scaler",        "value": best["scaler"]},
        {"label": "Best Imbalance",     "value": best["imbalance"]},
        {"label": "Best OOT AUC",       "value": str(best["oot_roc_auc"]), "variant": "success"},
        {"label": "Best Test AUC",      "value": str(best["test_roc_auc"])},
        {"label": "Best OOT KS",        "value": str(best["oot_ks"])},
    ]
    b.add_executive_summary(cards, narrative=(
        f"A total of <strong>{exp_count:,} experiments</strong> were executed across all combinations of "
        f"{len(feature_sets)} feature selection strategies, {len(model_configs)} model configurations, "
        f"scalers, and class imbalance handling techniques. "
        f"Every experiment was trained on Train, validated on Test, and scored on OOT. "
        f"Model rankings are computed on the OOT sample using a 0.01 ROC AUC tie-breaking tolerance (fewer features win ties).\n\n"
        f"The winning combination is "
        f"<strong>{best['model_label']}</strong> with feature set "
        f"<strong>{best['feature_set']}</strong>, scaled with "
        f"<strong>{best['scaler']}</strong> and imbalance strategy "
        f"<strong>{best['imbalance']}</strong>, achieving an OOT ROC AUC of "
        f"<strong>{best['oot_roc_auc']}</strong> "
        f"(Test AUC of {best['test_roc_auc']}) using {best['n_features']} features."
    ))

    # ── Top 30 experiments table ───────────────────────────────────────
    top30_rows = []
    for r in all_results[:30]:
        top30_rows.append({
            "Rank":        r["rank"],
            "Feature Set": r["feature_set"],
            "Model":       r["model_label"],
            "Scaler":      r["scaler"],
            "Imbalance":   r["imbalance"],
            "OOT AUC":     r["oot_roc_auc"],
            "Test AUC":    r["test_roc_auc"],
            "CV AUC":      f'{r["cv_roc_auc_mean"]} ± {r["cv_roc_auc_std"]}',
            "OOT KS":      r["oot_ks"],
            "OOT F1":      r["oot_f1"],
            "OOT Recall":  r["oot_recall"],
            "N Features":  r["n_features"],
        })
    b.add_section(
        "Top 30 Experiment Combinations",
        b.table(top30_rows, caption="Top 30 Ranked Experiments (by OOT ROC AUC + Tie Breaker)",
                interpretation=(
                    "All experiments ranked by OOT ROC AUC. "
                    "If OOT ROC AUC values are within 0.01, the tie-breaking rule ranks "
                    "the model with fewer features higher. "
                    "The full ranked table of all experiments is saved to "
                    "artifacts/experiment_results.csv."
                )),
        icon="🏆",
    )

    # ── Top 3 per feature set ─────────────────────────────────────────
    fs_rows = []
    for fs_name, top3 in sorted(top3_per_fs.items(),
                                 key=lambda x: x[1][0]["oot_roc_auc"], reverse=True):
        for pos, r in enumerate(top3, start=1):
            fs_rows.append({
                "Feature Set":  fs_name,
                "Pos":          f"#{pos}",
                "N Features":   r["n_features"],
                "Model":        r["model_label"],
                "Scaler":       r["scaler"],
                "Imbalance":    r["imbalance"],
                "OOT AUC":      r["oot_roc_auc"],
                "Test AUC":     r["test_roc_auc"],
                "OOT KS":       r["oot_ks"],
                "OOT Recall":   r["oot_recall"],
                "Overall Rank": r["rank"],
            })
    b.add_section(
        "Top 3 per Feature Selection Method",
        b.table(fs_rows, caption="Top 3 Experiments for each Feature Set (ranked by OOT ROC AUC)",
                interpretation=(
                    "Shows the top 3 model configurations for each feature selection method. "
                    "Compare #1 across feature sets to find which selection method yields "
                    "the highest ceiling performance. Compare #1 vs #3 within a feature set "
                    "to understand how sensitive that feature set is to model choice."
                )),
        icon="🔍",
    )

    # Feature set bar chart — top 1 AUC per set
    _fs_bar_data = [(fs, top3[0]["oot_roc_auc"])
                    for fs, top3 in sorted(top3_per_fs.items(),
                                           key=lambda x: x[1][0]["oot_roc_auc"], reverse=True)]
    fs_bar = model_metric_bar(
        [x[0] for x in _fs_bar_data],
        [x[1] for x in _fs_bar_data],
        metric_name="Best OOT ROC AUC",
        title="Best OOT AUC per Feature Selection Method",
    )
    b.add_section(
        "Feature Selection Method Comparison Chart",
        b.figure(fs_bar, "Feature Set OOT AUC Comparison",
                 interpretation=(
                     "Each bar is the #1 OOT AUC for that feature set (across all model configs). "
                     "A large spread means feature selection method is a dominant factor. "
                     "A small spread means model choice dominates over selection method."
                 )),
        icon="📊",
    )

    # ── Top 3 per model family ────────────────────────────────────────
    family_rows = []
    for fam, top3 in sorted(top3_per_family.items(),
                             key=lambda x: x[1][0]["oot_roc_auc"], reverse=True):
        for pos, r in enumerate(top3, start=1):
            family_rows.append({
                "Model Family":  fam,
                "Pos":           f"#{pos}",
                "Config":        r["model_label"],
                "Feature Set":   r["feature_set"],
                "Scaler":        r["scaler"],
                "Imbalance":     r["imbalance"],
                "OOT AUC":       r["oot_roc_auc"],
                "Test AUC":      r["test_roc_auc"],
                "OOT KS":        r["oot_ks"],
                "OOT Recall":    r["oot_recall"],
                "OOT F1":        r["oot_f1"],
                "Overall Rank":  r["rank"],
            })
    b.add_section(
        "Top 3 per Model Family",
        b.table(family_rows, caption="Top 3 Experiments for each Model Family",
                interpretation=(
                    "Top 3 configurations per model family. "
                    "#1 shows the ceiling for that family; #2 and #3 show robustness — "
                    "a family where #1 and #3 are close is consistently strong, not a lucky outlier."
                )),
        icon="🤖",
    )

    # Model family bar — best AUC per family
    _fam_bar_data = [(fam, top3[0]["oot_roc_auc"])
                     for fam, top3 in sorted(top3_per_family.items(),
                                             key=lambda x: x[1][0]["oot_roc_auc"], reverse=True)]
    family_bar = model_metric_bar(
        [x[0] for x in _fam_bar_data],
        [x[1] for x in _fam_bar_data],
        metric_name="Best OOT ROC AUC",
        title="Best OOT AUC per Model Family",
    )
    b.add_section(
        "Model Family Comparison Chart",
        b.figure(family_bar, "Model Family OOT AUC Comparison",
                 interpretation=(
                     "Best achievable OOT AUC per model family. "
                     "Tree-based models typically outperform LR on non-linear credit risk patterns. "
                     "The LR ceiling represents maximum linear discriminability."
                 )),
        icon="📈",
    )

    # ── Top 3 per scaler ──────────────────────────────────────────────
    scaler_rows = []
    for sc_name, top3 in sorted(top3_per_scaler.items(),
                                 key=lambda x: x[1][0]["oot_roc_auc"], reverse=True):
        for pos, r in enumerate(top3, start=1):
            scaler_rows.append({
                "Scaler":        sc_name,
                "Pos":           f"#{pos}",
                "Model":         r["model_label"],
                "Feature Set":   r["feature_set"],
                "Imbalance":     r["imbalance"],
                "OOT AUC":       r["oot_roc_auc"],
                "Test AUC":      r["test_roc_auc"],
                "Recall":        r["oot_recall"],
                "Specificity":   r["oot_specificity"],
                "Overall Rank":  r["rank"],
            })
    b.add_section(
        "Top 3 per Scaling Strategy",
        b.table(scaler_rows, caption="Top 3 Experiments for each Scaler",
                interpretation=(
                    "Scaling matters for LR models; tree models are scale-invariant (appear under 'No Scaling'). "
                    "RobustScaler is preferred when outliers exist. "
                    "StandardScaler assumes Gaussian distribution. MinMaxScaler maps to [0,1]. "
                    "Compare #1 vs #3 within each scaler — a tight spread means the scaler "
                    "is robust regardless of model choice."
                )),
        icon="⚙️",
    )

    # ── Top 3 per imbalance strategy ──────────────────────────────────
    imb_rows = []
    for imb_name, top3 in sorted(top3_per_imb.items(),
                                  key=lambda x: x[1][0]["oot_roc_auc"], reverse=True):
        for pos, r in enumerate(top3, start=1):
            imb_rows.append({
                "Imbalance Strategy": imb_name,
                "Pos":                f"#{pos}",
                "Model":              r["model_label"],
                "Feature Set":        r["feature_set"],
                "Scaler":             r["scaler"],
                "OOT AUC":            r["oot_roc_auc"],
                "Test AUC":           r["test_roc_auc"],
                "Recall":             r["oot_recall"],
                "Specificity":        r["oot_specificity"],
                "OOT KS":             r["oot_ks"],
                "Overall Rank":       r["rank"],
            })
    b.add_section(
        "Top 3 per Imbalance Strategy",
        b.table(imb_rows, caption="Top 3 Experiments for each Imbalance Handling Strategy",
                interpretation=(
                    "Recall vs Specificity trade-off shifts significantly with imbalance strategy. "
                    "class_weight=balanced boosts Recall (catches more defaults) at cost of Specificity. "
                    "Undersampling has a similar but stronger effect. "
                    "No-weighting maximises accuracy — misleading for imbalanced data. "
                    "For credit risk, high Recall is typically preferred to minimise missed defaults."
                )),
        icon="⚖️",
    )

    # ── ROC curves for best per family ────────────────────────────────
    if roc_data:
        roc_fig = roc_curves(roc_data, title="ROC Curves — Best per Model Family (OOT)")
        b.add_section(
            "ROC Curves",
            b.figure(roc_fig, "OOT ROC Curves",
                     interpretation=(
                         "ROC curves for the best experiment from each model family evaluated on the OOT hold-out set. "
                         "Higher and more left-bulging curves indicate better separation "
                         "between defaulters and non-defaulters across all thresholds."
                     ),
                     business_implication=(
                         "In credit risk scoring, ROC AUC > 0.75 is acceptable, "
                         "> 0.80 is good, > 0.85 is excellent. The KS statistic "
                         "(maximum vertical gap) is the standard bank metric."
                     )),
            icon="📉",
        )

    # ── Full ranked table (all experiments) ───────────────────────────
    all_rows = []
    for r in all_results:
        all_rows.append({
            "Rank":        r["rank"],
            "Feature Set": r["feature_set"],
            "Model":       r["model_label"],
            "Family":      r["model_family"],
            "Scaler":      r["scaler"],
            "Imbalance":   r["imbalance"],
            "OOT AUC":     r["oot_roc_auc"],
            "Test AUC":    r["test_roc_auc"],
            "CV AUC":      r["cv_roc_auc_mean"],
            "±":           r["cv_roc_auc_std"],
            "OOT KS":      r["oot_ks"],
            "OOT F1":      r["oot_f1"],
            "OOT Recall":  r["oot_recall"],
            "N Features":  r["n_features"],
            "Error":       r["error"] or "",
        })

    b.add_section(
        "Complete Ranked Results — All Experiments",
        b.callout(
            f"The full table below shows ALL {exp_count:,} experiments ranked by OOT ROC AUC. "
            f"Every combination of feature set, model, scaler, and imbalance strategy is included. "
            f"A machine-readable copy is saved to <code>artifacts/experiment_results.csv</code>.",
            kind="note",
        )
        + b.table(
            all_rows,
            caption=f"All {exp_count} Experiments — Ranked by OOT ROC AUC + Tie Breaker",
            interpretation=(
                "This is the complete combinatorial ranking. "
                "Each row is a unique pipeline configuration. "
                "Use this table to understand which dimensions (feature set, model, "
                "scaler, imbalance) contribute most to performance variation."
            ),
        ),
        icon="📋",
    )

    # ── Build & write ─────────────────────────────────────────────────
    html   = b.build()
    writer = ReportWriter(config)
    path   = writer.write(html, "07_Model_Comparison.html")
    writer.write_index()

    elapsed = time.perf_counter() - t0_stage
    log.stage_end("07 — Full Combinatorial Experiment Grid", elapsed)
    log.info(f"Total experiments: {exp_count} | Best OOT AUC: {best['oot_roc_auc']} using {best['n_features']} features")
    log.info(f"Best combo: {best['feature_set']} | {best['model_label']} | "
             f"{best['scaler']} | {best['imbalance']}")

    return path, {
        "results":      {r["experiment_id"]: r for r in all_results},
        "best_model":   best["model_label"],
        "best_exp":     best,
        "ranked":       [(r["model_label"], r) for r in all_results[:20]],
        "roc_data":     roc_data,
        "feat_cols":    feat_cols,
        "X_te":         X_te_raw,
        "y_te":         y_te,
        "all_results":  all_results,
    }

