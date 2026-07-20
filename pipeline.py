"""
pipeline.py
Risk Modelling Pipeline Orchestrator

Usage
-----
python pipeline.py                          # run all 10 stages
python pipeline.py --stages 1 2 3          # run specific stages
python pipeline.py --config path/to.yaml   # custom config
python pipeline.py --from-stage 5          # resume from stage 5

The pipeline goal: try every technique, score each, sort best.
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import yaml

from core.logger import get_logger
from core.data_loader import DataLoader
from core.schema_detector import SchemaDetector
from core.memory_manager import MemoryManager


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str = "config/pipeline_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_pipeline(
    config_path: str = "config/pipeline_config.yaml",
    stages:      list = None,   # None = all
    from_stage:  int  = 1,
):
    cfg = load_config(config_path)
    log = get_logger("Pipeline", cfg.get("paths", {}).get("logs_dir", "logs"))

    log.info("=" * 72)
    log.info("  RISK MODELLING PIPELINE — Binary Classification")
    log.info("  Goal: Try every technique, score each, sort best")
    log.info("=" * 72)

    # Shared objects
    loader = DataLoader(cfg)
    schema = SchemaDetector(loader, cfg)
    mem    = MemoryManager(cfg)

    mem_info = mem.get_system_info()
    log.info(f"RAM: {mem_info.get('total_ram_gb', '?')} GB total, "
             f"{mem_info.get('available_ram_gb', '?')} GB available")

    # Carry-over between stages with fallback disk loader for resuming
    artifacts_dir = Path(cfg.get("paths", {}).get("artifacts_dir", "artifacts"))
    
    import json
    
    # 1. fe_spec
    fe_spec = {}
    fe_spec_path = artifacts_dir / "feature_engineering_spec.json"
    if fe_spec_path.exists():
        try:
            with open(fe_spec_path, "r") as f:
                fe_spec = json.load(f)
        except Exception:
            pass

    # 2. selected_features
    selected_features = []
    selected_features_path = artifacts_dir / "selected_features.json"
    if selected_features_path.exists():
        try:
            with open(selected_features_path, "r") as f:
                selected_features = json.load(f)
        except Exception:
            pass

    # 3. prep_info
    prep_info = {}
    prep_info_path = artifacts_dir / "train_test_split_info.json"
    if prep_info_path.exists():
        try:
            with open(prep_info_path, "r") as f:
                prep_info = json.load(f)
        except Exception:
            pass

    # 4. model_output
    model_output = {}
    model_results_path = artifacts_dir / "model_results.json"
    if model_results_path.exists():
        try:
            with open(model_results_path, "r") as f:
                model_results = json.load(f)
                model_output = {
                    "results": {r["experiment_id"]: r for r in model_results.get("ranking", [])},
                    "best_model": model_results.get("best_model", ""),
                    "best_exp": model_results.get("best_experiment", {}),
                    "ranked": [(r["model_label"], r) for r in model_results.get("ranking", [])[:20]],
                    "feat_cols": model_results.get("best_experiment", {}).get("n_features", 0),
                }
        except Exception:
            pass

    all_stages = list(range(1, 11))
    run_stages  = stages if stages else [s for s in all_stages if s >= from_stage]

    stage_status = {}

    pipeline_t0  = time.perf_counter()

    # ── Stage 01 ─────────────────────────────────────────────────────
    if 1 in run_stages:
        try:
            from stages import data_overview as s01
            p = s01.run(loader, schema, cfg)
            stage_status[1] = f"✅ {p.name}"
        except Exception:
            log.error(f"Stage 01 failed:\n{traceback.format_exc()}")
            stage_status[1] = "❌ Failed"

    # ── Stage 02 ─────────────────────────────────────────────────────
    if 2 in run_stages:
        try:
            from stages import data_quality as s02
            p = s02.run(loader, schema, cfg)
            stage_status[2] = f"✅ {p.name}"
        except Exception:
            log.error(f"Stage 02 failed:\n{traceback.format_exc()}")
            stage_status[2] = "❌ Failed"

    # ── Stage 03 ─────────────────────────────────────────────────────
    if 3 in run_stages:
        try:
            from stages import eda as s03
            p = s03.run(loader, schema, cfg)
            stage_status[3] = f"✅ {p.name}"
        except Exception:
            log.error(f"Stage 03 failed:\n{traceback.format_exc()}")
            stage_status[3] = "❌ Failed"

    # ── Stage 04 ─────────────────────────────────────────────────────
    if 4 in run_stages:
        try:
            from stages import feature_engineering as s04
            p, fe_spec = s04.run(loader, schema, cfg)
            stage_status[4] = f"✅ {p.name}"
        except Exception:
            log.error(f"Stage 04 failed:\n{traceback.format_exc()}")
            stage_status[4] = "❌ Failed"

    # ── Stage 05 — FEATURE SELECTION ─────────────────────────────────
    if 5 in run_stages:
        try:
            from stages import feature_selection as s05
            p, selected_features = s05.run(loader, schema, cfg, fe_spec)
            stage_status[5] = f"✅ {p.name}  ({len(selected_features)} features selected)"
        except Exception:
            log.error(f"Stage 05 failed:\n{traceback.format_exc()}")
            stage_status[5] = "❌ Failed"
            # Fall back to all numeric features
            selected_features = schema.get_numeric_cols()[:200]

    # ── Stage 06 ─────────────────────────────────────────────────────
    if 6 in run_stages:
        try:
            from stages import model_preparation as s06
            p, prep_info = s06.run(loader, schema, cfg, selected_features)
            stage_status[6] = f"✅ {p.name}"
        except Exception:
            log.error(f"Stage 06 failed:\n{traceback.format_exc()}")
            stage_status[6] = "❌ Failed"

    # ── Stage 07 — MODEL COMPARISON ──────────────────────────────────
    if 7 in run_stages:
        try:
            from stages import model_comparison as s07
            p, model_output = s07.run(loader, schema, cfg, selected_features, prep_info)
            best = model_output.get("best_model", "?")
            best_auc = model_output.get("results", {}).get(best, {}).get("metrics", {}).get("roc_auc", "?")
            stage_status[7] = f"✅ {p.name}  (best: {best} AUC={best_auc})"
        except Exception:
            log.error(f"Stage 07 failed:\n{traceback.format_exc()}")
            stage_status[7] = "❌ Failed"

    # ── Stage 08 ─────────────────────────────────────────────────────
    if 8 in run_stages:
        try:
            from stages import best_model as s08
            p = s08.run(loader, cfg, model_output)
            stage_status[8] = f"✅ {p.name}"
        except Exception:
            log.error(f"Stage 08 failed:\n{traceback.format_exc()}")
            stage_status[8] = "❌ Failed"

    # ── Stage 09 ─────────────────────────────────────────────────────
    if 9 in run_stages:
        try:
            from stages import feature_importance as s09
            p = s09.run(loader, cfg, model_output, selected_features)
            stage_status[9] = f"✅ {p.name}"
        except Exception:
            log.error(f"Stage 09 failed:\n{traceback.format_exc()}")
            stage_status[9] = "❌ Failed"

    # ── Stage 10 ─────────────────────────────────────────────────────
    if 10 in run_stages:
        try:
            from stages import executive_summary as s10
            p = s10.run(loader, schema, cfg, model_output, selected_features)
            stage_status[10] = f"✅ {p.name}"
        except Exception:
            log.error(f"Stage 10 failed:\n{traceback.format_exc()}")
            stage_status[10] = "❌ Failed"

    # ── Final summary ─────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - pipeline_t0
    log.info("")
    log.info("=" * 72)
    log.info("  PIPELINE COMPLETE")
    log.info(f"  Total time: {total_elapsed/60:.1f} min")
    log.info("=" * 72)
    for stage_num, status in stage_status.items():
        log.info(f"  Stage {stage_num:02d}: {status}")
    log.info("=" * 72)
    reports_dir = Path(cfg.get("reporting", {}).get("output_dir", "reports"))
    log.info(f"  Reports: {reports_dir.resolve()}/index.html")
    log.info("=" * 72)


# ── Stage alias imports ───────────────────────────────────────────────────────
# Friendly module aliases so pipeline.py can import as "from stages import eda"
import importlib, os

def _alias(module_name: str, file_path: str):
    """Register a file-based module under stages.<alias>."""
    spec = importlib.util.spec_from_file_location(
        f"stages.{module_name}", file_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"stages.{module_name}"] = mod
    spec.loader.exec_module(mod)

_STAGE_FILES = {
    "data_overview":       "stages/01_data_overview.py",
    "data_quality":        "stages/02_data_quality.py",
    "eda":                 "stages/03_eda.py",
    "feature_engineering": "stages/04_feature_engineering.py",
    "feature_selection":   "stages/05_feature_selection.py",
    "model_preparation":   "stages/06_model_preparation.py",
    "model_comparison":    "stages/07_model_comparison.py",
    "best_model":          "stages/08_best_model.py",
    "feature_importance":  "stages/09_feature_importance.py",
    "executive_summary":   "stages/10_executive_summary.py",
}

for _alias_name, _fpath in _STAGE_FILES.items():
    if os.path.exists(_fpath):
        _alias(_alias_name, _fpath)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Risk Modelling Pipeline — Binary Classification"
    )
    parser.add_argument("--config",      default="config/pipeline_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--stages",      nargs="+", type=int,
                        help="Specific stages to run (e.g. --stages 1 2 5)")
    parser.add_argument("--from-stage",  type=int, default=1,
                        help="Run from this stage onwards")
    args = parser.parse_args()

    run_pipeline(
        config_path = args.config,
        stages      = args.stages,
        from_stage  = args.from_stage,
    )
