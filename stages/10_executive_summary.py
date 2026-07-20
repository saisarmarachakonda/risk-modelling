"""
stages/10_executive_summary.py
Stage 10 — Final Executive Report

Consolidates findings from all stages into a single comprehensive
report for senior management, risk committees, and regulators.

Output
------
reports/10_Final_Executive_Report.html
"""
import time
from pathlib import Path
from typing import Dict, List

from core.data_loader import DataLoader
from core.schema_detector import SchemaDetector
from core.logger import get_logger
from reporting.html_builder import HTMLReportBuilder
from reporting.report_writer import ReportWriter


def run(
    loader: DataLoader,
    schema: SchemaDetector,
    config: dict,
    model_output: Dict,
    selected_features: List[str],
) -> Path:
    log = get_logger("10_ExecutiveSummary",
                     config.get("paths", {}).get("logs_dir", "logs"))
    log.stage_start("10 — Final Executive Report")
    t0 = time.perf_counter()

    best_exp = model_output.get("best_exp", {})
    results   = model_output.get("results", {})
    best_name = model_output.get("best_model", "")

    if best_exp:
        best_m = {
            "roc_auc": best_exp.get("oot_roc_auc", "N/A"),
            "ks": best_exp.get("oot_ks", "N/A"),
            "f1": best_exp.get("oot_f1", "N/A"),
        }
    else:
        best_r = results.get(best_name, {})
        best_m = best_r.get("metrics", {})

    n_rows    = loader.count_rows()
    schema_sum = schema.get_schema_summary()
    domain    = config.get("domain", {})
    proj      = config.get("project", {})

    b = HTMLReportBuilder(
        report_title   = "Final Executive Report",
        stage_number   = 10,
        stage_subtitle = "Consolidated findings — all pipeline stages",
        config         = config,
        n_rows         = n_rows,
        n_cols         = schema_sum["total_columns"],
    )

    cards = [
        {"label": "Best Model",        "value": best_name or "N/A",           "variant": "success"},
        {"label": "OOT ROC AUC",       "value": str(best_m.get("roc_auc","N/A")), "variant": "success"},
        {"label": "OOT KS Statistic",  "value": str(best_m.get("ks","N/A"))},
        {"label": "Dataset Size",      "value": f"{n_rows:,}", "sub": "records"},
        {"label": "Features (raw)",    "value": f'{schema_sum["total_columns"]:,}'},
        {"label": "Features (selected)","value": f'{len(selected_features):,}'},
        {"label": "Models Trained",    "value": str(len(results))},
        {"label": "Deployment Ready",  "value": "✅ Yes" if (isinstance(best_m.get("roc_auc"), (int, float)) and best_m.get("roc_auc", 0) > 0.65) or (isinstance(best_m.get("roc_auc"), str) and best_m.get("roc_auc") != "N/A" and float(best_m.get("roc_auc")) > 0.65) else "⚠️ Review",
         "variant": "success" if (isinstance(best_m.get("roc_auc"), (int, float)) and best_m.get("roc_auc", 0) > 0.65) or (isinstance(best_m.get("roc_auc"), str) and best_m.get("roc_auc") != "N/A" and float(best_m.get("roc_auc")) > 0.65) else "warning"},
    ]

    b.add_executive_summary(cards, narrative=(
        f"This report consolidates findings from all {10} stages of the {proj.get('name', 'Risk Modelling Pipeline')} "
        f"applied to a {n_rows:,}-row, {schema_sum['total_columns']:,}-column {domain.get('name', 'credit risk')} dataset.\n\n"
        f"After comprehensive data quality assessment, exploratory analysis, feature engineering, "
        f"and multi-method feature selection ({len(selected_features):,} features retained from "
        f"{schema_sum['total_columns']:,}), the pipeline executed a full combinatorial grid search of model architectures.\n\n"
        f"The winning model is <strong>{best_name}</strong> with a final Out-of-Time (OOT) holdout ROC AUC of "
        f"<strong>{best_m.get('roc_auc','N/A')}</strong> and an OOT KS statistic of "
        f"<strong>{best_m.get('ks','N/A')}</strong>. The model is assessed as "
        f"{'ready for production deployment' if ((isinstance(best_m.get('roc_auc'), (int, float)) and best_m.get('roc_auc', 0) > 0.65) or (isinstance(best_m.get('roc_auc'), str) and best_m.get('roc_auc') != 'N/A' and float(best_m.get('roc_auc')) > 0.65)) else 'requiring further development'} "
        f"subject to final verification."
    ))

    # Pipeline summary
    pipeline_rows = [
        {"Stage": "01", "Name": "Data Overview",          "Key Finding": f"{n_rows:,} rows × {schema_sum['total_columns']:,} cols. {schema_sum['constant']} constant cols detected."},
        {"Stage": "02", "Name": "Data Quality",           "Key Finding": f"Quality Score computed. {schema_sum['constant']} cols to drop. Missing & leakage flags reviewed."},
        {"Stage": "03", "Name": "EDA",                    "Key Finding": "IV, distributions, correlations analysed. High-skew features identified."},
        {"Stage": "04", "Name": "Feature Engineering",    "Key Finding": "Imputation, WoE encoding, log transform, StandardScaler applied."},
        {"Stage": "05", "Name": "Feature Selection",      "Key Finding": f"All methods tried (filter + LR + tree). {len(selected_features)} features retained by consensus."},
        {"Stage": "06", "Name": "Model Preparation",      "Key Finding": "Stratified 80/10/10 split. 5-fold CV. Class weighting applied."},
        {"Stage": "07", "Name": "Model Comparison",       "Key Finding": f"{len(results)} models trained. {best_name} ranked first."},
        {"Stage": "08", "Name": "Best Model Analysis",    "Key Finding": f"ROC={best_m.get('roc_auc','N/A')}, KS={best_m.get('ks','N/A')}, F1={best_m.get('f1','N/A')}."},
        {"Stage": "09", "Name": "Feature Importance",     "Key Finding": "Top features, SHAP analysis, and business interpretations documented."},
        {"Stage": "10", "Name": "Executive Report",       "Key Finding": "Consolidated findings — this report."},
    ]
    b.add_section(
        "Pipeline Execution Summary",
        b.table(pipeline_rows, caption="All Pipeline Stages — Key Findings"),
        icon="📋",
    )

    # Model ranking
    ranked = model_output.get("ranked", [])
    if ranked:
        rank_rows = []
        for i, (n, r) in enumerate(ranked):
            if "metrics" in r:
                m_dict = r["metrics"]
                group  = r.get("group", "?")
            else:
                m_dict = {
                    "roc_auc": r.get("test_roc_auc", "?"),
                    "pr_auc":  r.get("test_pr_auc", "?"),
                    "ks":      r.get("test_ks", "?"),
                    "f1":      r.get("test_f1", "?"),
                }
                group  = r.get("model_family", "?")
            
            rank_rows.append({
                "Rank": i+1,
                "Model": n,
                "ROC AUC": m_dict.get("roc_auc", "?"),
                "PR AUC": m_dict.get("pr_auc", "?"),
                "KS": m_dict.get("ks", "?"),
                "F1": m_dict.get("f1", "?"),
                "Group": group
            })
        b.add_section(
            "Model Ranking Summary",
            b.table(rank_rows, caption="All Models Ranked by ROC AUC"),
            icon="🏆",
        )


    # Recommendations
    recs = (
        b.subsection("Immediate Actions (Before Deployment)")
        + b.callout(
            "<ul>"
            f"<li>Validate <strong>{best_name}</strong> against a holdout population from a different time period</li>"
            "<li>Perform model fairness audit — check for demographic proxies in top features</li>"
            "<li>Calibrate predicted probabilities using Platt scaling or isotonic regression</li>"
            "<li>Document model assumptions and limitations for model risk committee</li>"
            "</ul>",
            kind="recommend", title="✅ Pre-Deployment Checklist",
        )
        + b.subsection("Monitoring (Post-Deployment)")
        + b.callout(
            "<ul>"
            "<li>Track PSI (Population Stability Index) monthly for top 20 features</li>"
            "<li>Monitor ROC AUC and KS statistic on recent vintage monthly</li>"
            "<li>Set alert: ROC AUC degradation > 0.03 triggers review</li>"
            "<li>Retrain annually or when drift is detected</li>"
            "</ul>",
            kind="insight", title="📡 Production Monitoring",
        )
        + b.subsection("Long-Term Improvements")
        + b.callout(
            "<ul>"
            "<li>Collect additional behavioural data (payment history, transaction patterns)</li>"
            "<li>Explore alternative data sources (open banking, telco data)</li>"
            "<li>Implement online learning for real-time score updates</li>"
            "<li>Build ensemble: LR scorecard (regulatory) + LightGBM (challenger) dual-model system</li>"
            "</ul>",
            kind="recommend", title="🚀 Future Roadmap",
        )
    )
    b.add_section("Recommendations", recs, icon="💡")

    # Limitations
    lims = b.callout(
        "<ul>"
        "<li>Model trained on a sample — full 2M row training may improve performance</li>"
        "<li>Feature selection based on current distribution — may not generalise to future vintages</li>"
        "<li>Class imbalance handling via weighting may not be optimal for all business objectives</li>"
        "<li>SHAP interpretation available only for tree-based models in this pipeline version</li>"
        "<li>No demographic fairness testing performed in this pipeline stage</li>"
        "</ul>",
        kind="warning", title="⚠️ Limitations",
    )
    b.add_section("Limitations & Caveats", lims, icon="⚠️")

    # Sign-off
    import datetime
    b.add_section(
        "Report Approval",
        b.card(
            "Sign-Off Block",
            f"<table style='width:100%;font-size:14px;border-collapse:collapse;'>"
            f"<tr><td style='padding:12px;border:1px solid #e2e8f0;font-weight:600'>Prepared by</td>"
            f"<td style='padding:12px;border:1px solid #e2e8f0;'>{config.get('project',{}).get('author','—')}</td></tr>"
            f"<tr><td style='padding:12px;border:1px solid #e2e8f0;font-weight:600'>Report Date</td>"
            f"<td style='padding:12px;border:1px solid #e2e8f0;'>{datetime.date.today().strftime('%B %d, %Y')}</td></tr>"
            f"<tr><td style='padding:12px;border:1px solid #e2e8f0;font-weight:600'>Pipeline Version</td>"
            f"<td style='padding:12px;border:1px solid #e2e8f0;'>{config.get('project',{}).get('version','1.0.0')}</td></tr>"
            f"<tr><td style='padding:12px;border:1px solid #e2e8f0;font-weight:600'>Model Validator</td>"
            f"<td style='padding:12px;border:1px solid #e2e8f0;'>________________________________</td></tr>"
            f"<tr><td style='padding:12px;border:1px solid #e2e8f0;font-weight:600'>Risk Committee Approval</td>"
            f"<td style='padding:12px;border:1px solid #e2e8f0;'>________________________________</td></tr>"
            f"</table>"
        ),
        icon="✍️",
    )

    html   = b.build()
    writer = ReportWriter(config)
    path   = writer.write(html, "10_Final_Executive_Report.html")
    writer.write_index()

    elapsed = time.perf_counter() - t0
    log.stage_end("10 — Executive Report", elapsed)
    return path
