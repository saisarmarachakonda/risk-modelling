"""
stages/08_best_model.py
Stage 08 — Best Model Deep-Dive Report

Output
------
reports/08_Best_Model_Report.html
"""
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from core.data_loader import DataLoader
from core.logger import get_logger
from reporting.html_builder import HTMLReportBuilder
from reporting.figure_generator import (
    confusion_matrix_plot, roc_curves, pr_curves,
    ks_curve, gain_lift_curves, calibration_curve_plot,
    learning_curve_plot,
)
from reporting.report_writer import ReportWriter


def run(
    loader: DataLoader,
    config: dict,
    model_output: Dict,
) -> Path:
    log = get_logger("08_BestModel", config.get("paths", {}).get("logs_dir", "logs"))
    log.stage_start("08 — Best Model Deep-Dive")
    t0 = time.perf_counter()

    # Support both old per-model dict and new combinatorial grid output
    best_exp     = model_output.get("best_exp", {})
    all_results  = model_output.get("all_results", [])
    best_name    = model_output.get("best_model", best_exp.get("model_label", ""))
    feat_cols    = model_output.get("feat_cols", [])
    X_te         = model_output.get("X_te", np.array([]))
    y_te         = model_output.get("y_te", np.array([]))
    roc_data_all = model_output.get("roc_data", [])

    # Build metrics dict from best_exp (combinatorial) or old results
    if best_exp:
        m = {
            "roc_auc":       best_exp.get("oot_roc_auc",      "N/A"),
            "pr_auc":        best_exp.get("oot_pr_auc",       "N/A"),
            "f1":            best_exp.get("oot_f1",           "N/A"),
            "recall":        best_exp.get("oot_recall",       "N/A"),
            "specificity":   best_exp.get("oot_specificity",  "N/A"),
            "ks":            best_exp.get("oot_ks",           "N/A"),
            "test_roc_auc":  best_exp.get("test_roc_auc",     "N/A"),
            "cv_roc_auc_mean": best_exp.get("cv_roc_auc_mean","N/A"),
            "cv_roc_auc_std":  best_exp.get("cv_roc_auc_std", "N/A"),
            "train_time_s":  best_exp.get("train_time_s",    "N/A"),
        }
        group = best_exp.get("model_family", "Tree-Based")
    else:
        results = model_output.get("results", {})
        best_r  = results.get(best_name, {})
        m       = best_r.get("metrics", {})
        group   = best_r.get("group", "Tree-Based")


    b = HTMLReportBuilder(
        report_title   = f"Best Model Report — {best_name}",
        stage_number   = 8,
        stage_subtitle = f"Deep analysis of the winning model: {best_name}",
        config         = config,
        n_rows         = loader.count_rows(),
        n_cols         = len(feat_cols),
    )

    cards = [
        {"label": "Best Model",         "value": best_name,                       "variant": "success"},
        {"label": "OOT ROC AUC",        "value": str(m.get("roc_auc",  "N/A")),   "variant": "success"},
        {"label": "Test ROC AUC",       "value": str(m.get("test_roc_auc", "N/A"))},
        {"label": "OOT PR AUC",         "value": str(m.get("pr_auc",   "N/A"))},
        {"label": "OOT F1 Score",       "value": str(m.get("f1",       "N/A"))},
        {"label": "OOT Recall",         "value": str(m.get("recall",   "N/A")),   "variant": "warning"},
        {"label": "OOT Specificity",    "value": str(m.get("specificity","N/A"))},
        {"label": "OOT KS Statistic",   "value": str(m.get("ks",       "N/A"))},
    ]
    # group already set above based on best_exp or results

    why_won = (
        f"<strong>{best_name}</strong> achieved the highest ROC AUC of {m.get('roc_auc','N/A')} "
        f"on the held-out test set. "
        + (
            "As a tree-based model, it captures non-linear relationships and feature interactions "
            "that logistic regression cannot model without manual feature engineering. "
            "It is also robust to outliers and does not require feature scaling."
            if group == "Tree-Based"
            else
            "As a logistic regression model, it provides directly interpretable coefficients "
            "(log-odds) for each feature, strong regulatory transparency, and reliable "
            "probability calibration without post-hoc adjustments."
        )
    )
    b.add_executive_summary(cards, narrative=why_won)

    # Confusion matrix
    if len(y_te) > 0 and len(X_te) > 0:
        # Re-predict using best model's stored data via roc_data
        best_roc = next((r for r in roc_data_all if r["name"] == best_name), None)
        if best_roc:
            fpr_arr = np.array(best_roc["fpr"])
            tpr_arr = np.array(best_roc["tpr"])

            roc_fig = roc_curves(
                [best_roc],
                title=f"ROC Curve — {best_name}",
            )
            ks_fig  = ks_curve(fpr_arr, tpr_arr,
                               title=f"KS Curve — {best_name}")

            b.add_section(
                "ROC & KS Analysis",
                b.figure(roc_fig, "ROC Curve", interpretation=(
                    f"ROC AUC = {m.get('roc_auc','N/A')}. The curve shows how the model's "
                    "true positive rate and false positive rate trade off across all thresholds. "
                    "A KS statistic of " + str(m.get('ks','N/A')) + " indicates the model's "
                    "maximum separation between defaulters and non-defaulters."
                ))
                + b.figure(ks_fig, "KS Curve", interpretation=(
                    "The KS curve shows cumulative sensitivity (TPR) and 1-specificity (FPR) "
                    "as the threshold varies. The maximum vertical distance is the KS statistic. "
                    "In credit risk, KS > 40 is considered good; KS > 50 is excellent."
                )),
                icon="📈",
            )

    # Metrics deep-dive
    metrics_rows = [
        {"Metric": "ROC AUC",           "Value": str(m.get("roc_auc","?")),
         "Interpretation": "Probability that model ranks a defaulter above a non-defaulter"},
        {"Metric": "PR AUC",            "Value": str(m.get("pr_auc","?")),
         "Interpretation": "Area under precision-recall curve — more sensitive to class imbalance"},
        {"Metric": "Accuracy",          "Value": str(m.get("accuracy","?")),
         "Interpretation": "Overall correct classification rate — misleading for imbalanced data"},
        {"Metric": "Precision",         "Value": str(m.get("precision","?")),
         "Interpretation": "Of predicted defaults, what fraction are actual defaults (reduces false alarms)"},
        {"Metric": "Recall (Sensitivity)","Value": str(m.get("recall","?")),
         "Interpretation": "Of actual defaults, what fraction did the model catch (reduces missed defaults)"},
        {"Metric": "Specificity",       "Value": str(m.get("specificity","?")),
         "Interpretation": "Of actual non-defaults, what fraction correctly classified"},
        {"Metric": "F1 Score",          "Value": str(m.get("f1","?")),
         "Interpretation": "Harmonic mean of precision and recall"},
        {"Metric": "MCC",               "Value": str(m.get("mcc","?")),
         "Interpretation": "Matthews Correlation — balanced metric for imbalanced classes (-1 to +1)"},
        {"Metric": "KS Statistic",      "Value": str(m.get("ks","?")),
         "Interpretation": "Maximum separation between defaulter and non-defaulter score distributions"},
        {"Metric": "Brier Score",       "Value": str(m.get("brier","?")),
         "Interpretation": "Mean squared error of probability predictions — lower is better calibrated"},
        {"Metric": "Log Loss",          "Value": str(m.get("log_loss","?")),
         "Interpretation": "Cross-entropy loss — penalises confident wrong predictions heavily"},
        {"Metric": "Train Time (s)",    "Value": str(m.get("train_time_s","?")),
         "Interpretation": "Wall-clock training time on sample"},
    ]

    b.add_section(
        "Detailed Metric Analysis",
        b.p(
            "Every metric is reported with its interpretation to provide a complete picture "
            "of model performance. No single metric tells the full story — "
            "ROC AUC, PR AUC, and KS together provide a robust view for credit risk applications."
        )
        + b.table(metrics_rows, caption=f"Performance Metrics — {best_name}"),
        icon="📐",
    )

    # Strengths / weaknesses
    if group == "Tree-Based":
        strengths = ["Highest ROC AUC among all models",
                     "Handles non-linear risk patterns automatically",
                     "Robust to outliers in feature space",
                     "Scale-invariant - no normalisation required",
                     "Captures feature interactions"]
        weaknesses = ["Less interpretable than logistic regression",
                      "SHAP required for individual explanations",
                      "Can overfit on small datasets",
                      "May not be regulatory approved in all jurisdictions"]
    else:
        strengths = ["Highest ROC AUC among all models",
                     "Directly interpretable log-odds coefficients",
                     "Excellent probability calibration",
                     "Regularisation controls overfitting",
                     "Regulatory approved in many jurisdictions"]
        weaknesses = ["Cannot capture non-linear interactions",
                      "Sensitive to multicollinearity",
                      "Requires feature scaling",
                      "Sparsity needed for high-dimensional data"]

    str_html = "".join(f"<li>{s}</li>" for s in strengths)
    wk_html  = "".join(f"<li>{s}</li>" for s in weaknesses)
    sw_body = (
        "<table style='width:100%;font-size:14px;border-collapse:collapse'>"
        "<thead style='background:#0f2d52;color:white'><tr>"
        "<th style='padding:10px'>Strengths</th>"
        "<th style='padding:10px'>Weaknesses</th>"
        "</tr></thead><tbody><tr>"
        "<td style='padding:10px;vertical-align:top'><ul>" + str_html + "</ul></td>"
        "<td style='padding:10px;vertical-align:top'><ul>" + wk_html + "</ul></td>"
        "</tr></tbody></table>"
    )
    sw_content = b.card(f"Strengths and Weaknesses - {best_name}", sw_body)
    b.add_section("Strengths and Weaknesses", sw_content, icon="[+/-]")


    # Business recommendations
    biz_content = (
        b.callout(
            f"<strong>Production Deployment:</strong> {best_name} is recommended as the "
            f"primary risk scoring model. Predicted probability should be calibrated and "
            f"converted to a scorecard points system for operational use.",
            kind="recommend",
        )
        + b.callout(
            "<strong>Monitoring:</strong> Track ROC AUC, KS statistic, and PSI (Population "
            "Stability Index) monthly. Alert threshold: ROC AUC drops >0.03 or KS drops >0.05.",
            kind="insight",
        )
        + b.callout(
            "<strong>Threshold Setting:</strong> The default 0.5 threshold maximises accuracy "
            "but may not be optimal. For credit risk, set threshold to align with the portfolio's "
            "target bad rate or cost-benefit ratio (cost of default vs cost of lost business).",
            kind="warning",
        )
    )
    b.add_section("Business Recommendations", biz_content, icon="💼")

    html   = b.build()
    writer = ReportWriter(config)
    path   = writer.write(html, "08_Best_Model_Report.html")
    writer.write_index()

    elapsed = time.perf_counter() - t0
    log.stage_end("08 — Best Model", elapsed)
    return path
