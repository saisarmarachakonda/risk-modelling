# Risk Modelling & Binary Classification Pipeline

A high-performance machine learning pipeline built to handle large risk datasets (millions of records, thousands of columns) efficiently using **DuckDB**, **Polars**, and **Scikit-learn** with thread-based parallel execution.

The pipeline automatically runs a 10-stage process, executes a full combinatorial grid search of feature selection methods and model algorithms, and generates offline-compatible, static HTML reports for each stage.

---

## 🛠 Pipeline Architecture

The pipeline consists of the following 10 stages:

1. **`01_data_overview`**: Initial dataset analysis, column counting, and structure checks.
2. **`02_data_quality`**: Assesses missing values, constants, duplicates, and target leakage risks.
3. **`03_eda`**: Distribution, skewness, categoricals, and bivariate Information Value (IV) analysis.
4. **`04_feature_engineering`**: Imputes missing data, creates indicators, applies log1p transforms, clips outliers, standardises features, and encodes categoricals.
5. **`05_feature_selection`**: Evaluates 17+ filter and wrapper methods to rank and recommend features by consensus.
6. **`06_model_preparation`**: Splits data (stratified train/test) and configures Stratified K-Fold validation.
7. **`07_model_comparison`**: Performs a full combinatorial grid search (315 experiments) across:
   * **7 Feature Sets**
   * **9 Model Families** (Logistic Regression, Decision Trees, Random Forests, Extra Trees, Gradient Boosting, XGBoost, Naive Bayes)
   * **3 Scalers**
   * **3 Imbalance Strategies**
8. **`08_best_model`**: Deep-dives into the winning model's calibration, lift, gain, learning curves, and strengths/weaknesses.
9. **`09_feature_importance`**: Explains model behavior using global MDI, permutation importances, and SHAP.
10. **`10_executive_summary`**: Consolidated final report for stakeholders and auditors.

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install requirements:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
Run the entire 10-stage pipeline:
```bash
python3 pipeline.py
```

Resume from a specific stage (e.g. model comparison onwards):
```bash
python3 pipeline.py --from-stage 7
```

Run specific stages only:
```bash
python3 pipeline.py --stages 1 2 3
```

---

## 📊 Feature Replication & Serving

A standalone serving utility is provided to replicate all feature engineering, preprocessing, and feature selection steps on any raw dataset using high-performance Polars operations.

To recreate model-ready features from a raw CSV:
```bash
python3 scripts/recreate_features.py --data data/sample_credit_risk.csv --spec-dir artifacts --output data/engineered_features.csv
```

This ensures identical preprocessing (imputation, winsorisation, log transforms, and category mapping) is applied to new test data without data leakage.

---

## 📈 Dashboard Reports

Static, responsive HTML reports are generated in the `reports/` folder. Open the consolidated dashboard to browse findings:
👉 **`reports/index.html`**

---

## 🔒 Ignoring Large Data files

This project uses a `.gitignore` configured to prevent staging data CSVs, logs, or pipeline execution artifacts (`artifacts/`, `reports/`, `logs/`) to keep the source repository clean and protect sensitive credit risk datasets.
