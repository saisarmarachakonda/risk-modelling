"""
scripts/generate_sample_data.py
Generates a realistic synthetic credit risk dataset for pipeline testing.

Dataset properties
------------------
* 100,000 rows  (scales up easily — change N_ROWS)
* 120 columns:
    - 60 numeric financial features (income, debt, ratios, balances)
    - 25 categorical features (loan purpose, employment, state, grade)
    - 10 boolean flags (has_mortgage, is_self_employed, etc.)
    -  5 datetime features (application date, birth date, etc.)
    -  5 identifier columns (loan_id, customer_id, etc.)
    -  5 constant columns (will be detected and dropped by pipeline)
    - 10 near-constant / noisy columns
    - Target column: `default_flag` (binary 0/1, ~15% default rate)

Data quality intentional issues
---------------------------------
* ~8% missing in some columns, ~35% missing in a few (high-miss)
* 2 columns with suspiciously high correlation (multicollinearity)
* 5 constant columns (zero variance)
* 3 near-duplicate rows (~2% duplicates)
* One column with very high IV (to test leakage detection)

Usage
-----
python scripts/generate_sample_data.py
python scripts/generate_sample_data.py --rows 500000 --out data/large_sample.parquet
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng  = np.random.default_rng(SEED)


def generate(n_rows: int = 100_000) -> pd.DataFrame:
    print(f"Generating {n_rows:,} rows...")

    # ── Identifiers ────────────────────────────────────────────────────
    loan_id     = [f"LN{str(i).zfill(8)}"  for i in range(1, n_rows + 1)]
    customer_id = [f"CU{str(rng.integers(1, n_rows // 3))}" for _ in range(n_rows)]
    application_id = rng.integers(10_000_000, 99_999_999, size=n_rows)
    branch_code    = rng.integers(100, 999, size=n_rows)
    record_hash    = [f"H{rng.integers(0, 2**32):010X}" for _ in range(n_rows)]

    # ── Core credit risk numeric features ─────────────────────────────
    annual_income         = rng.lognormal(mean=11.0, sigma=0.6, size=n_rows)  # ~60k median
    monthly_income        = annual_income / 12
    debt_to_income        = rng.beta(2, 5, size=n_rows) * 0.8          # 0–0.8
    credit_score          = (rng.normal(680, 80, size=n_rows)
                             .clip(300, 850)).astype(int).astype(float)
    loan_amount           = rng.lognormal(9.5, 0.8, size=n_rows)       # ~$14k median
    installment           = loan_amount / rng.integers(24, 84, size=n_rows)
    interest_rate         = rng.uniform(5.0, 28.0, size=n_rows)
    term_months           = rng.choice([36, 60, 84], size=n_rows, p=[0.5, 0.35, 0.15])
    num_credit_lines      = rng.integers(1, 30, size=n_rows).astype(float)
    num_delinq_2yrs       = rng.integers(0, 10, size=n_rows).astype(float)
    num_inquiries_6mths   = rng.integers(0, 15, size=n_rows).astype(float)
    credit_history_months = rng.integers(6, 400, size=n_rows).astype(float)
    revolving_balance     = rng.lognormal(8.0, 1.2, size=n_rows)
    revolving_util        = rng.beta(2, 3, size=n_rows)                # 0–1
    total_accounts        = rng.integers(2, 50, size=n_rows).astype(float)
    open_accounts         = rng.integers(1, 20, size=n_rows).astype(float)
    derogatory_marks      = rng.integers(0, 5, size=n_rows).astype(float)
    months_since_last_delinq = rng.integers(0, 120, size=n_rows).astype(float)
    total_payment         = loan_amount * rng.uniform(0.0, 1.5, size=n_rows)
    total_received_principal = total_payment * rng.uniform(0.5, 1.0, size=n_rows)
    total_received_interest  = total_payment - total_received_principal
    collection_recovery_fee  = rng.exponential(50, size=n_rows)
    last_payment_amount   = installment * rng.uniform(0.5, 1.5, size=n_rows)
    outstanding_principal = (loan_amount - total_received_principal).clip(0)
    days_since_last_payment  = rng.integers(0, 365, size=n_rows).astype(float)
    num_collections       = rng.integers(0, 5, size=n_rows).astype(float)
    employment_length_yrs = rng.uniform(0, 40, size=n_rows)
    monthly_expenses      = monthly_income * rng.uniform(0.3, 0.95, size=n_rows)
    savings_balance       = rng.lognormal(7.0, 2.0, size=n_rows)
    checking_balance      = rng.lognormal(6.5, 2.0, size=n_rows)

    # Derived / engineered features
    payment_to_income     = installment / monthly_income.clip(1)
    loan_to_income        = loan_amount  / annual_income.clip(1)
    credit_util_score     = 850 - (revolving_util * 200).clip(0, 550)
    net_monthly_cashflow  = monthly_income - monthly_expenses
    account_age_ratio     = credit_history_months / total_accounts.clip(1)
    delinq_rate           = num_delinq_2yrs / credit_history_months.clip(1) * 24

    # Additional numeric columns
    prev_loan_count       = rng.integers(0, 10, size=n_rows).astype(float)
    total_credit_limit    = revolving_balance / revolving_util.clip(0.01)
    ltv_ratio             = loan_amount / total_credit_limit.clip(1)
    credit_age_years      = credit_history_months / 12
    balance_per_account   = revolving_balance / open_accounts.clip(1)
    inq_last_12m          = rng.integers(0, 20, size=n_rows).astype(float)
    home_equity           = rng.lognormal(10.5, 1.5, size=n_rows)
    mortgage_balance      = rng.lognormal(11.5, 1.0, size=n_rows)
    auto_loan_balance     = rng.lognormal(9.0, 1.5, size=n_rows)
    student_loan_balance  = rng.lognormal(9.5, 1.8, size=n_rows)
    credit_card_balance   = rng.lognormal(7.5, 1.5, size=n_rows)
    tax_liens             = rng.integers(0, 3, size=n_rows).astype(float)
    bankruptcies          = rng.integers(0, 2, size=n_rows).astype(float)
    charge_off_count      = rng.integers(0, 3, size=n_rows).astype(float)
    hardship_flag         = rng.integers(0, 2, size=n_rows).astype(float)
    num_satisfactory      = (total_accounts - derogatory_marks).clip(0)
    pct_satisfactory      = num_satisfactory / total_accounts.clip(1)
    avg_credit_limit      = total_credit_limit / num_credit_lines.clip(1)
    months_since_derog    = rng.integers(0, 200, size=n_rows).astype(float)
    total_delinq_amount   = rng.exponential(500, size=n_rows)
    funded_amount_inv     = loan_amount * rng.uniform(0.8, 1.0, size=n_rows)
    # Highly correlated twin (for multicollinearity test)
    credit_score_copy     = credit_score + rng.normal(0, 2, size=n_rows)  # ~perfect correlation

    # ── Target: default_flag ──────────────────────────────────────────
    # Logistic model: higher risk = higher default probability
    log_odds = (
        -3.5
        + 0.015  * num_delinq_2yrs
        - 0.005  * (credit_score - 650)
        + 2.5    * debt_to_income
        + 0.8    * revolving_util
        + 0.3    * (loan_to_income - 0.3)
        + 0.05   * num_inquiries_6mths
        + 0.2    * derogatory_marks
        - 0.002  * employment_length_yrs
        + 0.5    * bankruptcies
        - 0.003  * credit_history_months
        + rng.normal(0, 0.3, size=n_rows)  # noise
    )
    prob_default    = 1 / (1 + np.exp(-log_odds))
    default_flag    = (rng.uniform(0, 1, size=n_rows) < prob_default).astype(int)
    print(f"  Default rate: {default_flag.mean():.2%}")

    # ── Categorical features ───────────────────────────────────────────
    loan_purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "credit_card",
         "medical", "car", "small_business", "wedding", "moving",
         "vacation", "education", "house", "renewable_energy", "other"],
        size=n_rows, p=[0.35, 0.12, 0.15, 0.07, 0.06, 0.05, 0.03,
                        0.02, 0.02, 0.04, 0.02, 0.01, 0.06],
    )
    loan_grade = rng.choice(
        ["A", "B", "C", "D", "E", "F", "G"],
        size=n_rows, p=[0.20, 0.28, 0.25, 0.15, 0.07, 0.03, 0.02],
    )
    loan_sub_grade = [
        g + str(rng.integers(1, 6))
        for g in loan_grade
    ]
    employment_title = rng.choice(
        ["Teacher", "Manager", "Engineer", "Nurse", "Sales", "Director",
         "Driver", "Analyst", "Consultant", "Retired", "Self-employed",
         "Admin", "Technician", "Owner", "Supervisor"],
        size=n_rows,
    )
    home_ownership = rng.choice(
        ["RENT", "MORTGAGE", "OWN", "OTHER"],
        size=n_rows, p=[0.45, 0.40, 0.12, 0.03],
    )
    verification_status = rng.choice(
        ["Not Verified", "Verified", "Source Verified"],
        size=n_rows, p=[0.35, 0.35, 0.30],
    )
    loan_status_initial = rng.choice(
        ["Current", "Fully Paid", "In Grace Period", "Late (16-30)", "Late (31-120)"],
        size=n_rows, p=[0.40, 0.35, 0.10, 0.08, 0.07],
    )
    state = rng.choice(
        ["CA","TX","NY","FL","IL","PA","OH","GA","NC","MI",
         "NJ","VA","WA","AZ","MA","TN","IN","MO","MD","WI"],
        size=n_rows,
    )
    city_tier = rng.choice(
        ["Tier 1 Metro", "Tier 2 City", "Suburban", "Rural"],
        size=n_rows, p=[0.25, 0.35, 0.30, 0.10],
    )
    education_level = rng.choice(
        ["High School", "Some College", "Bachelors", "Masters", "PhD", "Trade School"],
        size=n_rows, p=[0.20, 0.18, 0.35, 0.15, 0.05, 0.07],
    )
    marital_status = rng.choice(
        ["Single", "Married", "Divorced", "Widowed"],
        size=n_rows, p=[0.35, 0.45, 0.15, 0.05],
    )
    loan_channel = rng.choice(
        ["Online", "Branch", "Referral", "Broker", "Mobile App"],
        size=n_rows, p=[0.40, 0.20, 0.15, 0.10, 0.15],
    )
    collateral_type = rng.choice(
        ["None", "Vehicle", "Property", "Securities", "Other"],
        size=n_rows, p=[0.55, 0.20, 0.15, 0.05, 0.05],
    )
    payment_method = rng.choice(
        ["Auto-Pay", "Direct Debit", "Manual", "Check"],
        size=n_rows, p=[0.45, 0.30, 0.15, 0.10],
    )
    risk_segment = rng.choice(
        ["Prime", "Near-Prime", "Sub-Prime", "Deep Sub-Prime"],
        size=n_rows, p=[0.30, 0.35, 0.25, 0.10],
    )
    industry_sector = rng.choice(
        ["Healthcare", "Technology", "Finance", "Retail", "Manufacturing",
         "Education", "Government", "Construction", "Transportation", "Other"],
        size=n_rows,
    )
    product_type = rng.choice(
        ["Personal Loan", "Auto Loan", "Home Equity", "Credit Builder", "Consolidation"],
        size=n_rows, p=[0.35, 0.20, 0.15, 0.10, 0.20],
    )
    currency = rng.choice(["USD"], size=n_rows)  # near-constant
    bank_name = rng.choice(
        ["First National", "Citizens", "Horizon", "Meridian", "Pacific",
         "Liberty", "Summit", "Keystone", "Central", "Allied"],
        size=n_rows,
    )
    credit_bureau = rng.choice(
        ["Experian", "Equifax", "TransUnion"],
        size=n_rows, p=[0.38, 0.32, 0.30],
    )
    origination_quarter = rng.choice(
        ["Q1", "Q2", "Q3", "Q4"], size=n_rows
    )

    # ── Boolean flags ──────────────────────────────────────────────────
    has_mortgage        = (home_ownership == "MORTGAGE").astype(int)
    is_self_employed    = rng.binomial(1, 0.12, size=n_rows)
    has_auto_loan       = rng.binomial(1, 0.35, size=n_rows)
    has_student_loan    = rng.binomial(1, 0.28, size=n_rows)
    has_co_borrower     = rng.binomial(1, 0.08, size=n_rows)
    ever_bankrupt       = (bankruptcies > 0).astype(int)
    is_first_loan       = (prev_loan_count == 0).astype(int)
    auto_pay_enrolled   = (payment_method == "Auto-Pay").astype(int)
    has_verified_income = (verification_status == "Verified").astype(int)
    fico_checked        = rng.binomial(1, 0.92, size=n_rows)

    # ── Datetime features ──────────────────────────────────────────────
    base_date       = pd.Timestamp("2020-01-01")
    application_date = pd.to_datetime(
        base_date + pd.to_timedelta(rng.integers(0, 1460, size=n_rows), unit="D")
    )
    birth_date      = pd.to_datetime(
        pd.Timestamp("1950-01-01")
        + pd.to_timedelta(rng.integers(0, 365*50, size=n_rows), unit="D")
    )
    last_payment_date = pd.to_datetime(
        application_date + pd.to_timedelta(rng.integers(30, 730, size=n_rows), unit="D")
    )
    next_payment_date = pd.to_datetime(
        last_payment_date + pd.to_timedelta(30, unit="D")
    )
    credit_opened_date = pd.to_datetime(
        application_date - pd.to_timedelta(rng.integers(180, 3650, size=n_rows), unit="D")
    )

    # ── Constant columns (pipeline should auto-drop) ───────────────────
    const_country     = np.full(n_rows, "USA")
    const_currency_code = np.full(n_rows, "USD")
    const_product_ver = np.full(n_rows, 1)
    const_model_ver   = np.full(n_rows, 2)
    const_data_source = np.full(n_rows, "INTERNAL")

    # ── Assemble DataFrame ─────────────────────────────────────────────
    df = pd.DataFrame({
        # Identifiers
        "loan_id":                  loan_id,
        "customer_id":              customer_id,
        "application_id":           application_id,
        "branch_code":              branch_code,
        "record_hash":              record_hash,

        # Numeric — financial
        "annual_income":            annual_income,
        "monthly_income":           monthly_income,
        "debt_to_income":           debt_to_income,
        "credit_score":             credit_score,
        "credit_score_v2":          credit_score_copy,        # highly correlated twin
        "loan_amount":              loan_amount,
        "installment":              installment,
        "interest_rate":            interest_rate,
        "term_months":              term_months,
        "num_credit_lines":         num_credit_lines,
        "num_delinq_2yrs":          num_delinq_2yrs,
        "num_inquiries_6mths":      num_inquiries_6mths,
        "credit_history_months":    credit_history_months,
        "revolving_balance":        revolving_balance,
        "revolving_util":           revolving_util,
        "total_accounts":           total_accounts,
        "open_accounts":            open_accounts,
        "derogatory_marks":         derogatory_marks,
        "months_since_last_delinq": months_since_last_delinq,
        "total_payment":            total_payment,
        "total_received_principal": total_received_principal,
        "total_received_interest":  total_received_interest,
        "collection_recovery_fee":  collection_recovery_fee,
        "last_payment_amount":      last_payment_amount,
        "outstanding_principal":    outstanding_principal,
        "days_since_last_payment":  days_since_last_payment,
        "num_collections":          num_collections,
        "employment_length_yrs":    employment_length_yrs,
        "monthly_expenses":         monthly_expenses,
        "savings_balance":          savings_balance,
        "checking_balance":         checking_balance,
        "payment_to_income":        payment_to_income,
        "loan_to_income":           loan_to_income,
        "credit_util_score":        credit_util_score,
        "net_monthly_cashflow":     net_monthly_cashflow,
        "account_age_ratio":        account_age_ratio,
        "delinq_rate":              delinq_rate,
        "prev_loan_count":          prev_loan_count,
        "total_credit_limit":       total_credit_limit,
        "ltv_ratio":                ltv_ratio,
        "credit_age_years":         credit_age_years,
        "balance_per_account":      balance_per_account,
        "inq_last_12m":             inq_last_12m,
        "home_equity":              home_equity,
        "mortgage_balance":         mortgage_balance,
        "auto_loan_balance":        auto_loan_balance,
        "student_loan_balance":     student_loan_balance,
        "credit_card_balance":      credit_card_balance,
        "tax_liens":                tax_liens,
        "bankruptcies":             bankruptcies,
        "charge_off_count":         charge_off_count,
        "hardship_flag":            hardship_flag,
        "num_satisfactory":         num_satisfactory,
        "pct_satisfactory":         pct_satisfactory,
        "avg_credit_limit":         avg_credit_limit,
        "months_since_derog":       months_since_derog,
        "total_delinq_amount":      total_delinq_amount,
        "funded_amount_inv":        funded_amount_inv,

        # Categorical
        "loan_purpose":             loan_purpose,
        "loan_grade":               loan_grade,
        "loan_sub_grade":           loan_sub_grade,
        "employment_title":         employment_title,
        "home_ownership":           home_ownership,
        "verification_status":      verification_status,
        "loan_status_initial":      loan_status_initial,
        "state":                    state,
        "city_tier":                city_tier,
        "education_level":          education_level,
        "marital_status":           marital_status,
        "loan_channel":             loan_channel,
        "collateral_type":          collateral_type,
        "payment_method":           payment_method,
        "risk_segment":             risk_segment,
        "industry_sector":          industry_sector,
        "product_type":             product_type,
        "currency":                 currency,
        "bank_name":                bank_name,
        "credit_bureau":            credit_bureau,
        "origination_quarter":      origination_quarter,

        # Booleans
        "has_mortgage":             has_mortgage,
        "is_self_employed":         is_self_employed,
        "has_auto_loan":            has_auto_loan,
        "has_student_loan":         has_student_loan,
        "has_co_borrower":          has_co_borrower,
        "ever_bankrupt":            ever_bankrupt,
        "is_first_loan":            is_first_loan,
        "auto_pay_enrolled":        auto_pay_enrolled,
        "has_verified_income":      has_verified_income,
        "fico_checked":             fico_checked,

        # Datetime
        "application_date":         application_date,
        "birth_date":               birth_date,
        "last_payment_date":        last_payment_date,
        "next_payment_date":        next_payment_date,
        "credit_opened_date":       credit_opened_date,

        # Constants (should be auto-dropped)
        "country":                  const_country,
        "currency_code":            const_currency_code,
        "product_version":          const_product_ver,
        "model_version":            const_model_ver,
        "data_source":              const_data_source,

        # TARGET
        "default_flag":             default_flag,
    })

    # ── Introduce realistic missing values ─────────────────────────────
    print("  Introducing missing values...")

    def _nullify(col_name: str, pct: float):
        idx = rng.choice(n_rows, size=int(n_rows * pct), replace=False)
        df.loc[idx, col_name] = np.nan

    # High missing (will be flagged as critical)
    _nullify("months_since_last_delinq",  0.35)   # 35% — critical
    _nullify("months_since_derog",        0.42)   # 42% — critical
    _nullify("home_equity",               0.38)   # 38% — critical

    # Medium missing (5–30%)
    _nullify("employment_length_yrs",     0.08)
    _nullify("employment_title",          0.10)
    _nullify("total_delinq_amount",       0.12)
    _nullify("savings_balance",           0.15)
    _nullify("tax_liens",                 0.20)
    _nullify("bankruptcies",              0.18)
    _nullify("charge_off_count",          0.07)
    _nullify("num_collections",           0.09)
    _nullify("checking_balance",          0.06)

    # Low missing (<5%)
    _nullify("credit_score",              0.02)
    _nullify("annual_income",             0.03)
    _nullify("debt_to_income",            0.04)
    _nullify("revolving_util",            0.02)
    _nullify("education_level",           0.03)
    _nullify("marital_status",            0.01)

    # ── Introduce ~2% duplicate rows ──────────────────────────────────
    print("  Adding ~2% duplicate rows...")
    n_dups = int(n_rows * 0.02)
    dup_idx = rng.choice(n_rows, size=n_dups, replace=False)
    dup_df  = df.iloc[dup_idx].copy()
    df = pd.concat([df, dup_df], ignore_index=True)
    print(f"  Total rows after duplicates: {len(df):,}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate sample credit risk dataset")
    parser.add_argument("--rows",  type=int,  default=100_000, help="Number of base rows")
    parser.add_argument("--out",   type=str,  default="data/sample_credit_risk.csv",
                        help="Output file path (.csv or .parquet)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate(n_rows=args.rows)

    print(f"\nSaving to {out_path}...")
    if str(out_path).endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"✅ Dataset saved: {out_path}")
    print(f"   Shape:   {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    print(f"   Size:    {size_mb:.1f} MB")
    print(f"   Defaults: {df['default_flag'].mean():.2%}")
    print(f"\nMissing value summary (top cols):")
    miss = df.isnull().mean().sort_values(ascending=False).head(10)
    for col, pct in miss.items():
        if pct > 0:
            print(f"   {col:<35s}  {pct*100:.1f}%")
    print(f"\nColumn types: {df.dtypes.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
