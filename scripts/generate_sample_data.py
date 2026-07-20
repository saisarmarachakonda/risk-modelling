"""
scripts/generate_sample_data.py
Generates the synthetically partitioned and distributed datasets 
for the Risk Modelling Machine Learning Contest.

Files Created:
--------------
* data/performance.csv: Contains record_id, target variable 'funded', and sample_group ('Train', 'Test', 'OOT')
* data/features_premiers.csv: Approved feature set "Premiers" (prefixed with premiers_) + prohibited score variables
* data/features_trended.csv: Approved feature set "Trended 3D" (prefixed with trended_) + prohibited reason code variables
* data/features_ccr.csv: Approved feature set "CCR" (prefixed with ccr_)
* data/features_cia.csv: Approved feature set "CIA" (prefixed with cia_)
* data/features_cfa.csv: Approved feature set "CFA" (prefixed with cfa_)
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
rng  = np.random.default_rng(SEED)

def generate_contest_data(n_rows: int = 60000):
    print(f"Generating {n_rows:,} records for the Machine Learning Contest...")
    
    # 1. Generate record_id and target variable 'funded'
    record_id = [f"REC{str(i).zfill(8)}" for i in range(1, n_rows + 1)]
    
    # Target variable 'funded' (binary 0/1, ~22% funded rate)
    funded = rng.choice([0, 1], size=n_rows, p=[0.78, 0.22])
    
    # Partition dataset into Train, Test, and OOT
    # Rules say Train (60%), Test (20%), OOT (20%)
    groups = rng.choice(
        ["Train", "Test", "OOT"], 
        size=n_rows, 
        p=[0.60, 0.20, 0.20]
    )
    
    performance_df = pd.DataFrame({
        "record_id": record_id,
        "funded": funded,
        "sample_group": groups
    })
    
    # 2. Approved features generator helpers
    def create_features(prefix, count, missing_cols=[]):
        data = {"record_id": record_id}
        for i in range(1, count + 1):
            col_name = f"{prefix}_feature_{i}"
            # Realistic data pattern: income-like log-normal, count-like poisson, flags, etc.
            pattern_type = i % 4
            if pattern_type == 0:
                val = rng.lognormal(mean=4.5, sigma=1.0, size=n_rows)
            elif pattern_type == 1:
                val = rng.poisson(lam=4.0, size=n_rows).astype(float)
            elif pattern_type == 2:
                val = rng.uniform(0, 100, size=n_rows)
            else:
                val = rng.choice([0.0, 1.0], size=n_rows, p=[0.85, 0.15])
            
            # Inject missing values for columns in missing_cols
            if i in missing_cols:
                mask = rng.choice([True, False], size=n_rows, p=[0.12, 0.88])
                val[mask] = np.nan
                
            data[col_name] = val
        return pd.DataFrame(data)

    print("Generating Premiers 1.3 features...")
    # FS1: Premiers (30 features, some missing)
    premiers_df = create_features("premiers", 30, missing_cols=[5, 12, 22])
    # Add prohibited variables: score variables
    premiers_df["premiers_fico_score"] = rng.uniform(300, 850, size=n_rows)
    premiers_df["premiers_internal_score_v3"] = rng.uniform(1, 10, size=n_rows)
    
    print("Generating Trended 3D 1.1 features...")
    # FS2: Trended (25 features)
    trended_df = create_features("trended", 25, missing_cols=[8])
    # Add prohibited variables: reason code variables
    trended_df["trended_reason_code_1"] = rng.choice(["A1", "B2", "C3", "None"], size=n_rows)
    trended_df["trended_reason_code_v2"] = rng.choice(["R102", "R108", "None"], size=n_rows)
    
    print("Generating CCR features...")
    # FS3: CCR (20 features)
    ccr_df = create_features("ccr", 20, missing_cols=[15])
    
    print("Generating CIA features...")
    # FS4: CIA (15 features)
    cia_df = create_features("cia", 15)
    
    print("Generating CFA features...")
    # FS5: CFA (10 features)
    cfa_df = create_features("cfa", 10)
    
    # Save datasets
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    
    performance_df.to_csv(out_dir / "performance.csv", index=False)
    premiers_df.to_csv(out_dir / "features_premiers.csv", index=False)
    trended_df.to_csv(out_dir / "features_trended.csv", index=False)
    ccr_df.to_csv(out_dir / "features_ccr.csv", index=False)
    cia_df.to_csv(out_dir / "features_cia.csv", index=False)
    cfa_df.to_csv(out_dir / "features_cfa.csv", index=False)
    
    print("\n" + "="*50)
    print("CONTEST DATASETS CREATED SUCCESSFULLY")
    print(f"Directory:           {out_dir.resolve()}")
    print(f"Total Records:       {n_rows:,}")
    print(f"Train / Test / OOT:  {performance_df['sample_group'].value_counts().to_dict()}")
    print(f"Target 'funded':     {performance_df['funded'].value_counts().to_dict()}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ML Contest Data")
    parser.add_argument("--rows", type=int, default=60000, help="Number of records to generate")
    args = parser.parse_args()
    
    generate_contest_data(args.rows)
