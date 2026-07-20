import os
import json
import argparse
import sys
from pathlib import Path
import polars as pl
import numpy as np

def parse_mode_value(val):
    """
    Handles fallback parsing in case a mode fill value was serialized 
    as a Polars Series string representation (e.g. starting with 'shape:').
    """
    if isinstance(val, str) and "shape:" in val:
        # Extract the value inside the bracket [ ... ]
        try:
            lines = val.strip().split("\n")
            for i, line in enumerate(lines):
                if line.strip() == "[":
                    val_str = lines[i+1].strip().strip('"').strip("'")
                    # Try to convert to int or float if possible
                    try:
                        return int(val_str)
                    except ValueError:
                        try:
                            return float(val_str)
                        except ValueError:
                            return val_str
        except Exception:
            return "UNKNOWN"
    return val

def recreate_pipeline(data_path: str, spec_dir: str, output_path: str = None) -> pl.DataFrame:
    """
    Loads raw data, applies all transformation steps in feature_engineering_spec.json,
    and filters down to the final selected_features.json.
    """
    print(f"Loading specs from: {spec_dir}")
    spec_path = Path(spec_dir) / "feature_engineering_spec.json"
    features_path = Path(spec_dir) / "selected_features.json"

    if not spec_path.exists():
        raise FileNotFoundError(f"Missing feature engineering specification: {spec_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing selected features specification: {features_path}")

    with open(spec_path, "r") as f:
        spec = json.load(f)
    with open(features_path, "r") as f:
        feat_info = json.load(f)

    # 1. Load Raw Dataset
    print(f"Loading raw dataset: {data_path}")
    # Read CSV auto-detecting types
    df = pl.read_csv(data_path, ignore_errors=True)
    orig_shape = df.shape
    print(f"Original shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    # Target column check
    target = "target"  # default target name
    if target not in df.columns:
        # check if target exists in raw columns
        possible_targets = [c for c in df.columns if c.lower() in ("target", "label", "default", "bad_indicator")]
        if possible_targets:
            target = possible_targets[0]
            print(f"Detected target column: '{target}'")
        else:
            target = None

    # 2. Drop constant and identifier columns
    dropped_cols = set(spec.get("dropped_constant", []) + spec.get("dropped_identifier", []))
    cols_to_drop = [c for c in dropped_cols if c in df.columns and c != target]
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} constant/identifier columns...")
        df = df.drop(cols_to_drop)

    # 3. Add missing value indicators
    missing_indicators = spec.get("missing_indicators", [])
    if missing_indicators:
        print(f"Creating {len(missing_indicators)} missing indicator binary columns...")
        indicator_exprs = []
        for c in missing_indicators:
            if c in df.columns:
                indicator_exprs.append(
                    pl.col(c).is_null().cast(pl.Int8).alias(f"{c}_is_missing")
                )
        if indicator_exprs:
            df = df.with_columns(indicator_exprs)

    # 4. Impute missing values
    imputations = spec.get("imputation", {})
    if imputations:
        print("Imputing missing values (median for numeric, mode for categorical)...")
        impute_exprs = []
        for col, imp_cfg in imputations.items():
            if col in df.columns:
                raw_val = imp_cfg["fill_value"]
                fill_val = parse_mode_value(raw_val)
                impute_exprs.append(
                    pl.col(col).fill_null(fill_val)
                )
        if impute_exprs:
            df = df.with_columns(impute_exprs)

    # 5. Winsorise Outliers (Clip values)
    winsorisation = spec.get("winsorisation", {})
    if winsorisation:
        print("Clipping outliers using winsorisation bounds...")
        clip_exprs = []
        for col, bounds in winsorisation.items():
            if col in df.columns:
                lower = bounds.get("lower")
                upper = bounds.get("upper")
                clip_exprs.append(
                    pl.col(col).clip(lower_bound=lower, upper_bound=upper)
                )
        if clip_exprs:
            df = df.with_columns(clip_exprs)

    # 6. Apply Log Transformations
    log_transforms = spec.get("log_transform", [])
    if log_transforms:
        print(f"Applying log1p transformation to {len(log_transforms)} right-skewed features...")
        log_exprs = []
        for col in log_transforms:
            if col in df.columns:
                # log1p(x) is safe for x >= 0
                log_exprs.append(
                    pl.col(col).log1p()
                )
        if log_exprs:
            df = df.with_columns(log_exprs)

    # 7. Apply Encoding
    encodings = spec.get("encoding", {})
    if encodings:
        print("Applying categorical encoding...")
        for col, enc_type in encodings.items():
            if col not in df.columns:
                continue
            if enc_type == "one_hot":
                # Polars OHE dummy columns creation
                df = df.to_dummies(columns=[col])
            elif enc_type in ("ordinal", "woe"):
                # Fast label encoding mapping string categories to index values
                df = df.with_columns(
                    pl.col(col).cast(pl.Categorical).to_physical().cast(pl.Int64)
                )

    # 8. Align & Select Final Features
    selected_features = feat_info.get("selected_features", [])
    
    # We must also support any dummy columns created by OHE
    all_final_features = []
    for f in selected_features:
        # Check direct match
        if f in df.columns:
            all_final_features.append(f)
        else:
            # Check OHE dummy names (e.g. origcol_category)
            ohe_dummies = [c for c in df.columns if c.startswith(f"{f}_")]
            if ohe_dummies:
                all_final_features.extend(ohe_dummies)
            else:
                # Missing indicator columns (e.g. col_is_missing)
                if f.endswith("_is_missing") and f in df.columns:
                    all_final_features.append(f)

    # Ensure target is kept
    keep_cols = all_final_features
    if target and target in df.columns:
        keep_cols = keep_cols + [target]

    # Filter columns
    df_out = df.select([c for c in keep_cols if c in df.columns])
    
    print("\n" + "="*50)
    print("RECREATION RUN COMPLETED SUCCESSFULLY")
    print(f"Raw Input Shape: {orig_shape[0]:,} rows x {orig_shape[1]:,} columns")
    print(f"Output Shape:    {df_out.shape[0]:,} rows x {df_out.shape[1]:,} columns")
    print(f"Selected consensus features matched: {len(all_final_features)}")
    print("="*50)

    if output_path:
        df_out.write_csv(output_path)
        print(f"Saved processed dataset to: {output_path}")

    return df_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recreate Model Features from Saved Spec")
    parser.add_argument("--data", type=str, default="data/sample_credit_risk.csv", help="Path to raw input CSV file")
    parser.add_argument("--spec-dir", type=str, default="artifacts", help="Directory containing the JSON specs")
    parser.add_argument("--output", type=str, default="data/engineered_features.csv", help="Path to save processed CSV output")
    
    args = parser.parse_args()
    
    try:
        recreate_pipeline(args.data, args.spec_dir, args.output)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
