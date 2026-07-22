"""
core/data_loader.py
DuckDB-backed data loader for the Risk Modelling Pipeline.

Design
------
* Registers the input file (CSV / Parquet) as a DuckDB virtual table.
* All heavy aggregations (counts, stats, cardinality) execute as SQL — 
  never loading the full dataset into Python memory.
* Column batching avoids DuckDB's SQL-length limits at 4k+ columns.
* Polars is used for lightweight post-processing of small result sets.
"""
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import polars as pl
import numpy as np


_BATCH_COLS = 400   # DuckDB SQL length limit guard


class DataLoader:
    """
    Registers and queries a large tabular dataset via DuckDB.

    Parameters
    ----------
    config : dict — pipeline_config.yaml loaded as dict
    """

    def __init__(self, config: dict):
        self.config      = config
        data_cfg         = config.get("data", {})
        self.input_path  = Path(data_cfg.get("input_path", "data/input.csv"))
        self.target_col  = data_cfg.get("target_column", "target")
        self.input_format = data_cfg.get("input_format", "csv").lower()
        self._sep        = data_cfg.get("csv_separator", ",")
        self._nulls      = data_cfg.get("null_values", ["", "NA", "N/A", "null", "NULL"])

        mem_cfg = config.get("memory", {})
        self._threads  = mem_cfg.get("duckdb_threads", 4)
        self._mem_lim  = mem_cfg.get("duckdb_memory_limit", "6GB")

        self.execution_mode = config.get("project", {}).get("execution_mode", "python").lower()
        self._spark = None
        self._con: Optional[duckdb.DuckDBPyConnection] = None
        self._table = "input_data"
        self._cols: Optional[List[str]] = None
        self._connect()

    # ── Connection & registration ─────────────────────────────────────

    def _connect(self):
        if self.execution_mode == "spark":
            try:
                from pyspark.sql import SparkSession
                spark_cfg = self.config.get("spark", {})
                builder = SparkSession.builder.appName(spark_cfg.get("app_name", "Risk Modelling Spark Processor"))
                builder = builder.master(spark_cfg.get("master", "local[*]"))
                
                builder = builder.config("spark.driver.memory", spark_cfg.get("driver_memory", "8g"))
                builder = builder.config("spark.executor.memory", spark_cfg.get("executor_memory", "8g"))
                builder = builder.config("spark.executor.cores", spark_cfg.get("executor_cores", 4))
                builder = builder.config("spark.sql.shuffle.partitions", spark_cfg.get("sql_shuffle_partitions", 200))
                
                self._spark = builder.getOrCreate()
                print("\n[INFO] Successfully initialized PySpark Session in SPARK mode!")
            except Exception as e:
                print(f"\n[WARNING] Failed to initialize Spark session ({e}). Falling back to DuckDB (PYTHON mode).")
                self.execution_mode = "python"

        self._con = duckdb.connect(":memory:")
        self._con.execute(f"SET threads={self._threads}")
        self._con.execute(f"SET memory_limit='{self._mem_lim}'")
        self._register()

    def _register(self):
        p = str(self.input_path)
        null_str = ", ".join(f"'{n}'" for n in self._nulls)
        feat_files = self.config.get("data", {}).get("feature_files", [])
        id_col = self.config.get("data", {}).get("id_column", "record_id")
        
        if self.execution_mode == "spark" and self._spark is not None:
            try:
                print(f"[INFO] Loading data via PySpark engine (Spark mode)...")
                df_spark = self._spark.read.options(header=True, inferSchema=True, delimiter=self._sep).csv(p)
                self._con.register("perf_raw_spark", df_spark)
                self._con.execute("CREATE OR REPLACE VIEW perf_raw AS SELECT * FROM perf_raw_spark")
                
                for idx, fpath in enumerate(feat_files):
                    tbl_alias = f"feat_{idx}"
                    df_feat_spark = self._spark.read.options(header=True, inferSchema=True, delimiter=self._sep).csv(fpath)
                    self._con.register(f"{tbl_alias}_spark", df_feat_spark)
                    self._con.execute(f"CREATE OR REPLACE VIEW {tbl_alias} AS SELECT * FROM {tbl_alias}_spark")
            except Exception as e:
                print(f"\n[WARNING] Spark loading failed: {e}. Falling back to default DuckDB loader.")
                self.execution_mode = "python"

        if self.execution_mode == "python" or self._spark is None:
            # Load performance file as VIEW
            self._con.execute(
                f"CREATE OR REPLACE VIEW perf_raw AS "
                f"SELECT * FROM read_csv_auto('{p}', sep='{self._sep}', nullstr=[{null_str}], header=true)"
            )
            
            # Load and join each feature file as VIEW
            for idx, fpath in enumerate(feat_files):
                tbl_alias = f"feat_{idx}"
                self._con.execute(
                    f"CREATE OR REPLACE VIEW {tbl_alias} AS "
                    f"SELECT * FROM read_csv_auto('{fpath}', sep='{self._sep}', nullstr=[{null_str}], header=true)"
                )
            
        # Join query builder
        join_query = "FROM perf_raw p"
        select_fields = ["p.*"]
        for idx in range(len(feat_files)):
            tbl_alias = f"feat_{idx}"
            join_query += f" LEFT JOIN {tbl_alias} f{idx} ON p.{id_col} = f{idx}.{id_col}"
            # Select all columns from this feature file except the join key
            # To get column names, we query information_schema in DuckDB
            cols_res = self._con.execute(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_name='{tbl_alias}' AND column_name != '{id_col}'"
            ).fetchall()
            for row in cols_res:
                c_name = row[0]
                select_fields.append(f"f{idx}.\"{c_name}\" AS \"{c_name}\"")
                
        # Create intermediate combined view
        combined_sql = f"CREATE OR REPLACE VIEW combined_raw AS SELECT {', '.join(select_fields)} {join_query}"
        self._con.execute(combined_sql)
        
        # Filter out prohibited columns
        all_cols_res = self._con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='combined_raw'"
        ).fetchall()
        
        import re
        prohibited_patterns = self.config.get("data", {}).get("prohibited_patterns", [])
        patterns = [re.compile(p, re.IGNORECASE) for p in prohibited_patterns]
        
        safe_selects = []
        dropped_prohibited = []
        for row in all_cols_res:
            c = row[0]
            is_prohibited = False
            for pat in patterns:
                if pat.search(c):
                    is_prohibited = True
                    break
            if is_prohibited:
                dropped_prohibited.append(c)
            else:
                safe_selects.append(f"\"{c}\"")
                
        if dropped_prohibited:
            print(f"\n[WARNING] Disqualification Guard: Dropped {len(dropped_prohibited)} prohibited columns matching patterns:")
            print(f"  {', '.join(dropped_prohibited)}")
            
        # Create final view
        self._con.execute(
            f"CREATE OR REPLACE VIEW {self._table} AS SELECT {', '.join(safe_selects)} FROM combined_raw"
        )


    # ── Schema ────────────────────────────────────────────────────────

    def get_columns(self) -> List[str]:
        if self._cols is None:
            if self.execution_mode == "spark" and self._spark is not None:
                self._cols = self._spark.table(self._table).columns
            else:
                res = self._con.execute(
                    f"SELECT column_name FROM information_schema.columns "
                    f"WHERE table_name='{self._table}' ORDER BY ordinal_position"
                ).fetchall()
                self._cols = [r[0] for r in res]
        return self._cols

    def get_db_types(self) -> Dict[str, str]:
        if self.execution_mode == "spark" and self._spark is not None:
            return {
                field.name: str(field.dataType).replace("Type", "").upper()
                for field in self._spark.table(self._table).schema.fields
            }
        res = self._con.execute(
            f"SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_name='{self._table}'"
        ).fetchall()
        return {r[0]: r[1] for r in res}

    # ── Row operations ────────────────────────────────────────────────

    def count_rows(self) -> int:
        if self.execution_mode == "spark" and self._spark is not None:
            return self._spark.table(self._table).count()
        return int(self._con.execute(
            f"SELECT COUNT(*) FROM {self._table}"
        ).fetchone()[0])

    def get_file_size_mb(self) -> float:
        try:
            return self.input_path.stat().st_size / (1024 ** 2)
        except Exception:
            return 0.0

    # ── Sampling ──────────────────────────────────────────────────────

    def sample(self, n: int, seed: int = 42) -> pl.DataFrame:
        """Return an n-row random sample as a Polars DataFrame (all columns)."""
        if self.execution_mode == "spark" and self._spark is not None:
            n_rows = self.count_rows()
            frac = min(1.0, n / max(n_rows, 1))
            sdf = self._spark.table(self._table).sample(withReplacement=False, fraction=frac, seed=seed)
            import pandas as pd
            return pl.from_pandas(sdf.limit(n).toPandas())
        n_rows = self.count_rows()
        frac   = min(1.0, n / max(n_rows, 1))
        q      = f"SELECT * FROM {self._table} USING SAMPLE {frac*100:.4f}% (bernoulli, {seed})"
        return pl.from_arrow(self._con.execute(q).arrow())

    def sample_columns(
        self,
        cols: List[str],
        n: int = 100_000,
        seed: int = 42,
    ) -> pl.DataFrame:
        """Return an n-row sample of specific columns."""
        if not cols:
            return pl.DataFrame()
        if self.execution_mode == "spark" and self._spark is not None:
            n_rows = self.count_rows()
            frac = min(1.0, n / max(n_rows, 1))
            escaped_cols = [f"`{c}`" for c in cols]
            sdf = self._spark.table(self._table).select(escaped_cols).sample(withReplacement=False, fraction=frac, seed=seed)
            import pandas as pd
            return pl.from_pandas(sdf.limit(n).toPandas())
        n_rows  = self.count_rows()
        frac    = min(1.0, n / max(n_rows, 1))
        col_sql = ", ".join(f'"{c}"' for c in cols)
        q       = (
            f"SELECT {col_sql} FROM {self._table} "
            f"USING SAMPLE {frac*100:.4f}% (bernoulli, {seed})"
        )
        return pl.from_arrow(self._con.execute(q).arrow())

    # ── Aggregate statistics ──────────────────────────────────────────

    def get_numeric_stats(self, cols: List[str]) -> pl.DataFrame:
        """Compute descriptive stats for numeric columns via Spark or DuckDB SQL."""
        if not cols:
            return pl.DataFrame()

        if self.execution_mode == "spark" and self._spark is not None:
            escaped_cols = [f"`{c}`" for c in cols]
            sdf = self._spark.table(self._table).select(escaped_cols)
            summary_df = sdf.summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max").toPandas()
            rows = []
            n_total = self.count_rows()
            for c in cols:
                try:
                    cnt = float(summary_df.loc[summary_df["summary"] == "count", c].values[0])
                    mean_val = float(summary_df.loc[summary_df["summary"] == "mean", c].values[0])
                    std_val = float(summary_df.loc[summary_df["summary"] == "stddev", c].values[0])
                    min_val = float(summary_df.loc[summary_df["summary"] == "min", c].values[0])
                    q1_val = float(summary_df.loc[summary_df["summary"] == "25%", c].values[0])
                    med_val = float(summary_df.loc[summary_df["summary"] == "50%", c].values[0])
                    q3_val = float(summary_df.loc[summary_df["summary"] == "75%", c].values[0])
                    max_val = float(summary_df.loc[summary_df["summary"] == "max", c].values[0])
                    rows.append({
                        "column": c,
                        "n": int(cnt),
                        "n_missing": int(n_total - cnt),
                        "mean": round(mean_val, 6),
                        "std": round(std_val, 6),
                        "min": min_val,
                        "q1": q1_val,
                        "median": med_val,
                        "q3": q3_val,
                        "max": max_val
                    })
                except Exception:
                    pass
            return pl.DataFrame(rows)

        rows = []
        for c in cols:
            qc = f'"{c}"'
            try:
                row = self._con.execute(
                    f"SELECT '{c}' AS column, "
                    f"COUNT({qc}) AS n, "
                    f"COUNT(*)-COUNT({qc}) AS n_missing, "
                    f"ROUND(AVG({qc}),6) AS mean, "
                    f"ROUND(STDDEV_POP({qc}),6) AS std, "
                    f"MIN({qc}) AS min, "
                    f"ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {qc}),6) AS q1, "
                    f"ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {qc}),6) AS median, "
                    f"ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {qc}),6) AS q3, "
                    f"MAX({qc}) AS max "
                    f"FROM {self._table}"
                ).fetchone()
                rows.append({
                    "column":    row[0],
                    "n":         row[1],
                    "n_missing": row[2],
                    "mean":      row[3],
                    "std":       row[4],
                    "min":       row[5],
                    "q1":        row[6],
                    "median":    row[7],
                    "q3":        row[8],
                    "max":       row[9],
                })
            except Exception:
                pass
        return pl.DataFrame(rows)

    def get_missing_counts(self) -> pl.DataFrame:
        """Return missing count and % for all columns."""
        cols    = self.get_columns()
        n_total = self.count_rows()
        
        if self.execution_mode == "spark" and self._spark is not None:
            from pyspark.sql import functions as F
            sdf = self._spark.table(self._table)
            rows = []
            for batch in _batched(cols, 100):
                null_exprs = [F.sum(F.col(f"`{c}`").isNull().cast("int")).alias(c) for c in batch]
                null_row = sdf.select(null_exprs).collect()[0].asDict()
                for col, cnt in null_row.items():
                    rows.append({
                        "column": col,
                        "missing_count": int(cnt or 0),
                        "missing_pct": round(int(cnt or 0) / max(n_total, 1) * 100, 4)
                    })
            return pl.DataFrame(rows)

        rows    = []
        for batch in _batched(cols, _BATCH_COLS):
            cases = ", ".join(
                f"SUM(CASE WHEN \"{c}\" IS NULL THEN 1 ELSE 0 END) AS \"{c}\""
                for c in batch
            )
            res = self._con.execute(
                f"SELECT {cases} FROM {self._table}"
            ).fetchone()
            for col, cnt in zip(batch, res):
                rows.append({
                    "column":       col,
                    "missing_count": int(cnt),
                    "missing_pct":  round(int(cnt) / max(n_total, 1) * 100, 4),
                })
        return pl.DataFrame(rows)

    def get_cardinality(self, cols: List[str]) -> Dict[str, int]:
        """Return approximate cardinality for each column."""
        if self.execution_mode == "spark" and self._spark is not None:
            from pyspark.sql import functions as F
            sdf = self._spark.table(self._table)
            result = {}
            for batch in _batched(cols, 100):
                card_exprs = [F.countDistinct(F.col(f"`{c}`")).alias(c) for c in batch]
                card_row = sdf.select(card_exprs).collect()[0].asDict()
                for col, card in card_row.items():
                    result[col] = int(card or 0)
            return result

        result = {}
        for batch in _batched(cols, _BATCH_COLS):
            cases = ", ".join(
                f"COUNT(DISTINCT \"{c}\") AS \"{c}\"" for c in batch
            )
            res = self._con.execute(
                f"SELECT {cases} FROM {self._table}"
            ).fetchone()
            for col, card in zip(batch, res):
                result[col] = int(card)
        return result

    def column_value_counts(self, col: str, top_n: int = 10) -> pl.DataFrame:
        """Top N value frequencies for a column."""
        n = self.count_rows()
        if self.execution_mode == "spark" and self._spark is not None:
            from pyspark.sql import functions as F
            sdf = self._spark.table(self._table)
            res = sdf.groupBy(f"`{col}`").count().orderBy(F.col("count").desc()).limit(top_n).collect()
            return pl.DataFrame([
                {
                    "value": r[0],
                    "count": int(r[1]),
                    "pct": round(int(r[1]) * 100.0 / max(n, 1), 4)
                }
                for r in res
            ])

        try:
            res = self._con.execute(
                f"SELECT \"{col}\" AS value, COUNT(*) AS count, "
                f"ROUND(COUNT(*)*100.0/{n},4) AS pct "
                f"FROM {self._table} "
                f"GROUP BY \"{col}\" ORDER BY count DESC LIMIT {top_n}"
            ).fetchall()
            return pl.DataFrame([
                {"value": r[0], "count": r[1], "pct": r[2]} for r in res
            ])
        except Exception:
            return pl.DataFrame()

    def get_target_distribution(self) -> pl.DataFrame:
        df = self.column_value_counts(self.target_col)
        if not df.is_empty() and "value" in df.columns:
            return df.rename({"value": "target_value"})
        return df

    # ── IV / WoE ──────────────────────────────────────────────────────

    def compute_iv_woe(
        self,
        col: str,
        n_bins: int = 10,
    ) -> Tuple[float, pl.DataFrame]:
        """
        Compute Information Value and WoE for a numeric column.
        """
        t = self.target_col
        if self.execution_mode == "spark" and self._spark is not None:
            from pyspark.ml.feature import QuantileDiscretizer
            from pyspark.sql import functions as F
            try:
                sdf = self._spark.table(self._table).select(f"`{col}`", f"`{t}`")
                sdf_clean = sdf.na.drop()
                qd = QuantileDiscretizer(numBuckets=n_bins, inputCol=col, outputCol="bin", relativeError=0.01, handleInvalid="skip")
                bucketed = qd.fit(sdf_clean).transform(sdf_clean)
                agg = bucketed.groupBy("bin").agg(
                    F.sum(F.col(f"`{t}`").cast("int")).alias("events"),
                    F.count("*").alias("total")
                ).collect()
                
                woe_rows = []
                total_ev = 0
                total_nev = 0
                bins_dict = {}
                for r in agg:
                    b = int(r["bin"])
                    ev = int(r["events"] or 0)
                    total = int(r["total"] or 0)
                    nev = total - ev
                    bins_dict[b] = {"ev": ev, "nev": nev}
                    total_ev += ev
                    total_nev += nev
                
                if total_ev == 0 or total_nev == 0:
                    return 0.0, pl.DataFrame()
                
                iv_total = 0.0
                for b in sorted(bins_dict):
                    ev = bins_dict[b]["ev"]
                    nev = bins_dict[b]["nev"]
                    dist_ev = ev / total_ev
                    dist_nev = nev / total_nev
                    if dist_ev > 0 and dist_nev > 0:
                        woe = np.log(dist_ev / dist_nev)
                        iv_c = (dist_ev - dist_nev) * woe
                    else:
                        woe = 0.0
                        iv_c = 0.0
                    iv_total += iv_c
                    woe_rows.append({
                        "bin": b,
                        "events": ev,
                        "non_events": nev,
                        "woe": round(woe, 6),
                        "iv_contrib": round(iv_c, 6)
                    })
                return round(iv_total, 6), pl.DataFrame(woe_rows)
            except Exception:
                return 0.0, pl.DataFrame()

        try:
            res = self._con.execute(
                f"SELECT "
                f"  NTILE({n_bins}) OVER (ORDER BY \"{col}\") AS bin, "
                f"  CAST(\"{t}\" AS INTEGER) AS target "
                f"FROM {self._table} "
                f"WHERE \"{col}\" IS NOT NULL AND \"{t}\" IS NOT NULL"
            ).fetchall()

            if not res:
                return 0.0, pl.DataFrame()

            from collections import defaultdict
            bins: Dict[int, Dict] = defaultdict(lambda: {"ev": 0, "nev": 0})
            for row in res:
                b, tgt = row[0], row[1]
                if tgt == 1:
                    bins[b]["ev"]  += 1
                else:
                    bins[b]["nev"] += 1

            total_ev  = sum(v["ev"]  for v in bins.values())
            total_nev = sum(v["nev"] for v in bins.values())
            if total_ev == 0 or total_nev == 0:
                return 0.0, pl.DataFrame()

            iv_total = 0.0
            woe_rows = []
            for b in sorted(bins):
                ev  = bins[b]["ev"]
                nev = bins[b]["nev"]
                dist_ev  = ev  / total_ev
                dist_nev = nev / total_nev
                if dist_ev > 0 and dist_nev > 0:
                    woe = np.log(dist_ev / dist_nev)
                    iv_c = (dist_ev - dist_nev) * woe
                else:
                    woe  = 0.0
                    iv_c = 0.0
                iv_total += iv_c
                woe_rows.append({
                    "bin":        b,
                    "events":     ev,
                    "non_events": nev,
                    "woe":        round(woe, 6),
                    "iv_contrib": round(iv_c, 6),
                })

            return round(iv_total, 6), pl.DataFrame(woe_rows)
        except Exception:
            return 0.0, pl.DataFrame()

    def get_feature_cols(self) -> List[str]:
        """Return all non-target columns."""
        return [c for c in self.get_columns() if c != self.target_col]

    def close(self):
        if self._con:
            self._con.close()
            self._con = None
        if self._spark:
            try:
                self._spark.stop()
            except Exception:
                pass
            self._spark = None


# ── Utilities ─────────────────────────────────────────────────────────────────

def _batched(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
