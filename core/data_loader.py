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

        self._con: Optional[duckdb.DuckDBPyConnection] = None
        self._table = "input_data"
        self._cols: Optional[List[str]] = None
        self._connect()

    # ── Connection & registration ─────────────────────────────────────

    def _connect(self):
        self._con = duckdb.connect(":memory:")
        self._con.execute(f"SET threads={self._threads}")
        self._con.execute(f"SET memory_limit='{self._mem_lim}'")
        self._register()

    def _register(self):
        p = str(self.input_path)
        null_str = ", ".join(f"'{n}'" for n in self._nulls)
        
        # Load performance file as VIEW
        self._con.execute(
            f"CREATE OR REPLACE VIEW perf_raw AS "
            f"SELECT * FROM read_csv_auto('{p}', sep='{self._sep}', nullstr=[{null_str}], header=true)"
        )
        
        # Load and join each feature file as VIEW
        feat_files = self.config.get("data", {}).get("feature_files", [])
        id_col = self.config.get("data", {}).get("id_column", "record_id")
        
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
            res = self._con.execute(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_name='{self._table}' ORDER BY ordinal_position"
            ).fetchall()
            self._cols = [r[0] for r in res]
        return self._cols

    def get_db_types(self) -> Dict[str, str]:
        res = self._con.execute(
            f"SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_name='{self._table}'"
        ).fetchall()
        return {r[0]: r[1] for r in res}

    # ── Row operations ────────────────────────────────────────────────

    def count_rows(self) -> int:
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
        """Compute descriptive stats for numeric columns via DuckDB SQL."""
        if not cols:
            return pl.DataFrame()

        parts = []
        for batch in _batched(cols, _BATCH_COLS):
            aggs = []
            for c in batch:
                qc = f'"{c}"'
                aggs += [
                    f"'{c}' AS column",
                    f"COUNT({qc})              AS n",
                    f"COUNT(*) - COUNT({qc})   AS n_missing",
                    f"AVG({qc})                AS mean",
                    f"STDDEV_POP({qc})         AS std",
                    f"MIN({qc})                AS min",
                    f"PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {qc}) AS q1",
                    f"PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {qc}) AS median",
                    f"PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {qc}) AS q3",
                    f"MAX({qc})                AS max",
                ]
                sub = (
                    f"SELECT {', '.join(aggs)} FROM {self._table}"
                )
                parts.append(sub)

        # Execute each column individually (safest for wide tables)
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
        t = self.target_col
        n = self.count_rows()
        try:
            res = self._con.execute(
                f"SELECT \"{t}\" AS target_value, COUNT(*) AS count, "
                f"ROUND(COUNT(*)*100.0/{n},4) AS pct "
                f"FROM {self._table} "
                f"GROUP BY \"{t}\" ORDER BY \"{t}\""
            ).fetchall()
            return pl.DataFrame([
                {"target_value": r[0], "count": int(r[1]), "pct": float(r[2])} for r in res
            ])
        except Exception:
            return pl.DataFrame()

    # ── IV / WoE ──────────────────────────────────────────────────────

    def compute_iv_woe(
        self,
        col: str,
        n_bins: int = 10,
    ) -> Tuple[float, pl.DataFrame]:
        """
        Compute Information Value and WoE for a numeric column.

        Returns
        -------
        iv     : float
        woe_df : pl.DataFrame with columns [bin, count, events, non_events, woe, iv_contrib]
        """
        t = self.target_col
        try:
            # Create decile bins via DuckDB NTILE
            res = self._con.execute(
                f"SELECT "
                f"  NTILE({n_bins}) OVER (ORDER BY \"{col}\") AS bin, "
                f"  CAST(\"{t}\" AS INTEGER) AS target "
                f"FROM {self._table} "
                f"WHERE \"{col}\" IS NOT NULL AND \"{t}\" IS NOT NULL"
            ).fetchall()

            if not res:
                return 0.0, pl.DataFrame()

            # Aggregate by bin
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


# ── Utilities ─────────────────────────────────────────────────────────────────

def _batched(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
