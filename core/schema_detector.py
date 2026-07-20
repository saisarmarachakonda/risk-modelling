"""
core/schema_detector.py
Automatic column type classification for the Risk Modelling Pipeline.

Classification hierarchy
------------------------
1. target     — the target column configured in YAML
2. constant   — one unique value (zero variance)
3. identifier — uniqueness_ratio > 0.95 or name matches ID patterns
4. boolean    — 2 unique values (or DuckDB BOOLEAN type)
5. datetime   — DuckDB DATE/TIMESTAMP type or name pattern
6. categorical — string/object type or low-cardinality numeric
7. numeric    — all remaining numeric types

All classification uses batched DuckDB SQL queries to handle 4k+ columns
without hitting SQL length limits or loading data into memory.
"""
import re
from typing import Dict, List, Optional, Tuple

from core.data_loader import DataLoader, _batched


_ID_PATTERNS = re.compile(
    r"^(id|_id|uuid|guid|row_id|pk|primary_key|account_no|customer_id"
    r"|loan_id|contract_id|application_id|user_id|member_id)$",
    re.IGNORECASE,
)
_DATE_PATTERNS = re.compile(
    r"(date|dt|time|timestamp|created|updated|modified|birth|inception|maturity)",
    re.IGNORECASE,
)
_BATCH = 400


class SchemaDetector:
    """
    Classifies all columns into semantic types.

    Usage
    -----
    >>> sd = SchemaDetector(loader, config)
    >>> schema = sd.detect()          # returns Dict[col_name, info_dict]
    >>> num_cols = sd.get_numeric_cols()
    """

    _NUMERIC_DB_TYPES = {
        "BIGINT", "HUGEINT", "INTEGER", "INT", "SMALLINT", "TINYINT",
        "UBIGINT", "UINTEGER", "USMALLINT", "UTINYINT",
        "DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC",
    }
    _DATETIME_DB_TYPES = {"DATE", "TIMESTAMP", "TIMESTAMPTZ", "TIME", "INTERVAL"}
    _BOOL_DB_TYPES     = {"BOOLEAN", "BOOL"}
    _STR_DB_TYPES      = {"VARCHAR", "TEXT", "CHAR", "STRING", "JSON", "BLOB"}

    def __init__(self, loader: DataLoader, config: dict):
        self._loader   = loader
        self._config   = config
        self._schema: Optional[Dict[str, dict]] = None

    def detect(self) -> Dict[str, dict]:
        """Return (cached) full schema classification."""
        if self._schema is None:
            self._schema = self._build_schema()
        return self._schema

    # ── Classification ────────────────────────────────────────────────

    def _build_schema(self) -> Dict[str, dict]:
        target  = self._loader.target_col
        columns = self._loader.get_columns()
        n_rows  = self._loader.count_rows()
        db_types = self._loader.get_db_types()
        cardinality = self._loader.get_cardinality(columns)
        missing = self._loader.get_missing_counts()
        miss_map = {
            r["column"]: r["missing_pct"]
            for r in missing.to_dicts()
        }
        cfg_num_thresh = self._config.get("preprocessing", {}).get(
            "categorical_cardinality_threshold", 20
        )

        schema = {}
        for col in columns:
            db_type = db_types.get(col, "VARCHAR").upper()
            n_unique = cardinality.get(col, 0)
            miss_pct = miss_map.get(col, 0.0)
            uniq_ratio = n_unique / max(n_rows, 1)

            # Classify
            if col == target:
                col_type = "target"
            elif n_unique <= 1:
                col_type = "constant"
            elif _ID_PATTERNS.match(col) or uniq_ratio > 0.95:
                col_type = "identifier"
            elif db_type in self._BOOL_DB_TYPES or n_unique == 2:
                col_type = "boolean"
            elif db_type in self._DATETIME_DB_TYPES or _DATE_PATTERNS.search(col):
                col_type = "datetime"
            elif db_type in self._STR_DB_TYPES:
                col_type = "categorical"
            elif db_type in self._NUMERIC_DB_TYPES:
                if n_unique <= cfg_num_thresh:
                    col_type = "categorical"
                else:
                    col_type = "numeric"
            else:
                col_type = "categorical"

            # Preprocessing suggestion
            suggestion = self._preproc_suggest(col_type, n_unique)

            schema[col] = {
                "col_type":             col_type,
                "col_type_label":       col_type.capitalize(),
                "db_type":              db_type,
                "n_unique":             n_unique,
                "uniqueness_ratio":     round(uniq_ratio, 4),
                "missing_pct":          round(miss_pct, 4),
                "preprocessing_suggestion": suggestion,
            }

        return schema

    @staticmethod
    def _preproc_suggest(col_type: str, n_unique: int) -> str:
        suggestions = {
            "target":     "Use as label — do not include as feature",
            "constant":   "Drop — zero variance, no predictive value",
            "identifier": "Drop — uniqueness ≥ 95%, likely ID/key column",
            "boolean":    "Encode as 0/1 integer — no other transform needed",
            "datetime":   "Extract year/month/day/weekday features or compute age",
            "numeric":    "Impute median, winsorise IQR 3×, StandardScaler for LR",
            "categorical": (
                "OHE (≤10 unique)" if n_unique <= 10
                else "Ordinal Encode (≤50 unique)" if n_unique <= 50
                else "WoE / Target Encode (>50 unique)"
            ),
        }
        return suggestions.get(col_type, "Review manually")

    # ── Typed column lists ────────────────────────────────────────────

    def get_numeric_cols(self) -> List[str]:
        return [c for c, v in self.detect().items() if v["col_type"] == "numeric"]

    def get_categorical_cols(self) -> List[str]:
        return [c for c, v in self.detect().items() if v["col_type"] == "categorical"]

    def get_boolean_cols(self) -> List[str]:
        return [c for c, v in self.detect().items() if v["col_type"] == "boolean"]

    def get_datetime_cols(self) -> List[str]:
        return [c for c, v in self.detect().items() if v["col_type"] == "datetime"]

    def get_constant_cols(self) -> List[str]:
        return [c for c, v in self.detect().items() if v["col_type"] == "constant"]

    def get_identifier_cols(self) -> List[str]:
        return [c for c, v in self.detect().items() if v["col_type"] == "identifier"]

    def get_feature_cols(self) -> List[str]:
        """All columns that are usable as model features."""
        exclude = {"target", "constant", "identifier", "datetime"}
        return [c for c, v in self.detect().items() if v["col_type"] not in exclude]

    def get_schema_summary(self) -> Dict[str, int]:
        s = self.detect()
        total = len(s)
        return {
            "total_columns":   total,
            "numeric":         sum(1 for v in s.values() if v["col_type"] == "numeric"),
            "categorical":     sum(1 for v in s.values() if v["col_type"] == "categorical"),
            "boolean":         sum(1 for v in s.values() if v["col_type"] == "boolean"),
            "datetime":        sum(1 for v in s.values() if v["col_type"] == "datetime"),
            "identifier":      sum(1 for v in s.values() if v["col_type"] == "identifier"),
            "constant":        sum(1 for v in s.values() if v["col_type"] == "constant"),
            "target":          sum(1 for v in s.values() if v["col_type"] == "target"),
            "feature_columns": sum(1 for v in s.values()
                                   if v["col_type"] not in {"target","constant","identifier","datetime"}),
        }
