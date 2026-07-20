"""
core/memory_manager.py
Memory-safe chunking, sampling, and system monitoring utilities.
Designed for datasets with 4,000+ columns and 2,000,000+ rows.
"""
import math
import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import psutil


class MemoryManager:
    """
    Utilities for safe, incremental processing of very large datasets.

    Key responsibilities
    --------------------
    * Monitor available system RAM before heavy operations.
    * Compute safe sample sizes that fit within memory budgets.
    * Generate chunk (start, end) index pairs for batch processing.
    * Report hardware information for report appendices.
    """

    def __init__(self, config: dict):
        self.config = config
        mem_cfg = config.get("memory", {})
        self.max_memory_gb: float = mem_cfg.get("max_memory_gb", 8.0)
        self.chunk_size: int     = mem_cfg.get("chunk_size", 100_000)

    # ------------------------------------------------------------------
    # System memory
    # ------------------------------------------------------------------

    @staticmethod
    def available_memory_gb() -> float:
        """Available system RAM in GB."""
        return psutil.virtual_memory().available / (1024 ** 3)

    @staticmethod
    def total_memory_gb() -> float:
        """Total system RAM in GB."""
        return psutil.virtual_memory().total / (1024 ** 3)

    @staticmethod
    def memory_usage_pct() -> float:
        """Current memory usage as percentage (0–100)."""
        return psutil.virtual_memory().percent

    # ------------------------------------------------------------------
    # Estimation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_df_memory_gb(
        n_rows: int,
        n_cols: int,
        avg_bytes_per_cell: float = 8.0,
    ) -> float:
        """
        Estimate how many GB a dense float64 DataFrame would occupy.

        Parameters
        ----------
        n_rows, n_cols : int
        avg_bytes_per_cell : float
            Default 8 bytes (float64). Use 4 for float32, 1-2 for bool/int8.
        """
        return (n_rows * n_cols * avg_bytes_per_cell) / (1024 ** 3)

    def safe_sample_size(
        self,
        n_rows: int,
        n_cols: int,
        target_gb: float = 1.0,
    ) -> int:
        """
        Return the maximum row count that fits within *target_gb* RAM,
        capped at *n_rows*.
        """
        bytes_per_row = n_cols * 8
        target_bytes = target_gb * (1024 ** 3)
        max_rows = int(target_bytes / max(bytes_per_row, 1))
        return min(max_rows, n_rows)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunk_indices(
        self,
        total_rows: int,
        chunk_size: Optional[int] = None,
    ) -> Iterator[Tuple[int, int]]:
        """
        Yield ``(start, end)`` row-index tuples for chunked processing.

        Example
        -------
        >>> for start, end in mgr.chunk_indices(1_000_000):
        ...     process(df[start:end])
        """
        cs = chunk_size or self.chunk_size
        for start in range(0, total_rows, cs):
            yield start, min(start + cs, total_rows)

    def n_chunks(self, total_rows: int, chunk_size: Optional[int] = None) -> int:
        """Number of chunks for the given total rows."""
        cs = chunk_size or self.chunk_size
        return math.ceil(total_rows / cs)

    # ------------------------------------------------------------------
    # System information
    # ------------------------------------------------------------------

    @staticmethod
    def get_system_info() -> dict:
        """
        Return a dictionary of hardware/OS information for report appendices.
        """
        vm   = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        freq = psutil.cpu_freq()

        return {
            "cpu_logical_cores":   psutil.cpu_count(logical=True),
            "cpu_physical_cores":  psutil.cpu_count(logical=False),
            "cpu_freq_mhz":        round(freq.current, 0) if freq else "N/A",
            "total_ram_gb":        round(vm.total   / (1024 ** 3), 2),
            "available_ram_gb":    round(vm.available / (1024 ** 3), 2),
            "ram_used_pct":        vm.percent,
            "disk_total_gb":       round(disk.total / (1024 ** 3), 2),
            "disk_free_gb":        round(disk.free  / (1024 ** 3), 2),
            "os":                  os.uname().sysname if hasattr(os, "uname") else os.name,
            "os_release":          os.uname().release if hasattr(os, "uname") else "N/A",
        }

    def warn_if_low_memory(self, threshold_gb: float = 2.0, logger=None) -> bool:
        """
        Return True (and optionally log a warning) if free RAM is below
        *threshold_gb*.
        """
        avail = self.available_memory_gb()
        if avail < threshold_gb:
            msg = (
                f"Low memory warning: only {avail:.2f} GB available "
                f"(threshold={threshold_gb:.1f} GB)."
            )
            if logger:
                logger.warning(msg)
            else:
                print(f"WARNING: {msg}")
            return True
        return False
