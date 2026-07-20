"""
core/logger.py
Structured, colourised logging for the Risk Modelling Pipeline.
Writes simultaneously to console and a date-stamped log file.
"""
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class _ColourFormatter(logging.Formatter):
    """ANSI colour codes for console output."""

    GREY    = "\x1b[38;5;240m"
    CYAN    = "\x1b[36m"
    YELLOW  = "\x1b[33m"
    RED     = "\x1b[31m"
    BOLD_RED = "\x1b[31;1m"
    RESET   = "\x1b[0m"

    LEVEL_COLORS = {
        logging.DEBUG:    GREY,
        logging.INFO:     CYAN,
        logging.WARNING:  YELLOW,
        logging.ERROR:    RED,
        logging.CRITICAL: BOLD_RED,
    }

    BASE_FMT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        formatter = logging.Formatter(
            f"{colour}{self.BASE_FMT}{self.RESET}",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return formatter.format(record)


class PipelineLogger:
    """
    Centralised logger for the entire pipeline.

    Features
    --------
    * Singleton per name — safe to call get_logger() anywhere.
    * Console handler with ANSI colours.
    * File handler (full DEBUG level) written to ``logs/`` directory.
    * Helper methods ``stage_start`` / ``stage_end`` for pipeline stages.
    """

    _instances: dict = {}

    def __init__(self, name: str, log_dir: str = "logs", level: int = logging.INFO):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._logger = self._build(level)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build(self, level: int) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)

        if logger.handlers:
            return logger

        # Console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(_ColourFormatter())

        # File
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"pipeline_{ts}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)-20s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        logger.addHandler(ch)
        logger.addHandler(fh)
        return logger

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def debug(self, msg: str)    -> None: self._logger.debug(msg)
    def info(self, msg: str)     -> None: self._logger.info(msg)
    def warning(self, msg: str)  -> None: self._logger.warning(msg)
    def error(self, msg: str)    -> None: self._logger.error(msg)
    def critical(self, msg: str) -> None: self._logger.critical(msg)

    def stage_start(self, stage_name: str) -> None:
        sep = "═" * 70
        self._logger.info(sep)
        self._logger.info(f"  STARTING STAGE: {stage_name}")
        self._logger.info(sep)

    def stage_end(self, stage_name: str, elapsed_seconds: float) -> None:
        self._logger.info(
            f"  COMPLETED: {stage_name} | elapsed={elapsed_seconds:.2f}s"
        )
        self._logger.info("═" * 70)


def get_logger(name: str = "pipeline", log_dir: str = "logs") -> PipelineLogger:
    """
    Factory function — returns a singleton PipelineLogger per name.

    Parameters
    ----------
    name : str
        Logger name (shown in log output).
    log_dir : str
        Directory where log files are written.
    """
    key = f"{name}::{log_dir}"
    if key not in PipelineLogger._instances:
        PipelineLogger._instances[key] = PipelineLogger(name, log_dir)
    return PipelineLogger._instances[key]
