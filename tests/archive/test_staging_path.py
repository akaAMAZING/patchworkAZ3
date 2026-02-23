"""Unit test: staging path creation works on Windows (WinError 267 regression)."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest


def test_staging_path_training_log_creation():
    """Create staging/iter_000, attach FileHandler to training.log, verify no WinError 267."""
    from src.training.run_layout import staging_dir
    from src.training.main import _attach_staging_log_handler, _detach_staging_log_handler

    with tempfile.TemporaryDirectory(prefix="pw_staging_test_") as td:
        run_root = Path(td) / "runs" / "test_run"
        staging_path = staging_dir(run_root, 0)
        staging_path.mkdir(parents=True, exist_ok=True)

        _attach_staging_log_handler(staging_path)
        try:
            root_logger = logging.getLogger()
            root_logger.info("Test log entry")
            for h in root_logger.handlers:
                if getattr(h, "stream", None) is not None:
                    h.flush()
            log_file = staging_path / "training.log"
            assert log_file.exists(), f"training.log not created at {log_file}"
            assert log_file.is_file(), f"{log_file} is not a file"
        finally:
            _detach_staging_log_handler()
