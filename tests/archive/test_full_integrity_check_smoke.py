"""
Full Integrity Check Smoke Tests

Calls the integrity script functions directly (not subprocess).
Guarded by RUN_FULL_INTEGRITY=1 — skip unless explicitly enabled.
run_e2e=false by default for faster CI.
"""

import os
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

CONFIG_PATH = REPO_ROOT / "configs" / "config_best.yaml"


@pytest.mark.skipif(
    not os.environ.get("RUN_FULL_INTEGRITY"),
    reason="Full integrity tests skipped unless RUN_FULL_INTEGRITY=1",
)
class TestFullIntegritySmoke:
    """Run full integrity check (Step 0-3, optionally 4-5)."""

    def test_full_integrity_no_e2e(self):
        """Run integrity check with run_e2e=false (Steps 0-3 only)."""
        from tools.full_integrity_check import run_full_integrity

        with tempfile.TemporaryDirectory(prefix="pw_integrity_") as td:
            report = run_full_integrity(
                str(CONFIG_PATH),
                tmp_dir=td,
                device="cpu",
                run_slow=False,
                run_e2e=False,
            )
        assert report["overall_pass"], f"Integrity check failed: {report}"
        assert any(s.get("step") == "0_environment" and s.get("pass") for s in report["steps"])
        assert any(s.get("step") == "3_schedule" and s.get("pass") for s in report["steps"])
