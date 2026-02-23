"""
Resume and graceful shutdown correctness tests.

A) Stop requested during an iteration → process exits only after commit marker exists.
B) Restart → starts at next iteration and uses the expected checkpoint for selfplay.
C) Double SIGINT during commit → still exits only after manifest exists (_commit_in_progress guard).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import sys
from pathlib import Path

import pytest
import yaml

from tests.conftest import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.run_layout import (
    COMMIT_MANIFEST,
    clear_commit_test_callbacks,
    committed_dir,
    get_staging_cleanup_plan,
    reconcile_run_state,
    register_commit_test_callback,
)


def _minimal_resume_config():
    """Config with 2 games, 1 epoch, 0 eval for fast iteration. Uses config_best as base for completeness."""
    cfg_path = REPO_ROOT / "configs" / "config_best.yaml"
    if not cfg_path.exists():
        cfg_path = REPO_ROOT / "configs" / "config_e2e_smoke.yaml"
    if not cfg_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = dict(cfg)
    cfg["hardware"] = cfg.get("hardware") or {}
    cfg["hardware"]["device"] = "cpu"  # Fast for CI
    cfg["selfplay"] = cfg.get("selfplay") or {}
    cfg["selfplay"]["games_per_iteration"] = 2
    cfg["selfplay"]["bootstrap"] = cfg["selfplay"].get("bootstrap") or {}
    cfg["selfplay"]["bootstrap"]["games"] = 2
    cfg["training"] = cfg.get("training") or {}
    cfg["training"]["epochs_per_iteration"] = 1
    cfg["evaluation"] = cfg.get("evaluation") or {}
    cfg["evaluation"]["games_vs_best"] = 0
    cfg["evaluation"]["games_vs_pure_mcts"] = 0
    cfg["evaluation"]["elo"] = cfg["evaluation"].get("elo") or {
        "enabled": False, "initial_rating": 1500, "k_factor": 32
    }
    cfg["iteration"] = cfg.get("iteration") or {}
    cfg["iteration"]["max_iterations"] = 3
    return cfg


class TestGracefulShutdown:
    """A) Stop requested during iteration → exit only after commit."""

    def test_shutdown_exits_after_commit_marker_exists(self):
        """When _shutdown_requested is set, process exits only after commit_manifest exists."""
        cfg = _minimal_resume_config()
        if cfg is None:
            pytest.skip("config_e2e_smoke.yaml not found")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            run_root = td / "resume_test_run"
            run_root.mkdir(parents=True)
            cfg["paths"] = cfg.get("paths") or {}
            cfg["paths"]["run_dir"] = str(run_root)
            cfg["paths"]["run_id"] = "resume_test"
            cfg["paths"]["checkpoints_dir"] = str(run_root / "checkpoints")
            cfg["paths"]["logs_dir"] = str(run_root / "logs")
            cfg_path = td / "config_resume_test.yaml"
            with open(cfg_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)

            # Import here so we can mutate the module-level flag
            import src.training.main as main_mod

            # Reset shutdown state (may be set by previous test)
            main_mod._shutdown_requested = False
            main_mod._hard_stop_requested = False

            from src.training.main import AlphaZeroTrainer

            trainer = AlphaZeroTrainer(str(cfg_path), cli_run_dir=str(run_root), cli_run_id="resume_test")
            train_done = threading.Event()
            exc = []

            def run_train():
                try:
                    trainer.train(start_iteration=0, resume_checkpoint=None)
                except Exception as e:
                    exc.append(e)
                finally:
                    train_done.set()

            t = threading.Thread(target=run_train, daemon=True)
            t.start()

            # Wait for at least one iteration to complete (2 games + 1 epoch can take 30-90s on CPU)
            # We set shutdown after a delay; iteration will finish then we break at start of next
            time.sleep(15)
            main_mod._shutdown_requested = True

            train_done.wait(timeout=120)
            if exc:
                raise exc[0]

            # Assert: at least one committed iteration has commit_manifest.json
            last_comm, _ = reconcile_run_state(run_root, -1)
            assert last_comm >= 0, "Expected at least one committed iteration after graceful shutdown"
            manifest = committed_dir(run_root, last_comm) / COMMIT_MANIFEST
            assert manifest.exists(), f"commit_manifest.json missing for iter{last_comm:03d}"

    def test_double_sigint_during_commit_still_exits_after_manifest(self):
        """Second Ctrl+C during commit: _commit_in_progress guard prevents os._exit; manifest exists."""
        cfg = _minimal_resume_config()
        if cfg is None:
            pytest.skip("config_best.yaml not found")

        os.environ["PATCHWORK_SLOW_COMMIT_FOR_TEST"] = "1"
        try:
            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                run_root = td / "resume_test_run"
                run_root.mkdir(parents=True)
                cfg["paths"] = cfg.get("paths") or {}
                cfg["paths"]["run_dir"] = str(run_root)
                cfg["paths"]["run_id"] = "resume_test"
                cfg["paths"]["checkpoints_dir"] = str(run_root / "checkpoints")
                cfg["paths"]["logs_dir"] = str(run_root / "logs")
                cfg_path = td / "config_resume_test.yaml"
                with open(cfg_path, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False)

                import src.training.main as main_mod

                main_mod._shutdown_requested = False
                main_mod._hard_stop_requested = False

                def simulate_second_ctrl_c():
                    main_mod._hard_stop_requested = True

                register_commit_test_callback(simulate_second_ctrl_c)
                try:
                    from src.training.main import AlphaZeroTrainer

                    trainer = AlphaZeroTrainer(str(cfg_path), cli_run_dir=str(run_root), cli_run_id="resume_test")
                    train_done = threading.Event()
                    exc = []

                    def run_train():
                        try:
                            trainer.train(start_iteration=0, resume_checkpoint=None)
                        except Exception as e:
                            exc.append(e)
                        finally:
                            train_done.set()

                    t = threading.Thread(target=run_train, daemon=True)
                    t.start()

                    time.sleep(12)
                    main_mod._shutdown_requested = True

                    train_done.wait(timeout=120)
                    if exc:
                        raise exc[0]

                    last_comm, _ = reconcile_run_state(run_root, -1)
                    assert last_comm >= 0
                    manifest = committed_dir(run_root, last_comm) / COMMIT_MANIFEST
                    assert manifest.exists(), (
                        f"commit_manifest.json missing for iter{last_comm:03d} "
                        "— double SIGINT during commit must not corrupt"
                    )
                finally:
                    clear_commit_test_callbacks()
        finally:
            os.environ.pop("PATCHWORK_SLOW_COMMIT_FOR_TEST", None)


class TestResumeCorrectness:
    """B) Restart uses next iteration and expected checkpoint."""

    def test_resume_starts_at_next_iteration(self):
        """After one committed iteration, resume starts at next_iter and uses best_model for selfplay."""
        cfg = _minimal_resume_config()
        if cfg is None:
            pytest.skip("config_e2e_smoke.yaml not found")

        with tempfile.TemporaryDirectory() as td:
            run_root = Path(td) / "resume_test_run"
            run_root.mkdir(parents=True)
            ckpt_dir = run_root / "checkpoints"
            ckpt_dir.mkdir()
            committed_base = run_root / "committed"
            committed_base.mkdir(parents=True)

            # Create minimal committed iter 0 layout (enough for auto-resume to work)
            import torch
            from src.network.model import create_network

            net = create_network(cfg)
            best_path = ckpt_dir / "best_model.pt"
            torch.save({"model_state_dict": net.state_dict(), "global_step": 0}, best_path)
            latest_path = ckpt_dir / "latest_model.pt"
            torch.save({"model_state_dict": net.state_dict(), "global_step": 0}, latest_path)

            iter0_dir = committed_base / "iter_000"
            iter0_dir.mkdir()
            manifest = {
                "iteration": 0,
                "commit_method": "rename",
                "best_model_path": str(best_path),
                "latest_model_path": str(latest_path),
            }
            (iter0_dir / COMMIT_MANIFEST).write_text(json.dumps(manifest, indent=2))
            # Minimal selfplay.h5 so replay can load
            import h5py
            import numpy as np
            sp_path = iter0_dir / "selfplay.h5"
            with h5py.File(sp_path, "w") as f:
                n = 50
                ch = int(cfg.get("network", {}).get("input_channels", 56))
                f.create_dataset("states", shape=(n, ch, 9, 9), dtype=np.float32)
                f.create_dataset("action_masks", shape=(n, 2026), dtype=np.float32)
                f.create_dataset("policies", shape=(n, 2026), dtype=np.float32)
                f.create_dataset("values", shape=(n,), dtype=np.float32)
                f.create_dataset("ownerships", shape=(n, 2, 9, 9), dtype=np.float32)
                f.attrs["num_games"] = 2
                f.attrs["num_positions"] = n
                f.attrs["selfplay_complete"] = True

            run_state_path = run_root / "run_state.json"
            run_state = {
                "last_committed_iteration": 0,
                "best_model_path": str(best_path),
                "best_iteration": 0,
                "latest_model_path": str(latest_path),
                "latest_checkpoint": str(latest_path),
                "global_step": 0,
                "consecutive_rejections": 0,
            }
            run_state_path.write_text(json.dumps(run_state, indent=2))

            # Replay buffer state so get_training_data has iter 0 data
            replay_state_path = run_root / "replay_state.json"
            replay_state_path.write_text(
                json.dumps([{"iteration": 0, "path": str(sp_path), "positions": 50}])
            )

            cfg["paths"] = cfg.get("paths") or {}
            cfg["paths"]["run_dir"] = str(run_root)
            cfg["paths"]["run_id"] = "resume_test"
            cfg["paths"]["checkpoints_dir"] = str(ckpt_dir)
            cfg["paths"]["logs_dir"] = str(run_root / "logs")
            cfg_path = Path(td) / "config_resume_test.yaml"
            with open(cfg_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)

            from src.training.main import AlphaZeroTrainer

            trainer = AlphaZeroTrainer(str(cfg_path), cli_run_dir=str(run_root), cli_run_id="resume_test")
            start_iter, resume_ckpt = trainer._try_auto_resume(start_iteration=0, resume_checkpoint=None)

            assert start_iter == 1, f"Expected next_iteration=1, got {start_iter}"
            assert resume_ckpt is not None, "Expected checkpoint for selfplay"
            assert Path(resume_ckpt).exists(), f"Checkpoint {resume_ckpt} should exist"

    def test_resume_plan_consistent_with_cleanup_logic(self):
        """get_staging_cleanup_plan matches cleanup_staging behavior."""
        cfg = _minimal_resume_config()
        if cfg is None:
            pytest.skip("config_e2e_smoke.yaml not found")

        with tempfile.TemporaryDirectory() as td:
            run_root = Path(td) / "plan_test"
            run_root.mkdir()
            (run_root / "run_state.json").write_text('{"last_committed_iteration": 0}')
            staging = run_root / "staging"
            staging.mkdir()
            (staging / "iter_001").mkdir()

            plan = get_staging_cleanup_plan(run_root, 0, cfg)
            deletes = [(i, r) for i, a, r in plan if a == "delete"]
            assert len(deletes) >= 1, "Partial staging iter_001 should be marked for deletion"
            assert any("staging" in r.lower() and "discard" in r.lower() for _, r in deletes)
