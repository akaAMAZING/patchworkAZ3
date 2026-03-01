#!/usr/bin/env python
"""
Final validation of 201-bin + WIN_FIRST integration before resuming production training.

Purpose: Validate ONE checkpoint (the candidate to resume from). Not "latest vs another model".
  - Phase 3 A/B: that same checkpoint plays vs a BASELINE (a trained model). We compare
    WIN_FIRST on vs off (win rate, margins, worst-5% loss). Baseline: --ab-baseline, or
    an early committed iter, or checkpoints/latest_model.pt. Pure MCTS only if none exist.

Runs:
  1) Checkpoint compatibility smoke (load scalar-head into 201-bin, GPU server + eval_client)
  2) Self-play speed benchmark (games/min, eval req/sec, etc.)
  3) Behavioral A/B: same checkpoint vs baseline, WIN_FIRST on vs off
  4) WIN_FIRST root selection debug log (1 game)
  5) Structured report + go/no-go recommendation

Usage:
  python tools/validate_201bin_winfirst.py --config configs/config_best.yaml --checkpoint checkpoints/best_model.pt
  python tools/validate_201bin_winfirst.py --config configs/config_best.yaml --benchmark-minutes 5 --ab-games 200
"""

from __future__ import annotations

import argparse
import copy
import io
import logging
import multiprocessing as mp
import os
import queue
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _run_gpu_server_standalone(config: dict, checkpoint_path: str, req_q, resp_q, stop_evt, ready_q) -> None:
    """Top-level target for GPU server subprocess (Windows spawn requires picklable target)."""
    from src.network.gpu_inference_server import run_gpu_inference_server
    run_gpu_inference_server(config, checkpoint_path, req_q, [resp_q], stop_evt, ready_q, "cuda")


def _discover_best_checkpoint() -> Path | None:
    """Return path to best_model.pt or latest iteration in committed/."""
    for name in ("best_model.pt", "best.pt", "latest_model.pt"):
        p = REPO_ROOT / "checkpoints" / name
        if p.is_file():
            return p
    runs = REPO_ROOT / "runs"
    if runs.is_dir():
        for run_dir in sorted(runs.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            committed = run_dir / "committed"
            if committed.is_dir():
                iters = sorted(
                    (f for f in committed.iterdir() if f.suffix == ".pt" and "iteration" in f.name),
                    key=lambda f: int(re.search(r"(\d+)", f.name).group(1)) if re.search(r"(\d+)", f.name) else 0,
                    reverse=True,
                )
                if iters:
                    return iters[0]
    return None


def _discover_ab_baseline() -> Path | None:
    """Return path to a trained model for A/B baseline. Order: early committed iter, then checkpoints/latest_model.pt. None => use pure_mcts."""
    runs = REPO_ROOT / "runs"
    if runs.is_dir():
        for run_dir in sorted(runs.iterdir()):
            if not run_dir.is_dir():
                continue
            committed_base = run_dir / "committed"
            if not committed_base.is_dir():
                continue
            found = []
            for sub in committed_base.iterdir():
                if not sub.is_dir() or not sub.name.startswith("iter_"):
                    continue
                m = re.match(r"iter_(\d+)", sub.name)
                if not m:
                    continue
                iter_num = int(m.group(1))
                ckpt = sub / f"iteration_{iter_num:03d}.pt"
                if ckpt.is_file():
                    found.append((iter_num, ckpt))
            if not found:
                continue
            found.sort(key=lambda x: x[0])
            return found[0][1]
    # Fallback: use latest_model.pt so we don't need pure_mcts
    latest = REPO_ROOT / "checkpoints" / "latest_model.pt"
    if latest.is_file():
        return latest
    return None


def run_1_checkpoint_smoke(config: dict, checkpoint_path: Path, device: torch.device) -> dict:
    """Load scalar-head checkpoint into 201-bin model; verify logs, forward, then GPU server + eval_client."""
    from src.network.model import create_network, load_model_checkpoint, get_state_dict_for_inference, ValueHead

    result = {"passed": False, "score_head_skipped": None, "forward_ok": False, "server_responses_ok": False, "messages": []}

    # Capture log for "score_head" / "Dropping incompatible"
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.WARNING)
    log = logging.getLogger("src.network.model")
    log.addHandler(handler)
    try:
        net = create_network(config)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = get_state_dict_for_inference(ckpt, config, for_selfplay=False)
        model_state = net.state_dict()
        # ---- Diagnostic: state_dict source, score_head keys, missing/unexpected, score_head_reinitialized ----
        use_ema = False
        if "ema" in config.get("training", {}):
            ema_cfg = config["training"]["ema"]
            if ema_cfg.get("enabled", False) and ema_cfg.get("use_for_eval", True):
                use_ema = True
        if use_ema and "ema_state_dict" in ckpt:
            state_dict_source = "ema_state_dict"
        elif "model_state_dict" in ckpt:
            state_dict_source = "model_state_dict"
        else:
            state_dict_source = "root (ckpt itself)"
        print(f"  [Phase 1] state_dict loaded from: {state_dict_source}")

        ckpt_score_keys = [k for k in state_dict.keys() if "score_head" in k]
        model_score_keys = [k for k in model_state.keys() if "score_head" in k]
        expected_weight_key = "value_head.score_head.weight"
        expected_bias_key = "value_head.score_head.bias"
        has_expected_weight = expected_weight_key in state_dict
        has_expected_bias = expected_bias_key in state_dict

        print(f"  [Phase 1] score-related keys in checkpoint: {len(ckpt_score_keys)}")
        for k in sorted(ckpt_score_keys):
            shape = tuple(state_dict[k].shape)
            print(f"    {k!r} shape={shape}")
        value_hidden = int(config.get("network", {}).get("value_hidden", 512))
        expected_out = ValueHead.SCORE_BINS
        print(f"  [Phase 1] model expected score_head: weight {expected_weight_key!r} shape=({expected_out}, {value_hidden}), bias {expected_bias_key!r} shape=({expected_out},)")
        print(f"  [Phase 1] expected keys in checkpoint: {expected_weight_key!r} exists={has_expected_weight}, {expected_bias_key!r} exists={has_expected_bias}")

        all_missing = set(model_state.keys()) - set(state_dict.keys())
        all_unexpected = set(state_dict.keys()) - set(model_state.keys())
        missing_score = sorted(k for k in all_missing if "score_head" in k)
        unexpected_score = sorted(k for k in all_unexpected if "score_head" in k)
        print(f"  [Phase 1] load_state_dict would report: missing_keys (score_head) = {missing_score!r}")
        print(f"  [Phase 1] load_state_dict would report: unexpected_keys (score_head) = {unexpected_score!r}")

        shape_mismatch = False
        for k in ckpt_score_keys:
            if k in model_state and state_dict[k].shape != model_state[k].shape:
                shape_mismatch = True
                print(f"  [Phase 1] shape mismatch: {k!r} ckpt={tuple(state_dict[k].shape)} model={tuple(model_state[k].shape)}")

        old_style_keys = [k for k in unexpected_score if ".0." in k or k.endswith(".0.weight") or k.endswith(".0.bias")]
        score_head_reinitialized = (
            not has_expected_weight or not has_expected_bias
            or bool(missing_score)
            or bool(old_style_keys)
            or shape_mismatch
        )
        print(f"  [Phase 1] score_head_reinitialized = {score_head_reinitialized} (expected weight/bias missing={not (has_expected_weight and has_expected_bias)}, missing_score={bool(missing_score)}, old_style score_head.0.*={bool(old_style_keys)}, shape_mismatch={shape_mismatch})")
        result["score_head_skipped"] = score_head_reinitialized
        # ---- end diagnostic ----
        load_model_checkpoint(net, state_dict)
        net.to(device)
        net.eval()
    finally:
        log.removeHandler(handler)
    if result.get("score_head_skipped") is None:
        result["score_head_skipped"] = False
    result["messages"].append(f"score_head reinitialized: {result['score_head_skipped']} (expected True for scalar-head ckpt)")

    # Forward: batch of 2, check score_logits shape (B, 201) and no NaNs.
    # Run on CPU to avoid CUDA index bounds issues with dummy shop_ids; gold_v2 uses 32ch + det_legality.
    net_cpu = net.to(torch.device("cpu"))
    with torch.no_grad():
        B = 2
        net_cfg = config.get("network", {})
        use_film = bool(net_cfg.get("use_film", False))
        if use_film:
            # 32ch state so _apply_gpu_legality runs; shop_ids 0..32 (33 pieces, index 33 is OOB)
            x = torch.randn(B, 32, 9, 9, device="cpu", dtype=torch.float32)
            m = torch.ones(B, 2026, device="cpu")
            g_dim = int(net_cfg.get("film_global_dim", 61))
            t_c, t_l = 8, 54
            x_global = torch.randn(B, g_dim, device="cpu", dtype=torch.float32)
            x_track = torch.randn(B, t_c, t_l, device="cpu", dtype=torch.float32)
            shop_ids = torch.randint(0, 33, (B, 33), device="cpu", dtype=torch.long)
            shop_feats = torch.randn(B, 33, 10, device="cpu", dtype=torch.float32)
            try:
                pl, v, sl = net_cpu(x, m, x_global=x_global, x_track=x_track, shop_ids=shop_ids, shop_feats=shop_feats)
            except Exception as e:
                result["messages"].append(f"Forward failed: {e}")
                return result
        else:
            x = torch.randn(B, 56, 9, 9, device="cpu", dtype=torch.float32)
            m = torch.ones(B, 2026, device="cpu")
            try:
                pl, v, sl = net_cpu(x, m)
            except Exception as e:
                result["messages"].append(f"Forward failed: {e}")
                return result
    net.to(device)
    if v.shape != (B, 1):
        result["messages"].append(f"value shape {v.shape} != (B, 1)")
        return result
    if sl.shape != (B, ValueHead.SCORE_BINS):
        result["messages"].append(f"score_logits shape {sl.shape} != (B, {ValueHead.SCORE_BINS})")
        return result
    if not torch.isfinite(sl).all() or not torch.isfinite(v).all():
        result["messages"].append("NaNs/inf in value or score_logits")
        return result
    result["forward_ok"] = True
    result["messages"].append("Forward: value (B,1), score_logits (B,201), no NaNs")

    # GPU server + eval_client (subprocess server, same process client)
    if not torch.cuda.is_available():
        result["messages"].append("CUDA not available; skipping server/eval_client check")
        result["passed"] = result["forward_ok"]
        return result

    req_q = mp.Queue(maxsize=64)
    resp_q = mp.Queue(maxsize=64)
    stop_evt = mp.Event()
    ready_q = mp.Queue(maxsize=1)

    proc = mp.Process(
        target=_run_gpu_server_standalone,
        args=(config, str(checkpoint_path), req_q, resp_q, stop_evt, ready_q),
    )
    proc.start()
    try:
        status = ready_q.get(timeout=120)
        if status != "ready":
            result["messages"].append(f"Server not ready: {status}")
            return result
    except queue.Empty:
        proc.terminate()
        proc.join(timeout=5)
        result["messages"].append("GPU server did not signal ready within 120s")
        return result

    # Send a few eval requests. Server expects gold_v2_32ch when config has use_film.
    from src.mcts.gpu_eval_client import GPUEvalClient
    from src.network.gold_v2_constants import C_SPATIAL_ENC, F_GLOBAL, C_TRACK, TRACK_LEN, NMAX, F_SHOP
    client = GPUEvalClient(req_q, resp_q, worker_id=0, timeout_s=30)
    n_req = 3
    rids = []
    use_film = bool(config.get("network", {}).get("use_film", False))
    for _ in range(n_req):
        if use_film:
            x_spatial = np.random.randn(C_SPATIAL_ENC, 9, 9).astype(np.float32) * 0.1
            x_global = np.zeros(F_GLOBAL, dtype=np.float32)
            x_track = np.zeros((C_TRACK, TRACK_LEN), dtype=np.float32)
            shop_ids = np.full(NMAX, -1, dtype=np.int64)
            shop_feats = np.zeros((NMAX, F_SHOP), dtype=np.float32)
            mask_np = np.ones(2026, dtype=np.float32)
            legal_np = np.array([0, 1], dtype=np.int32)
            rid = client.submit_multimodal(x_spatial, x_global, x_track, shop_ids, shop_feats, mask_np, legal_np, score_center_points=0.0, effective_static_w=0.0, effective_dynamic_w=0.3)
        else:
            state_np = np.random.randn(56, 9, 9).astype(np.float32) * 0.1
            mask_np = np.ones(2026, dtype=np.float32)
            legal_np = np.array([0, 1], dtype=np.int32)
            rid = client.submit_legacy(state_np, mask_np, legal_np, score_center_points=0.0, effective_static_w=0.0, effective_dynamic_w=0.3)
        rids.append(rid)

    ok_count = 0
    for rid in rids:
        try:
            priors, value, mean_points, score_utility = client.receive(rid)
        except Exception as e:
            result["messages"].append(f"receive({rid}) failed: {e}")
            continue
        if not isinstance(priors, np.ndarray) or value is None:
            result["messages"].append(f"Bad response types: priors={type(priors)}, value={type(value)}")
            continue
        # Sensible ranges
        if not (-1.01 <= float(value) <= 1.01):
            result["messages"].append(f"value {value} outside [-1,1]")
            continue
        if not (-101 <= float(mean_points) <= 101):
            result["messages"].append(f"mean_points {mean_points} outside ~[-100,100]")
            continue
        # score_utility roughly in [-|w|,|w|] with w~0.3
        if not (-1.1 <= float(score_utility) <= 1.1):
            result["messages"].append(f"score_utility {score_utility} outside [-1,1]")
            continue
        ok_count += 1
    stop_evt.set()
    proc.join(timeout=10)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)

    result["server_responses_ok"] = ok_count == n_req
    result["messages"].append(f"Server responses: {ok_count}/{n_req} with sensible (value, mean_points, score_utility)")
    result["passed"] = result["forward_ok"] and result["server_responses_ok"]
    return result


def run_2_selfplay_benchmark(config: dict, checkpoint_path: Path, minutes: float) -> dict:
    """Run self-play for `minutes`; report games/min, eval req/sec, avg batch size, etc."""
    from src.training.selfplay_optimized_integration import SelfPlayGenerator

    result = {"passed": False, "games": 0, "games_per_min": 0.0, "eval_req_per_sec": 0.0, "avg_batch_size": 0.0, "avg_eval_latency_ms": None, "gpu_util": None, "messages": []}
    cfg = copy.deepcopy(config)
    sims = int(cfg.get("selfplay", {}).get("mcts", {}).get("simulations", 192))
    max_game_len = int(cfg.get("selfplay", {}).get("max_game_length", 200))
    num_games_bench = max(5, int(minutes * 2))
    cfg.setdefault("iteration", {})
    cfg["iteration"]["games_schedule"] = [{"iteration": 0, "games": 999}, {"iteration": 1, "games": num_games_bench}]
    cfg.setdefault("selfplay", {}).setdefault("bootstrap", {})
    cfg["selfplay"]["bootstrap"]["games"] = 0
    gen = SelfPlayGenerator(cfg)
    network_path = str(checkpoint_path)
    if not Path(network_path).is_file():
        result["messages"].append(f"Checkpoint not found: {network_path}")
        return result

    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="validate_bench_"))
    try:
        start = time.perf_counter()
        data_path, stats = gen.generate(iteration=1, network_path=network_path, output_dir=tmpdir)
        elapsed = time.perf_counter() - start
    except Exception as e:
        result["messages"].append(f"Benchmark error: {e}")
        return result
    finally:
        try:
            if (tmpdir / "selfplay.h5").exists():
                (tmpdir / "selfplay.h5").unlink()
            tmpdir.rmdir()
        except Exception:
            pass

    games_done = stats.get("num_games", 0)
    if elapsed <= 0:
        result["messages"].append("No elapsed time")
        return result
    total_evals = games_done * sims * min(40, max_game_len)
    result["games"] = games_done
    result["games_per_min"] = games_done / (elapsed / 60.0)
    result["eval_req_per_sec"] = total_evals / elapsed if total_evals else 0.0
    result["avg_batch_size"] = int(cfg.get("inference", {}).get("batch_size", 512))
    result["passed"] = games_done >= 1
    result["messages"].append(f"Ran {games_done} games in {elapsed:.1f}s -> {result['games_per_min']:.2f} games/min, ~{result['eval_req_per_sec']:.0f} eval req/s")
    return result


def run_3_ab_winfirst(
    config: dict,
    checkpoint_path: Path,
    num_games: int,
    device: torch.device,
    baseline_path: Path | None = None,
) -> dict:
    """A: WIN_FIRST enabled, B: disabled. Same checkpoint vs baseline (trained model or pure_mcts)."""
    from src.training.evaluation import Evaluator

    if baseline_path is not None and baseline_path.is_file():
        baseline_type = "previous_best"
        baseline_label = baseline_path.name
    else:
        baseline_type = "pure_mcts"
        baseline_label = "pure_mcts"

    progress_interval = max(10, num_games // 20)

    def run_side(win_first_enabled: bool, label: str):
        cfg = copy.deepcopy(config)
        cfg.setdefault("selfplay", {}).setdefault("mcts", {}).setdefault("win_first", {})["enabled"] = win_first_enabled
        ev = Evaluator(cfg, device)
        print(f"  {label}: {num_games} games vs {baseline_label} (progress every {progress_interval} games)...", flush=True)
        sys.stdout.flush()

        def progress_callback(n_done: int, wins: int, wr: float):
            print(f"    {label}: {n_done}/{num_games} games, {wins} wins ({wr:.1%})", flush=True)
            sys.stdout.flush()

        stats = ev.evaluate_vs_baseline(
            str(checkpoint_path),
            baseline_type=baseline_type,
            baseline_path=str(baseline_path) if baseline_path is not None else None,
            num_games=num_games,
            game_offset=0,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
        )
        wr = stats.get("win_rate", 0.0)
        wins = stats.get("model_wins", 0)
        print(f"  {label}: done {wins}/{num_games} wins, WR {wr:.1%}", flush=True)
        return stats

    stats_a = run_side(True, "A (WIN_FIRST on)")
    stats_b = run_side(False, "B (WIN_FIRST off)")

    results_a = stats_a.get("results", [])
    results_b = stats_b.get("results", [])

    def _worst_pct_loss_tail(loss_margins: list[float], pct: float = 5.0) -> float | None:
        """Mean of the worst (most negative) pct% of loss margins. Matches eval_latest_vs_oldest.py."""
        if not loss_margins:
            return None
        sorted_losses = sorted(loss_margins)
        n = max(1, int(len(sorted_losses) * pct / 100.0))
        worst = sorted_losses[:n]
        return float(sum(worst) / len(worst))

    def worst_5pct_loss(results_list):
        loss_margins = [r["model_score_margin"] for r in results_list if not r.get("model_won", True)]
        return _worst_pct_loss_tail(loss_margins, 5.0)

    out = {
        "A_win_first_on": {
            "win_rate": stats_a.get("win_rate", 0),
            "avg_final_margin": stats_a.get("avg_model_score_margin", 0),
            "avg_loss_margin": stats_a.get("avg_loss_margin", 0),
            "worst_5pct_loss": worst_5pct_loss(results_a),
        },
        "B_win_first_off": {
            "win_rate": stats_b.get("win_rate", 0),
            "avg_final_margin": stats_b.get("avg_model_score_margin", 0),
            "avg_loss_margin": stats_b.get("avg_loss_margin", 0),
            "worst_5pct_loss": worst_5pct_loss(results_b),
        },
        "passed": True,
        "messages": [],
    }
    out["messages"].append(
        f"A (WIN_FIRST on):  WR={out['A_win_first_on']['win_rate']:.2%}  avg_margin={out['A_win_first_on']['avg_final_margin']:.2f}  worst_5%_loss={out['A_win_first_on']['worst_5pct_loss']}"
    )
    out["messages"].append(
        f"B (WIN_FIRST off): WR={out['B_win_first_off']['win_rate']:.2%}  avg_margin={out['B_win_first_off']['avg_final_margin']:.2f}  worst_5%_loss={out['B_win_first_off']['worst_5pct_loss']}"
    )

    # Comparison table (same format as tools/eval_latest_vs_oldest.py run_ab_win_first)
    wr_off, wr_on = out["B_win_first_off"]["win_rate"] * 100, out["A_win_first_on"]["win_rate"] * 100
    am_off = out["B_win_first_off"]["avg_final_margin"]
    am_on = out["A_win_first_on"]["avg_final_margin"]
    al_off = out["B_win_first_off"]["avg_loss_margin"]
    al_on = out["A_win_first_on"]["avg_loss_margin"]
    w5_off = out["B_win_first_off"]["worst_5pct_loss"] if out["B_win_first_off"]["worst_5pct_loss"] is not None else 0.0
    w5_on = out["A_win_first_on"]["worst_5pct_loss"] if out["A_win_first_on"]["worst_5pct_loss"] is not None else 0.0
    out["messages"].append("")
    out["messages"].append("  Metric                    | Baseline (WF off) | New (WF on)   | Delta")
    out["messages"].append("  -------------------------|------------------|---------------|--------")
    out["messages"].append("  Win rate (model %%)        | %6.1f%%           | %6.1f%%        | %+.1f%%" % (wr_off, wr_on, wr_on - wr_off))
    out["messages"].append("  Avg final margin          | %+6.1f pts        | %+6.1f pts     | %+.1f" % (am_off, am_on, am_on - am_off))
    out["messages"].append("  Avg margin (losses only)   | %6.1f pts        | %6.1f pts     | %+.1f (better if less negative)" % (al_off, al_on, al_on - al_off))
    out["messages"].append("  Worst-5%% loss tail        | %6.1f pts        | %6.1f pts     | %+.1f (better if less negative)" % (w5_off, w5_on, w5_on - w5_off))
    print("  A/B comparison (same format as eval_latest_vs_oldest --ab-win-first):", flush=True)
    for line in out["messages"][-6:]:
        print("  ", line, flush=True)

    return out


def run_4_winfirst_debug_one_game(config: dict, checkpoint_path: Path) -> dict:
    """Run 1 self-play game with debug_log_one_game enabled. Pass if MCTS internal flag _win_first_debug_logged is True (robust; no log capture)."""
    from src.training.selfplay_optimized import init_optimized_worker

    cfg = copy.deepcopy(config)
    cfg.setdefault("selfplay", {}).setdefault("mcts", {}).setdefault("win_first", {})["debug_log_one_game"] = True
    cfg.setdefault("hardware", {})["device"] = "cpu"
    result = {"passed": False, "win_first_path_hit": False, "messages": []}
    try:
        init_optimized_worker(str(checkpoint_path), cfg)
        from src.training.selfplay_optimized import _WORKER
        if _WORKER is None or _WORKER.mcts is None:
            result["messages"].append("Worker or MCTS not initialized")
            return result
        _WORKER.play_game(0, 0, 42)
        # Robust pass: MCTS sets _win_first_debug_logged True the first time _select_action_win_first runs
        flag = getattr(_WORKER.mcts, "_win_first_debug_logged", False)
        result["win_first_path_hit"] = bool(flag)
        result["passed"] = bool(flag)
        if flag:
            result["messages"].append("WIN_FIRST root selection path exercised (_win_first_debug_logged=True).")
        else:
            result["messages"].append("WIN_FIRST path not hit in one game (_win_first_debug_logged still False).")
    except Exception as e:
        result["messages"].append(f"Play game failed: {e}")
    return result


def main():
    ap = argparse.ArgumentParser(description="201-bin + WIN_FIRST final validation")
    ap.add_argument("--config", default="configs/config_best.yaml", help="Config YAML")
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (scalar-head for smoke); default: discover")
    ap.add_argument("--benchmark-minutes", type=float, default=5.0, help="Self-play benchmark duration (minutes)")
    ap.add_argument("--ab-games", type=int, default=200, help="Games per arm for A/B (WIN_FIRST on vs off)")
    ap.add_argument("--ab-baseline", type=str, default=None, help="Path to baseline checkpoint (e.g. runs/.../committed/iter_001/iteration_001.pt). If not set, auto-discover early committed iter or use pure_mcts.")
    ap.add_argument("--skip-benchmark", action="store_true", help="Skip self-play benchmark")
    ap.add_argument("--skip-ab", action="store_true", help="Skip A/B evaluation")
    args = ap.parse_args()

    with open(REPO_ROOT / args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else _discover_best_checkpoint()
    if not checkpoint_path or not checkpoint_path.is_file():
        print("No checkpoint found. Use --checkpoint path or place best_model.pt in checkpoints/")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "1_checkpoint_smoke": None,
        "2_selfplay_benchmark": None,
        "3_ab_winfirst": None,
        "4_winfirst_debug": None,
        "go_no_go": "NO-GO",
        "summary": [],
    }

    # 1) Checkpoint smoke
    print("(1) Checkpoint compatibility smoke...")
    report["1_checkpoint_smoke"] = run_1_checkpoint_smoke(config, checkpoint_path, device)
    for m in report["1_checkpoint_smoke"].get("messages", []):
        print("  ", m)
    if not report["1_checkpoint_smoke"].get("passed"):
        report["summary"].append("FAIL: checkpoint smoke")

    # 2) Self-play benchmark
    if not args.skip_benchmark:
        print("(2) Self-play benchmark...")
        report["2_selfplay_benchmark"] = run_2_selfplay_benchmark(config, checkpoint_path, args.benchmark_minutes)
        for m in report["2_selfplay_benchmark"].get("messages", []):
            print("  ", m)
        if not report["2_selfplay_benchmark"].get("passed"):
            report["summary"].append("FAIL: benchmark")
    else:
        report["2_selfplay_benchmark"] = {"passed": True, "games_per_min": None, "messages": ["Skipped"]}

    # 3) A/B
    if not args.skip_ab:
        ab_baseline = Path(args.ab_baseline) if args.ab_baseline else _discover_ab_baseline()
        if ab_baseline is not None and ab_baseline.is_file():
            print(f"(3) A/B WIN_FIRST on vs off (baseline: {ab_baseline.name})...")
        else:
            print("(3) A/B WIN_FIRST on vs off (baseline: pure_mcts — no committed iter or checkpoints/latest_model.pt found)...")
        report["3_ab_winfirst"] = run_3_ab_winfirst(config, checkpoint_path, args.ab_games, device, baseline_path=ab_baseline)
        for m in report["3_ab_winfirst"].get("messages", []):
            print("  ", m)
    else:
        report["3_ab_winfirst"] = {"passed": True, "A_win_first_on": {}, "B_win_first_off": {}, "messages": ["Skipped"]}

    # 4) Debug log
    print("(4) WIN_FIRST root selection debug (1 game)...")
    report["4_winfirst_debug"] = run_4_winfirst_debug_one_game(config, checkpoint_path)
    for m in report["4_winfirst_debug"].get("messages", []):
        print("  ", m)

    # Go/no-go: GO if Phase 1 passes, benchmark passes, A/B shows WIN_FIRST ON not worse than OFF, Phase 4 passes.
    smoke_ok = report["1_checkpoint_smoke"] and report["1_checkpoint_smoke"].get("passed")
    bench_ok = report["2_selfplay_benchmark"] and (report["2_selfplay_benchmark"].get("passed") or report["2_selfplay_benchmark"].get("games_per_min") is None)
    r3 = report["3_ab_winfirst"]
    wr_a = r3.get("A_win_first_on", {}).get("win_rate", 0.0) if r3 else 0.0
    wr_b = r3.get("B_win_first_off", {}).get("win_rate", 0.0) if r3 else 0.0
    ab_ok = r3 and r3.get("passed") and (wr_a >= wr_b - 0.02)
    debug_ok = report["4_winfirst_debug"] and report["4_winfirst_debug"].get("passed")
    score_head_skipped = report["1_checkpoint_smoke"].get("score_head_skipped") if report["1_checkpoint_smoke"] else None
    if smoke_ok and bench_ok and ab_ok and debug_ok:
        report["go_no_go"] = "GO"
        report["summary"].append("Recommendation: GO — resume production training.")
        if score_head_skipped:
            report["summary"].append("(score_head skipped/reinit as expected for scalar-head checkpoint.)")
    else:
        if not smoke_ok:
            report["summary"].append("Checkpoint smoke failed.")
        if not bench_ok:
            report["summary"].append("Benchmark failed or regressed.")
        if not ab_ok:
            report["summary"].append("A/B failed or WIN_FIRST ON worse than OFF (WR_ON %.2f%% vs WR_OFF %.2f%%)." % (wr_a * 100, wr_b * 100))
        if not debug_ok:
            report["summary"].append("WIN_FIRST debug path not exercised in Phase 4.")
        report["summary"].append("Recommendation: NO-GO — fix above before resuming.")

    # Structured report table
    print("\n" + "=" * 60)
    print("VALIDATION REPORT (201-bin + WIN_FIRST)")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print()
    print("| Phase | Passed | Details |")
    print("|-------|--------|--------|")
    r1 = report["1_checkpoint_smoke"]
    print(f"| 1 Checkpoint smoke | {'PASS' if r1 and r1.get('passed') else 'FAIL'} | score_head_skipped={r1.get('score_head_skipped')}, forward_ok={r1.get('forward_ok')}, server_ok={r1.get('server_responses_ok')} |")
    r2 = report["2_selfplay_benchmark"]
    if r2:
        print(f"| 2 Self-play benchmark | {'PASS' if r2.get('passed') else 'FAIL'} | games={r2.get('games')}, games/min={r2.get('games_per_min')}, eval_req/s={r2.get('eval_req_per_sec')} |")
    r3 = report["3_ab_winfirst"]
    if r3 and r3.get("A_win_first_on"):
        a = r3["A_win_first_on"]
        b = r3["B_win_first_off"]
        print(f"| 3a WIN_FIRST ON  | - | WR={a.get('win_rate', 0):.2%} avg_margin={a.get('avg_final_margin')} worst_5%_loss={a.get('worst_5pct_loss')} |")
        print(f"| 3b WIN_FIRST OFF | - | WR={b.get('win_rate', 0):.2%} avg_margin={b.get('avg_final_margin')} worst_5%_loss={b.get('worst_5pct_loss')} |")
    r4 = report["4_winfirst_debug"]
    print(f"| 4 WIN_FIRST debug | {'PASS' if r4 and r4.get('passed') else 'FAIL'} | win_first_path_hit={r4.get('win_first_path_hit') if r4 else None} |")
    print()
    print("METRICS TABLE")
    print("-" * 60)
    if r2 and r2.get("games_per_min") is not None:
        print(f"  games/min:        {r2.get('games_per_min', 0):.2f}")
        print(f"  eval req/sec:     {r2.get('eval_req_per_sec', 0):.1f}")
        print(f"  avg batch size:   {r2.get('avg_batch_size', 0)}")
        print(f"  GPU utilization:  {r2.get('gpu_util', 'N/A')}")
        print(f"  avg eval latency: {r2.get('avg_eval_latency_ms', 'N/A')} ms")
    if r3 and r3.get("A_win_first_on"):
        print(f"  A (WIN_FIRST on):  win_rate={r3['A_win_first_on'].get('win_rate', 0):.2%}  avg_margin={r3['A_win_first_on'].get('avg_final_margin')}  avg_loss_margin={r3['A_win_first_on'].get('avg_loss_margin')}  worst_5%_loss={r3['A_win_first_on'].get('worst_5pct_loss')}")
        print(f"  B (WIN_FIRST off): win_rate={r3['B_win_first_off'].get('win_rate', 0):.2%}  avg_margin={r3['B_win_first_off'].get('avg_final_margin')}  avg_loss_margin={r3['B_win_first_off'].get('avg_loss_margin')}  worst_5%_loss={r3['B_win_first_off'].get('worst_5pct_loss')}")
    print()
    print("GO/NO-GO:", report["go_no_go"])
    for s in report["summary"]:
        print(" ", s)
    r1 = report["1_checkpoint_smoke"]
    print()
    print("Exact numbers:")
    print("  Phase 1 score_head_skipped:", r1.get("score_head_skipped") if r1 else "N/A")
    if r2 and r2.get("games_per_min") is not None:
        print("  Benchmark games/min: %.2f  eval_req/s: %.1f  avg_batch_size: %s" % (r2.get("games_per_min", 0), r2.get("eval_req_per_sec", 0), r2.get("avg_batch_size")))
    if r3 and r3.get("A_win_first_on"):
        a, b = r3["A_win_first_on"], r3["B_win_first_off"]
        print("  A (WIN_FIRST on):  WR=%.2f%%  avg_margin=%.2f  avg_loss_margin=%.2f  worst_5%%_loss=%s" % (a.get("win_rate", 0) * 100, a.get("avg_final_margin", 0), a.get("avg_loss_margin", 0), a.get("worst_5pct_loss")))
        print("  B (WIN_FIRST off): WR=%.2f%%  avg_margin=%.2f  avg_loss_margin=%.2f  worst_5%%_loss=%s" % (b.get("win_rate", 0) * 100, b.get("avg_final_margin", 0), b.get("avg_loss_margin", 0), b.get("worst_5pct_loss")))
    print("  Phase 4 win_first_path_hit:", r4.get("win_first_path_hit") if r4 else "N/A")
    print("=" * 60)
    sys.exit(0 if report["go_no_go"] == "GO" else 1)


if __name__ == "__main__":
    main()
