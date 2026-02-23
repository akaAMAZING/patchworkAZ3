#!/usr/bin/env python
"""
Evaluate two model checkpoints head-to-head.

Modes:
  1. Interactive (default): Lists all available checkpoints with indices, prompts you to pick two.
  2. Index args:  --model_a 3 --model_b 7  (use indices from the list)
  3. Path args:   --model_a path/to/a.pt --model_b path/to/b.pt
  4. Legacy:      --ckpt_dir X --window N  (latest vs oldest in last N checkpoints)

Run from repo root:
    python tools/eval_latest_vs_oldest.py              # GUI (default)
    python tools/eval_latest_vs_oldest.py --model_a 0 --model_b 5 --games 20
    python tools/eval_latest_vs_oldest.py --no-gui     # Interactive CLI instead of GUI

With --elo (default): persists Glicko-2 ratings to elo_ratings.json.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from src.network.model import create_network, load_model_checkpoint
from src.network.encoder import ActionEncoder, StateEncoder
from src.mcts.alphazero_mcts_optimized import create_optimized_mcts
from src.game.patchwork_engine import (
    apply_action_unchecked,
    compute_score_fast,
    current_player_fast,
    get_winner_fast,
    new_game,
    terminal_fast,
)
from elo_system import (
    expected_score,
    load_ratings,
    model_elo_id,
    save_ratings,
    update_ratings_from_match,
    update_ratings_standard_elo_batch,
    wr_to_elo_gap,
)

# Match iteration_*.pt (committed snapshots) and best_model_iter*.pt (gate-promoted copies)
_ITER_PATTERNS = [
    re.compile(r"iteration_(\d+)\.pt$"),
    re.compile(r"best_model_iter(\d+)\.pt$"),
]


def iter_num(p: Path) -> int | None:
    """Extract iteration number if it's an iter checkpoint, else None."""
    for pattern in _ITER_PATTERNS:
        m = pattern.search(p.name)
        if m:
            return int(m.group(1))
    return None


def _label_for(p: Path) -> str:
    """Short readable label for a checkpoint."""
    n = iter_num(p)
    if n is not None:
        return f"iter{n:03d}"
    if p.name == "best_model.pt":
        return "best (current)"
    if p.name == "latest_model.pt":
        return "latest"
    return p.stem


def discover_checkpoints(repo_root: Path) -> list[tuple[Path, str]]:
    """
    Discover all model checkpoints in checkpoints/ and runs/.
    Returns list of (path, label) sorted by label.
    """
    results: list[tuple[Path, str]] = []
    seen_paths: set[Path] = set()

    def add(p: Path, label: str) -> None:
        p = p.resolve()
        if p in seen_paths:
            return
        seen_paths.add(p)
        results.append((p, label))

    # 1. checkpoints/ - best_model.pt, best_model_iter*.pt, latest_model.pt, iteration_*.pt
    ckpt_dir = repo_root / "checkpoints"
    if ckpt_dir.is_dir():
        for name in ("best_model.pt", "latest_model.pt"):
            f = ckpt_dir / name
            if f.is_file():
                add(f, _label_for(f))
        for f in ckpt_dir.glob("best_model_iter*.pt"):
            if f.is_file():
                add(f, _label_for(f))
        for f in ckpt_dir.glob("iteration_*.pt"):
            if f.is_file():
                add(f, _label_for(f))

    # 2. runs/<run>/committed/iter_XXX/iteration_XXX.pt
    runs_dir = repo_root / "runs"
    if runs_dir.is_dir():
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            committed = run_dir / "committed"
            if not committed.is_dir():
                continue
            for iter_dir in sorted(committed.iterdir()):
                if iter_dir.is_dir() and iter_dir.name.startswith("iter_"):
                    for f in iter_dir.glob("iteration_*.pt"):
                        if f.is_file():
                            add(f, f"committed/{_label_for(f)}")

    # 3. runs/<run>/staging/iter_XXX/iteration_XXX.pt
    if runs_dir.is_dir():
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            staging = run_dir / "staging"
            if not staging.is_dir():
                continue
            for iter_dir in sorted(staging.iterdir()):
                if iter_dir.is_dir() and iter_dir.name.startswith("iter_"):
                    for f in iter_dir.glob("iteration_*.pt"):
                        if f.is_file():
                            add(f, f"staging/{_label_for(f)}")

    # Sort: prefer iter number, then alphabetically by label
    def sort_key(item: tuple[Path, str]) -> tuple[int, str]:
        path, label = item
        n = iter_num(path)
        return (n if n is not None else 9999, label)

    results.sort(key=sort_key)
    return results


def torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_eval(
    cfg: dict,
    model_a_path: Path,
    model_b_path: Path,
    games: int,
    sims: int,
    cpuct: float,
    device: torch.device,
    log: logging.Logger,
    elo_file: Path | None = None,
    elo_style: str = "glicko2",
) -> dict | None:
    """Run head-to-head evaluation between two checkpoints."""
    id_a = model_elo_id(model_a_path)
    id_b = model_elo_id(model_b_path)
    ratings = load_ratings(elo_file) if elo_file else {}
    if elo_file:
        ra = ratings.get(id_a, {}).get("rating", 1500)
        rda = ratings.get(id_a, {}).get("rd", 350)
        rb = ratings.get(id_b, {}).get("rating", 1500)
        rdb = ratings.get(id_b, {}).get("rd", 350)
        exp_a = expected_score(ra, rb, rda, rdb)
        log.info(f"ELO (pre):  A {id_a} {ra:.0f} ±{rda:.0f}  |  B {id_b} {rb:.0f} ±{rdb:.0f}  |  E[A win]={exp_a*100:.0f}%\n")

    state_enc = StateEncoder()
    action_enc = ActionEncoder()

    model_a = create_network(cfg)
    ckpt_a = torch_load(str(model_a_path), device)
    load_model_checkpoint(model_a, ckpt_a["model_state_dict"])
    model_a.to(device).eval()

    model_b = create_network(cfg)
    ckpt_b = torch_load(str(model_b_path), device)
    load_model_checkpoint(model_b, ckpt_b["model_state_dict"])
    model_b.to(device).eval()

    mcts_a = create_optimized_mcts(model_a, cfg, device, state_enc, action_enc)
    mcts_a.config.simulations = sims
    mcts_a.config.cpuct = cpuct

    mcts_b = create_optimized_mcts(model_b, cfg, device, state_enc, action_enc)
    mcts_b.config.simulations = sims
    mcts_b.config.cpuct = cpuct

    max_moves = int(cfg["selfplay"]["max_game_length"])
    base_seed = int(cfg.get("seed", 42))

    a_wins = 0
    b_wins = 0
    margins: list[float] = []  # A_score - B_score per game (from A's perspective)
    move_counts: list[int] = []
    a_wins_as_p0 = 0
    a_games_as_p0 = 0

    label_a = model_a_path.name
    label_b = model_b_path.name

    for game_idx in range(games):
        a_is_p0 = (game_idx % 2 == 0)
        a_player = 0 if a_is_p0 else 1
        b_player = 1 - a_player

        seed = base_seed + game_idx * 1000
        state = new_game(seed=seed)

        move_number = 0
        while move_number < max_moves and not terminal_fast(state):
            to_move = current_player_fast(state)

            if to_move == a_player:
                vc, *_ = mcts_a.search(state, to_move, move_number, add_noise=False)
                action = mcts_a.select_action(vc, temperature=0.0, deterministic=True)
            else:
                vc, *_ = mcts_b.search(state, to_move, move_number, add_noise=False)
                action = mcts_b.select_action(vc, temperature=0.0, deterministic=True)

            state = apply_action_unchecked(state, action)
            move_number += 1

        winner = get_winner_fast(state)
        p0_score = compute_score_fast(state, 0)
        p1_score = compute_score_fast(state, 1)

        a_score = p0_score if a_is_p0 else p1_score
        b_score = p1_score if a_is_p0 else p0_score
        margin = float(a_score - b_score)
        margins.append(margin)
        move_counts.append(move_number)
        if a_is_p0:
            a_games_as_p0 += 1
            if winner == a_player:
                a_wins_as_p0 += 1

        if winner == a_player:
            a_wins += 1
            winner_str = "A"
        else:
            b_wins += 1
            winner_str = "B"

        log.info(
            f"Game {game_idx:03d} | A={'P0' if a_is_p0 else 'P1'} vs B={'P1' if a_is_p0 else 'P0'} "
            f"| winner={winner_str} | score P0={p0_score:.1f} P1={p1_score:.1f} | moves={move_number}"
        )

    total = a_wins + b_wins
    wr = (a_wins / total) if total else 0.0
    avg_margin = sum(margins) / len(margins) if margins else 0.0
    win_margins = [m for m in margins if m > 0]
    loss_margins = [m for m in margins if m < 0]
    avg_win_margin = sum(win_margins) / len(win_margins) if win_margins else 0.0
    avg_loss_margin = sum(loss_margins) / len(loss_margins) if loss_margins else 0.0

    # Win rate standard error (approx ±1.96*SE = 95% CI)
    wr_se = (wr * (1 - wr) / total) ** 0.5 if total else 0.0

    # Margin stats
    margin_std = (sum((m - avg_margin) ** 2 for m in margins) / len(margins)) ** 0.5 if len(margins) > 1 else 0.0
    max_win = max(margins) if margins else 0.0
    max_loss = min(margins) if margins else 0.0

    # Game length
    avg_moves = sum(move_counts) / len(move_counts) if move_counts else 0.0
    min_moves = min(move_counts) if move_counts else 0
    max_moves_actual = max(move_counts) if move_counts else 0

    # Side balance (first-player advantage)
    a_games_as_p1 = total - a_games_as_p0
    wr_p0 = (a_wins_as_p0 / a_games_as_p0 * 100) if a_games_as_p0 else 0.0
    wr_p1 = ((a_wins - a_wins_as_p0) / a_games_as_p1 * 100) if a_games_as_p1 else 0.0

    log.info(f"\nSUMMARY: A wins={a_wins}, B wins={b_wins}, A winrate={wr*100:.1f}% (SE ±{wr_se*100:.1f}%, n={total})")
    log.info(f"  Score margin (A - B): avg={avg_margin:+.1f} ±{margin_std:.1f} pts  |  when A wins: +{avg_win_margin:.1f}  |  when A loses: {avg_loss_margin:.1f}")
    log.info(f"  Largest margin: A +{max_win:.0f}  |  A {max_loss:+.0f}")
    log.info(f"  Game length: avg={avg_moves:.0f} moves  (min={min_moves}, max={max_moves_actual})")
    log.info(f"  A as P0: {a_wins_as_p0}/{a_games_as_p0} ({wr_p0:.0f}%)  |  A as P1: {a_wins - a_wins_as_p0}/{a_games_as_p1} ({wr_p1:.0f}%)")
    log.info(f"  A: {label_a}")
    log.info(f"  B: {label_b}")

    # Calibration: standard Elo implied gap (64.5% WR -> ~104 Elo)
    implied_elo = wr_to_elo_gap(wr)
    log.info(f"  [Calibration] Observed WR {wr*100:.1f}% implies ~{implied_elo:.0f} Elo gap (standard 400-scale)")

    # ELO update
    if elo_file and margins:
        outcomes = [1.0 if m > 0 else 0.0 for m in margins]
        if elo_style == "standard":
            ratings, (new_ra, new_rda), (new_rb, new_rdb) = update_ratings_standard_elo_batch(
                ratings, id_a, id_b, outcomes
            )
            log.info(f"\nELO (post, standard): A {id_a} {new_ra:.0f} ±{new_rda:.0f}  |  B {id_b} {new_rb:.0f} ±{new_rdb:.0f}  |  gap={new_ra - new_rb:.0f}")
        else:
            ratings, (new_ra, new_rda), (new_rb, new_rdb) = update_ratings_from_match(
                ratings, id_a, id_b, outcomes, margins
            )
            log.info(f"\nELO (post, Glicko-2): A {id_a} {new_ra:.0f} ±{new_rda:.0f}  |  B {id_b} {new_rb:.0f} ±{new_rdb:.0f}  |  gap={new_ra - new_rb:.0f}  (note: Glicko-2 gaps often 2–3x standard Elo)"        )
        save_ratings(elo_file, ratings)

    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "total": total,
        "wr": wr,
        "wr_se": wr_se,
        "implied_elo": implied_elo,
    }


def run_champion_vs_field(
    cfg: dict,
    main_path: Path,
    opponent_paths: list[Path],
    games_per_opponent: int,
    sims: int,
    cpuct: float,
    device: torch.device,
    log: logging.Logger,
    elo_file: Path | None = None,
    elo_style: str = "standard",
) -> None:
    """
    Champion vs field: one main model plays each opponent.
    E.g. Iter 24 vs {Iter 1, 6, 12, 18} with 50 games each.
    """
    if games_per_opponent < 1:
        log.error("Games per opponent must be at least 1.")
        return
    if not opponent_paths:
        log.error("Select at least one opponent.")
        return
    for opp in opponent_paths:
        if opp == main_path:
            log.error("Main model cannot be in opponents list.")
            return

    main_id = model_elo_id(main_path)
    log.info(f"Champion vs field: Main={main_path.name} vs {len(opponent_paths)} opponents")
    log.info(f"  {games_per_opponent} games per opponent, {sims} sims")
    for p in opponent_paths:
        log.info(f"  opponent: {p.name}")

    results: list[dict] = []
    for i, opp_path in enumerate(opponent_paths):
        opp_id = model_elo_id(opp_path)
        log.info(f"\n--- Match {i+1}/{len(opponent_paths)}: {main_id} vs {opp_id} ---")
        res = run_eval(
            cfg, main_path, opp_path, games_per_opponent, sims, cpuct, device, log,
            elo_file=elo_file, elo_style=elo_style
        )
        if res:
            res["path_a"] = main_path
            res["path_b"] = opp_path
            res["id_a"] = main_id
            res["id_b"] = opp_id
            results.append(res)

    log.info("\n" + "=" * 60)
    log.info("CHAMPION VS FIELD SUMMARY (Win rate = main model's perspective)")
    log.info("=" * 60)
    for r in results:
        wr = r["wr"]
        se = r["wr_se"]
        ci_95 = 1.96 * se * 100
        elo_gap = r["implied_elo"]
        log.info(
            f"  {r['id_a']} vs {r['id_b']}: "
            f"WR={wr*100:.1f}% ±{ci_95:.1f}% (95% CI)  |  Elo gap ~{elo_gap:.0f}  "
            f"({r['a_wins']}-{r['b_wins']}, n={r['total']})"
        )


def run_gui(
    choices: list[tuple[Path, str]],
    cfg_path: Path,
    default_games: int,
    default_sims: int,
    cpuct: float,
    elo_file: Path | None = None,
    elo_style: str = "standard",
) -> None:
    """Simple Tkinter GUI to pick two models (or round-robin) and run eval."""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except ImportError:
        print("Tkinter not available. Use interactive mode: python tools/eval_latest_vs_oldest.py --no-gui")
        sys.exit(1)

    def do_eval() -> None:
        try:
            games_val = int(games_var.get())
            sims_val = int(sims_var.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Error", "Games and Sims must be integers.")
            return
        if games_val < 1:
            messagebox.showerror("Error", "Games must be at least 1.")
            return
        if sims_val < 1:
            messagebox.showerror("Error", "Sims must be at least 1.")
            return

        if rr_var.get():
            main_sel = list_main.curselection()
            opp_sel = list_opp.curselection()
            if not main_sel:
                messagebox.showerror("Error", "Select the main (champion) model.")
                return
            if not opp_sel:
                messagebox.showerror("Error", "Select at least one opponent.")
                return
            if int(main_sel[0]) in [int(i) for i in opp_sel]:
                messagebox.showerror("Error", "Main model cannot be in opponents list.")
                return
        else:
            i_a = list_a.curselection()
            i_b = list_b.curselection()
            if not i_a or not i_b:
                messagebox.showerror("Error", "Select one model in each column.")
                return
            if int(i_a[0]) == int(i_b[0]):
                messagebox.showerror("Error", "Pick two different models.")
                return

        root.destroy()

        logging.basicConfig(level=logging.INFO, format="%(message)s")
        log = logging.getLogger("eval")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        device = torch.device("cuda" if (cfg["hardware"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu")

        if rr_var.get():
            main_path = choices[int(main_sel[0])][0]
            opponent_paths = [choices[int(i)][0] for i in opp_sel]
            log.info(f"Champion vs field: main={main_path.name}, {len(opponent_paths)} opponents, "
                     f"{games_val} games each, {sims_val} sims\n")
            run_champion_vs_field(
                cfg, main_path, opponent_paths, games_val, sims_val, cpuct, device, log,
                elo_file=elo_file, elo_style=elo_style
            )
        else:
            path_a, _ = choices[int(i_a[0])]
            path_b, _ = choices[int(i_b[0])]
            log.info(f"Model A: {path_a.name}")
            log.info(f"Model B: {path_b.name}")
            log.info(f"Games: {games_val}  Sims: {sims_val}\n")
            run_eval(cfg, path_a, path_b, games_val, sims_val, cpuct, device, log, elo_file=elo_file, elo_style=elo_style)

    def toggle_mode() -> None:
        if rr_var.get():
            frame_h2h.pack_forget()
            frame_rr.pack(fill=tk.BOTH, expand=True)
            games_label.config(text="Games per opponent:")
            games_var.set(50)
        else:
            frame_rr.pack_forget()
            frame_h2h.pack(fill=tk.BOTH, expand=True)
            games_label.config(text="Games:")

    root = tk.Tk()
    root.title("Patchwork Eval - Select Models")
    root.geometry("600x520")

    main = ttk.Frame(root, padding=10)
    main.pack(fill=tk.BOTH, expand=True)

    rr_var = tk.BooleanVar(value=False)

    # Games and Sims row
    opts_frame = ttk.Frame(main)
    opts_frame.pack(fill=tk.X, pady=(0, 5))
    ttk.Checkbutton(opts_frame, text="Champion vs field", variable=rr_var, command=toggle_mode).pack(side=tk.LEFT, padx=(0, 15))
    games_label = ttk.Label(opts_frame, text="Games:")
    games_label.pack(side=tk.LEFT, padx=(0, 5))
    games_var = tk.IntVar(value=default_games)
    games_spin = tk.Spinbox(opts_frame, from_=1, to=10000, width=8, textvariable=games_var)
    games_spin.pack(side=tk.LEFT, padx=(0, 15))
    ttk.Label(opts_frame, text="Sims:").pack(side=tk.LEFT, padx=(0, 5))
    sims_var = tk.IntVar(value=default_sims)
    sims_spin = tk.Spinbox(opts_frame, from_=1, to=2000, width=8, textvariable=sims_var)
    sims_spin.pack(side=tk.LEFT)

    # Head-to-head frame (default)
    frame_h2h = ttk.Frame(main)
    frame_h2h.pack(fill=tk.BOTH, expand=True)
    ttk.Label(frame_h2h, text="Model A (first player in game 0)").pack(anchor=tk.W)
    list_a = tk.Listbox(frame_h2h, height=10, exportselection=False, selectmode=tk.SINGLE)
    list_a.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
    ttk.Label(frame_h2h, text="Model B (second player in game 0)").pack(anchor=tk.W)
    list_b = tk.Listbox(frame_h2h, height=10, exportselection=False, selectmode=tk.SINGLE)
    list_b.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Champion vs field frame (hidden by default)
    frame_rr = ttk.Frame(main)
    row_rr = ttk.Frame(frame_rr)
    row_rr.pack(fill=tk.BOTH, expand=True)
    col_main = ttk.Frame(row_rr)
    col_main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
    ttk.Label(col_main, text="Main (champion)").pack(anchor=tk.W)
    list_main = tk.Listbox(col_main, height=10, exportselection=False, selectmode=tk.SINGLE)
    list_main.pack(fill=tk.BOTH, expand=True)
    col_opp = ttk.Frame(row_rr)
    col_opp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    ttk.Label(col_opp, text="Opponents (Ctrl+click for multi)").pack(anchor=tk.W)
    list_opp = tk.Listbox(col_opp, height=10, exportselection=False, selectmode=tk.EXTENDED)
    list_opp.pack(fill=tk.BOTH, expand=True)
    ttk.Label(frame_rr, text="E.g. Iter 24 vs {1, 6, 12, 18} — games per opponent above.", font=("", 8)).pack(anchor=tk.W)

    for i, (p, label) in enumerate(choices):
        entry = f"{i}: {label}"
        if len(str(p)) < 55:
            entry += f"  ({p})"
        list_a.insert(tk.END, entry)
        list_b.insert(tk.END, entry)
        list_main.insert(tk.END, entry)
        list_opp.insert(tk.END, entry)

    btn_frame = ttk.Frame(main)
    btn_frame.pack(fill=tk.X, pady=10)
    ttk.Button(btn_frame, text="Run Eval", command=do_eval).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Cancel", command=root.destroy).pack(side=tk.LEFT)

    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Evaluate two Patchwork AlphaZero checkpoints head-to-head.")
    ap.add_argument("--config", default="configs/overnight_strong.yaml", help="Config file")
    ap.add_argument("--games", type=int, default=20, help="Number of games")
    ap.add_argument("--override_sims", type=int, default=0, help="0 = use config eval sims")
    ap.add_argument("--override_cpuct", type=float, default=0.0, help="0 = use config eval cpuct")

    # Model selection
    ap.add_argument("--model_a", help="Model A: index (0,1,2...) or path to .pt file")
    ap.add_argument("--model_b", help="Model B: index or path to .pt file")
    ap.add_argument("--champion-vs-field", action="store_true", help="Champion vs field: main model vs opponents")
    ap.add_argument("--main", help="Main model index for champion-vs-field (e.g. 24)")
    ap.add_argument("--opponents", help="Comma-separated opponent indices (e.g. 1,6,12,18)")
    ap.add_argument("--games-per-opponent", type=int, default=50, help="Games per opponent (champion-vs-field)")
    ap.add_argument("--no-gui", action="store_true", help="Use interactive CLI instead of GUI (when not specifying models)")

    # ELO
    ap.add_argument("--elo", action="store_true", default=True, help="Update ELO ratings (default)")
    ap.add_argument("--no-elo", action="store_false", dest="elo", help="Disable ELO tracking")
    ap.add_argument("--elo-file", type=str, default="elo_ratings.json", help="Path to ELO ratings JSON")
    ap.add_argument("--elo-style", choices=["glicko2", "standard"], default="standard",
                    help="glicko2: Glicko-2 (gaps ~2–3x standard Elo); standard: 64.5%% WR -> ~104 Elo (default)")
    ap.add_argument("--show-elo", action="store_true", help="Print ELO leaderboard and exit")

    # Legacy: latest vs oldest in window
    ap.add_argument("--ckpt_dir", help="(Legacy) Only search this dir for checkpoints")
    ap.add_argument("--window", type=int, default=0, help="(Legacy) Use last N checkpoints; 0=disable")

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("eval")

    if args.show_elo:
        elo_path = REPO_ROOT / args.elo_file
        ratings = load_ratings(elo_path)
        if not ratings:
            log.info("No ELO ratings yet. Run an eval first.")
        else:
            sorted_ids = sorted(ratings.keys(), key=lambda k: ratings[k]["rating"], reverse=True)
            log.info("ELO Leaderboard (Glicko-2):")
            for i, mid in enumerate(sorted_ids, 1):
                r = ratings[mid]
                log.info(f"  {i:2d}. {mid:20s}  {r['rating']:6.0f} ±{r['rd']:.0f}  ({r.get('games', 0)} games)")
        return

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg["evaluation"]["eval_mcts"]
    sims = int(args.override_sims) if args.override_sims > 0 else int(eval_cfg["simulations"])
    cpuct = float(args.override_cpuct) if args.override_cpuct > 0 else float(eval_cfg["cpuct"])
    device = torch.device("cuda" if (cfg["hardware"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu")
    elo_file = (REPO_ROOT / args.elo_file) if args.elo else None

    # Resolve model paths
    path_a: Path | None = None
    path_b: Path | None = None

    # Legacy: --ckpt_dir + --window
    if args.ckpt_dir and args.window > 0:
        ckpt_dir = Path(args.ckpt_dir)
        if not ckpt_dir.is_absolute():
            ckpt_dir = REPO_ROOT / ckpt_dir
        # Old-style discovery in single dir
        seen: dict[int, Path] = {}
        for p in list(ckpt_dir.glob("iteration_*.pt")) + list(ckpt_dir.glob("best_model_iter*.pt")):
            if p.is_file():
                n = iter_num(p)
                if n is not None and n not in seen:
                    seen[n] = p
        iters = sorted(seen.values(), key=iter_num)
        if len(iters) < 2:
            raise SystemExit(f"Need at least 2 checkpoints in {ckpt_dir}")
        window = iters[-args.window:] if len(iters) >= args.window else iters
        path_a = window[0]
        path_b = iters[-1]
        log.info(f"Legacy mode: {path_a.name} vs {path_b.name}\n")
    else:
        choices = discover_checkpoints(REPO_ROOT)
        if len(choices) < 2:
            raise SystemExit("Need at least 2 checkpoints. Found none in checkpoints/ or runs/.")

        # Champion vs field CLI: --champion-vs-field --main 24 --opponents 1,6,12,18
        if getattr(args, "champion_vs_field", False) and args.main is not None and args.opponents:
            try:
                main_idx = int(args.main)
            except ValueError:
                raise SystemExit("--main must be an integer index")
            opp_indices = [int(x.strip()) for x in args.opponents.split(",") if x.strip()]
            if main_idx < 0 or main_idx >= len(choices):
                raise SystemExit(f"Main index {main_idx} out of range (0-{len(choices)-1})")
            if not opp_indices:
                raise SystemExit("Specify at least one opponent: --opponents 1,6,12,18")
            main_path = choices[main_idx][0]
            opponent_paths = []
            for idx in opp_indices:
                if 0 <= idx < len(choices):
                    if idx == main_idx:
                        raise SystemExit("Main model cannot be in opponents list")
                    opponent_paths.append(choices[idx][0])
                else:
                    raise SystemExit(f"Opponent index {idx} out of range (0-{len(choices)-1})")
            games_per = getattr(args, "games_per_opponent", 50)
            log.info(f"Champion vs field: main={main_path.name}, {len(opponent_paths)} opponents, "
                     f"{games_per} games each, {sims} sims\n")
            run_champion_vs_field(cfg, main_path, opponent_paths, games_per, sims, cpuct, device, log,
                                 elo_file=elo_file, elo_style=args.elo_style)
            return

        # Default to GUI unless --no-gui or explicit model args
        use_gui = not args.no_gui
        if use_gui and args.model_a is None and args.model_b is None and not getattr(args, "champion_vs_field", False):
            run_gui(choices, cfg_path, args.games, sims, cpuct, elo_file=elo_file, elo_style=args.elo_style)
            return

        if args.model_a is not None and args.model_b is not None:
            # Resolve indices or paths
            def resolve(s: str) -> Path:
                if s.isdigit():
                    idx = int(s)
                    if 0 <= idx < len(choices):
                        return choices[idx][0]
                    raise SystemExit(f"Index {idx} out of range (0-{len(choices)-1})")
                p = Path(s)
                if not p.is_absolute():
                    p = REPO_ROOT / p
                if not p.exists():
                    raise SystemExit(f"File not found: {p}")
                return p

            path_a = resolve(args.model_a)
            path_b = resolve(args.model_b)
        else:
            # Interactive: list and prompt
            log.info("Available checkpoints:")
            for i, (p, label) in enumerate(choices):
                try:
                    rel = p.relative_to(REPO_ROOT)
                except ValueError:
                    rel = p
                log.info(f"  {i:2d}: {label:20s}  {rel}")
            log.info("")

            def prompt(name: str) -> int:
                while True:
                    try:
                        s = input(f"{name} index (0-{len(choices)-1}): ").strip()
                        idx = int(s)
                        if 0 <= idx < len(choices):
                            return idx
                    except ValueError:
                        pass
                    log.info("Invalid. Enter a number in range.")

            idx_a = prompt("Model A")
            idx_b = prompt("Model B")
            if idx_a == idx_b:
                raise SystemExit("Pick two different models.")
            path_a = choices[idx_a][0]
            path_b = choices[idx_b][0]

    if path_a is None or path_b is None:
        raise SystemExit("Could not resolve model paths.")

    log.info(f"Model A: {path_a.name}")
    log.info(f"Model B: {path_b.name}")
    log.info(f"Games: {args.games}\n")

    run_eval(cfg, path_a, path_b, args.games, sims, cpuct, device, log, elo_file=elo_file, elo_style=args.elo_style)


if __name__ == "__main__":
    main()
