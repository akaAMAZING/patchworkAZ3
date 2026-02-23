"""
Replay buffer occupancy analysis: Does window_iterations_schedule matter?

Answers: When games_schedule drops (e.g. 400 -> 320 at iter 80), does the buffer
actually fall below max_size=500k due to window=8? Shows buffer occupancy
over time and per-iteration sample counts after subsampling.

Usage:
  python tools/analyze_replay_buffer_occupancy.py --config configs/config_best.yaml
  python tools/analyze_replay_buffer_occupancy.py --config configs/config_best.yaml --positions_per_game 150 200
"""

import argparse
from pathlib import Path

import yaml


def step_lookup(schedule: list, iteration: int, key: str, base: float) -> float:
    """Last entry with entry.iteration <= iteration."""
    if not schedule:
        return base
    for e in sorted(schedule, key=lambda x: x["iteration"], reverse=True):
        if iteration >= e["iteration"]:
            return float(e.get(key, base))
    return base


def run_analysis(config: dict, positions_per_game: float = 170.0) -> None:
    """Simulate buffer occupancy over iterations and print tables."""
    max_size = int(config.get("replay_buffer", {}).get("max_size", 500_000))
    games_schedule = config.get("iteration", {}).get("games_schedule", [])
    win_schedule = config.get("iteration", {}).get("window_iterations_schedule", [])
    base_games = config.get("selfplay", {}).get("games_per_iteration", 400)
    base_window = int(config.get("replay_buffer", {}).get("window_iterations", 8))

    # Per-iteration positions (simulate forward in time)
    # iter 0: bootstrap games (200) -> fewer positions; then full games
    bootstrap_games = config.get("selfplay", {}).get("bootstrap", {}).get("games", 200)
    pos_iter = []  # positions added at each iteration

    for it in range(450):  # 0..449
        games = int(step_lookup(games_schedule, it, "games", float(base_games)))
        if it == 0:
            games = bootstrap_games  # First iter uses bootstrap
        pos_iter.append(games * positions_per_game)

    # Now simulate buffer state at each iteration (after add_iteration, before merge)
    print("=" * 100)
    print("REPLAY BUFFER OCCUPANCY ANALYSIS")
    print("=" * 100)
    print(f"positions_per_game: {positions_per_game:.0f}  (from logs: ~170)")
    print(f"max_size (cap):    {max_size:,}")
    print()

    # Table 1: Buffer occupancy over time
    print("Table 1 — Buffer occupancy over time")
    print("-" * 100)
    print(f"{'iter':>5} {'games':>6} {'window':>6} {'raw_pos':>10} {'after_cap':>10} {'below_500k?':>12} {'iters_in_buf':>12}")
    print("-" * 100)

    entries_by_iter = []  # list of (iter, positions) for last N iters
    below_500k_count = 0

    for it in range(450):
        games = int(step_lookup(games_schedule, it, "games", float(base_games)))
        if it == 0:
            games = bootstrap_games
        window = int(step_lookup(win_schedule, it, "window_iterations", float(base_window)))
        positions_this_iter = games * positions_per_game

        # Simulate: we now have entries for iters [max(0, it-window+1), ..., it]
        # But actually replay buffer keeps last WINDOW iterations. When we add iter N,
        # we have iters [N-window+1 .. N] (after eviction). So we sum positions from
        # those iterations. Positions from iter i were added when we committed iter i.
        start = max(0, it - window + 1)
        raw = sum(pos_iter[i] for i in range(start, it + 1))
        after_cap = min(raw, max_size)
        below = raw < max_size
        if below:
            below_500k_count += 1
        n_iters = min(window, it + 1)
        print(f"{it:5d} {games:6d} {window:6d} {raw:10,.0f} {after_cap:10,.0f} {'YES' if below else 'no':>12} {n_iters:12d}")

        if it in (0, 7, 8, 59, 60, 61, 79, 80, 81, 199, 200, 201, 399, 400, 401):
            # Highlight key breakpoints
            pass  # already printing all

    print("-" * 100)
    print(f"Iterations where raw buffer < 500k: {below_500k_count} / 450")
    print()

    # Table 2: Per-iteration sample counts after subsampling (at key breakpoints)
    print("Table 2 — Per-iteration sample counts after subsampling (newest_fraction=0.25)")
    print("  (At breakpoints: how many samples come from each iteration in the window)")
    print("-" * 100)
    newest_frac = float(config.get("replay_buffer", {}).get("newest_fraction", 0.25))

    for breakpoint_iter in [7, 8, 60, 80, 200, 300, 400]:
        if breakpoint_iter >= 450:
            continue
        window = int(step_lookup(win_schedule, breakpoint_iter, "window_iterations", float(base_window)))
        start = max(0, breakpoint_iter - window + 1)
        iters_in_window = list(range(start, breakpoint_iter + 1))
        raw_total = sum(pos_iter[i] for i in iters_in_window)
        target = min(raw_total, max_size)

        # Proportional allocation (simplified: we use recency with newest_fraction)
        # Newest iteration gets newest_frac of target; rest split proportionally
        newest_pos = pos_iter[breakpoint_iter]
        rest_positions = [(i, pos_iter[i]) for i in iters_in_window[:-1]]
        rest_total = sum(p for _, p in rest_positions)

        if raw_total <= max_size:
            takes = {i: pos_iter[i] for i in iters_in_window}
        else:
            newest_take = min(int(target * newest_frac), int(newest_pos))
            rest_take = target - newest_take
            takes = {breakpoint_iter: newest_take}
            if rest_total > 0 and rest_take > 0:
                for i, p in rest_positions:
                    takes[i] = int(p * rest_take / rest_total)
                # Adjust for rounding
                takes[breakpoint_iter] = target - sum(takes.get(ii, 0) for ii in iters_in_window if ii != breakpoint_iter)
            else:
                takes[breakpoint_iter] = target

        print(f"\n  iter {breakpoint_iter} (window={window}, raw={raw_total:,.0f}, target={target:,}):")
        for i in sorted(takes.keys()):
            t = takes.get(i, 0)
            pct = 100 * t / target if target else 0
            print(f"    iter {i:3d}: {t:8,.0f} samples ({pct:5.1f}%)  [positions in buffer: {pos_iter[i]:,.0f}]")

    # Table 3: With vs without schedule
    print("\nTable 3 — WITH vs WITHOUT window_iterations_schedule (fixed window=8)")
    print("-" * 100)
    print(f"{'iter':>5} {'games':>6} {'w/ sched':>10} {'no sched':>10} {'diff':>10}")
    print("-" * 100)
    for it in [7, 60, 80, 88, 200, 250, 300, 400]:
        if it >= 450:
            continue
        games = int(step_lookup(games_schedule, it, "games", float(base_games)))
        if it == 0:
            games = bootstrap_games
        win_sched = int(step_lookup(win_schedule, it, "window_iterations", float(base_window)))
        start_sched = max(0, it - win_sched + 1)
        raw_sched = sum(pos_iter[i] for i in range(start_sched, it + 1))
        # Fixed window=8
        start_fixed = max(0, it - 8 + 1)
        raw_fixed = sum(pos_iter[i] for i in range(start_fixed, it + 1))
        diff = raw_sched - raw_fixed
        print(f"{it:5d} {games:6d} {raw_sched:10,.0f} {raw_fixed:10,.0f} {diff:+10,.0f}")

    print()
    print("=" * 100)
    print("CONCLUSION: Is window_iterations_schedule necessary?")
    print("=" * 100)
    first_games_drop = next((e["iteration"] for e in sorted(games_schedule, key=lambda x: x["iteration"]) if e.get("games", 400) < 400), 999)
    if first_games_drop < 450:
        games_at_drop = int(step_lookup(games_schedule, first_games_drop, "games", 400))
        raw_with_window8 = 8 * games_at_drop * positions_per_game
        raw_with_window10 = 10 * games_at_drop * positions_per_game
        print(f"When games drop to {games_at_drop} at iter {first_games_drop}:")
        print(f"  - Fixed window=8:  raw = 8 x {games_at_drop} x {positions_per_game:.0f} = {raw_with_window8:,.0f}  {'< 500k - BUFFER BELOW CAP' if raw_with_window8 < max_size else '(ok)'}")
        print(f"  - Schedule w=10:   raw = 10 x {games_at_drop} x {positions_per_game:.0f} = {raw_with_window10:,.0f}  (subsample to 500k)")
    print()
    if below_500k_count > 0:
        print(f"YES - The buffer falls below 500k in {below_500k_count} iterations. The schedule increases")
        print("     the window when games decrease, keeping the buffer nearer 500k for richer training.")
    else:
        print("The buffer stays at or above 500k throughout; window schedule has minor effect.")
    print()


def main():
    ap = argparse.ArgumentParser(description="Analyze replay buffer occupancy over training")
    ap.add_argument("--config", type=str, default="configs/config_best.yaml", help="Config path")
    ap.add_argument("--positions_per_game", type=float, default=170.0,
                    help="Positions per game (from logs: 400 games ~68k pos => 170)")
    args = ap.parse_args()

    path = Path(args.config)
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_analysis(config, args.positions_per_game)


if __name__ == "__main__":
    main()
