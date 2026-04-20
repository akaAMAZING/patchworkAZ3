"""
Standalone ladder Elo eval — run head-to-head between two checkpoints and update ladder_elo.json.
Always writes results to TensorBoard (ladder/ prefix) so backfill shows up cleanly in a
dedicated section separate from the existing eval/ metrics.

Usage examples:

  # Backfill the full ladder from all committed checkpoints (run this now):
  python scripts/run_ladder_eval.py --backfill

  # Run a single match — iter30 vs iter25:
  python scripts/run_ladder_eval.py --iter-a 30 --iter-b 25

  # Inject a known result without re-running games:
  python scripts/run_ladder_eval.py --inject --iter-a 29 --wins 55 --losses 45

  # Just print the current ladder (no games):
  python scripts/run_ladder_eval.py --print-only

  # Run anchor eval manually (e.g. iter30 vs iter192):
  python scripts/run_ladder_eval.py --anchor --iter-a 30 --games 40

The script reads configs/config_fresh_run.yaml by default. Override with --config.
Results go to <run-dir>/ladder_elo.json and logs/tensorboard/.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# ── ANSI palette ─────────────────────────────────────────────────────────────
R  = "\033[0m"          # reset
BD = "\033[1m"          # bold
DM = "\033[2m"          # dim
IT = "\033[3m"          # italic

K  = "\033[90m"         # dark gray
RD = "\033[91m"         # red
GN = "\033[92m"         # green
YL = "\033[93m"         # yellow
BL = "\033[94m"         # blue
MG = "\033[95m"         # magenta
CY = "\033[96m"         # cyan
WH = "\033[97m"         # white

BGN = "\033[32m"        # dark green
BRD = "\033[31m"        # dark red

# block chars
FULL  = "█"
SEVEN = "▉"
HALF  = "▌"
SHADE = "░"
TIP   = "▏"


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOR = _supports_color()


def _c(*parts: str) -> str:
    """Join ANSI codes — strips them when color is disabled."""
    if _COLOR:
        return "".join(parts)
    # strip escape sequences
    import re
    text = "".join(parts)
    return re.sub(r"\033\[[0-9;]*m", "", text)


def _bar(filled: float, total: float, width: int = 38) -> str:
    ratio = max(0.0, min(1.0, filled / total)) if total > 0 else 0.0
    filled_w = ratio * width
    full_blocks = int(filled_w)
    frac = filled_w - full_blocks
    empty = width - full_blocks - (1 if frac > 0 else 0)
    bar = FULL * full_blocks
    if frac >= 0.75:
        bar += SEVEN
    elif frac >= 0.5:
        bar += HALF
    elif frac > 0:
        bar += TIP
    bar += SHADE * max(0, empty)
    return bar


def _wr_badge(wr: float) -> str:
    if wr >= 0.70:
        return _c(GN, BD, f" {wr*100:5.1f}% ", R)
    if wr >= 0.58:
        return _c(GN, f" {wr*100:5.1f}% ", R)
    if wr >= 0.50:
        return _c(YL, f" {wr*100:5.1f}% ", R)
    if wr >= 0.42:
        return _c(YL, f" {wr*100:5.1f}% ", R)
    return _c(RD, f" {wr*100:5.1f}% ", R)


def _elo_badge(gap: float) -> str:
    sign = "+" if gap >= 0 else ""
    if gap >= 40:
        return _c(GN, BD, f"{sign}{gap:.1f} Elo", R)
    if gap >= 20:
        return _c(GN, f"{sign}{gap:.1f} Elo", R)
    if gap >= 5:
        return _c(YL, f"{sign}{gap:.1f} Elo", R)
    if gap > -5:
        return _c(K, f"{sign}{gap:.1f} Elo", R)
    return _c(RD, f"{sign}{gap:.1f} Elo", R)


def _verdict(wr: float) -> str:
    if wr >= 0.70: return _c(GN, BD, "▲▲ DOMINANT", R)
    if wr >= 0.58: return _c(GN,     "▲  LEADING",  R)
    if wr >= 0.53: return _c(GN,     "↗  AHEAD",    R)
    if wr >= 0.47: return _c(YL,     "→  EVEN",     R)
    if wr >= 0.40: return _c(YL,     "↘  BEHIND",   R)
    return             _c(RD,        "▼▼ LOSING",   R)


def _separator(char: str = "─", width: int = 68) -> str:
    return _c(K, char * width, R)


def _match_header(iter_a: int, iter_b: int, total_games: int) -> None:
    label = f"  LADDER MATCH  iter{iter_a:03d} vs iter{iter_b:03d}  •  {total_games} games  "
    bar = "━" * (len(label) + 2)
    print()
    print(_c(CY, BD, f"  ┌{bar}┐", R))
    print(_c(CY, BD, f"  │{label}│", R))
    print(_c(CY, BD, f"  └{bar}┘", R))
    print()


def _make_progress_callback(iter_a: int, iter_b: int, total_games: int):
    """Returns a callback that prints a live per-batch update every 10 games."""
    prev = {"games": 0, "wins": 0}
    start_ts = time.time()

    def callback(games: int, wins: int, wr: float) -> None:
        losses = games - wins
        batch_games = games - prev["games"]
        batch_wins  = wins  - prev["wins"]
        batch_losses = batch_games - batch_wins
        batch_wr = batch_wins / batch_games if batch_games > 0 else 0.5
        prev["games"] = games
        prev["wins"]  = wins

        pct = games / total_games
        elapsed = time.time() - start_ts
        eta_s = (elapsed / pct * (1 - pct)) if pct > 0 else 0
        eta = f"{eta_s/60:.0f}m{eta_s%60:.0f}s" if eta_s >= 60 else f"{eta_s:.0f}s"

        # progress bar row
        bar_color = GN if wr >= 0.55 else (YL if wr >= 0.45 else RD)
        prog_bar = _c(bar_color, _bar(games, total_games, 38), R)
        pct_str  = _c(WH, f"{pct*100:4.0f}%", R)
        game_str = _c(K, f"Game {games:>3}/{total_games}", R)

        # batch pill
        bw_str = _c(GN, f"W{batch_wins}", R)
        bl_str = _c(RD, f"L{batch_losses}", R)
        bwr_cl = GN if batch_wr >= 0.55 else (YL if batch_wr >= 0.45 else RD)
        bwr_str = _c(bwr_cl, f"{batch_wr*100:.0f}%", R)
        batch_pill = f"batch {games-batch_games+1}-{games}: {bw_str} {bl_str} {bwr_str}"

        # running stats
        cur_elo = _compute_implied_elo(wr)
        elo_str = _elo_badge(cur_elo)
        vrd     = _verdict(wr)

        print(f"  {game_str}  [{prog_bar}] {pct_str}  ETA {eta}")
        print(f"  {batch_pill}  •  total W{wins} L{losses}  WR{_wr_badge(wr)}  {elo_str}  {vrd}")
        print()

    return callback


# ── config / io helpers ───────────────────────────────────────────────────────

def _load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _tb_log_dir(config: dict, run_root: Path) -> Path:
    tb_cfg = (config.get("logging") or {}).get("tensorboard") or {}
    log_dir = tb_cfg.get("log_dir") or str(Path(config["paths"]["logs_dir"]) / "tensorboard")
    p = Path(log_dir)
    return p if p.is_absolute() else run_root.parents[1] / p


def _write_ladder_to_tb(state: dict, tb_log_dir: Path) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        for e in state["entries"]:
            step = e["iter"]
            writer.add_scalar("ladder/cumulative_elo", e["cumulative_elo"], step)
            writer.add_scalar("ladder/step_gap_elo", e["implied_gap"], step)
            writer.add_scalar("ladder/win_rate", e["win_rate"], step)
        writer.close()
        print(_c(K, f"  TensorBoard: wrote {len(state['entries'])} ladder entries -> {tb_log_dir}", R))
    except Exception as ex:
        print(_c(YL, f"  [WARN] TensorBoard write failed: {ex}", R))


def _checkpoint_path(run_root: Path, iteration: int) -> Path:
    return run_root / "committed" / f"iter_{iteration:03d}" / f"iteration_{iteration:03d}.pt"


def _committed_iters(run_root: Path):
    committed = run_root / "committed"
    if not committed.exists():
        return []
    iters = []
    for item in committed.iterdir():
        if not item.is_dir() or not item.name.startswith("iter_"):
            continue
        try:
            n = int(item.name.split("_")[1])
            if (item / f"iteration_{n:03d}.pt").exists():
                iters.append(n)
        except (ValueError, IndexError):
            continue
    return sorted(iters)


def _compute_implied_elo(win_rate: float) -> float:
    wr = max(0.01, min(0.99, float(win_rate)))
    return 400.0 * math.log10(wr / (1.0 - wr))


def _load_ladder(path: Path, lookback: int, base_elo: float) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(_c(YL, f"[WARN] Could not load {path}: {e} — starting fresh", R))
    return {"lookback": lookback, "base_elo": base_elo, "entries": []}


def _save_ladder(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _has_entry(state: dict, iter_n: int) -> bool:
    return any(e["iter"] == iter_n for e in state["entries"])


def _add_result(state: dict, iter_n: int, wins: int, losses: int) -> dict:
    total = wins + losses
    win_rate = wins / total if total > 0 else 0.5
    implied_gap = _compute_implied_elo(win_rate)
    prev_elo = state["entries"][-1]["cumulative_elo"] if state["entries"] else state["base_elo"]
    entry = {
        "iter": int(iter_n),
        "vs_iter": int(iter_n) - state["lookback"],
        "wins": int(wins),
        "losses": int(losses),
        "total": int(total),
        "win_rate": round(float(win_rate), 4),
        "implied_gap": round(float(implied_gap), 1),
        "cumulative_elo": round(float(prev_elo + implied_gap), 1),
    }
    state["entries"].append(entry)
    return entry


# ── pretty table ─────────────────────────────────────────────────────────────

def _print_table(state: dict) -> None:
    entries = state["entries"]
    if not entries:
        print(_c(K, "\n  No ladder entries yet.\n", R))
        return

    # compute bar scale
    elos = [e["cumulative_elo"] for e in entries]
    elo_min, elo_max = min(elos), max(elos)
    elo_range = max(elo_max - elo_min, 1.0)
    BAR_W = 24

    # header
    print()
    print(_c(CY, BD, "  ╔══ LADDER PROGRESSION ", "═" * 46, "╗", R))
    n_entries = len(entries)
    sub = f"  {n_entries} checkpoints  •  base Elo {state['base_elo']:.0f}  •  lookback {state['lookback']}"
    print(_c(CY, BD, f"  ║{sub:<69}║", R))
    print(_c(CY, BD, "  ╠══════════════════════════════════════════════════════════════════════╣", R))

    # column headers
    hdr = (
        f"  ║  {'ITER':>4}  {'vs':>4}  {'W':>4}  {'L':>4}  {'WR%':>5}  "
        f"{'Gap':>7}  {'Cumul':>6}  {'Progress':<26}  {'Rating':<12}  ║"
    )
    print(_c(BD, WH, hdr, R))
    print(_c(CY, "  ╠══════════════════════════════════════════════════════════════════════╣", R))

    prev_elo = state["base_elo"]
    for i, e in enumerate(entries):
        gap      = e["implied_gap"]
        wr       = e["win_rate"]
        cum      = e["cumulative_elo"]
        delta    = cum - prev_elo
        prev_elo = cum

        # mini Elo bar — proportional to cumulative position
        filled = round((cum - elo_min) / elo_range * BAR_W)
        filled = max(1, min(BAR_W, filled))
        if gap >= 30:
            bar_col = GN
        elif gap >= 10:
            bar_col = YL
        else:
            bar_col = K
        mini_bar = _c(bar_col, FULL * filled, K, SHADE * (BAR_W - filled), R)

        # rating label
        if gap < 5:
            rating_label = _c(K,  DM, "plateau",      R)
        elif gap < 15:
            rating_label = _c(YL,     "slow gain",    R)
        elif gap < 30:
            rating_label = _c(YL,     "steady",       R)
        elif gap < 50:
            rating_label = _c(GN,     "healthy",      R)
        else:
            rating_label = _c(GN, BD, "strong leap",  R)

        # trend arrow (vs previous entry)
        if i == 0:
            arrow = _c(K, "  ─", R)
        elif gap > entries[i-1]["implied_gap"] + 5:
            arrow = _c(GN, " ↑↑", R)
        elif gap > entries[i-1]["implied_gap"] - 5:
            arrow = _c(YL, "  →", R)
        else:
            arrow = _c(RD, " ↓↓", R)

        gap_str = _elo_badge(gap)
        cum_str = _c(WH, f"{cum:6.0f}", R)
        wr_str  = _wr_badge(wr)

        # strip ansi for length accounting, pad manually
        row = (
            f"  ║  {e['iter']:>4}  {e['vs_iter']:>4}  {e['wins']:>4}  {e['losses']:>4}  "
            f"{wr_str}  {gap_str}  {cum_str}  {mini_bar}{arrow}  {rating_label}"
        )
        print(row)

    print(_c(CY, "  ╠══════════════════════════════════════════════════════════════════════╣", R))

    # trend summary
    last3_gaps = [e["implied_gap"] for e in entries[-3:]]
    avg3 = sum(last3_gaps) / len(last3_gaps)
    if all(g >= 20 for g in last3_gaps):
        trend_msg = _c(GN, BD, "IMPROVING  — keep training, gaps are healthy", R)
    elif all(g >= 10 for g in last3_gaps):
        trend_msg = _c(GN, "STEADY     — model still making gains", R)
    elif avg3 < 10:
        trend_msg = _c(YL, "SLOWING    — consider LR drop or longer steps", R)
    else:
        trend_msg = _c(YL, "MIXED      — watch next 2 checkpoints", R)

    last_gap  = entries[-1]["implied_gap"]
    last_cum  = entries[-1]["cumulative_elo"]
    gain_tot  = last_cum - state["base_elo"]
    print(f"  ║  Recent trend ({len(last3_gaps)} steps): {trend_msg}")
    print(f"  ║  Last gap: {_elo_badge(last_gap)}  •  Total Elo gained from base: {_elo_badge(gain_tot)}")
    print(_c(CY, BD, "  ╚══════════════════════════════════════════════════════════════════════╝", R))

    # decision rule reminder
    print(_c(K, IT, "  ↳ plateau rule: gap < 15 Elo for 2 consecutive steps → LR has done its work", R))
    print()


# ── eval runner ──────────────────────────────────────────────────────────────

def _run_eval(config: dict, iter_a: int, iter_b: int,
              model_path: str, ref_path: str, games: int) -> tuple:
    """Run games between model_path (A) and ref_path (B). Returns (wins_A, losses_A, stats)."""
    import torch
    from src.training.evaluation import Evaluator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_workers = max(1, int((config.get("evaluation") or {}).get("eval_parallel_games", 1)))
    dev_label = _c(CY, str(device), R)
    print(f"  Device: {dev_label}  •  {games} max games  •  {n_workers} worker{'s' if n_workers > 1 else ''}")
    print()

    # Read SPRT config from ladder_eval.sprt (if enabled)
    _ladder_cfg  = (config.get("evaluation") or {}).get("ladder_eval") or {}
    _sprt_block  = _ladder_cfg.get("sprt") or {}
    sprt_cfg     = _sprt_block if _sprt_block.get("enabled", False) else None

    if sprt_cfg:
        p1  = float(sprt_cfg.get("p1",  0.55))
        mn  = int(sprt_cfg.get("min_games", 60))
        print(
            _c(K, f"  SPRT enabled  H0≤50%  H1≥{p1*100:.0f}%  "
               f"α={float(sprt_cfg.get('alpha', 0.05))*100:.0f}%  "
               f"β={float(sprt_cfg.get('beta',  0.05))*100:.0f}%  "
               f"min={mn} games  max={games} games", R)
        )
        print()

    evaluator = Evaluator(config, device)
    callback  = _make_progress_callback(iter_a, iter_b, games)

    results = evaluator.evaluate_vs_baseline(
        model_path=model_path,
        baseline_type="previous_best",
        baseline_path=ref_path,
        num_games=games,
        progress_callback=callback,
        progress_interval=10,
        sprt_cfg=sprt_cfg,
    )

    wins   = results["model_wins"]
    losses = results["baseline_wins"]
    wr     = results["win_rate"]
    margin = results.get("avg_model_score_margin", 0)
    elo    = _compute_implied_elo(wr)
    n_played = results.get("total_games", wins + losses)

    # SPRT summary line (only if SPRT was active)
    sprt_info = results.get("sprt")
    if sprt_info:
        decision = sprt_info["decision"]
        saved    = games - n_played
        dec_col  = GN if decision == "accept" else (RD if decision == "reject" else YL)
        dec_str  = _c(dec_col, BD, decision.upper(), R)
        saved_str = _c(GN, f"saved {saved} games", R) if saved > 0 else _c(K, "ran to max", R)
        print(f"  SPRT {dec_str}  •  {n_played}/{games} games played  •  {saved_str}")
        print()

    print(_separator())
    print(
        f"  FINAL  W:{_c(GN, BD, str(wins), R)}  L:{_c(RD, str(losses), R)}"
        f"  ({n_played} games)"
        f"  WR:{_wr_badge(wr)}  {_elo_badge(elo)}  margin {margin:+.1f}pts  {_verdict(wr)}"
    )
    print(_separator())
    print()
    return wins, losses, results


def _run_anchor_eval(config: dict, anchor_cfg: dict, model_path: str, games: int) -> None:
    """Run games vs the fixed anchor and print results. Does not update ladder."""
    import torch
    import yaml

    anchor_section = (config.get("evaluation") or {}).get("anchor_checkpoint") or {}
    anchor_path = anchor_section.get("path")
    anchor_config_path = anchor_section.get("config")

    if not anchor_path or not anchor_config_path:
        print(_c(RD, "[ERROR] No anchor_checkpoint configured in evaluation section.", R))
        return

    acfg_path = Path(anchor_config_path)
    if not acfg_path.is_absolute():
        acfg_path = _ROOT / acfg_path
    if not acfg_path.exists():
        print(_c(RD, f"[ERROR] Anchor config not found: {acfg_path}", R))
        return
    with open(acfg_path, "r", encoding="utf-8") as f:
        anchor_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.training.evaluation import Evaluator
    evaluator = Evaluator(config, device, anchor_config=anchor_config, anchor_path=anchor_path)

    if evaluator._anchor_mcts is None:
        print(_c(RD, "[ERROR] Anchor model failed to load.", R))
        return

    print(f"  Running {games} games vs anchor ({Path(anchor_path).name})...")
    results = evaluator.evaluate_vs_baseline(model_path, baseline_type="anchor", num_games=games)
    wins   = results["model_wins"]
    losses = results["baseline_wins"]
    wr     = results["win_rate"]
    margin = results.get("avg_model_score_margin", 0)
    implied = _compute_implied_elo(wr)
    print(f"  vs anchor: {wins}W / {losses}L  WR={wr*100:.1f}%  margin={margin:+.1f}pts")
    print(f"  Implied Elo gap vs anchor: {_elo_badge(implied)}  (400-scale)")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ladder Elo eval for AlphaZero training")
    parser.add_argument("--config", default="configs/config_fresh_run.yaml")
    parser.add_argument("--run-dir", default=None, help="Run root dir (default: from config)")
    parser.add_argument("--iter-a", type=int, default=None, help="Candidate iteration (newer)")
    parser.add_argument("--iter-b", type=int, default=None, help="Reference iteration (older, default=iter-a minus lookback)")
    parser.add_argument("--games", type=int, default=None, help="Number of games (default: from config)")
    parser.add_argument("--backfill", action="store_true", help="Populate full ladder from all committed checkpoints")
    parser.add_argument("--inject", action="store_true", help="Inject known result without running games")
    parser.add_argument("--wins", type=int, default=None, help="Known wins (for --inject)")
    parser.add_argument("--losses", type=int, default=None, help="Known losses (for --inject)")
    parser.add_argument("--lookback", type=int, default=None, help="Ladder lookback (default: from config)")
    parser.add_argument("--print-only", action="store_true", help="Print current ladder and exit")
    parser.add_argument("--anchor", action="store_true", help="Run anchor eval for --iter-a (manual, does not update ladder)")
    parser.add_argument("--no-tb", action="store_true", help="Skip TensorBoard write")
    args = parser.parse_args()

    config = _load_config(args.config)

    if args.run_dir:
        run_root = Path(args.run_dir)
    else:
        paths = config.get("paths", {})
        run_root = Path(paths.get("run_root", "runs")) / paths.get("run_id", "patchwork_fresh_v2")

    tb_dir = _tb_log_dir(config, run_root)
    ladder_path = run_root / "ladder_elo.json"
    base_elo = float(config["evaluation"]["elo"]["initial_rating"])

    _ladder_cfg = (config.get("evaluation") or {}).get("ladder_eval") or {}
    lookback      = args.lookback or int(_ladder_cfg.get("lookback", 5))
    default_games = args.games or int(_ladder_cfg.get("games", 200))

    state = _load_ladder(ladder_path, lookback, base_elo)

    # ── anchor eval ──────────────────────────────────────────────────────────
    if args.anchor:
        if args.iter_a is None:
            print("--anchor requires --iter-a")
            sys.exit(1)
        model_p = _checkpoint_path(run_root, args.iter_a)
        if not model_p.exists():
            print(_c(RD, f"Checkpoint not found: {model_p}", R))
            sys.exit(1)
        games = args.games or 40
        print(f"\nAnchor eval: iter{args.iter_a:03d} ({games} games)")
        _run_anchor_eval(config, {}, str(model_p), games)
        return

    # ── print only ───────────────────────────────────────────────────────────
    if args.print_only:
        _print_table(state)
        return

    # ── inject known result ──────────────────────────────────────────────────
    if args.inject:
        if args.iter_a is None or args.wins is None or args.losses is None:
            print("--inject requires --iter-a, --wins, --losses")
            sys.exit(1)
        iter_a = args.iter_a
        if _has_entry(state, iter_a):
            print(_c(YL, f"  Entry for iter{iter_a:03d} already exists. Skipping.", R))
        else:
            entry = _add_result(state, iter_a, args.wins, args.losses)
            _save_ladder(ladder_path, state)
            print(
                f"  Injected iter{iter_a:03d}:"
                f"  gap={_elo_badge(entry['implied_gap'])}  cumulative={entry['cumulative_elo']:.0f}"
            )
            if not args.no_tb:
                _write_ladder_to_tb(state, tb_dir)
        _print_table(state)
        return

    # ── backfill ─────────────────────────────────────────────────────────────
    if args.backfill:
        committed  = _committed_iters(run_root)
        candidates = [n for n in committed if n >= lookback and n % lookback == 0]
        todo       = [n for n in sorted(candidates) if not _has_entry(state, n)]
        already    = len(candidates) - len(todo)

        print()
        print(_c(CY, BD, "  BACKFILL MODE", R))
        print(_c(K,  f"  {len(committed)} committed checkpoints  •  {len(candidates)} ladder candidates  •  {already} already done  •  {len(todo)} to run", R))
        if not todo:
            print(_c(GN, "  Nothing new to run — ladder is up to date.", R))
            _print_table(state)
            return

        ran_any = False
        for idx, iter_n in enumerate(todo, 1):
            ref_iter = iter_n - lookback
            model_p  = _checkpoint_path(run_root, iter_n)
            ref_p    = _checkpoint_path(run_root, ref_iter)

            if not model_p.exists():
                print(_c(YL, f"  iter{iter_n:03d}: checkpoint missing, skipping", R))
                continue
            if not ref_p.exists():
                print(_c(YL, f"  iter{iter_n:03d}: ref iter{ref_iter:03d} checkpoint missing, skipping", R))
                continue

            queue_label = _c(K, f"({idx}/{len(todo)})", R)
            print(_c(K, f"  {queue_label} queued: iter{iter_n:03d} vs iter{ref_iter:03d}", R))

        print()

        for idx, iter_n in enumerate(todo, 1):
            ref_iter = iter_n - lookback
            model_p  = _checkpoint_path(run_root, iter_n)
            ref_p    = _checkpoint_path(run_root, ref_iter)

            if not model_p.exists() or not ref_p.exists():
                continue

            _match_header(iter_n, ref_iter, default_games)
            wins, losses, _ = _run_eval(config, iter_n, ref_iter, str(model_p), str(ref_p), default_games)
            entry = _add_result(state, iter_n, wins, losses)
            _save_ladder(ladder_path, state)

            # mini confirmation
            cum_str = _c(WH, BD, f"{entry['cumulative_elo']:.0f}", R)
            print(
                f"  {_c(GN, BD, 'SAVED', R)}  iter{iter_n:03d}"
                f"  gap={_elo_badge(entry['implied_gap'])}"
                f"  cumulative={cum_str}  {_c(K, f'({idx}/{len(todo)} done)', R)}"
            )
            print()
            ran_any = True

        if ran_any and not args.no_tb:
            _write_ladder_to_tb(state, tb_dir)

        _print_table(state)
        return

    # ── single match ─────────────────────────────────────────────────────────
    if args.iter_a is None:
        print("Specify --iter-a (and optionally --iter-b; default = iter-a minus lookback)")
        sys.exit(1)

    iter_a = args.iter_a
    iter_b = args.iter_b if args.iter_b is not None else (iter_a - lookback)
    model_p = _checkpoint_path(run_root, iter_a)
    ref_p   = _checkpoint_path(run_root, iter_b)

    if not model_p.exists():
        print(_c(RD, f"Checkpoint not found: {model_p}", R))
        sys.exit(1)
    if not ref_p.exists():
        print(_c(RD, f"Ref checkpoint not found: {ref_p}", R))
        sys.exit(1)

    if _has_entry(state, iter_a):
        print(_c(YL, f"  Entry for iter{iter_a:03d} already in ladder. Use --inject to override.", R))
        _print_table(state)
        return

    _match_header(iter_a, iter_b, default_games)
    wins, losses, _ = _run_eval(config, iter_a, iter_b, str(model_p), str(ref_p), default_games)
    entry = _add_result(state, iter_a, wins, losses)
    _save_ladder(ladder_path, state)
    if not args.no_tb:
        _write_ladder_to_tb(state, tb_dir)
    _print_table(state)


if __name__ == "__main__":
    main()
