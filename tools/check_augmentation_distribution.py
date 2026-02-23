#!/usr/bin/env python3
"""
Sanity: print transform distribution over selfplay games or a shard.
Usage:
  python -m tools.check_augmentation_distribution --games 5   # run 5 games, print tag dist
  python -m tools.check_augmentation_distribution path/to/shard.h5  # if shard has transform_tags
"""
import sys
from pathlib import Path
# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collections import Counter

def main():
    args = sys.argv[1:]
    if "--games" in args:
        idx = args.index("--games")
        n = int(args[idx + 1]) if idx + 1 < len(args) else 5
        from src.training.selfplay_optimized import OptimizedSelfPlayWorker
        cfg = {
            "selfplay": {
                "augmentation": "d4",
                "bootstrap": {"mcts_simulations": 32},
                "mcts": {"simulations": 64, "parallel_leaves": 8},
                "max_game_length": 200,
            },
        }
        worker = OptimizedSelfPlayWorker(network_path=None, config=cfg)
        all_tags = []
        for i in range(n):
            r = worker.play_game(i, 0, 42 + i)
            tags = r.get("stored_flip_types", [])
            all_tags.extend(tags)
        c = Counter(all_tags)
        total = len(all_tags)
        print(f"D4 transform distribution over {n} games ({total} samples):")
        for tag, count in sorted(c.items()):
            pct = 100 * count / total if total else 0
            print(f"  {tag}: {count} ({pct:.1f}%)")
        return

    if not args:
        print(__doc__)
        sys.exit(1)
    path = Path(args[0])
    if not path.exists():
        print(f"Not found: {path}")
        sys.exit(1)

    try:
        import h5py
    except ImportError:
        print("h5py required for shard inspection")
        sys.exit(1)

    with h5py.File(path, "r") as f:
        if "transform_tags" in f:
            tags = f["transform_tags"][:]
            c = Counter(tag.decode() if isinstance(tag, bytes) else str(tag) for tag in tags)
            total = len(tags)
            print(f"Transform distribution ({total} samples):")
            for tag, count in sorted(c.items()):
                pct = 100 * count / total
                print(f"  {tag}: {count} ({pct:.1f}%)")
        else:
            print("No transform_tags in shard. Run with --games N to sample from live selfplay.")

if __name__ == "__main__":
    main()
