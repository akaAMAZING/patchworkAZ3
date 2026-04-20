"""
Replay Buffer for Patchwork AlphaZero

Maintains a sliding window of self-play data across multiple iterations.
Supports merging multiple HDF5 files into a single training dataset and
enforcing a maximum buffer size by dropping the oldest data first.

Transactional design:
- add_iteration() adds to in-memory entries (path may be staging for current iter)
- get_training_data() writes merged HDF5 to specified output_path (staging dir)
- finalize_iteration_for_commit() replaces staging path with committed path, then persists
- restore_state() loads only from committed paths (replay_state.json)
"""

import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Dual-Head: refuse to merge legacy HDF5 (no score_margins) when training score head
_REJECT_LEGACY_IF_SCORE_HEAD = True

# Score margin validation: raw integer margin in points (KataGo Dual-Head)
SCORE_MARGIN_MAX_ABS = 120


def _validate_score_margins(
    score_margins: np.ndarray,
    max_abs: float = SCORE_MARGIN_MAX_ABS,
    source: str = "",
) -> None:
    """Validate score_margins: finite and in range. Raise with clear message on failure."""
    if score_margins.size == 0:
        return
    if not np.all(np.isfinite(score_margins)):
        bad = np.logical_not(np.isfinite(score_margins)).sum()
        raise ValueError(
            f"score_margins validation failed{': ' + source if source else ''}: "
            f"{bad} non-finite value(s). "
            "Wipe runs/<run_id>/staging and committed and regenerate self-play."
        )
    abs_max = float(np.abs(score_margins).max())
    if abs_max > max_abs:
        raise ValueError(
            f"score_margins validation failed{': ' + source if source else ''}: "
            f"max |score_margin| = {abs_max} > {max_abs}. "
            "Wipe runs/<run_id>/staging and committed and regenerate self-play."
        )


def _bounded_hamilton(weights: List[float], caps: List[int], total: int) -> List[int]:
    """
    Allocate ``total`` integers proportionally to ``weights``, capped per entry.

    Uses binary search (water-filling) to find the continuous-quota multiplier
    alpha, then a single Hamilton rounding pass.  No early rounding or iterative
    redistribute loop — the result is always exactly ``total`` (summed).

    Args:
        weights: Non-negative float weight for each entry.
        caps:    Maximum integers that can be taken from each entry.
        total:   Target total to allocate (clamped to [0, sum(caps)]).

    Returns:
        List of integer takes, one per entry, summing to ``total``.
    """
    n = len(weights)
    assert len(caps) == n
    total = max(0, min(total, sum(caps)))
    if n == 0 or total == 0:
        return [0] * n

    # Entries eligible for allocation (positive weight and positive cap)
    pos = [i for i, (w, c) in enumerate(zip(weights, caps)) if w > 0 and c > 0]
    if not pos:
        # Fallback: allocate cap-proportionally when all weights are zero
        weights = [float(c) for c in caps]
        pos = [i for i, c in enumerate(caps) if c > 0]

    # Binary search for alpha such that sum(min(cap_i, alpha * w_i)) == total
    lo = 0.0
    hi = max(caps[i] / weights[i] for i in pos)
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if sum(min(caps[i], mid * weights[i]) for i in pos) >= total:
            hi = mid
        else:
            lo = mid

    alpha = hi
    quotas = [0.0] * n
    for i in pos:
        quotas[i] = min(caps[i], alpha * weights[i])

    # Floor each quota, then distribute remainder via largest-fraction (Hamilton) rule.
    # Tie-break by index (higher = newer, since _entries is sorted oldest-first).
    takes = [min(caps[i], math.floor(quotas[i])) for i in range(n)]
    rem = total - sum(takes)
    order = sorted(
        [i for i in range(n) if takes[i] < caps[i]],
        key=lambda i: (quotas[i] - takes[i], i),  # higher i = newer; reverse=True below
        reverse=True,
    )
    for i in order[:rem]:
        takes[i] += 1
    return takes


class ReplayBuffer:
    """
    Sliding-window replay buffer backed by HDF5 files.

    Keeps the last ``window_size`` iterations of self-play data and merges
    them into a single HDF5 file for training. Respects ``max_size`` by
    randomly subsampling if total positions exceed the limit.

    State is only persisted at commit time; all persisted paths must point
    to committed iteration directories.
    """

    def __init__(self, config: dict, state_path: Optional[Path] = None):
        self.config = config
        rb_config = config.get("replay_buffer", {}) or {}
        self.max_size = int(rb_config.get("max_size", 500_000))
        self.min_size = int(rb_config.get("min_size", 8_000))
        self.window_size = max(1, int(rb_config.get("window_iterations", 5)))
        self.newest_fraction = float(rb_config.get("newest_fraction", 0.0))

        # League recency-biased sampling overrides newest_fraction when enabled
        league_cfg = config.get("league", {}) or {}
        if league_cfg.get("enabled", False):
            recency_frac = float(league_cfg.get("recency_newest_frac", 0.70))
            recency_window = float(league_cfg.get("recency_newest_window", 0.15))
            if recency_frac > 0:
                self.newest_fraction = recency_frac
                self._recency_window = recency_window
                logger.info(
                    "Replay buffer: league recency bias enabled "
                    "(%.0f%% from newest %.0f%% of buffer)",
                    recency_frac * 100, recency_window * 100,
                )
            else:
                self._recency_window = 0.0
        else:
            self._recency_window = 0.0

        self.recency_weight_lambda = float(rb_config.get("recency_weight_lambda", 1.0))
        if not (0.0 < self.recency_weight_lambda <= 1.0):
            raise ValueError(
                f"recency_weight_lambda must be in (0, 1], got {self.recency_weight_lambda}"
            )

        self.buffer_dir = Path(config["paths"].get("data_dir", "data")) / "replay_buffer"
        self._state_path = Path(state_path) if state_path else self.buffer_dir / "replay_state.json"
        self._entries: List[Tuple[int, str, int]] = []  # (iteration, path, num_positions)
        # Only create buffer_dir when state lives there (main uses run_root for state; buffer_dir unused)
        if state_path is None:
            self.buffer_dir.mkdir(parents=True, exist_ok=True)
        if self._state_path:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)

    def save_state(self) -> None:
        """Persist replay buffer entries to disk. Only call at commit time."""
        state = [{"iteration": it, "path": p, "positions": n} for it, p, n in self._entries]
        tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._state_path)

    def clear_persisted_state(self) -> None:
        """Clear in-memory entries and delete persisted replay_state.json (e.g. for --flush-replay-on-resume)."""
        self._entries = []
        if self._state_path and self._state_path.exists():
            try:
                self._state_path.unlink()
                logger.info("Replay buffer: cleared persisted state at %s", self._state_path)
            except OSError as e:
                logger.warning("Replay buffer: could not remove %s: %s", self._state_path, e)

    def restore_state(self) -> bool:
        """Restore replay buffer entries from disk after crash/resume.

        Only loads entries whose paths exist (committed iterations only).
        Enforces current window_size by evicting oldest entries if needed.
        Replaces _entries (idempotent: safe to call after repair or twice).
        Returns True if state was restored successfully.
        """
        if not self._state_path.exists():
            return False
        try:
            with open(self._state_path, "r", encoding="utf-8-sig") as f:
                state = json.load(f)
            new_entries: List[Tuple[int, str, int]] = []
            for entry in state:
                path = str(entry["path"])
                # Normalize path for existence check (handles Windows backslashes)
                if Path(path).exists():
                    new_entries.append((int(entry["iteration"]), path, int(entry["positions"])))
                else:
                    if "staging" in path:
                        logger.debug("Replay buffer restore: missing file %s, skipping (staging discarded)", path)
                    else:
                        logger.warning("Replay buffer restore: missing file %s, skipping", path)
            if new_entries:
                new_entries.sort(key=lambda x: x[0])
                # Enforce window_size by evicting oldest
                while len(new_entries) > self.window_size:
                    new_entries.pop(0)
                self._entries = new_entries
                return True
            return False
        except Exception as e:
            logger.warning("Failed to restore replay buffer state: %s", e)
            return False

    def finalize_iteration_for_commit(
        self, iteration: int, committed_selfplay_path: str, num_positions: int
    ) -> None:
        """
        Replace the in-memory entry for this iteration with the committed path,
        evict beyond window, then persist. Call only at commit time after staging->committed move.
        Ensures replay_state never exceeds window_iterations.
        """
        self._entries = [(it, p, n) for it, p, n in self._entries if it != iteration]
        self._entries.append((iteration, committed_selfplay_path, num_positions))
        self._entries.sort(key=lambda x: x[0])
        # Evict oldest beyond window (important for repair path which adds multiple iters)
        while len(self._entries) > self.window_size:
            evicted = self._entries.pop(0)
            logger.info(
                "Replay buffer: evicted iteration %d (%d positions) at commit",
                evicted[0],
                evicted[2],
            )
        self.save_state()

    @property
    def total_positions(self) -> int:
        return sum(n for _, _, n in self._entries)

    @property
    def num_iterations(self) -> int:
        return len(self._entries)

    def add_iteration(self, iteration: int, data_path: str) -> bool:
        """
        Register a new iteration's self-play data.

        The HDF5 file is not copied — we reference it in place.
        Old iterations beyond the window are evicted.
        Idempotent: if the iteration already exists, its entry is replaced
        (handles crash-restart where selfplay was regenerated).
        """
        try:
            with h5py.File(data_path, "r") as f:
                states_key = "spatial_states" if "spatial_states" in f else "states"
                num_positions = int(f[states_key].shape[0])
        except Exception as e:
            logger.error(f"Failed to read {data_path}: {e}")
            return False

        # Remove existing entry for this iteration (crash-restart idempotency)
        self._entries = [(it, p, n) for it, p, n in self._entries if it != iteration]

        self._entries.append((iteration, str(data_path), num_positions))
        
        logger.debug(
            f"Replay buffer: added iteration {iteration} "
            f"({num_positions} positions, {self.total_positions} total)"
        )

        # Evict oldest beyond window. Only delete if NOT in committed/ (never delete archive)
        while len(self._entries) > self.window_size:
            evicted = self._entries.pop(0)
            logger.info("Replay buffer: evicted iteration %d (%d positions)", evicted[0], evicted[2])
            evicted_path = Path(evicted[1])
            if "committed" not in evicted_path.parts:
                try:
                    if evicted_path.exists():
                        evicted_path.unlink()
                        logger.debug("Deleted evicted selfplay file: %s", evicted_path)
                except Exception as e:
                    logger.warning("Failed to delete evicted file %s: %s", evicted_path, e)

        # Persist state for crash recovery
        self.save_state()
        return True

    def get_training_data(self, seed: int = 42, output_path: Optional[str] = None) -> str:
        """
        Merge buffered iterations into a single HDF5 file for training.

        If total positions exceed ``max_size``, subsample. When ``newest_fraction`` > 0,
        a fixed fraction of samples is taken from the newest iteration and the rest
        from older iterations (proportionally). This keeps newest self-play data
        from being diluted when the buffer is large.

        Args:
            seed: Random seed for subsampling.
            output_path: If provided, write merged file here (atomically via .tmp).
                        Use for staging per-iteration; otherwise uses buffer_dir.
        """
        if not self._entries:
            raise RuntimeError("Replay buffer is empty")

        if output_path is not None:
            merged_path = Path(output_path)
            merged_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            merged_path = self.buffer_dir / "merged_training.h5"

        # Atomic write: write to .tmp then replace
        tmp_path = merged_path.with_suffix(merged_path.suffix + ".tmp")

        data_cfg = self.config.get("data", {}) or {}
        # data.expected_spatial_channels is the channel count actually stored on disk.
        # network.input_channels is the model trunk input (may differ: e.g. gold_v2_32ch
        # stores 32ch but the trunk sees 56ch after DeterministicLegalityModule).
        _stored_ch = data_cfg.get("expected_spatial_channels")
        expected_channels = int(_stored_ch) if _stored_ch is not None else int(
            (self.config.get("network", {}) or {}).get("input_channels", 56)
        )
        # is_multimodal: True for any gold_v2 encoding (has spatial/global/track/shop arrays)
        _enc_ver = data_cfg.get("encoding_version", "")
        is_multimodal = _enc_ver.startswith("gold_v2") or expected_channels == 56
        state_shape = (expected_channels, 9, 9)
        mask_shape = (2026,)
        pol_shape = (2026,)
        own_shape = (2, 9, 9)
        # Validate first file exists
        first = None
        for _, h5_path, _ in self._entries:
            try:
                first = h5py.File(h5_path, "r")
                break
            except Exception:
                continue
        if first is None:
            raise RuntimeError("No data could be read from replay buffer")
        with first:
            has_ownership = "ownerships" in first
            has_score_margins = "score_margins" in first
            merge_multimodal = "spatial_states" in first and is_multimodal

        total = sum(n for _, _, n in self._entries)
        logger.debug(f"Replay buffer merge: {total} total positions from {len(self._entries)} iterations")

        # Decide per-file sample counts.
        #
        # Three paths (priority order):
        #   1. recency_weight_lambda < 1.0 AND league disabled: proportional oversampling.
        #      Newer iters contribute > their raw share (sampled with replacement if needed),
        #      older iters contribute < their share.  Total = exactly target_total.
        #      Active regardless of buffer fill level — does not require overflow.
        #   2. use_newest_bias (league enabled + overflow): newest_fraction split.
        #   3. Uniform: use-all when fits, proportional trim when overflow.
        #
        target_total = min(total, self.max_size)
        rng = np.random.default_rng(seed)
        per_file: List[Tuple[int, str, int, int]] = []

        # League path requires an explicit recency window; without it newest_fraction is a no-op.
        use_newest_bias = (
            self.newest_fraction > 0
            and len(self._entries) > 1
            and getattr(self, "_recency_window", 0.0) > 0
        )

        if self.recency_weight_lambda < 1.0 and not use_newest_bias:
            # Proportional recency oversampling — always active when lambda < 1.0.
            # w_i = lambda^age * n_i.  Each iter's share = w_i / sum(w_j) * target_total.
            # Newer iters (high weight) receive MORE than their raw n_i → oversampled
            #   with replacement (some positions appear twice in the merged file).
            # Older iters receive LESS → subsampled without replacement as usual.
            max_iter = max(it for it, _, _ in self._entries)
            weights = [
                (self.recency_weight_lambda ** (max_iter - it)) * n
                for it, _, n in self._entries
            ]
            total_weight = sum(weights)
            raw = [w / total_weight * target_total for w in weights]
            takes = [math.floor(r) for r in raw]
            rem = target_total - sum(takes)
            # Hamilton rounding: largest fractional remainder first; higher index = newer wins ties.
            order = sorted(range(len(raw)), key=lambda i: (raw[i] - takes[i], i), reverse=True)
            for i in order[:rem]:
                takes[i] += 1
            for i, (it, p, n) in enumerate(self._entries):
                per_file.append((it, p, n, takes[i]))
            scale = target_total / total_weight if total_weight > 0 else 1.0
            logger.debug(
                "Replay buffer: recency oversampling %d -> %d positions "
                "(lambda=%.2f, scale=%.3fx, takes=%s)",
                total, target_total, self.recency_weight_lambda, scale, takes,
            )
        elif total <= self.max_size:
            # Buffer fits entirely: use all positions uniformly.
            for it, p, n in self._entries:
                per_file.append((it, p, n, n))
        else:
            if use_newest_bias:
                # League recency-biased sampling: newest_fraction of samples come from
                # the newest _recency_window fraction of the buffer (by position count).
                recency_window = getattr(self, '_recency_window', 0.0)
                if recency_window > 0 and len(self._entries) > 1:
                    # Identify "newest" entries: those in the top recency_window of positions
                    threshold_positions = int(total * recency_window)
                    newest_entries = []
                    rest_entries = []
                    cumulative = 0
                    for it, p, n in reversed(self._entries):
                        if cumulative < threshold_positions:
                            newest_entries.append((it, p, n))
                            cumulative += n
                        else:
                            rest_entries.append((it, p, n))
                    newest_entries.reverse()
                    rest_entries.reverse()
                    if not newest_entries:
                        newest_entries = [self._entries[-1]]
                        rest_entries = self._entries[:-1]
                else:
                    # Fallback: newest iteration only
                    newest_entries = [self._entries[-1]]
                    rest_entries = self._entries[:-1]

                n_newest_total = sum(n for _, _, n in newest_entries)
                rest_total = sum(n for _, _, n in rest_entries)
                newest_take = min(int(target_total * self.newest_fraction), n_newest_total)
                rest_take = target_total - newest_take
                if rest_take < 0:
                    rest_take = 0

                # Proportional allocation within newest group
                takes_by_it: Dict[int, int] = {}
                if n_newest_total > 0:
                    raw = [n * newest_take / n_newest_total for _, _, n in newest_entries]
                    base = [int(x) for x in raw]
                    rem = newest_take - sum(base)
                    frac_idx = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
                    for k in range(min(rem, len(frac_idx))):
                        base[frac_idx[k]] += 1
                    for (it, _, n), take in zip(newest_entries, base):
                        takes_by_it[it] = take

                # Proportional allocation within rest group
                if rest_take > 0 and rest_total > 0:
                    raw = [n * rest_take / rest_total for _, _, n in rest_entries]
                    base = [int(x) for x in raw]
                    rem = rest_take - sum(base)
                    frac_idx = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
                    for k in range(min(rem, len(frac_idx))):
                        base[frac_idx[k]] += 1
                    for (it, _, n), take in zip(rest_entries, base):
                        takes_by_it[it] = take
                else:
                    for it, _, _ in rest_entries:
                        takes_by_it[it] = 0

                # Preserve _entries order (oldest first)
                for it, p, n in self._entries:
                    per_file.append((it, p, n, takes_by_it.get(it, 0)))
                logger.debug(
                    f"Replay buffer: subsampling {total} -> {target_total} positions "
                    f"(newest={newest_take}, rest={rest_take}, newest_frac={self.newest_fraction:.0%})"
                )
            else:
                # Uniform proportional overflow (lambda == 1.0, league disabled)
                raw = [n * target_total / total for _, _, n in self._entries]
                base = [int(x) for x in raw]
                rem = target_total - sum(base)
                frac_idx = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
                for k in range(rem):
                    base[frac_idx[k]] += 1
                for (it, p, n), take in zip(self._entries, base):
                    per_file.append((it, p, n, take))
                logger.debug(f"Replay buffer: subsampling {total} -> {target_total} positions")

        # KataGo Dual-Head: refuse legacy HDF5 without score_margins when training score head
        score_loss_weight = float(
            (self.config.get("training", {}) or {}).get("score_loss_weight", 0.0)
        )
        score_margin_max_abs = float(
            (self.config.get("data", {}) or {}).get("score_margin_max_abs", SCORE_MARGIN_MAX_ABS)
        )

        # Stream-write merged file (no giant RAM concat).
        # The merged file is a temporary staging artifact loaded immediately into RAM
        # (PatchworkDataset bulk-reads it then discards it). Compression costs ~90s write
        # + forces decompression at load time — wasted CPU. Use no compression so writes
        # and reads are purely disk-bandwidth bound (~10s total vs ~120s with lzf).
        compression = None
        comp_opts = None
        # Write to tmp then replace for atomicity
        with h5py.File(tmp_path, "w") as out:
            chunk_rows = min(4096, max(256, total // 8))
            if merge_multimodal:
                from src.network.gold_v2_constants import C_SPATIAL_ENC, F_GLOBAL, C_TRACK, TRACK_LEN, NMAX, F_SHOP
                out_states = out.create_dataset("spatial_states", shape=(0, C_SPATIAL_ENC, 9, 9), maxshape=(None, C_SPATIAL_ENC, 9, 9),
                                                dtype=np.float32, chunks=(chunk_rows, C_SPATIAL_ENC, 9, 9),
                                                compression=compression, compression_opts=comp_opts)
                out_global = out.create_dataset("global_states", shape=(0, F_GLOBAL), maxshape=(None, F_GLOBAL),
                                                dtype=np.float16, chunks=(chunk_rows, F_GLOBAL),
                                                compression=compression, compression_opts=comp_opts)
                out_track = out.create_dataset("track_states", shape=(0, C_TRACK, TRACK_LEN), maxshape=(None, C_TRACK, TRACK_LEN),
                                                dtype=np.float16, chunks=(chunk_rows, C_TRACK, TRACK_LEN),
                                                compression=compression, compression_opts=comp_opts)
                out_shop_ids = out.create_dataset("shop_ids", shape=(0, NMAX), maxshape=(None, NMAX),
                                                  dtype=np.int16, chunks=(chunk_rows, NMAX),
                                                  compression=compression, compression_opts=comp_opts)
                out_shop_feats = out.create_dataset("shop_feats", shape=(0, NMAX, F_SHOP), maxshape=(None, NMAX, F_SHOP),
                                                    dtype=np.float16, chunks=(chunk_rows, NMAX, F_SHOP),
                                                    compression=compression, compression_opts=comp_opts)
            else:
                out_states = out.create_dataset("states", shape=(0, *state_shape), maxshape=(None, *state_shape),
                                                dtype=np.float32, chunks=(chunk_rows, *state_shape),
                                                compression=compression, compression_opts=comp_opts)
            out_masks = out.create_dataset("action_masks", shape=(0, *mask_shape), maxshape=(None, *mask_shape),
                                           dtype=np.float32, chunks=(chunk_rows, *mask_shape),
                                           compression=compression, compression_opts=comp_opts)
            out_pols = out.create_dataset("policies", shape=(0, *pol_shape), maxshape=(None, *pol_shape),
                                          dtype=np.float32, chunks=(chunk_rows, *pol_shape),
                                          compression=compression, compression_opts=comp_opts)
            out_vals = out.create_dataset("values", shape=(0,), maxshape=(None,), dtype=np.float32,
                                          chunks=(chunk_rows,), compression=compression, compression_opts=comp_opts)
            out_scores = out.create_dataset("score_margins", shape=(0,), maxshape=(None,), dtype=np.float32,
                                            chunks=(chunk_rows,), compression=compression, compression_opts=comp_opts)
            out_owns = out.create_dataset("ownerships", shape=(0, *own_shape), maxshape=(None, *own_shape),
                                          dtype=np.float32, chunks=(chunk_rows, *own_shape),
                                          compression=compression, compression_opts=comp_opts)
            # slot_piece_ids optional - created only when merging canonical data
            out_slot_ids = None

            cur = 0
            read_chunk = 8192
            for it, h5_path, n, take in per_file:
                if take <= 0:
                    continue
                try:
                    with h5py.File(h5_path, "r") as f:
                        file_has_ownership = "ownerships" in f
                        file_has_scores = "score_margins" in f
                        if (
                            _REJECT_LEGACY_IF_SCORE_HEAD
                            and score_loss_weight > 0
                            and not file_has_scores
                        ):
                            raise RuntimeError(
                                f"Legacy HDF5 without score_margins: {h5_path}. "
                                "Dual-Head training cannot use zero-filled scores (cripples Score Head). "
                                "Wipe runs/<run_id>/staging and runs/<run_id>/committed and restart "
                                "so all self-play data uses the new schema."
                            )
                        idx = np.arange(n, dtype=np.int64)
                        if take != n:
                            # replace=True when take > n (oversampling newer iterations).
                            idx = rng.choice(idx, size=take, replace=take > n)
                            idx.sort()
                        # batched gather for locality
                        file_multimodal = "spatial_states" in f
                        if file_multimodal != merge_multimodal:
                            raise ValueError(
                                f"Replay buffer: cannot mix multimodal and legacy. "
                                f"File {h5_path} has_multimodal={file_multimodal}, merge expects {merge_multimodal}"
                            )
                        for s0 in range(0, len(idx), read_chunk):
                            sl = idx[s0:s0 + read_chunk]
                            # h5py requires strictly increasing (no duplicate) indices.
                            # When oversampling (take > n), sl may contain duplicates.
                            # Deduplicate for h5py, then expand back via inverse mapping.
                            if take > n:
                                sl_h5, sl_inv = np.unique(sl, return_inverse=True)
                            else:
                                sl_h5, sl_inv = sl, None
                            if merge_multimodal:
                                st_raw = np.asarray(f["spatial_states"][sl_h5], dtype=np.float32)
                                if sl_inv is not None:
                                    st_raw = st_raw[sl_inv]
                            else:
                                st_raw = np.asarray(f["states"][sl_h5], dtype=np.float32)
                                if sl_inv is not None:
                                    st_raw = st_raw[sl_inv]
                            file_ch = st_raw.shape[1]
                            if file_ch != expected_channels:
                                allow_legacy = bool(
                                    (self.config.get("data", {}) or {}).get("allow_legacy_state_channels", False)
                                )
                                if allow_legacy:
                                    if file_ch < expected_channels:
                                        pad = np.zeros(
                                            (st_raw.shape[0], expected_channels - file_ch, 9, 9),
                                            dtype=np.float32,
                                        )
                                        st = np.concatenate([st_raw, pad], axis=1)
                                    else:
                                        st = st_raw[:, :expected_channels, :, :].copy()
                                else:
                                    raise ValueError(
                                        f"Replay buffer: HDF5 file {h5_path} has states with "
                                        f"{file_ch} channels, expected {expected_channels}. "
                                        "Regenerate self-play data or set data.allow_legacy_state_channels: true."
                                    )
                            else:
                                st = st_raw
                            am = np.asarray(f["action_masks"][sl_h5], dtype=np.float32)
                            pi = np.asarray(f["policies"][sl_h5], dtype=np.float32)
                            va = np.asarray(f["values"][sl_h5], dtype=np.float32)
                            if sl_inv is not None:
                                am = am[sl_inv]
                                pi = pi[sl_inv]
                                va = va[sl_inv]
                            if file_has_scores:
                                sc = np.asarray(f["score_margins"][sl_h5], dtype=np.float32)
                                if sl_inv is not None:
                                    sc = sc[sl_inv]
                                _validate_score_margins(sc, max_abs=score_margin_max_abs, source=h5_path)
                            else:
                                sc = np.zeros(st.shape[0], dtype=np.float32)
                            if file_has_ownership:
                                ow = np.asarray(f["ownerships"][sl_h5], dtype=np.float32)
                                if sl_inv is not None:
                                    ow = ow[sl_inv]
                            else:
                                # Fill with -1 sentinel for old data without ownership
                                ow = np.full((st.shape[0], *own_shape), -1.0, dtype=np.float32)

                            file_has_slot_ids = "slot_piece_ids" in f
                            if file_has_slot_ids:
                                sid = np.asarray(f["slot_piece_ids"][sl_h5], dtype=np.int16)
                                if sl_inv is not None:
                                    sid = sid[sl_inv]
                            else:
                                sid = np.full((st.shape[0], 3), -1, dtype=np.int16)

                            new = cur + st.shape[0]
                            out_states.resize((new,) + out_states.shape[1:])
                            if merge_multimodal:
                                out_global.resize((new, F_GLOBAL))
                                out_track.resize((new,) + out_track.shape[1:])
                                out_shop_ids.resize((new,) + out_shop_ids.shape[1:])
                                out_shop_feats.resize((new,) + out_shop_feats.shape[1:])
                            out_masks.resize((new, *mask_shape))
                            out_pols.resize((new, *pol_shape))
                            out_vals.resize((new,))
                            out_scores.resize((new,))
                            out_owns.resize((new, *own_shape))
                            if out_slot_ids is None:
                                out_slot_ids = out.create_dataset(
                                    "slot_piece_ids", shape=(0, 3), maxshape=(None, 3),
                                    dtype=np.int16, chunks=(chunk_rows, 3),
                                    compression=compression, compression_opts=comp_opts
                                )
                            if out_slot_ids is not None:
                                out_slot_ids.resize((new, 3))
                                out_slot_ids[cur:new] = sid

                            out_states[cur:new] = st
                            if merge_multimodal:
                                g = np.asarray(f["global_states"][sl_h5], dtype=np.float16)
                                tr = np.asarray(f["track_states"][sl_h5], dtype=np.float16)
                                si = np.asarray(f["shop_ids"][sl_h5])
                                sf = np.asarray(f["shop_feats"][sl_h5], dtype=np.float16)
                                if sl_inv is not None:
                                    g, tr, si, sf = g[sl_inv], tr[sl_inv], si[sl_inv], sf[sl_inv]
                                out_global[cur:new] = g
                                out_track[cur:new] = tr
                                out_shop_ids[cur:new] = si
                                out_shop_feats[cur:new] = sf
                            out_masks[cur:new] = am
                            out_pols[cur:new] = pi
                            out_vals[cur:new] = va
                            out_scores[cur:new] = sc
                            out_owns[cur:new] = ow
                            cur = new
                except Exception as e:
                    logger.error(f"Failed to read {h5_path} for merging: {e}")
                    raise RuntimeError(
                        f"Replay buffer merge failed reading {h5_path}. "
                        "Cannot train on partial data. Fix or remove corrupted self-play file."
                    ) from e

            out.attrs["num_positions"] = int(cur)
            out.attrs["num_source_iterations"] = len(self._entries)
            out.attrs["source_iterations"] = [it for it, _, _ in self._entries]

        os.replace(tmp_path, merged_path)
        logger.debug(
            "Replay buffer: wrote %d positions to %s",
            cur, merged_path,
        )
        return str(merged_path)

    def has_enough_data(self) -> bool:
        """Check if buffer has enough data to start training."""
        return self.total_positions >= self.min_size