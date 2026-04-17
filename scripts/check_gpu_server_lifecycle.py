#!/usr/bin/env python3
"""
Check GPU server lifecycle behaviour (for repeated-cycle failure debugging).

Prints a summary of:
- Whether the GPU server is started once or recreated per cycle
- Where cleanup (queue close, process join, SHM destroy) happens
- Where to look for PID/exitcode logs

Usage:
  python scripts/check_gpu_server_lifecycle.py

No GPU or training run required; this script only inspects the codebase.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main() -> int:
    print("GPU server lifecycle check (codebase inspection)\n")
    print("1. Is the GPU server long-lived or recreated?")
    print("   -> Recreated every cycle. Evidence:")
    print("      - main.py: for iteration in range(...): _generate_selfplay_data(...)")
    print("      - selfplay_optimized_integration.generate(): _start_gpu_server() at start,")
    print("        finally: _stop_gpu_server() before return.")
    print()

    print("2. Where is old-process cleanup done?")
    print("   -> src/training/selfplay_optimized_integration.py _stop_gpu_server()")
    print("      - stop_evt.set(), gpu_process.join(5), optional terminate(), join(2)")
    print("      - req_q.close(), resp_qs[].close(), cancel_join_thread() on each")
    print("      - gpu_process = None, req_q = None, resp_qs = None, stop_evt = None")
    print("      - SHM destroy, _worker_shm_bufs/names clear")
    print()

    print("3. Where are PID and exit code logged?")
    print("   -> _start_gpu_server: logger.info('GPU server process started PID=%s', ...)")
    print("   -> _stop_gpu_server:  logger.info('GPU server stopped PID=%s exitcode=%s', ...)")
    print("   Set logging to INFO to see these (e.g. logging.basicConfig(level=logging.INFO)).")
    print()

    print("4. Per-cycle CUDA memory (parent) logging?")
    print("   -> main.py after empty_cache/sleep(3): logger.debug('[LIFECYCLE] iter N end: CUDA allocated=...')")
    print("   Set logging to DEBUG to see these.")
    print()

    # Quick sanity: can we import the module and see the attributes?
    try:
        from src.training.selfplay_optimized_integration import SelfPlayGenerator
        gen = SelfPlayGenerator({"paths": {}, "selfplay": {"num_workers": 2}, "hardware": {"device": "cuda"}})
        assert gen.gpu_process is None
        assert gen.req_q is None
        assert gen.resp_qs is None
        assert gen.stop_evt is None
        print("5. SelfPlayGenerator initial state: gpu_process/req_q/resp_qs/stop_evt are None -> OK")
    except Exception as e:
        print(f"5. SelfPlayGenerator check: {e}")
    print()

    print("Done. Run training with INFO logging to observe PID/exitcode per cycle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
