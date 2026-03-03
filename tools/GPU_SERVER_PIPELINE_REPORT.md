# GPU Server Pipeline Check Report

## Config summary
- Config file: `C:\Users\Shanks\Desktop\Codes\patchworkaz - Copy - v2\configs\config_best.yaml`
- `selfplay.num_workers`: 12
- `selfplay.mcts.parallel_leaves`: 64
- `iteration.parallel_leaves_schedule`: [{'iteration': 0, 'parallel_leaves': 64}, {'iteration': 200, 'parallel_leaves': 48}, {'iteration': 380, 'parallel_leaves': 32}]

## Expected slot computation
- Logic: max(schedule) if schedule else base.
- **expected_n_slots**: 64
- **WorkerSharedBuffer.SLOT_BYTES**: 29940

## First cycle SHM details
- cycle=0 expected_n_slots=64 SLOT_BYTES=29940 | wid0: name='wnsm_8b947519' size=1916160 derived=64 | wid1: name='wnsm_a4464a6b' size=1916160 derived=64 | wid2: name='wnsm_0301c24a' size=1916160 derived=64 | wid3: name='wnsm_737a19fd' size=1916160 derived=64 | wid4: name='wnsm_d98ac886' size=1916160 derived=64 | wid5: name='wnsm_2808616e' size=1916160 derived=64 | wid6: name='wnsm_46aab63c' size=1916160 derived=64 | wid7: name='wnsm_1f107cc9' size=1916160 derived=64 | wid8: name='wnsm_9779957c' size=1916160 derived=64 | wid9: name='wnsm_acb73d84' size=1916160 derived=64 | wid10: name='wnsm_057d327b' size=1916160 derived=64 | wid11: name='wnsm_19a88e7d' size=1916160 derived=64

## GPU server attach logs (from gpu_inference_server)
- Child logs one line per worker after attach, e.g.:
- `[GPU Server] SHM attach wid=0 name=wnsm_XXXX size=1916928 SLOT_BYTES=29940 derived=64 expected_n_slots=64`
- (On Windows, child size may be page-rounded, e.g. 1916928 vs parent 1916160; derived=64 in both.)
- Parent-side first cycle: see First cycle SHM details above.

## Burst (live request path)
- **Request throughput**: 61.9 req/s
- **Timeouts**: 0
- **Malformed responses**: 0
- **Burst PASS**: yes

## Result
- **Pass**: 10
- **Fail**: 0
- **Duration (s)**: 2963.29

---