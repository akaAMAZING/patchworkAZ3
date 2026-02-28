# Map: package context files → repo paths

| Package file | Repo path |
|--------------|-----------|
| `PROMPT.md` | (instructions only) |
| `context/value_targets.py` | `src/training/value_targets.py` |
| `context/model_value_and_score_loss.py` | `src/network/model.py` (ValueHead + `get_loss` value/score parts) |
| `context/selfplay_value_assignment.py` | `src/training/selfplay_optimized.py` (value/score fill loop) |
| `context/mcts_utility_and_terminal.py` | `src/mcts/alphazero_mcts_optimized.py` (MCTSConfig, `_compute_score_utility`, `_get_terminal_value`, backup) |
| `context/config_snippet.yaml` | `configs/config_best.yaml` (training + selfplay.mcts weights) |

All paths are relative to the repository root (e.g. `patchworkaz - Copy`).
