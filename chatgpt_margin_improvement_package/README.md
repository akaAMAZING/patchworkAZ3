# ChatGPT package: margin maximization in all situations

This folder is a **self-contained package** to paste into ChatGPT so it can propose code changes that make the agent **maximize score margin from its own perspective in every situation** (win by more, lose by less).

## How to use

1. **Open a new ChatGPT conversation** (or a thread where you want a detailed, code-level answer).
2. **Paste the contents of `PROMPT.md`** first. That is the main task and constraints.
3. **Paste the contents of each file in `context/`** (and optionally `FILE_MAP.md`) so ChatGPT has the exact code and structure:
   - `context/value_targets.py`
   - `context/model_value_and_score_loss.py`
   - `context/selfplay_value_assignment.py`
   - `context/mcts_utility_and_terminal.py`
   - `context/config_snippet.yaml`
4. Ask ChatGPT to output a concrete design and **exact code changes** (file paths, function names, and suggested edits) so that:
   - The value target (or combined value/score target) encourages “maximize margin” in all cases.
   - MCTS and self-play stay consistent.
   - Backward compatibility with existing replay data is addressed.

## What’s in the package

- **PROMPT.md** – Full task description, current behavior, desired behavior, constraints, and what we need (design + code changes + compatibility + sanity checks).
- **context/** – Relevant source excerpts and one full file (`value_targets.py`) so ChatGPT can suggest minimal, consistent edits.
- **FILE_MAP.md** – Mapping from package context files to real repo paths.
- **README.md** – This file.

After you get the suggested changes, apply them in your repo at the paths given in `FILE_MAP.md` (and in ChatGPT’s response).
