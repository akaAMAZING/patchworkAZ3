import numpy as np
import torch


def _state_with_three_empties_for_patch(player: int = 0) -> np.ndarray:
    """
    Construct a state where the current player has 3 empty cells:
      - two adjacent empties at (0,0) and (0,1)
      - one isolated empty at (8,8)
    and a pending patch to place.
    """
    from src.game.patchwork_engine import state_from_dict

    rows = []
    for r in range(9):
        s = ["1"] * 9
        rows.append(s)
    rows[0][0] = "."
    rows[0][1] = "."
    rows[8][8] = "."
    p_board = ["".join(r) for r in rows]

    empty_board = ["." * 9 for _ in range(9)]
    players = [{"position": 0, "buttons": 5, "income": 0, "board": empty_board} for _ in range(2)]
    players[player]["board"] = p_board

    d = {
        "edition": "revised",
        "players": players,
        "circle": [],
        "neutral": 0,
        "bonus_owner": -1,
        "pending_patches": 1,
        "pending_owner": int(player),
        "tie_player": int(player),
        "randomize_circle": False,
    }
    return state_from_dict(d)


def test_patch_tiebreak_prefers_better_packing_when_q_tied():
    from src.game.patchwork_engine import AT_PATCH
    from src.mcts.alphazero_mcts_optimized import (
        MCTSConfig,
        PatchTiebreakConfig,
        MCTSNode,
        OptimizedAlphaZeroMCTS,
    )

    st = _state_with_three_empties_for_patch(player=0)
    to_move = 0

    # Two PATCH placements:
    # - idx 0 fills one of the adjacent empties → tends to increase isolated holes
    # - idx 80 fills the isolated one → improves fragmentation
    a_bad = (AT_PATCH, 0, 0, 0, 0, 0)
    a_good = (AT_PATCH, 80, 0, 0, 0, 0)

    root = MCTSNode(st, to_move)
    root.legal_actions = [a_bad, a_good]
    root._init_arrays()

    # Make them tied in value/Q (sure-win): Qv=1.0 for both.
    root._visit_count[:] = 10
    root._value_sum[:] = 10.0
    root._score_sum[:] = 0.0
    root._total_value[:] = 10.0
    root.n_total = int(root._visit_count.sum())

    cfg = MCTSConfig()
    cfg.patch_tiebreak = PatchTiebreakConfig(
        enabled=True,
        mode="packing",
        value_tie_eps=0.0,
        win_prob_floor=0.98,
        weights={"empty_squares": 1.0, "empty_components": 2.0, "isolated_1x1": 3.0},
        score_weight=0.0,
    )

    mcts = OptimizedAlphaZeroMCTS(
        network=None,
        config=cfg,
        device=torch.device("cpu"),
        state_encoder=None,
        action_encoder=None,
        eval_client=None,
        inference_settings=None,
        full_config=None,
    )
    mcts._root = root

    picked = mcts._maybe_apply_patch_tiebreak(selected_action=a_bad)
    assert picked == a_good

