import random

from src.game.patchwork_engine import apply_action_unchecked, get_winner_fast, legal_actions_fast, new_game, terminal_fast
from src.mcts.alphazero_mcts_optimized import engine_action_to_flat_index
from src.network.encoder import ActionEncoder
from src.training.evaluation import build_eval_schedule
from src.training.value_targets import terminal_value_from_scores


def test_action_indexing_alignment_and_mask_legality():
    encoder = ActionEncoder()
    rng = random.Random(7)
    state = new_game(seed=7)

    for _ in range(12):
        legal = legal_actions_fast(state)
        indices, mask = encoder.encode_legal_actions(legal)
        assert len(legal) > 0
        assert mask.sum() >= 1.0
        for action in legal:
            idx_fast = int(engine_action_to_flat_index(action))
            idx_enc = int(encoder.encode_action(action))
            assert idx_fast == idx_enc
            assert mask[idx_enc] == 1.0

        for idx in indices[: min(32, len(indices))]:
            decoded = encoder.decode_action(int(idx))
            assert encoder.encode_action(decoded) == int(idx)

        state = apply_action_unchecked(state, rng.choice(legal))
        if terminal_fast(state):
            state = new_game(seed=rng.randint(1, 10_000))


def test_terminal_value_tie_and_non_tie_semantics():
    # KataGo Dual-Head: value is strictly 1.0 / -1.0 / 0.0
    v0 = terminal_value_from_scores(12, 8, winner=0, to_move=0)
    v1 = terminal_value_from_scores(12, 8, winner=0, to_move=1)
    assert v0 == 1.0
    assert v1 == -1.0

    # Tie: strictly 0.0
    t0 = terminal_value_from_scores(10, 10, winner=0, to_move=0)
    t1 = terminal_value_from_scores(10, 10, winner=0, to_move=1)
    assert t0 == 0.0
    assert t1 == 0.0

    # Another non-tie
    wl0 = terminal_value_from_scores(8, 6, winner=0, to_move=0)
    wl1 = terminal_value_from_scores(8, 6, winner=0, to_move=1)
    assert wl0 == 1.0
    assert wl1 == -1.0


def test_terminal_winner_consistency_random_rollout():
    rng = random.Random(99)
    state = new_game(seed=99)
    for _ in range(200):
        if terminal_fast(state):
            break
        state = apply_action_unchecked(state, rng.choice(legal_actions_fast(state)))

    assert terminal_fast(state)
    assert get_winner_fast(state) in (0, 1)


def test_paired_eval_schedule_fairness():
    schedule = build_eval_schedule(num_games=8, base_seed=42, paired_eval=True)
    assert len(schedule) == 8
    # Every pair uses same seed and opposite first player.
    for i in range(0, len(schedule), 2):
        seed_a, first_a = schedule[i]
        seed_b, first_b = schedule[i + 1]
        assert seed_a == seed_b
        assert first_a is True
        assert first_b is False
