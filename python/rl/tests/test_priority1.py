from __future__ import annotations

import unittest

import numpy as np
import torch

from python.rl.config import MCTSConfig, selfplay_exploration_scale
from python.rl.encoder import encode_state
from python.rl.game import GameConfig, GameState, Move, Stone
from python.rl.mcts import MCTS


class DummyPolicyValueNet(torch.nn.Module):
    def __init__(
        self,
        action_size: int,
        preferred_action: int,
        *,
        secondary_action: int | None = None,
        preferred_logit: float = 10.0,
        secondary_logit: float = 0.0,
        value: float = 0.0,
        score: float = 0.0,
    ) -> None:
        super().__init__()
        self.action_size = action_size
        self.preferred_action = preferred_action
        self.secondary_action = secondary_action
        self.preferred_logit = preferred_logit
        self.secondary_logit = secondary_logit
        self.value = value
        self.score = score

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        logits = torch.full((batch, self.action_size), -10.0, dtype=torch.float32, device=x.device)
        logits[:, self.preferred_action] = self.preferred_logit
        if self.secondary_action is not None:
            logits[:, self.secondary_action] = self.secondary_logit
        values = torch.full((batch,), self.value, dtype=torch.float32, device=x.device)
        ownership = torch.zeros(batch, self.action_size - 1, dtype=torch.float32, device=x.device)
        score = torch.full((batch,), self.score, dtype=torch.float32, device=x.device)
        return logits, values, ownership, score


class LastMoveValueNet(torch.nn.Module):
    def __init__(self, action_size: int) -> None:
        super().__init__()
        self.action_size = action_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        logits = torch.full((batch, self.action_size), -100.0, dtype=torch.float32, device=x.device)
        logits[:, 0] = 0.0
        logits[:, 1] = 0.0
        values = torch.zeros((batch,), dtype=torch.float32, device=x.device)
        for row in range(batch):
            if float(x[row, 3, 0].item()) > 0.5:
                values[row] = 1.0
            elif float(x[row, 3, 1].item()) > 0.5:
                values[row] = -1.0
        ownership = torch.zeros(batch, self.action_size - 1, dtype=torch.float32, device=x.device)
        score = torch.zeros(batch, dtype=torch.float32, device=x.device)
        return logits, values, ownership, score


class EqualPolicyNet(torch.nn.Module):
    def __init__(self, action_size: int) -> None:
        super().__init__()
        self.action_size = action_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        logits = torch.zeros((batch, self.action_size), dtype=torch.float32, device=x.device)
        values = torch.zeros((batch,), dtype=torch.float32, device=x.device)
        ownership = torch.zeros(batch, self.action_size - 1, dtype=torch.float32, device=x.device)
        score = torch.zeros(batch, dtype=torch.float32, device=x.device)
        return logits, values, ownership, score


class Priority1Tests(unittest.TestCase):
    def test_encode_state_includes_real_history_planes(self) -> None:
        state = GameState(GameConfig(side_length=2, opening_no_pass_moves=0))
        state.apply_move(Move.place(11))
        state.apply_move(Move.place(0))

        planes = encode_state(state, history=3)

        self.assertEqual(planes.shape, (8, state.topology.vertex_count))
        np.testing.assert_array_equal(planes[0], np.array([1.0 if stone == Stone.BLACK else 0.0 for stone in state.board], dtype=np.float32))
        np.testing.assert_array_equal(planes[1], np.array([1.0 if stone == Stone.WHITE else 0.0 for stone in state.board], dtype=np.float32))
        self.assertEqual(float(planes[4][11]), 1.0)
        self.assertAlmostEqual(float(planes[5].sum()), 0.0)
        self.assertAlmostEqual(float(planes[6].sum()), 0.0)
        self.assertAlmostEqual(float(planes[7].sum()), 0.0)

    def test_opening_pass_is_not_legal_until_threshold(self) -> None:
        state = GameState(GameConfig(side_length=2, opening_no_pass_moves=2))

        self.assertFalse(any(move.kind == "pass" for move in state.legal_moves()))
        state.apply_move(Move.place(11))
        self.assertFalse(any(move.kind == "pass" for move in state.legal_moves()))
        state.apply_move(Move.place(0))
        self.assertTrue(any(move.kind == "pass" for move in state.legal_moves()))

    def test_mcts_excludes_pass_when_opening_pass_is_disabled(self) -> None:
        state = GameState(GameConfig(side_length=2, opening_no_pass_moves=12))
        pass_action = state.topology.vertex_count
        model = DummyPolicyValueNet(action_size=pass_action + 1, preferred_action=pass_action)
        search = MCTS(
            model,
            MCTSConfig(simulations=1, root_exploration_fraction=0.0),
            torch.device("cpu"),
            input_history=2,
        )

        policy = search.run(state).policy

        self.assertEqual(float(policy[pass_action]), 0.0)
        self.assertGreater(float(policy[:-1].sum()), 0.0)

    def test_mcts_penalizes_pass_prior_when_other_moves_exist(self) -> None:
        state = GameState(GameConfig(side_length=2, opening_no_pass_moves=0))
        pass_action = state.topology.vertex_count
        place_action = 11
        model = DummyPolicyValueNet(
            action_size=pass_action + 1,
            preferred_action=pass_action,
            secondary_action=place_action,
            preferred_logit=1.0,
            secondary_logit=0.95,
        )
        search = MCTS(
            model,
            MCTSConfig(simulations=1, root_exploration_fraction=0.0, pass_prior_scale=0.2),
            torch.device("cpu"),
            input_history=2,
            root_noise_enabled=False,
        )

        policy = search.run(state).policy

        self.assertEqual(int(policy.argmax()), place_action)
        self.assertEqual(float(policy[pass_action]), 0.0)

    def test_mcts_blocks_losing_reply_pass_after_single_pass(self) -> None:
        state = GameState(GameConfig(side_length=2, opening_no_pass_moves=0))
        state.apply_move(Move.pass_turn())
        pass_action = state.topology.vertex_count
        model = DummyPolicyValueNet(
            action_size=pass_action + 1,
            preferred_action=pass_action,
            value=-0.25,
            score=-1.5,
        )
        search = MCTS(
            model,
            MCTSConfig(
                simulations=1,
                root_exploration_fraction=0.0,
                consecutive_pass_min_value=0.0,
                consecutive_pass_min_score_margin=0.0,
            ),
            torch.device("cpu"),
            input_history=2,
            root_noise_enabled=False,
        )

        policy = search.run(state).policy

        self.assertEqual(float(policy[pass_action]), 0.0)
        self.assertGreater(float(policy[:-1].sum()), 0.0)

    def test_mcts_can_skip_consecutive_pass_guard_when_disabled(self) -> None:
        state = GameState(GameConfig(side_length=2, opening_no_pass_moves=0))
        state.apply_move(Move.pass_turn())
        pass_action = state.topology.vertex_count
        model = DummyPolicyValueNet(
            action_size=pass_action + 1,
            preferred_action=pass_action,
            value=-0.25,
            score=-1.5,
        )
        search = MCTS(
            model,
            MCTSConfig(
                simulations=1,
                root_exploration_fraction=0.0,
                consecutive_pass_min_value=0.0,
                consecutive_pass_min_score_margin=0.0,
                disable_consecutive_pass_guard=True,
                pass_prior_scale=1.0,
            ),
            torch.device("cpu"),
            input_history=2,
            root_noise_enabled=False,
        )

        policy = search.run(state).policy

        self.assertGreater(float(policy[pass_action]), 0.0)

    def test_mcts_interprets_child_values_from_parent_perspective(self) -> None:
        state = GameState(GameConfig(side_length=2, allow_suicide=True, opening_no_pass_moves=999))
        state.board = [Stone.BLACK] * state.topology.vertex_count
        state.board[0] = Stone.EMPTY
        state.board[1] = Stone.EMPTY
        state.board_history = [state.board.copy()]
        state.hash_history = [state.compute_hash()]

        search = MCTS(
            LastMoveValueNet(state.topology.vertex_count + 1),
            MCTSConfig(simulations=10, root_exploration_fraction=0.0, c_puct=0.1),
            torch.device("cpu"),
            input_history=2,
            root_noise_enabled=False,
        )

        policy = search.run(state).policy

        self.assertEqual(int(policy.argmax()), 1)
        self.assertGreater(float(policy[1]), float(policy[0]))

    def test_mcts_root_prior_shaper_changes_root_visit_distribution(self) -> None:
        state = GameState(GameConfig(side_length=2, allow_suicide=True, opening_no_pass_moves=999))
        state.board = [Stone.BLACK] * state.topology.vertex_count
        state.board[0] = Stone.EMPTY
        state.board[1] = Stone.EMPTY
        state.board_history = [state.board.copy()]
        state.hash_history = [state.compute_hash()]
        model = DummyPolicyValueNet(
            action_size=state.topology.vertex_count + 1,
            preferred_action=0,
            secondary_action=1,
            preferred_logit=0.0,
            secondary_logit=0.0,
        )
        config = MCTSConfig(simulations=24, root_exploration_fraction=0.0, c_puct=2.0)

        baseline = MCTS(model, config, torch.device("cpu"), input_history=2, root_noise_enabled=False).run(state).policy

        def prefer_action_one(_state: GameState, priors: np.ndarray, legal_actions: list[int]) -> np.ndarray:
            adjusted = np.zeros_like(priors)
            adjusted[legal_actions] = priors[legal_actions]
            adjusted[1] = 0.95
            adjusted[0] = 0.05
            return adjusted

        shaped = MCTS(
            model,
            config,
            torch.device("cpu"),
            input_history=2,
            root_noise_enabled=False,
            root_prior_shaper=prefer_action_one,
        ).run(state).policy

        self.assertAlmostEqual(float(baseline[0]), 0.5, places=6)
        self.assertAlmostEqual(float(baseline[1]), 0.5, places=6)
        self.assertGreater(float(shaped[1]), float(baseline[1]))
        self.assertLess(float(shaped[0]), float(baseline[0]))

    def test_mcts_converts_terminal_result_to_leaf_player_perspective(self) -> None:
        state = GameState(GameConfig(side_length=2, opening_no_pass_moves=0, komi=1.5))
        state.apply_move(Move.place(0))
        state.apply_move(Move.pass_turn())

        pass_action = state.topology.vertex_count
        model = EqualPolicyNet(action_size=pass_action + 1)
        search = MCTS(
            model,
            MCTSConfig(simulations=12, root_exploration_fraction=0.0, c_puct=0.1, pass_prior_scale=1.0),
            torch.device("cpu"),
            input_history=2,
            root_noise_enabled=False,
        )

        policy = search.run(state).policy

        self.assertLess(float(policy[pass_action]), 0.5)
        self.assertGreater(float(policy[:-1].sum()), float(policy[pass_action]))

    def test_selfplay_exploration_scale_decays_with_iteration(self) -> None:
        cfg = MCTSConfig(selfplay_exploration_scale=1.4, selfplay_exploration_decay_iterations=4)

        self.assertAlmostEqual(selfplay_exploration_scale(cfg, 1), 1.4)
        self.assertAlmostEqual(selfplay_exploration_scale(cfg, 2), 1.2666666666666666)
        self.assertAlmostEqual(selfplay_exploration_scale(cfg, 4), 1.0)
        self.assertAlmostEqual(selfplay_exploration_scale(cfg, 10), 1.0)

    def test_terminal_scoring_cleans_obvious_dead_group(self) -> None:
        state = GameState(GameConfig(side_length=2, komi=0.5, cleanup_dead_stones=True))
        board = [Stone.EMPTY] * state.topology.vertex_count
        board[11] = Stone.WHITE
        for index in (1, 5, 10, 14):
            board[index] = Stone.BLACK
        state.board = board
        state.board_history = [board.copy()]
        state.finished = True
        state.end_reason = "double_pass"

        score = state.score()

        self.assertEqual(state.cleaned_dead_stones, 1)
        self.assertEqual(state.board[11], Stone.EMPTY)
        self.assertEqual(score.white, 0.5)
        self.assertGreater(score.black, 4.0)

    def test_apply_move_removes_captured_stones_and_tracks_last_move_captures(self) -> None:
        state = GameState(GameConfig(side_length=2, opening_no_pass_moves=0))
        state.board[11] = Stone.WHITE
        state.board[0] = Stone.BLACK
        state.board[10] = Stone.BLACK
        state.board_history = [state.board.copy()]
        state.hash_history = [state.compute_hash()]

        state.apply_move(Move.place(14))

        self.assertEqual(state.board[14], Stone.BLACK)
        self.assertEqual(state.board[11], Stone.EMPTY)
        self.assertEqual(state.last_move_captures, 1)
        self.assertEqual(state.to_play, Stone.WHITE)

    def test_simple_cleanup_removes_only_groups_with_one_or_fewer_liberties(self) -> None:
        state = GameState(GameConfig(side_length=2, cleanup_dead_stones=True, cleanup_dead_stones_mode="simple"))
        board = [Stone.EMPTY] * state.topology.vertex_count
        board[11] = Stone.WHITE
        board[0] = Stone.BLACK
        board[10] = Stone.BLACK
        state.board = board
        state.board_history = [board.copy()]
        state.finished = True
        state.end_reason = "double_pass"

        state.score()

        self.assertEqual(state.cleaned_dead_stones, 1)
        self.assertEqual(state.board[11], Stone.EMPTY)
        self.assertEqual(state.board[0], Stone.BLACK)
        self.assertEqual(state.board[10], Stone.BLACK)

    def test_simple_cleanup_preserves_two_liberty_group(self) -> None:
        state = GameState(GameConfig(side_length=2, cleanup_dead_stones=True, cleanup_dead_stones_mode="simple"))
        board = [Stone.EMPTY] * state.topology.vertex_count
        board[3] = Stone.BLACK
        board[1] = Stone.WHITE
        state.board = board
        state.board_history = [board.copy()]
        state.finished = True
        state.end_reason = "double_pass"

        state.score()

        self.assertEqual(state.cleaned_dead_stones, 0)
        self.assertEqual(state.board[3], Stone.BLACK)
        self.assertEqual(state.board[1], Stone.WHITE)

    def test_finalize_score_cleans_dead_group_when_game_hits_max_moves(self) -> None:
        state = GameState(GameConfig(side_length=2, komi=0.5, cleanup_dead_stones=True))
        board = [Stone.EMPTY] * state.topology.vertex_count
        board[11] = Stone.WHITE
        for index in (1, 5, 10, 14):
            board[index] = Stone.BLACK
        state.board = board
        state.board_history = [board.copy()]
        state.move_history = [Move.place(0)] * 12

        score = state.finalize_score("max_moves")

        self.assertTrue(state.finished)
        self.assertEqual(state.end_reason, "max_moves")
        self.assertEqual(state.cleaned_dead_stones, 1)
        self.assertEqual(state.board[11], Stone.EMPTY)
        self.assertEqual(score.white, 0.5)
        self.assertGreater(score.black, 4.0)

    def test_simple_ko_recapture_is_illegal(self) -> None:
        state = GameState(GameConfig(side_length=4, opening_no_pass_moves=0))
        for index in [3, 6, 7, 10, 11, 12, 21, 23, 24, 26, 31, 34, 38, 39, 41, 49, 56, 58, 61, 66, 87, 89, 90, 91, 94]:
            state.board[index] = Stone.BLACK
        for index in [1, 5, 15, 16, 18, 19, 28, 29, 32, 35, 37, 46, 47, 51, 52, 55, 59, 62, 73, 74, 78, 79, 84, 85]:
            state.board[index] = Stone.WHITE
        state.to_play = Stone.WHITE
        state.board_history = [state.board.copy()]
        state.hash_history = [17066100656971067507, 13294700249936357501]

        self.assertFalse(state.is_legal(Move.place(48)))


if __name__ == "__main__":
    unittest.main()
