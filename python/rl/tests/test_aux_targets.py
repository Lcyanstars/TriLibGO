from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from python.rl.analysis import suggest_dead_groups_from_ownership
from python.rl.config import ExperimentConfig, input_planes_for_history
from python.rl.encoder import LIBERTY_CLASS_COUNT, encode_state, liberty_global_features, liberty_target_classes
from python.rl.game import GameConfig, GameState, Stone
from python.rl import selfplay as selfplay_module
from python.rl.selfplay import TrainingSample, generate_selfplay_game, sampling_temperature


class FixedTerminalModel(torch.nn.Module):
    def __init__(self, action_size: int) -> None:
        super().__init__()
        self.action_size = action_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        logits = torch.full((batch, self.action_size), -10.0, dtype=torch.float32, device=x.device)
        logits[:, self.action_size - 1] = 10.0
        value = torch.zeros(batch, dtype=torch.float32, device=x.device)
        ownership = torch.zeros(batch, self.action_size - 1, dtype=torch.float32, device=x.device)
        score = torch.zeros(batch, dtype=torch.float32, device=x.device)
        liberty_logits = torch.zeros(batch, LIBERTY_CLASS_COUNT, self.action_size - 1, dtype=torch.float32, device=x.device)
        return logits, value, ownership, score, liberty_logits


class AuxiliaryTargetTests(unittest.TestCase):
    def test_liberty_global_features_and_targets_use_current_player_perspective(self) -> None:
        state = GameState(GameConfig(side_length=2))
        state.to_play = Stone.BLACK
        board = [Stone.EMPTY] * state.topology.vertex_count
        board[11] = Stone.BLACK
        board[0] = Stone.WHITE
        board[10] = Stone.WHITE
        board[15] = Stone.WHITE
        state.board = board
        state.board_history = [board.copy()]

        features = liberty_global_features(state)
        targets = liberty_target_classes(state)
        scale = float(state.topology.vertex_count)

        self.assertEqual(features.shape, (8,))
        self.assertAlmostEqual(float(features[0]), 1.0 / scale)
        self.assertEqual(int(targets[11]), 1)
        self.assertAlmostEqual(float(features[4]), 2.0 / scale)
        self.assertEqual(int(targets[0]), 5)
        self.assertEqual(int(targets[10]), 5)
        self.assertAlmostEqual(float(features[5]), 1.0 / scale)
        self.assertEqual(float(features[6]), 1.0)
        self.assertEqual(float(features[7]), 0.0)
        self.assertEqual(int(targets[15]), 6)
        self.assertEqual(int(targets[1]), 0)

        encoded = encode_state(state, history=2)
        self.assertEqual(encoded.shape[0], input_planes_for_history(2))
        self.assertEqual(encoded.shape[1], state.topology.vertex_count)

    def test_liberty_count_treats_shared_eye_as_one_liberty(self) -> None:
        state = GameState(GameConfig(side_length=2))
        state.to_play = Stone.BLACK
        eye = 11
        board = [Stone.BLACK] * state.topology.vertex_count
        board[eye] = Stone.EMPTY
        state.board = board
        state.board_history = [board.copy()]

        group, liberties = state.collect_group_with_liberties(0)
        features = liberty_global_features(state)
        targets = liberty_target_classes(state)
        expected_count_plane_value = 1.0 / float(state.topology.vertex_count)

        self.assertEqual(len(group), state.topology.vertex_count - 1)
        self.assertEqual(liberties, {eye})
        self.assertAlmostEqual(float(features[0]), expected_count_plane_value)
        self.assertAlmostEqual(float(features[1:6].sum()), 0.0)
        self.assertEqual(float(features[6]), 1.0)
        self.assertEqual(float(features[7]), 0.0)
        self.assertEqual(int(targets[0]), 1)
        self.assertEqual(int(targets[eye]), 0)

    def test_finished_state_ownership_map_marks_territory(self) -> None:
        state = GameState(GameConfig(side_length=2, komi=0.5, cleanup_dead_stones=True))
        board = [Stone.EMPTY] * state.topology.vertex_count
        board[11] = Stone.WHITE
        for index in (1, 5, 10, 14):
            board[index] = Stone.BLACK
        state.board = board
        state.board_history = [board.copy()]
        state.finished = True
        state.end_reason = "double_pass"
        state.score()

        ownership = state.ownership_map()

        self.assertEqual(state.board[11], Stone.EMPTY)
        self.assertTrue(np.all(ownership >= 0.0))
        self.assertEqual(float(ownership[11]), 1.0)

    def test_ownership_suggests_dead_group_for_surrounded_stones(self) -> None:
        state = GameState(GameConfig(side_length=2))
        board = [Stone.EMPTY] * state.topology.vertex_count
        board[11] = Stone.WHITE
        for index in (1, 5, 10, 14):
            board[index] = Stone.BLACK
        state.board = board
        ownership = np.zeros(state.topology.vertex_count, dtype=np.float32)
        ownership[11] = 0.95

        suggestions = suggest_dead_groups_from_ownership(state, ownership, threshold=0.75, max_liberties=3)

        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0].color, "W")
        self.assertIn("B3", suggestions[0].labels)

    def test_selfplay_samples_include_auxiliary_targets(self) -> None:
        cfg = ExperimentConfig()
        cfg.model.board_side = 2
        cfg.model.input_history = 2
        cfg.model.action_size = 25
        cfg.selfplay.max_moves = 4
        cfg.mcts.simulations = 1
        cfg.mcts.root_exploration_fraction = 0.0
        cfg.rules.opening_no_pass_moves = 0
        cfg.rules.cleanup_dead_stones = True
        cfg.rules.komi = 0.5
        cfg.telemetry.top_policy_moves = 4

        samples, trace = generate_selfplay_game(FixedTerminalModel(cfg.model.action_size), cfg, torch.device("cpu"))

        self.assertGreater(len(samples), 0)
        first = samples[0]
        self.assertIsInstance(first, TrainingSample)
        self.assertEqual(first.ownership_target.shape[0], cfg.model.action_size - 1)
        self.assertEqual(first.state_planes.shape[0], input_planes_for_history(cfg.model.input_history))
        self.assertEqual(first.liberty_target.shape[0], cfg.model.action_size - 1)
        self.assertIsInstance(first.score_target, float)
        self.assertAlmostEqual(first.curriculum_value_bonus, 0.0, places=6)
        self.assertFalse(first.is_curriculum_sample)
        self.assertEqual(trace.total_passes, 2)
        self.assertEqual(trace.cleanup_rule_resolved_groups, 0)
        self.assertEqual(trace.cleanup_local_search_resolved_groups, 0)
        self.assertEqual(trace.cleanup_preserved_seki_groups, 0)
        self.assertEqual(trace.unresolved_dead_groups, 0)
        self.assertAlmostEqual(trace.avg_abs_predicted_score_error, 0.5, places=6)
        self.assertAlmostEqual(trace.total_curriculum_value_bonus, 0.0, places=6)
        self.assertTrue(all(abs(move.root_score_margin_black_minus_white) < 1e-6 for move in trace.moves))
        self.assertTrue(all(abs(move.score_margin_error_black_minus_white - 0.5) < 1e-6 for move in trace.moves))
        self.assertEqual(trace.abnormal_tags, ["early_pass", "short_game"])
        self.assertAlmostEqual(trace.sample_weight, cfg.selfplay.abnormal_sample_weight)
        self.assertTrue(all(abs(sample.sample_weight - cfg.selfplay.abnormal_sample_weight) < 1e-6 for sample in samples))

    def test_sampling_temperature_anneals_after_opening(self) -> None:
        cfg = ExperimentConfig()
        cfg.mcts.temperature = 1.0
        cfg.mcts.temperature_opening_moves = 4
        cfg.selfplay.sampling_final_temperature = 0.25
        cfg.selfplay.sampling_decay_moves = 4

        self.assertAlmostEqual(sampling_temperature(0, cfg), 1.0)
        self.assertAlmostEqual(sampling_temperature(3, cfg), 1.0)
        self.assertAlmostEqual(sampling_temperature(4, cfg), 0.8125)
        self.assertAlmostEqual(sampling_temperature(7, cfg), 0.25)
        self.assertAlmostEqual(sampling_temperature(20, cfg), 0.25)

    def test_generate_selfplay_game_uses_curriculum_strength_for_selected_move_stats(self) -> None:
        cfg = ExperimentConfig()
        cfg.model.board_side = 2
        cfg.model.input_history = 2
        cfg.model.action_size = 25
        cfg.selfplay.max_moves = 1
        cfg.mcts.simulations = 1
        cfg.mcts.root_exploration_fraction = 0.0
        cfg.mcts.pass_prior_scale = 1.0
        cfg.rules.opening_no_pass_moves = 99
        cfg.curriculum.enabled = True
        cfg.curriculum.start_iteration = 1
        cfg.curriculum.full_strength_until_iteration = 1
        cfg.curriculum.end_iteration = 1

        strengths: list[float] = []
        original_curriculum_move_stats = selfplay_module.curriculum_move_stats

        def spy_curriculum_move_stats(state, action, cfg_arg, strength, **kwargs):
            strengths.append(float(strength))
            return original_curriculum_move_stats(state, action, cfg_arg, strength, **kwargs)

        with patch(
            "python.rl.selfplay.apply_curriculum_policy_shaping",
            side_effect=lambda state, policy, cfg_arg, strength, legal_actions=None: (policy, {}),
        ), patch("python.rl.selfplay.curriculum_move_stats", side_effect=spy_curriculum_move_stats):
            generate_selfplay_game(FixedTerminalModel(cfg.model.action_size), cfg, torch.device("cpu"), iteration=1)

        self.assertEqual(strengths, [1.0])


if __name__ == "__main__":
    unittest.main()
