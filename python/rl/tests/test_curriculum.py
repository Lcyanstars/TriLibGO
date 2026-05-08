from __future__ import annotations

import unittest

import numpy as np

from python.rl.config import ExperimentConfig, action_size_for_side
from python.rl.game import GameConfig, GameState, Stone
from python.rl.selfplay import (
    CurriculumMoveStats,
    SelfPlayMoveTrace,
    _backfill_next_move_capture_signal,
    _curriculum_value_bonus,
    _curriculum_sample_weight,
    _shaped_value_target,
    _small_true_eye_cache,
    _small_true_eye_owner_cache,
    apply_curriculum_policy_shaping,
    curriculum_move_stats,
    curriculum_strength,
    eye_fill_bad_action_mask,
)


class CurriculumTests(unittest.TestCase):
    def _make_cfg(self) -> ExperimentConfig:
        cfg = ExperimentConfig()
        cfg.model.board_side = 2
        cfg.model.action_size = action_size_for_side(cfg.model.board_side)
        cfg.curriculum.enabled = True
        cfg.curriculum.start_iteration = 3
        cfg.curriculum.full_strength_until_iteration = 4
        cfg.curriculum.end_iteration = 6
        return cfg

    def test_curriculum_strength_respects_schedule(self) -> None:
        cfg = self._make_cfg()

        self.assertEqual(curriculum_strength(cfg, 1), 0.0)
        self.assertEqual(curriculum_strength(cfg, 3), 1.0)
        self.assertEqual(curriculum_strength(cfg, 4), 1.0)
        self.assertAlmostEqual(curriculum_strength(cfg, 5), 1.0)
        self.assertAlmostEqual(curriculum_strength(cfg, 6), 0.5)
        self.assertEqual(curriculum_strength(cfg, 7), 0.0)

    def test_capture_count_for_move_counts_captured_group(self) -> None:
        state = GameState(GameConfig(side_length=2))
        state.board[11] = Stone.WHITE
        state.board[0] = Stone.BLACK
        state.board[10] = Stone.BLACK

        self.assertEqual(state.capture_count_for_move(14, player=Stone.BLACK), 1)
        self.assertEqual(state.capture_count_for_move(14, player=Stone.WHITE), 0)

    def test_small_true_eye_cache_marks_single_point_eye(self) -> None:
        state = GameState(GameConfig(side_length=2))
        for vertex in (0, 5, 10, 12, 13, 14, 19, 20, 21):
            state.board[vertex] = Stone.BLACK

        cache = _small_true_eye_cache(state)

        self.assertTrue(cache[11])

    def test_small_true_eye_cache_marks_two_point_eye(self) -> None:
        state = GameState(GameConfig(side_length=2))
        for vertex in (0, 1, 2, 5):
            state.board[vertex] = Stone.BLACK

        cache = _small_true_eye_cache(state)

        self.assertTrue(cache[3])
        self.assertTrue(cache[4])

    def test_small_true_eye_cache_rejects_false_eye_when_border_groups_are_disconnected(self) -> None:
        state = GameState(GameConfig(side_length=2))
        state.board[2] = Stone.WHITE
        state.board[5] = Stone.WHITE

        cache = _small_true_eye_cache(state)

        self.assertFalse(cache[3])
        self.assertFalse(cache[4])

    def test_opponent_true_eye_is_not_penalized_as_self_eye_fill(self) -> None:
        state = GameState(GameConfig(side_length=2))
        for vertex in (0, 5, 10, 12, 13, 14, 19, 20, 21):
            state.board[vertex] = Stone.WHITE
        state.to_play = Stone.BLACK

        cache = _small_true_eye_cache(state)
        owner_cache = _small_true_eye_owner_cache(state)
        stats = curriculum_move_stats(state, 11, self._make_cfg(), strength=1.0, true_eye_cache=cache, true_eye_owner_cache=owner_cache)

        self.assertTrue(cache[11])
        self.assertEqual(owner_cache[11], Stone.WHITE)
        self.assertFalse(stats.fills_small_true_eye)

    def test_curriculum_move_stats_detects_true_eye_without_precomputed_cache(self) -> None:
        state = GameState(GameConfig(side_length=2))
        for vertex in (0, 5, 10, 12, 13, 14, 19, 20, 21):
            state.board[vertex] = Stone.BLACK
        state.to_play = Stone.BLACK

        stats = curriculum_move_stats(state, 11, self._make_cfg(), strength=1.0)

        self.assertTrue(stats.fills_small_true_eye)

    def test_small_true_eye_cache_rejects_region_fill_that_captures(self) -> None:
        state = GameState(GameConfig(side_length=2))
        state.board[0] = Stone.BLACK
        state.board[10] = Stone.BLACK
        state.board[14] = Stone.BLACK
        state.board[1] = Stone.WHITE
        state.board[5] = Stone.WHITE

        cache = _small_true_eye_cache(state)

        self.assertFalse(cache[11])

    def test_policy_shaping_boosts_capture_and_penalizes_eye_fill(self) -> None:
        cfg = self._make_cfg()
        cfg.curriculum.selfplay_capture_bonus = 2.0
        cfg.curriculum.selfplay_capture_bonus_per_stone = 0.0
        cfg.curriculum.selfplay_capture_bonus_cap = 2.0
        cfg.curriculum.selfplay_true_eye_penalty = 0.25

        state = GameState(GameConfig(side_length=2))
        state.board[2] = Stone.BLACK
        state.board[1] = Stone.BLACK
        state.board[0] = Stone.BLACK
        state.board[5] = Stone.BLACK
        state.board[9] = Stone.WHITE
        policy = np.zeros(cfg.model.action_size, dtype=np.float32)
        legal_moves = state.legal_moves()
        legal_actions = [cfg.model.action_size - 1 if move.kind == "pass" else move.index for move in legal_moves]
        policy[legal_actions] = 1.0 / len(legal_actions)

        adjusted, stats = apply_curriculum_policy_shaping(state, policy, cfg, strength=1.0)

        self.assertGreater(float(adjusted[8]), float(policy[8]))
        self.assertLess(float(adjusted[3]), float(policy[3]))
        self.assertLess(float(adjusted[4]), float(policy[4]))
        self.assertEqual(stats[8].captures_by_move, 1)
        self.assertTrue(stats[3].fills_small_true_eye)
        self.assertTrue(stats[4].fills_small_true_eye)

    def test_policy_shaping_returns_move_stats_even_when_disabled(self) -> None:
        cfg = self._make_cfg()

        state = GameState(GameConfig(side_length=2))
        state.board[2] = Stone.BLACK
        state.board[1] = Stone.BLACK
        state.board[0] = Stone.BLACK
        state.board[5] = Stone.BLACK
        state.board[9] = Stone.WHITE
        policy = np.zeros(cfg.model.action_size, dtype=np.float32)
        legal_moves = state.legal_moves()
        legal_actions = [cfg.model.action_size - 1 if move.kind == "pass" else move.index for move in legal_moves]
        policy[legal_actions] = 1.0 / len(legal_actions)

        adjusted, stats = apply_curriculum_policy_shaping(state, policy, cfg, strength=0.0)

        self.assertTrue(np.allclose(adjusted, policy))
        self.assertIn(8, stats)
        self.assertTrue(stats[3].fills_small_true_eye)
        self.assertTrue(stats[4].fills_small_true_eye)

    def test_eye_fill_bad_action_mask_marks_non_capture_self_eye_fills(self) -> None:
        cfg = self._make_cfg()
        state = GameState(GameConfig(side_length=2))
        state.board[2] = Stone.BLACK
        state.board[1] = Stone.BLACK
        state.board[0] = Stone.BLACK
        state.board[5] = Stone.BLACK
        state.board[9] = Stone.WHITE
        state.to_play = Stone.BLACK

        mask = eye_fill_bad_action_mask(state, cfg)

        self.assertEqual(float(mask[3]), 1.0)
        self.assertEqual(float(mask[4]), 1.0)
        self.assertEqual(float(mask[8]), 0.0)
        self.assertEqual(float(mask[-1]), 0.0)

    def test_policy_shaping_penalizes_static_immediate_capture_risk(self) -> None:
        cfg = self._make_cfg()
        cfg.curriculum.sample_immediate_capture_risk_weight = 1.5
        cfg.curriculum.sample_immediate_capture_risk_per_stone = 0.0
        cfg.curriculum.sample_immediate_capture_risk_cap = 1.5

        state = GameState(GameConfig(side_length=2, opening_no_pass_moves=0))
        state.board[0] = Stone.WHITE
        state.board[10] = Stone.WHITE
        policy = np.zeros(cfg.model.action_size, dtype=np.float32)
        legal_moves = state.legal_moves()
        legal_actions = [cfg.model.action_size - 1 if move.kind == "pass" else move.index for move in legal_moves]
        policy[legal_actions] = 1.0 / len(legal_actions)

        adjusted, stats = apply_curriculum_policy_shaping(state, policy, cfg, strength=1.0)

        self.assertEqual(stats[11].immediate_capture_risk_stones, 1)
        self.assertEqual(stats[4].immediate_capture_risk_stones, 0)
        self.assertLess(float(adjusted[11]), float(policy[11]))
        self.assertGreater(float(adjusted[4]), float(adjusted[11]))

    def test_backfill_next_move_capture_updates_previous_weight(self) -> None:
        cfg = self._make_cfg()
        cfg.curriculum.sample_immediate_capture_risk_weight = 1.25
        cfg.curriculum.sample_immediate_capture_risk_per_stone = 0.25
        cfg.curriculum.sample_immediate_capture_risk_cap = 2.0
        previous = CurriculumMoveStats(captures_by_move=1, curriculum_weight=1.5)
        trace = SelfPlayMoveTrace(
            turn=1,
            player="B",
            action=0,
            move="A1",
            root_value=0.0,
            root_score_margin=0.0,
            root_score_margin_black_minus_white=0.0,
            score_margin_error_black_minus_white=0.0,
            policy=[],
            policy_top=[],
            captures_by_move=1,
            curriculum_weight=1.5,
        )

        _backfill_next_move_capture_signal(previous, trace, 2, cfg, strength=1.0)

        self.assertEqual(previous.next_move_capture_stones, 2)
        self.assertEqual(trace.next_move_capture_stones, 2)
        self.assertEqual(previous.curriculum_weight, 2.25)
        self.assertEqual(trace.curriculum_weight, 2.25)
        self.assertAlmostEqual(previous.curriculum_value_bonus, -0.10, places=6)
        self.assertAlmostEqual(trace.curriculum_value_bonus, -0.10, places=6)

    def test_sample_weight_applies_capture_risk_and_penalty_with_clamp(self) -> None:
        cfg = self._make_cfg()
        cfg.curriculum.sample_capture_weight = 1.5
        cfg.curriculum.sample_capture_weight_per_stone = 0.25
        cfg.curriculum.sample_capture_weight_cap = 2.0
        cfg.curriculum.sample_immediate_capture_risk_weight = 1.25
        cfg.curriculum.sample_immediate_capture_risk_per_stone = 0.25
        cfg.curriculum.sample_immediate_capture_risk_cap = 2.0
        cfg.curriculum.sample_true_eye_penalty_weight = 0.1
        cfg.curriculum.sample_weight_min = 0.2
        cfg.curriculum.sample_weight_max = 2.0

        boosted = _curriculum_sample_weight(
            cfg,
            CurriculumMoveStats(captures_by_move=3, next_move_capture_stones=2, fills_small_true_eye=False),
            strength=1.0,
        )
        penalized = _curriculum_sample_weight(
            cfg,
            CurriculumMoveStats(captures_by_move=0, next_move_capture_stones=0, fills_small_true_eye=True),
            strength=1.0,
        )

        self.assertEqual(boosted, 2.0)
        self.assertEqual(penalized, 0.2)

    def test_curriculum_value_bonus_rewards_capture_and_penalizes_blunder(self) -> None:
        cfg = self._make_cfg()
        bonus = _curriculum_value_bonus(
            cfg,
            CurriculumMoveStats(captures_by_move=2, next_move_capture_stones=1),
            strength=1.0,
        )

        self.assertAlmostEqual(bonus, 0.0, places=6)

    def test_shaped_value_target_ignores_curriculum_bonus(self) -> None:
        self.assertAlmostEqual(_shaped_value_target(0.95, 0.20), 0.95, places=6)
        self.assertAlmostEqual(_shaped_value_target(-0.95, -0.20), -0.95, places=6)


if __name__ == "__main__":
    unittest.main()
