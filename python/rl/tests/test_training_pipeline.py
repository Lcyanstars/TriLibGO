from __future__ import annotations

import json
import io
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from torch import optim

from python.rl.config import (
    ExperimentConfig,
    action_size_for_side,
    apply_board_side,
    effective_training_steps,
    input_planes_for_history,
    load_experiment_config,
)
from python.rl.model import PolicyValueNet
from python.rl.replay_buffer import ReplayBuffer
from python.rl.selfplay import SelfPlayGameTrace, SelfPlayMoveTrace, TrainingSample, terminal_supervision_weight
from python.rl.train import (
    apply_komi_adjustment,
    checkpoint_iteration,
    compute_learning_rate,
    curriculum_metrics_satisfied,
    curriculum_should_stop_for_metrics,
    eligible_komi_margins,
    evaluation_plan,
    iteration_selfplay_path,
    load_recent_selfplay_records,
    load_resume_state,
    print_iteration_summary,
    prune_checkpoints,
    resolved_evaluation_interval,
    reset_selfplay_logs,
    run_training,
    save_checkpoint,
    serialize_game_trace,
    should_adjust_komi,
    summarize_komi_adjustment,
    summarize_selfplay_games,
    train_iteration,
    trim_jsonl_to_iteration,
    trim_selfplay_logs,
)


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_root = Path("artifacts/test_tmp/training_pipeline")
        if self._tmp_root.exists():
            shutil.rmtree(self._tmp_root)
        self._tmp_root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(self._cleanup_tmp_root)

    def _cleanup_tmp_root(self) -> None:
        if self._tmp_root.exists():
            shutil.rmtree(self._tmp_root)

    def _make_cfg(self) -> ExperimentConfig:
        cfg = ExperimentConfig()
        cfg.model.board_side = 2
        cfg.model.input_history = 2
        cfg.model.channels = 8
        cfg.model.residual_blocks = 1
        cfg.model.policy_head_channels = 4
        cfg.model.value_head_channels = 4
        cfg.model.ownership_head_channels = 4
        cfg.model.score_head_channels = 4
        cfg.model.action_size = action_size_for_side(cfg.model.board_side)
        cfg.selfplay.replay_window = 32
        cfg.training.batch_size = 2
        cfg.training.gradient_clip_norm = 0.01
        return cfg

    def _make_sample(
        self,
        cfg: ExperimentConfig,
        action: int,
        weight: float,
        value: float,
        score: float,
        *,
        policy_weight: float | None = None,
        terminal_weight: float | None = None,
        curriculum_value_bonus: float = 0.0,
        next_move_capture_stones: int = 0,
        is_curriculum_sample: bool = False,
    ) -> TrainingSample:
        state_planes = np.zeros(
            (input_planes_for_history(cfg.model.input_history), cfg.model.action_size - 1),
            dtype=np.float32,
        )
        policy_target = np.zeros(cfg.model.action_size, dtype=np.float32)
        policy_target[action] = 1.0
        ownership_target = np.zeros(cfg.model.action_size - 1, dtype=np.float32)
        ownership_target[action % ownership_target.shape[0]] = 1.0
        return TrainingSample(
            state_planes=state_planes,
            policy_target=policy_target,
            value_target=value,
            ownership_target=ownership_target,
            score_target=score,
            sample_weight=weight,
            policy_weight=weight if policy_weight is None else policy_weight,
            terminal_weight=weight if terminal_weight is None else terminal_weight,
            curriculum_value_bonus=curriculum_value_bonus,
            next_move_capture_stones=next_move_capture_stones,
            curriculum_scale_at_generation=1.0 if is_curriculum_sample else 0.0,
            is_curriculum_sample=is_curriculum_sample,
        )

    def _make_trace(
        self,
        *,
        black_score: float,
        white_score: float,
        abnormal_tags: list[str] | None = None,
        unresolved_dead_groups: int = 0,
        end_reason: str = "double_pass",
    ) -> SelfPlayGameTrace:
        abnormal_tags = abnormal_tags or []
        return SelfPlayGameTrace(
            komi=3.0,
            move_count=64,
            black_score=black_score,
            white_score=white_score,
            result_value=1.0 if black_score > white_score else (-1.0 if black_score < white_score else 0.0),
            winner="B" if black_score > white_score else ("W" if black_score < white_score else "draw"),
            first_player_win=black_score > white_score,
            first_pass_turn=60,
            total_passes=2,
            cleaned_dead_stones=0,
            cleanup_rule_resolved_groups=0,
            cleanup_local_search_resolved_groups=0,
            cleanup_preserved_seki_groups=0,
            unresolved_dead_groups=unresolved_dead_groups,
            end_reason=end_reason,
            avg_abs_predicted_score_error=0.0,
            sample_weight=1.0,
            abnormal_tags=abnormal_tags,
            moves=[],
        )

    def test_effective_training_steps_respects_minimum_floor(self) -> None:
        self.assertEqual(effective_training_steps(64, 64, 0, 2, 128), 128)
        self.assertEqual(effective_training_steps(5000, 64, 0, 2, 128), 158)
        self.assertEqual(effective_training_steps(5000, 64, 96, 2, 128), 96)

    def test_compute_learning_rate_supports_warmup_and_cosine_decay(self) -> None:
        base = 1e-3
        self.assertAlmostEqual(compute_learning_rate(base, 0, 4, 12, 0.1, "cosine"), 0.00025)
        self.assertAlmostEqual(compute_learning_rate(base, 3, 4, 12, 0.1, "cosine"), 0.001)
        self.assertAlmostEqual(compute_learning_rate(base, 11, 4, 12, 0.1, "cosine"), 0.0001, places=7)
        self.assertAlmostEqual(compute_learning_rate(base, 20, 0, 0, 0.1, "constant"), 0.001)

    def test_should_adjust_komi_respects_warmup_and_interval(self) -> None:
        cfg = self._make_cfg()
        cfg.rules.auto_adjust_komi = True
        cfg.rules.komi_adjust_interval = 4
        cfg.rules.komi_adjust_warmup_iterations = 8

        self.assertFalse(should_adjust_komi(cfg, 4))
        self.assertFalse(should_adjust_komi(cfg, 7))
        self.assertTrue(should_adjust_komi(cfg, 8))
        self.assertFalse(should_adjust_komi(cfg, 10))
        self.assertTrue(should_adjust_komi(cfg, 12))

    def test_komi_adjustment_ignores_large_margin_and_abnormal_games(self) -> None:
        cfg = self._make_cfg()
        cfg.rules.auto_adjust_komi = True
        cfg.rules.komi = 3.0
        cfg.rules.selfplay_komi = [2.5, 3.0, 3.5]
        cfg.rules.komi_adjust_min_samples = 3
        cfg.rules.komi_adjust_max_margin = 3.0
        cfg.rules.komi_adjust_alpha = 0.25
        cfg.rules.komi_adjust_delta_max = 0.5

        traces = [
            self._make_trace(black_score=6.0, white_score=5.0),
            self._make_trace(black_score=5.5, white_score=4.0),
            self._make_trace(black_score=5.0, white_score=3.0),
            self._make_trace(black_score=8.0, white_score=4.0),
            self._make_trace(black_score=5.0, white_score=4.0, abnormal_tags=["early_pass"]),
            self._make_trace(black_score=5.0, white_score=4.0, unresolved_dead_groups=1),
            self._make_trace(black_score=5.0, white_score=4.0, end_reason="max_moves"),
        ]

        eligible = eligible_komi_margins(traces, cfg)
        summary = summarize_komi_adjustment(traces, cfg)

        self.assertEqual(eligible, [1.0, 1.5, 2.0])
        self.assertTrue(summary.checked)
        self.assertTrue(summary.applied)
        self.assertEqual(summary.eligible_games, 3)
        self.assertAlmostEqual(summary.weighted_margin or 0.0, 1.388888888888889, places=6)
        self.assertAlmostEqual(summary.delta, 0.34722222222222227, places=6)
        self.assertAlmostEqual(summary.next_komi, 3.3472222222222223, places=6)

        apply_komi_adjustment(cfg, summary)

        self.assertAlmostEqual(cfg.rules.komi, 3.3472222222222223, places=6)
        self.assertAlmostEqual(cfg.rules.selfplay_komi[0], 2.8472222222222223, places=6)
        self.assertAlmostEqual(cfg.rules.selfplay_komi[1], 3.3472222222222223, places=6)
        self.assertAlmostEqual(cfg.rules.selfplay_komi[2], 3.8472222222222223, places=6)

    def test_komi_adjustment_waits_for_enough_samples(self) -> None:
        cfg = self._make_cfg()
        cfg.rules.auto_adjust_komi = True
        cfg.rules.komi = 3.0
        cfg.rules.komi_adjust_min_samples = 4
        cfg.rules.komi_adjust_max_margin = 3.0

        traces = [
            self._make_trace(black_score=6.0, white_score=5.0),
            self._make_trace(black_score=5.5, white_score=4.5),
            self._make_trace(black_score=5.0, white_score=4.0),
        ]

        summary = summarize_komi_adjustment(traces, cfg)

        self.assertTrue(summary.checked)
        self.assertFalse(summary.applied)
        self.assertEqual(summary.reason, "insufficient_samples")
        self.assertEqual(summary.eligible_games, 3)
        self.assertAlmostEqual(summary.next_komi, 3.0)

    def test_komi_adjustment_allows_wider_margin_window(self) -> None:
        cfg = self._make_cfg()
        cfg.rules.komi_adjust_max_margin = 10.0

        traces = [
            self._make_trace(black_score=9.0, white_score=3.0),
            self._make_trace(black_score=6.5, white_score=2.5),
            self._make_trace(black_score=5.0, white_score=4.0),
        ]

        eligible = eligible_komi_margins(traces, cfg)

        self.assertEqual(eligible, [6.0, 4.0, 1.0])

    def test_komi_adjustment_ignores_max_move_games(self) -> None:
        cfg = self._make_cfg()
        cfg.rules.komi = 3.0
        cfg.rules.komi_adjust_min_samples = 3
        cfg.rules.komi_adjust_max_margin = 10.0
        cfg.rules.komi_adjust_alpha = 0.25
        cfg.rules.komi_adjust_delta_max = 0.5

        traces = [
            self._make_trace(black_score=100.0, white_score=3.0, end_reason="max_moves"),
            self._make_trace(black_score=90.0, white_score=3.0, end_reason="max_moves"),
            self._make_trace(black_score=80.0, white_score=3.0, end_reason="max_moves"),
        ]

        eligible = eligible_komi_margins(traces, cfg)
        summary = summarize_komi_adjustment(traces, cfg)

        self.assertEqual(eligible, [])
        self.assertFalse(summary.applied)
        self.assertEqual(summary.reason, "insufficient_samples")
        self.assertIsNone(summary.weighted_margin)
        self.assertAlmostEqual(summary.delta, 0.0, places=6)
        self.assertAlmostEqual(summary.next_komi, 3.0, places=6)

    def test_evaluation_plan_skips_unscheduled_iterations_and_keeps_full_override(self) -> None:
        cfg = self._make_cfg()
        cfg.mcts.simulations = 24
        cfg.evaluation.interval = 6
        cfg.evaluation.interval_schedule = [
            {"until_iteration": 16, "interval": 16},
            {"until_iteration": 32, "interval": 8},
        ]
        cfg.evaluation.games = 6
        cfg.evaluation.simulations = 24
        cfg.evaluation.full_games_every = 12
        cfg.evaluation.full_games = 9
        cfg.evaluation.full_simulations = 20

        skipped = evaluation_plan(cfg, 3)
        early = evaluation_plan(cfg, 16)
        middle = evaluation_plan(cfg, 24)
        late_skipped = evaluation_plan(cfg, 33)
        late = evaluation_plan(cfg, 36)
        full = evaluation_plan(cfg, 12)

        self.assertFalse(skipped.run)
        self.assertEqual(skipped.mode, "skipped")
        self.assertEqual(skipped.games, 0)
        self.assertEqual(skipped.interval, 16)
        self.assertTrue(early.run)
        self.assertEqual(early.mode, "fast")
        self.assertEqual(early.games, 6)
        self.assertEqual(early.simulations, 24)
        self.assertEqual(early.interval, 16)
        self.assertTrue(middle.run)
        self.assertEqual(middle.interval, 8)
        self.assertFalse(late_skipped.run)
        self.assertEqual(late_skipped.interval, 6)
        self.assertTrue(late.run)
        self.assertEqual(late.interval, 6)
        self.assertTrue(full.run)
        self.assertEqual(full.mode, "full")
        self.assertEqual(full.games, 9)
        self.assertEqual(full.simulations, 20)
        self.assertEqual(full.interval, 16)

    def test_resolved_evaluation_interval_uses_schedule_then_falls_back(self) -> None:
        cfg = self._make_cfg()
        cfg.evaluation.interval = 6
        cfg.evaluation.interval_schedule = [
            {"until_iteration": 16, "interval": 16},
            {"until_iteration": 32, "interval": 8},
        ]

        self.assertEqual(resolved_evaluation_interval(cfg, 1), 16)
        self.assertEqual(resolved_evaluation_interval(cfg, 16), 16)
        self.assertEqual(resolved_evaluation_interval(cfg, 17), 8)
        self.assertEqual(resolved_evaluation_interval(cfg, 32), 8)
        self.assertEqual(resolved_evaluation_interval(cfg, 33), 6)

    def test_train_iteration_reports_weighted_batch_and_clipped_gradients(self) -> None:
        torch.manual_seed(7)
        cfg = self._make_cfg()
        model = PolicyValueNet(cfg.model)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        replay = ReplayBuffer(cfg.selfplay.replay_window)
        replay.extend(
            [
                self._make_sample(cfg, action=0, weight=1.0, value=1.0, score=8.0),
                self._make_sample(cfg, action=1, weight=0.25, value=-1.0, score=-8.0),
            ]
        )

        metrics = train_iteration(model, optimizer, replay, cfg, torch.device("cpu"))

        self.assertAlmostEqual(metrics["effective_batch_size"], 1.25, places=6)
        self.assertAlmostEqual(metrics["avg_batch_weight"], 0.625, places=6)
        self.assertGreaterEqual(metrics["policy_entropy"], 0.0)
        self.assertEqual(metrics["entropy_bonus"], 0.0)
        self.assertGreater(metrics["grad_norm"], 0.0)
        self.assertLessEqual(metrics["clipped_grad_norm"], cfg.training.gradient_clip_norm + 1e-6)
        self.assertGreaterEqual(metrics["grad_norm"], metrics["clipped_grad_norm"])

    def test_policy_entropy_weight_reduces_total_loss(self) -> None:
        torch.manual_seed(7)
        base_cfg = self._make_cfg()
        base_model = PolicyValueNet(base_cfg.model)
        base_state = {key: value.clone() for key, value in base_model.state_dict().items()}

        def run_with_entropy(weight: float) -> dict[str, float]:
            cfg = self._make_cfg()
            cfg.training.policy_entropy_weight = weight
            model = PolicyValueNet(cfg.model)
            model.load_state_dict(base_state)
            optimizer = optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0)
            replay = ReplayBuffer(cfg.selfplay.replay_window)
            first = self._make_sample(cfg, action=0, weight=1.0, value=0.0, score=0.0)
            second = self._make_sample(cfg, action=1, weight=1.0, value=0.0, score=0.0)
            first.policy_target[1] = 0.5
            first.policy_target[0] = 0.5
            second.policy_target[0] = 0.5
            second.policy_target[1] = 0.5
            replay.extend([first, second])
            return train_iteration(model, optimizer, replay, cfg, torch.device("cpu"))

        disabled = run_with_entropy(0.0)
        enabled = run_with_entropy(0.01)

        self.assertGreater(enabled["policy_entropy"], 0.0)
        self.assertGreater(enabled["entropy_bonus"], 0.0)
        self.assertAlmostEqual(disabled["policy_loss"], enabled["policy_loss"], places=6)
        self.assertLess(enabled["total_loss"], disabled["total_loss"])

    def test_policy_entropy_uses_policy_target_support(self) -> None:
        class FixedLogitModel(torch.nn.Module):
            def __init__(self, action_size: int, vertex_count: int) -> None:
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(1))
                logits = torch.full((action_size,), -20.0, dtype=torch.float32)
                logits[0] = 0.0
                logits[1] = 0.0
                logits[2] = 20.0
                self.register_buffer("logits", logits)
                self.vertex_count = vertex_count

            def forward(self, states):
                batch = states.shape[0]
                logits = self.logits.unsqueeze(0).repeat(batch, 1) + self.bias
                value = torch.zeros(batch, dtype=torch.float32, device=states.device) + self.bias
                ownership = torch.zeros(batch, self.vertex_count, dtype=torch.float32, device=states.device) + self.bias
                score = torch.zeros(batch, dtype=torch.float32, device=states.device) + self.bias
                return logits, value, ownership, score

        cfg = self._make_cfg()
        cfg.training.policy_entropy_weight = 0.01
        sample = self._make_sample(cfg, action=0, weight=1.0, value=0.0, score=0.0)
        sample.policy_target[:] = 0.0
        sample.policy_target[0] = 0.5
        sample.policy_target[1] = 0.5
        replay = ReplayBuffer(cfg.selfplay.replay_window)
        replay.extend([sample])
        model = FixedLogitModel(cfg.model.action_size, cfg.model.action_size - 1)
        optimizer = optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0)

        metrics = train_iteration(model, optimizer, replay, cfg, torch.device("cpu"))

        self.assertAlmostEqual(metrics["policy_entropy"], float(np.log(2.0)), places=5)
        self.assertAlmostEqual(metrics["entropy_bonus"], 0.01 * float(np.log(2.0)), places=5)

    def test_eye_fill_loss_penalizes_bad_action_probability_without_changing_policy_loss(self) -> None:
        class FixedLogitModel(torch.nn.Module):
            def __init__(self, action_size: int, vertex_count: int) -> None:
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(1))
                self.action_size = action_size
                self.vertex_count = vertex_count

            def forward(self, states):
                batch = states.shape[0]
                logits = torch.zeros(batch, self.action_size, dtype=torch.float32, device=states.device) + self.bias
                value = torch.zeros(batch, dtype=torch.float32, device=states.device) + self.bias
                ownership = torch.zeros(batch, self.vertex_count, dtype=torch.float32, device=states.device) + self.bias
                score = torch.zeros(batch, dtype=torch.float32, device=states.device) + self.bias
                return logits, value, ownership, score

        def run(weight: float) -> dict[str, float]:
            cfg = self._make_cfg()
            cfg.training.eye_fill_loss_weight = weight
            sample = self._make_sample(cfg, action=0, weight=1.0, value=0.0, score=0.0)
            sample.eye_fill_bad_action_mask = np.zeros(cfg.model.action_size, dtype=np.float32)
            sample.eye_fill_bad_action_mask[2] = 1.0
            replay = ReplayBuffer(cfg.selfplay.replay_window)
            replay.extend([sample])
            model = FixedLogitModel(cfg.model.action_size, cfg.model.action_size - 1)
            optimizer = optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0)
            return train_iteration(model, optimizer, replay, cfg, torch.device("cpu"))

        disabled = run(0.0)
        enabled = run(0.2)

        self.assertGreater(enabled["eye_fill_loss"], 0.0)
        self.assertAlmostEqual(disabled["policy_loss"], enabled["policy_loss"], places=6)
        self.assertGreater(enabled["total_loss"], disabled["total_loss"])

    def test_train_iteration_separates_policy_and_terminal_weights(self) -> None:
        torch.manual_seed(7)
        cfg = self._make_cfg()
        base_model = PolicyValueNet(cfg.model)
        base_state = {key: value.clone() for key, value in base_model.state_dict().items()}
        state_planes = np.zeros(
            (input_planes_for_history(cfg.model.input_history), cfg.model.action_size - 1),
            dtype=np.float32,
        )
        with torch.no_grad():
            _, value_pred, ownership_pred, score_pred = base_model(torch.tensor(state_planes).unsqueeze(0))[:4]
        first_value_target = float(value_pred.item())
        second_value_target = -1.0 if first_value_target > 0.0 else 1.0
        first_score_target = float(score_pred.item())
        second_score_target = first_score_target + 32.0
        first_ownership_target = ownership_pred.squeeze(0).numpy()
        second_ownership_target = first_ownership_target + 2.0

        def run(second_terminal_weight: float) -> dict[str, float]:
            model = PolicyValueNet(cfg.model)
            model.load_state_dict(base_state)
            optimizer = optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0)
            replay = ReplayBuffer(cfg.selfplay.replay_window)
            first = self._make_sample(cfg, action=0, weight=1.0, value=first_value_target, score=first_score_target, terminal_weight=1.0)
            second = self._make_sample(
                cfg,
                action=1,
                weight=1.0,
                value=second_value_target,
                score=second_score_target,
                terminal_weight=second_terminal_weight,
            )
            first.ownership_target[:] = first_ownership_target
            second.ownership_target[:] = second_ownership_target
            replay.extend([first, second])
            return train_iteration(model, optimizer, replay, cfg, torch.device("cpu"))

        full = run(1.0)
        downweighted = run(0.25)

        self.assertAlmostEqual(full["policy_loss"], downweighted["policy_loss"], places=6)
        self.assertGreater(full["value_loss"], downweighted["value_loss"])
        self.assertGreater(full["ownership_loss"], downweighted["ownership_loss"])
        self.assertGreater(full["score_loss"], downweighted["score_loss"])

    def test_score_loss_scale_reduces_large_margin_dominance(self) -> None:
        torch.manual_seed(7)
        base_cfg = self._make_cfg()
        base_model = PolicyValueNet(base_cfg.model)
        base_state = {key: value.clone() for key, value in base_model.state_dict().items()}

        def run_with_scale(scale: float) -> float:
            cfg = self._make_cfg()
            cfg.training.score_loss_scale = scale
            model = PolicyValueNet(cfg.model)
            model.load_state_dict(base_state)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
            replay = ReplayBuffer(cfg.selfplay.replay_window)
            replay.extend(
                [
                    self._make_sample(cfg, action=0, weight=1.0, value=1.0, score=32.0),
                    self._make_sample(cfg, action=1, weight=1.0, value=-1.0, score=-32.0),
                ]
            )
            return train_iteration(model, optimizer, replay, cfg, torch.device("cpu"))["score_loss"]

        unscaled_loss = run_with_scale(1.0)
        scaled_loss = run_with_scale(16.0)

        self.assertGreater(unscaled_loss, 10.0)
        self.assertLess(scaled_loss, unscaled_loss * 0.2)

    def test_replay_summary_tracks_weighted_samples(self) -> None:
        cfg = self._make_cfg()
        replay = ReplayBuffer(cfg.selfplay.replay_window)
        replay.extend(
            [
                self._make_sample(cfg, action=0, weight=1.0, value=0.0, score=0.0, curriculum_value_bonus=0.10, is_curriculum_sample=True),
                self._make_sample(
                    cfg,
                    action=1,
                    weight=0.5,
                    value=0.0,
                    score=0.0,
                    curriculum_value_bonus=-0.20,
                    next_move_capture_stones=1,
                    is_curriculum_sample=True,
                ),
                self._make_sample(cfg, action=2, weight=0.25, value=0.0, score=0.0),
            ]
        )

        summary = replay.summary()

        self.assertAlmostEqual(summary["buffer_effective_size"], 1.75, places=6)
        self.assertAlmostEqual(summary["buffer_avg_sample_weight"], 1.75 / 3.0, places=6)
        self.assertAlmostEqual(summary["buffer_downweighted_rate"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(summary["buffer_next_move_capture_rate"], 1.0 / 3.0, places=6)

    def test_terminal_supervision_weight_downweights_noisy_endings(self) -> None:
        cfg = self._make_cfg()
        trace = self._make_trace(black_score=28.0, white_score=-20.0, unresolved_dead_groups=1, end_reason="max_moves")
        trace.cleaned_dead_stones = 16
        trace.total_passes = 8

        weight = terminal_supervision_weight(trace, cfg)

        self.assertEqual(weight, 0.25)

    def test_serialize_game_trace_includes_viewer_metadata(self) -> None:
        cfg = self._make_cfg()
        cfg.model.board_side = 4
        cfg.rules.allow_suicide = False
        trace = self._make_trace(black_score=8.0, white_score=6.5)
        trace.curriculum_scale = 0.75
        trace.total_captured_stones = 3
        trace.total_prev_turn_losses = 2
        trace.total_immediate_capture_risk_stones = 4
        trace.total_next_move_capture_stones = 5
        trace.total_curriculum_value_bonus = -0.35
        trace.true_eye_fills = 1
        trace.pure_capture_moves = 2
        trace.immediate_capture_risk_rate_moves = 3
        trace.next_move_capture_rate_moves = 2
        trace.curriculum_sample_moves = 8
        trace.positive_curriculum_bonus_moves = 2
        trace.negative_curriculum_bonus_moves = 3
        trace.true_eye_penalized_moves = 1
        trace.cleaned_dead_vertices = [3, 4]
        trace.final_board = [0, 1, 2]
        trace.moves = [
            SelfPlayMoveTrace(
                turn=1,
                player="B",
                action=3,
                move="A1",
                root_value=0.1,
                root_score_margin=0.2,
                root_score_margin_black_minus_white=0.2,
                score_margin_error_black_minus_white=-0.1,
                policy=[0.2, 0.8],
                policy_top=[(1, 0.8)],
                captures_by_move=1,
                stones_lost_prev_turn=0,
                immediate_capture_risk_stones=2,
                next_move_capture_stones=1,
                curriculum_value_bonus=0.05,
                is_curriculum_sample=True,
                fills_small_true_eye=False,
                curriculum_weight=1.25,
            )
        ]

        payload = serialize_game_trace(3, 2, trace, cfg)

        self.assertEqual(payload["format"], "trilibgo-selfplay-trace-v1")
        self.assertEqual(payload["source"], "selfplay")
        self.assertEqual(payload["side_length"], 4)
        self.assertEqual(payload["allow_suicide"], False)
        self.assertEqual(payload["iteration"], 3)
        self.assertEqual(payload["game_index"], 2)
        self.assertEqual(payload["curriculum_scale"], 0.75)
        self.assertEqual(payload["total_captured_stones"], 3)
        self.assertEqual(payload["total_prev_turn_losses"], 2)
        self.assertEqual(payload["total_immediate_capture_risk_stones"], 4)
        self.assertEqual(payload["total_next_move_capture_stones"], 5)
        self.assertEqual(payload["total_curriculum_value_bonus"], -0.35)
        self.assertEqual(payload["true_eye_fills"], 1)
        self.assertEqual(payload["pure_capture_moves"], 2)
        self.assertEqual(payload["immediate_capture_risk_rate_moves"], 3)
        self.assertEqual(payload["next_move_capture_rate_moves"], 2)
        self.assertEqual(payload["curriculum_sample_moves"], 8)
        self.assertEqual(payload["positive_curriculum_bonus_moves"], 2)
        self.assertEqual(payload["negative_curriculum_bonus_moves"], 3)
        self.assertEqual(payload["true_eye_penalized_moves"], 1)
        self.assertEqual(payload["cleaned_dead_vertices"], [3, 4])
        self.assertEqual(payload["final_board"], [0, 1, 2])
        self.assertEqual(payload["moves"][0]["immediate_capture_risk_stones"], 2)

    def test_print_iteration_summary_emits_multiline_grouped_output(self) -> None:
        cfg = self._make_cfg()
        metrics = {
            "iteration": 4,
            "selfplay_duration_sec": 12.5,
            "training_duration_sec": 0.75,
            "evaluation_duration_sec": 3.2,
            "eval_ran": True,
            "eval_mode": "fast",
            "eval_wins": 4,
            "eval_losses": 2,
            "eval_draws": 0,
            "eval_games": 6,
            "eval_interval": 8,
            "eval_simulations": 48,
            "eval_win_rate": 4 / 6,
            "bootstrap_best": False,
            "promoted": True,
            "best_updated": True,
            "policy_loss": 2.9,
            "policy_entropy": 1.23,
            "eye_fill_loss": 0.04,
            "value_loss": 0.42,
            "ownership_loss": 0.18,
            "score_loss": 0.33,
            "liberty_loss": 0.21,
            "total_loss": 3.83,
            "selfplay_samples": 1536,
            "buffer_size": 4096,
            "buffer_effective_size": 3584.0,
            "buffer_avg_sample_weight": 0.875,
            "buffer_downweighted_rate": 0.125,
            "training_steps": 24,
            "learning_rate_start": 1.0e-3,
            "learning_rate_end": 7.5e-4,
            "avg_game_length": 118.25,
            "avg_first_pass_turn": 102.5,
            "avg_total_passes": 2.0,
            "max_moves_rate": 0.125,
            "avg_score_margin_black_minus_white": 1.75,
            "draw_rate": 0.0,
            "first_player_win_rate": 0.625,
            "abnormal_games": 1.0,
            "selfplay_games": 16.0,
            "avg_abs_predicted_score_error": 4.2,
            "avg_cleaned_dead_stones": 1.25,
            "avg_cleanup_rule_resolved_groups": 0.5,
            "avg_cleanup_local_search_resolved_groups": 1.5,
            "avg_cleanup_preserved_seki_groups": 0.25,
            "avg_unresolved_dead_groups": 0.0,
            "komi_used": 3.0,
            "komi_next": 3.125,
            "komi_adjustment_checked": True,
            "komi_adjustment_reason": "applied",
            "komi_adjustment_eligible_games": 6,
            "komi_adjustment_considered_games": 8,
            "komi_adjustment_delta": 0.125,
            "curriculum_scale": 0.5,
            "avg_captures_by_move": 0.125,
            "avg_stones_lost_prev_turn": 0.0625,
            "avg_immediate_capture_risk_stones": 0.15625,
            "avg_next_move_capture_stones": 0.1875,
            "avg_curriculum_value_bonus": -0.0234375,
            "true_eye_fill_rate": 0.03125,
            "pure_capture_rate": 0.125,
            "immediate_capture_risk_rate": 0.1875,
            "next_move_capture_rate": 0.125,
            "positive_curriculum_bonus_rate": 0.0625,
            "negative_curriculum_bonus_rate": 0.09375,
            "eye_fill_penalized_rate": 0.015625,
        }

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_iteration_summary(cfg, 24, metrics)
        output = buffer.getvalue()

        self.assertIn("[Iter 4/24]", output)
        self.assertIn("Loss     policy=2.900 ent=1.230 eye=0.040 value=0.420 ownership=0.180 score=0.330 lib=0.210 total=3.830", output)
        self.assertIn("Buffer   samples=1536 buffer=4096 eff=3584.0 avg_w=0.875 down=12.5% steps=24", output)
        self.assertIn("Games    len=118.2 pass=102.5/2.00 margin=+1.75 draw=0.0% first=62.5% max=12.5%", output)
        self.assertIn("Cleanup  dead=1.25 rule=0.50 local=1.50", output)
        self.assertIn(
            "Tactic   curr=0.50 cap=0.125 risk=0.156 next_cap=0.188 eye_fill=3.1% cap_rate=12.5% risk_rate=18.8% next_cap_rate=12.5% bonus=-0.023 +b=6.2% -b=9.4% pen=1.6%",
            output,
        )
        self.assertIn("komi=3.00 -> 3.12 (applied, eligible=6/8, delta=+0.12)", output)
        self.assertGreaterEqual(output.count("\n"), 5)

    def test_summarize_selfplay_games_reports_max_moves_rate(self) -> None:
        cfg = self._make_cfg()
        traces = [
            self._make_trace(black_score=6.0, white_score=5.0, end_reason="double_pass"),
            self._make_trace(black_score=4.0, white_score=6.0, end_reason="max_moves"),
            self._make_trace(black_score=5.0, white_score=5.0, end_reason="max_moves"),
        ]

        summary = summarize_selfplay_games(traces)

        self.assertAlmostEqual(summary["max_moves_rate"], 2.0 / 3.0, places=6)

    def test_summarize_selfplay_games_reports_immediate_capture_risk_metrics(self) -> None:
        first = self._make_trace(black_score=6.0, white_score=5.0)
        second = self._make_trace(black_score=4.0, white_score=5.0)
        placeholder_move = SelfPlayMoveTrace(
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
        )
        first.moves = [placeholder_move, placeholder_move, placeholder_move]
        second.moves = [placeholder_move]
        first.total_immediate_capture_risk_stones = 3
        second.total_immediate_capture_risk_stones = 1
        first.immediate_capture_risk_rate_moves = 2
        second.immediate_capture_risk_rate_moves = 1
        first.pure_capture_moves = 1
        second.pure_capture_moves = 0

        summary = summarize_selfplay_games([first, second])

        self.assertAlmostEqual(summary["avg_immediate_capture_risk_stones"], 1.0, places=6)
        self.assertAlmostEqual(summary["immediate_capture_risk_rate"], 0.75, places=6)
        self.assertAlmostEqual(summary["capture_weighted_rate"], 0.25, places=6)

    def test_curriculum_metrics_satisfied_uses_capture_and_eye_fill_window(self) -> None:
        cfg = self._make_cfg()
        cfg.curriculum.stop_on_metrics = True
        cfg.curriculum.stop_min_iteration = 20
        cfg.curriculum.stop_max_iteration = 40
        cfg.curriculum.stop_min_capture_rate = 0.12
        cfg.curriculum.stop_max_true_eye_fill_rate = 0.04
        cfg.curriculum.stop_max_eye_fill_penalized_rate = 0.02

        acceptable = {
            "pure_capture_rate": 0.20,
            "true_eye_fill_rate": 0.03,
            "eye_fill_penalized_rate": 0.015,
        }
        capture_too_low = {
            "pure_capture_rate": 0.08,
            "true_eye_fill_rate": 0.03,
            "eye_fill_penalized_rate": 0.015,
        }
        eye_fill_too_high = {
            "pure_capture_rate": 0.20,
            "true_eye_fill_rate": 0.06,
            "eye_fill_penalized_rate": 0.03,
        }

        self.assertFalse(curriculum_metrics_satisfied(acceptable, cfg, 19))
        self.assertTrue(curriculum_metrics_satisfied(acceptable, cfg, 20))
        self.assertTrue(curriculum_metrics_satisfied(acceptable, cfg, 32))
        self.assertFalse(curriculum_metrics_satisfied(acceptable, cfg, 41))
        self.assertFalse(curriculum_metrics_satisfied(capture_too_low, cfg, 24))
        self.assertFalse(curriculum_metrics_satisfied(eye_fill_too_high, cfg, 24))

    def test_curriculum_metric_stop_respects_patience_and_max_iteration(self) -> None:
        cfg = self._make_cfg()
        cfg.curriculum.stop_on_metrics = True
        cfg.curriculum.stop_min_iteration = 20
        cfg.curriculum.stop_max_iteration = 40
        cfg.curriculum.stop_metric_patience = 2

        self.assertFalse(curriculum_should_stop_for_metrics(1, cfg, 24))
        self.assertTrue(curriculum_should_stop_for_metrics(2, cfg, 24))
        self.assertTrue(curriculum_should_stop_for_metrics(0, cfg, 40))

    def test_load_config_accepts_legacy_capture_weighted_stop_key(self) -> None:
        path = self._tmp_root / "legacy_config.json"
        path.write_text(
            json.dumps(
                {
                    "curriculum": {
                        "stop_min_capture_weighted_rate": 0.23,
                    }
                }
            ),
            encoding="utf-8",
        )

        cfg = load_experiment_config(path)

        self.assertAlmostEqual(cfg.curriculum.stop_min_capture_rate, 0.23)

    def test_apply_board_side_updates_action_size_and_stage_paths(self) -> None:
        cfg = load_experiment_config("python/rl/configs/stage5_side3_cpu_i5_12500h.json")

        apply_board_side(cfg, 4)

        self.assertEqual(cfg.model.board_side, 4)
        self.assertEqual(cfg.model.action_size, action_size_for_side(4))
        self.assertIn("side4", cfg.name)
        self.assertIn("stage5_side4", cfg.training.checkpoint_dir.as_posix())
        self.assertIn("stage5_side4", cfg.telemetry.selfplay_dir.as_posix())

    def test_resume_load_preserves_incumbent_and_global_step(self) -> None:
        cfg = self._make_cfg()
        cfg.training.best_dir = self._tmp_root / "best"
        checkpoint_path = self._tmp_root / "iter3.pt"
        candidate = PolicyValueNet(cfg.model)
        incumbent = PolicyValueNet(cfg.model)
        with torch.no_grad():
            for parameter in candidate.parameters():
                parameter.add_(0.5)
            for parameter in incumbent.parameters():
                parameter.sub_(0.25)
        optimizer = optim.AdamW(candidate.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        replay = ReplayBuffer(cfg.selfplay.replay_window)
        replay.extend([self._make_sample(cfg, action=0, weight=1.0, value=0.0, score=0.0)])

        save_checkpoint(
            checkpoint_path,
            candidate,
            incumbent,
            optimizer,
            replay,
            cfg,
            {"iteration": 3, "global_train_step": 17},
        )

        loaded_candidate, loaded_incumbent, _, loaded_replay, start_iteration, global_train_step, curriculum_metric_hit_streak, pending_komi_traces = load_resume_state(
            checkpoint_path,
            cfg,
            torch.device("cpu"),
        )

        candidate_norm = sum(parameter.detach().abs().sum().item() for parameter in loaded_candidate.parameters())
        incumbent_norm = sum(parameter.detach().abs().sum().item() for parameter in loaded_incumbent.parameters())
        self.assertNotAlmostEqual(candidate_norm, incumbent_norm)
        self.assertEqual(len(loaded_replay), 1)
        self.assertEqual(start_iteration, 4)
        self.assertEqual(global_train_step, 17)
        self.assertEqual(curriculum_metric_hit_streak, 0)
        self.assertEqual(pending_komi_traces, [])

    def test_resume_load_restores_runtime_config_state(self) -> None:
        cfg = self._make_cfg()
        cfg.curriculum.enabled = True
        cfg.rules.komi = 3.0
        cfg.rules.selfplay_komi = [2.5, 3.0, 3.5]
        checkpoint_path = self._tmp_root / "iter40.pt"
        candidate = PolicyValueNet(cfg.model)
        incumbent = PolicyValueNet(cfg.model)
        optimizer = optim.AdamW(candidate.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        replay = ReplayBuffer(cfg.selfplay.replay_window)
        saved_cfg = self._make_cfg()
        saved_cfg.curriculum.enabled = False
        saved_cfg.rules.komi = 4.25
        saved_cfg.rules.selfplay_komi = [3.75, 4.25, 4.75]

        save_checkpoint(
            checkpoint_path,
            candidate,
            incumbent,
            optimizer,
            replay,
            saved_cfg,
            {"iteration": 40, "global_train_step": 128},
        )

        load_resume_state(checkpoint_path, cfg, torch.device("cpu"))

        self.assertFalse(cfg.curriculum.enabled)
        self.assertAlmostEqual(cfg.rules.komi, 4.25)
        self.assertEqual(cfg.rules.selfplay_komi, [3.75, 4.25, 4.75])

    def test_resume_load_restores_runtime_progress_state(self) -> None:
        cfg = self._make_cfg()
        checkpoint_path = self._tmp_root / "iter24.pt"
        candidate = PolicyValueNet(cfg.model)
        incumbent = PolicyValueNet(cfg.model)
        optimizer = optim.AdamW(candidate.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        replay = ReplayBuffer(cfg.selfplay.replay_window)
        pending_komi_traces = [
            self._make_trace(black_score=5.0, white_score=3.0),
            self._make_trace(black_score=4.0, white_score=3.0),
        ]

        save_checkpoint(
            checkpoint_path,
            candidate,
            incumbent,
            optimizer,
            replay,
            cfg,
            {"iteration": 24, "global_train_step": 96, "curriculum_metric_hit_streak": 1},
            pending_komi_traces,
        )

        _, _, _, _, start_iteration, global_train_step, curriculum_metric_hit_streak, loaded_pending_komi_traces = load_resume_state(
            checkpoint_path,
            cfg,
            torch.device("cpu"),
        )

        self.assertEqual(start_iteration, 25)
        self.assertEqual(global_train_step, 96)
        self.assertEqual(curriculum_metric_hit_streak, 1)
        self.assertEqual(len(loaded_pending_komi_traces), 2)
        self.assertEqual(loaded_pending_komi_traces[0].black_score, 5.0)
        self.assertEqual(loaded_pending_komi_traces[1].white_score, 3.0)

    def test_trim_jsonl_to_iteration_discards_future_entries(self) -> None:
        path = self._tmp_root / "metrics.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"iteration": 7, "value": "keep"}),
                    json.dumps({"iteration": 8, "value": "keep"}),
                    json.dumps({"iteration": 9, "value": "drop"}),
                    json.dumps({"event": "metadata"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        removed = trim_jsonl_to_iteration(path, 8)

        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(removed, 1)
        self.assertEqual(lines, [{"iteration": 7, "value": "keep"}, {"iteration": 8, "value": "keep"}, {"event": "metadata"}])

    def test_checkpoint_pruning_keeps_recent_and_decimated_history(self) -> None:
        cfg = self._make_cfg()
        cfg.name = "checkpoint_prune"
        cfg.training.checkpoint_dir = self._tmp_root / "checkpoints"
        cfg.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg.training.checkpoint_keep_recent = 3
        cfg.training.checkpoint_keep_every = 10

        for iteration in range(1, 26):
            (cfg.training.checkpoint_dir / f"{cfg.name}_iter{iteration}.pt").write_bytes(b"x")

        removed = prune_checkpoints(cfg, 25)

        self.assertGreater(len(removed), 0)
        kept = sorted(
            checkpoint_iteration(path, cfg.name)
            for path in cfg.training.checkpoint_dir.glob(f"{cfg.name}_iter*.pt")
            if checkpoint_iteration(path, cfg.name) is not None
        )
        self.assertEqual(kept, [10, 20, 23, 24, 25])

    def test_selfplay_logs_are_sharded_trimmed_and_loaded_recently(self) -> None:
        cfg = self._make_cfg()
        cfg.name = "selfplay_shards"
        cfg.telemetry.selfplay_dir = self._tmp_root / "selfplay"
        cfg.telemetry.selfplay_dir.mkdir(parents=True, exist_ok=True)
        cfg.telemetry.report_recent_games = 3

        for iteration in range(1, 5):
            path = iteration_selfplay_path(cfg, iteration)
            for game_index in range(1, 3):
                payload = {"iteration": iteration, "game_index": game_index, "moves": []}
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload) + "\n")

        recent = load_recent_selfplay_records(cfg)
        self.assertEqual([(row["iteration"], row["game_index"]) for row in recent], [(3, 2), (4, 1), (4, 2)])

        removed = trim_selfplay_logs(cfg, 2)
        self.assertEqual(removed, 2)
        remaining = sorted(path.name for path in cfg.telemetry.selfplay_dir.glob("*.jsonl"))
        self.assertEqual(remaining, [f"{cfg.name}_iter0001.jsonl", f"{cfg.name}_iter0002.jsonl"])

        reset_selfplay_logs(cfg)
        self.assertFalse(any(cfg.telemetry.selfplay_dir.glob("*.jsonl")))

    def test_run_training_bootstraps_best_when_sparse_eval_is_skipped(self) -> None:
        cfg = self._make_cfg()
        cfg.name = "sparse_eval_smoke"
        cfg.seed = 7
        cfg.mcts.simulations = 1
        cfg.mcts.root_exploration_fraction = 0.0
        cfg.selfplay.games_per_iteration = 1
        cfg.selfplay.workers = 1
        cfg.selfplay.max_moves = 4
        cfg.selfplay.iterations = 1
        cfg.rules.opening_no_pass_moves = 0
        cfg.training.batch_size = 4
        cfg.training.epochs_per_iteration = 1
        cfg.training.min_steps_per_iteration = 1
        cfg.training.checkpoint_dir = self._tmp_root / "checkpoints"
        cfg.training.best_dir = self._tmp_root / "best"
        cfg.training.replay_dir = self._tmp_root / "replay"
        cfg.evaluation.interval = 4
        cfg.evaluation.games = 5
        cfg.evaluation.simulations = 1
        cfg.export.onnx_path = self._tmp_root / "export" / "sparse_eval_smoke.onnx"
        cfg.telemetry.enabled = False
        cfg.telemetry.save_game_records = False
        cfg.telemetry.show_progress = False
        cfg.telemetry.selfplay_dir = self._tmp_root / "selfplay"
        cfg.telemetry.report_path = self._tmp_root / "report.html"

        run_training(cfg)

        metrics_path = cfg.training.checkpoint_dir / f"{cfg.name}_metrics.jsonl"
        lines = metrics_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        metrics = json.loads(lines[0])
        best_update = json.loads(lines[1])
        self.assertFalse(metrics["eval_ran"])
        self.assertEqual(metrics["eval_mode"], "skipped")
        self.assertTrue(metrics["bootstrap_best"])
        self.assertTrue(metrics["best_updated"])
        self.assertEqual(best_update["event"], "best_update")
        self.assertTrue((cfg.training.best_dir / f"{cfg.name}_best.pt").exists())


if __name__ == "__main__":
    unittest.main()
