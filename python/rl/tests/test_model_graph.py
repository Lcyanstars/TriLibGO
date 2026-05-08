from __future__ import annotations

import unittest

import numpy as np
import torch

from python.rl.analysis import analyze_state
from python.rl.config import ModelConfig, input_planes_for_history
from python.rl.encoder import LIBERTY_CLASS_COUNT
from python.rl.game import GameConfig, GameState, Stone
from python.rl.mcts import MCTS
from python.rl.model import PolicyValueNet, build_spatial_layout
from python.rl.replay_buffer import ReplayBuffer
from python.rl.selfplay import TrainingSample
from python.rl.topology import BoardTopology
from python.rl.train import train_iteration


class FixedAnalysisModel(torch.nn.Module):
    def __init__(self, action_size: int, ownership: torch.Tensor, value: float = 0.0, score: float = 0.0) -> None:
        super().__init__()
        self.action_size = action_size
        self.register_buffer("_ownership", ownership)
        self.value = float(value)
        self.score = float(score)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        logits = torch.zeros(batch, self.action_size, dtype=torch.float32, device=x.device)
        value = torch.full((batch,), self.value, dtype=torch.float32, device=x.device)
        ownership = self._ownership.to(device=x.device).unsqueeze(0).repeat(batch, 1)
        score = torch.full((batch,), self.score, dtype=torch.float32, device=x.device)
        liberty_logits = torch.zeros(batch, LIBERTY_CLASS_COUNT, self.action_size - 1, dtype=torch.float32, device=x.device)
        return logits, value, ownership, score, liberty_logits


class GraphModelTests(unittest.TestCase):
    def test_spatial_layout_covers_all_vertices_once(self) -> None:
        topology = BoardTopology(2)
        height, width, index, mask = build_spatial_layout(2)

        self.assertGreaterEqual(height, 1)
        self.assertGreaterEqual(width, 1)
        self.assertEqual(index.shape[0], topology.vertex_count)
        self.assertEqual(int(mask.sum().item()), topology.vertex_count)
        self.assertEqual(len(set(index.tolist())), topology.vertex_count)

    def test_spatial_layout_window_matches_topology_neighbors(self) -> None:
        topology = BoardTopology(4)
        _, width, index, _ = build_spatial_layout(4)
        vertex_by_flat = {int(flat): vertex for vertex, flat in enumerate(index.tolist())}

        false_neighbors = []
        far_true_neighbors = []
        for vertex, flat in enumerate(index.tolist()):
            row, col = divmod(int(flat), width)
            for delta_row in range(-2, 3):
                for delta_col in range(-1, 2):
                    if delta_row == 0 and delta_col == 0:
                        continue
                    neighbor_row = row + delta_row
                    neighbor_col = col + delta_col
                    if neighbor_row < 0 or neighbor_col < 0 or neighbor_col >= width:
                        continue
                    neighbor = vertex_by_flat.get(neighbor_row * width + neighbor_col)
                    if neighbor is not None and neighbor not in topology.adjacency[vertex]:
                        false_neighbors.append((vertex, neighbor))
            for neighbor in topology.adjacency[vertex]:
                neighbor_row, neighbor_col = divmod(int(index[neighbor]), width)
                if abs(row - neighbor_row) > 2 or abs(col - neighbor_col) > 1:
                    far_true_neighbors.append((vertex, neighbor))

        self.assertEqual(false_neighbors, [])
        self.assertEqual(far_true_neighbors, [])

    def test_policy_value_net_forward_and_backward(self) -> None:
        cfg = ModelConfig(
            board_side=4,
            input_history=4,
            channels=32,
            residual_blocks=2,
            policy_head_channels=16,
            value_head_channels=16,
            ownership_head_channels=16,
            score_head_channels=16,
            action_size=97,
        )
        model = PolicyValueNet(cfg)
        batch = torch.randn(3, input_planes_for_history(cfg.input_history), cfg.action_size - 1)

        policy_logits, value, ownership, score, liberty_logits = model(batch)
        loss = policy_logits.mean() + value.mean() + ownership.mean() + score.mean() + liberty_logits.mean()
        loss.backward()

        self.assertEqual(tuple(policy_logits.shape), (3, cfg.action_size))
        self.assertEqual(tuple(value.shape), (3,))
        self.assertEqual(tuple(ownership.shape), (3, cfg.action_size - 1))
        self.assertEqual(tuple(score.shape), (3,))
        self.assertEqual(tuple(liberty_logits.shape), (3, LIBERTY_CLASS_COUNT, cfg.action_size - 1))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_analyze_state_exposes_auxiliary_heads(self) -> None:
        cfg = ModelConfig(
            board_side=2,
            input_history=2,
            channels=16,
            residual_blocks=1,
            policy_head_channels=8,
            value_head_channels=8,
            ownership_head_channels=8,
            score_head_channels=8,
            action_size=25,
        )
        state = GameState(GameConfig(side_length=2))
        analysis = analyze_state(PolicyValueNet(cfg), state, type("Cfg", (), {"model": cfg})())

        self.assertEqual(analysis.policy.shape[0], cfg.action_size)
        self.assertEqual(analysis.ownership.shape[0], cfg.action_size - 1)
        self.assertIsInstance(analysis.predicted_score_margin, float)

    def test_analyze_state_flips_auxiliary_heads_for_white_to_play(self) -> None:
        cfg = ModelConfig(board_side=2, input_history=2, action_size=25)
        state = GameState(GameConfig(side_length=2))
        state.to_play = Stone.WHITE
        board = [Stone.EMPTY] * state.topology.vertex_count
        board[11] = Stone.BLACK
        for index in (1, 5, 10, 14):
            board[index] = Stone.WHITE
        state.board = board
        model_ownership = torch.zeros(cfg.action_size - 1, dtype=torch.float32)
        model_ownership[11] = 0.9

        analysis = analyze_state(FixedAnalysisModel(cfg.action_size, model_ownership, score=3.0), state, type("Cfg", (), {"model": cfg})())

        self.assertAlmostEqual(float(analysis.ownership[11]), -0.9, places=6)
        self.assertAlmostEqual(analysis.predicted_score_margin, -3.0, places=6)
        self.assertEqual(len(analysis.suggested_dead_groups), 1)
        self.assertEqual(analysis.suggested_dead_groups[0].color, "B")

    def test_train_iteration_reports_auxiliary_losses(self) -> None:
        cfg = type(
            "Cfg",
            (),
            {
                "training": type(
                    "TrainingCfg",
                    (),
                    {
                        "batch_size": 2,
                        "gradient_clip_norm": 1.0,
                        "policy_loss_weight": 1.0,
                        "value_loss_weight": 1.0,
                        "ownership_loss_weight": 0.5,
                        "score_loss_weight": 0.25,
                        "liberty_loss_weight": 0.2,
                    },
                )(),
            },
        )()
        model_cfg = ModelConfig(
            board_side=4,
            input_history=2,
            channels=16,
            residual_blocks=1,
            policy_head_channels=8,
            value_head_channels=8,
            ownership_head_channels=8,
            score_head_channels=8,
            action_size=97,
        )
        model = PolicyValueNet(model_cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        replay = ReplayBuffer(8)
        for value_target, score_target in ((1.0, 3.0), (-1.0, -2.0)):
            replay.extend(
                [
                    TrainingSample(
                        state_planes=np.zeros((input_planes_for_history(model_cfg.input_history), model_cfg.action_size - 1), dtype=np.float32),
                        policy_target=np.full(model_cfg.action_size, 1.0 / model_cfg.action_size, dtype=np.float32),
                        value_target=value_target,
                        ownership_target=np.full(model_cfg.action_size - 1, value_target, dtype=np.float32),
                        score_target=score_target,
                    )
                ]
            )

        metrics = train_iteration(model, optimizer, replay, cfg, torch.device("cpu"))

        self.assertIn("ownership_loss", metrics)
        self.assertIn("score_loss", metrics)
        self.assertIn("liberty_loss", metrics)
        self.assertGreaterEqual(metrics["ownership_loss"], 0.0)
        self.assertGreaterEqual(metrics["score_loss"], 0.0)
        self.assertGreaterEqual(metrics["liberty_loss"], 0.0)

    def test_mcts_evaluation_uses_eval_mode_without_leaving_model_changed(self) -> None:
        cfg = ModelConfig(
            board_side=2,
            input_history=2,
            channels=16,
            residual_blocks=1,
            policy_head_channels=8,
            value_head_channels=8,
            ownership_head_channels=8,
            score_head_channels=8,
            action_size=25,
        )
        model = PolicyValueNet(cfg)
        model.train()
        mcts_cfg = type(
            "MctsCfg",
            (),
            {
                "simulations": 1,
                "c_puct": 1.0,
                "root_dirichlet_alpha": 0.3,
                "root_exploration_fraction": 0.0,
                "pass_prior_scale": 1.0,
                "disable_consecutive_pass_guard": False,
                "consecutive_pass_min_value": -1.0,
                "consecutive_pass_min_score_margin": -99.0,
            },
        )()
        search = MCTS(model, mcts_cfg, torch.device("cpu"), input_history=cfg.input_history, root_noise_enabled=False)

        search.run(GameState(GameConfig(side_length=2)))

        self.assertTrue(model.training)

    def test_train_iteration_sets_model_to_train_mode(self) -> None:
        cfg = type(
            "Cfg",
            (),
            {
                "training": type(
                    "TrainingCfg",
                    (),
                    {
                        "batch_size": 1,
                        "gradient_clip_norm": 1.0,
                        "policy_loss_weight": 1.0,
                        "policy_entropy_weight": 0.0,
                        "eye_fill_loss_weight": 0.0,
                        "value_loss_weight": 1.0,
                        "ownership_loss_weight": 0.5,
                        "score_loss_weight": 0.25,
                        "score_loss_scale": 16.0,
                    },
                )(),
            },
        )()
        model_cfg = ModelConfig(
            board_side=2,
            input_history=2,
            channels=16,
            residual_blocks=1,
            policy_head_channels=8,
            value_head_channels=8,
            ownership_head_channels=8,
            score_head_channels=8,
            action_size=25,
        )
        model = PolicyValueNet(model_cfg)
        model.eval()
        replay = ReplayBuffer(4)
        replay.extend(
            [
                TrainingSample(
                    state_planes=np.zeros((input_planes_for_history(model_cfg.input_history), model_cfg.action_size - 1), dtype=np.float32),
                    policy_target=np.full(model_cfg.action_size, 1.0 / model_cfg.action_size, dtype=np.float32),
                    value_target=0.0,
                    ownership_target=np.zeros(model_cfg.action_size - 1, dtype=np.float32),
                    score_target=0.0,
                )
            ]
        )

        train_iteration(model, torch.optim.AdamW(model.parameters(), lr=0.0), replay, cfg, torch.device("cpu"))

        self.assertTrue(model.training)

    def test_replay_buffer_ignores_zero_weight_samples(self) -> None:
        replay = ReplayBuffer(8)
        replay.extend(
            [
                TrainingSample(
                    state_planes=np.zeros((2, 24), dtype=np.float32),
                    policy_target=np.zeros(25, dtype=np.float32),
                    value_target=0.0,
                    ownership_target=np.zeros(24, dtype=np.float32),
                    score_target=0.0,
                    sample_weight=0.0,
                ),
                TrainingSample(
                    state_planes=np.zeros((2, 24), dtype=np.float32),
                    policy_target=np.zeros(25, dtype=np.float32),
                    value_target=0.0,
                    ownership_target=np.zeros(24, dtype=np.float32),
                    score_target=0.0,
                    sample_weight=1.0,
                ),
            ]
        )

        self.assertEqual(len(replay), 1)


if __name__ == "__main__":
    unittest.main()
