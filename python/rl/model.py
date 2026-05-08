from __future__ import annotations

from math import sqrt

import torch
from torch import nn

from .config import ModelConfig, input_planes_for_history
from .topology import BoardTopology

ModelForwardOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def build_spatial_layout(side_length: int) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    topology = BoardTopology(side_length)
    min_x = min(position.x for position in topology.positions)
    min_y = min(position.y for position in topology.positions)
    vertex_coords = [
        (
            round((position.y - min_y) / 0.5),
            round((position.x - min_x) / (sqrt(3.0) / 2.0)),
        )
        for position in topology.positions
    ]
    height = max(row for row, _ in vertex_coords) + 1
    width = max(col for _, col in vertex_coords) + 1
    mask = torch.zeros(height, width, dtype=torch.float32)
    vertex_to_flat = []
    for row, col in vertex_coords:
        vertex_to_flat.append(row * width + col)
        mask[row, col] = 1.0
    return height, width, torch.tensor(vertex_to_flat, dtype=torch.long), mask


class ConvResidualBlock2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(channels, channels, kernel_size=(5, 3), padding=(2, 1), bias=False)
        self.norm_1 = nn.BatchNorm2d(channels)
        self.conv_2 = nn.Conv2d(channels, channels, kernel_size=(5, 3), padding=(2, 1), bias=False)
        self.norm_2 = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_1(x)
        y = self.norm_1(y)
        y = self.activation(y)
        y = self.conv_2(y)
        y = self.norm_2(y)
        return self.activation(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.board_side = cfg.board_side
        self.vertex_count = cfg.action_size - 1
        self.spatial_height, self.spatial_width, spatial_index, valid_mask = build_spatial_layout(cfg.board_side)
        self.register_buffer("spatial_index", spatial_index)
        self.register_buffer("valid_mask", valid_mask.view(1, 1, self.spatial_height, self.spatial_width))

        input_planes = input_planes_for_history(cfg.input_history)
        self.global_feature_count = int(getattr(cfg, "global_feature_count", 8))
        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, cfg.channels, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(cfg.channels),
            nn.ReLU(inplace=True),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_feature_count, cfg.channels),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.channels, cfg.channels),
        )
        self.blocks = nn.ModuleList([ConvResidualBlock2d(cfg.channels) for _ in range(cfg.residual_blocks)])

        self.policy_board_head = nn.Sequential(
            nn.Conv2d(cfg.channels, cfg.policy_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.policy_head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.policy_head_channels, 1, kernel_size=1),
        )
        self.policy_pass_head = nn.Sequential(
            nn.Linear(cfg.channels, cfg.policy_head_channels),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.policy_head_channels, 1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(cfg.channels, cfg.value_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.value_head_channels),
            nn.ReLU(inplace=True),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(cfg.value_head_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        self.ownership_head = nn.Sequential(
            nn.Conv2d(cfg.channels, cfg.ownership_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.ownership_head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.ownership_head_channels, 1, kernel_size=1),
            nn.Tanh(),
        )
        self.score_head = nn.Sequential(
            nn.Conv2d(cfg.channels, cfg.score_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.score_head_channels),
            nn.ReLU(inplace=True),
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(cfg.score_head_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.liberty_head = nn.Sequential(
            nn.Conv2d(cfg.channels, cfg.liberty_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.liberty_head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.liberty_head_channels, 7, kernel_size=1),
        )

    def _to_spatial(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, vertices = x.shape
        flat = x.new_zeros((batch, channels, self.spatial_height * self.spatial_width))
        index = self.spatial_index.view(1, 1, -1).expand(batch, channels, -1)
        flat = flat.scatter(2, index, x)
        return flat.view(batch, channels, self.spatial_height, self.spatial_width)

    def _from_spatial(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.shape[0], x.shape[1], -1)
        return flat.index_select(2, self.spatial_index)

    def _masked_mean(self, x: torch.Tensor) -> torch.Tensor:
        masked = x * self.valid_mask
        summed = masked.sum(dim=(2, 3))
        denom = self.valid_mask.sum(dim=(2, 3)).clamp(min=1.0).to(dtype=x.dtype)
        return summed / denom

    def _apply_backbone(self, x: torch.Tensor, global_features: torch.Tensor | None = None) -> torch.Tensor:
        features = self.stem(self._to_spatial(x)) * self.valid_mask
        if global_features is None:
            global_features = x.new_zeros((x.shape[0], self.global_feature_count))
        global_embedding = self.global_encoder(global_features.to(dtype=x.dtype, device=x.device))
        features = (features + global_embedding.view(x.shape[0], -1, 1, 1)) * self.valid_mask
        for block in self.blocks:
            features = block(features) * self.valid_mask
        return features

    def forward(self, x: torch.Tensor, global_features: torch.Tensor | None = None) -> ModelForwardOutputs:
        features = self._apply_backbone(x, global_features)

        policy_board = self.policy_board_head(features) * self.valid_mask
        policy_logits = self._from_spatial(policy_board).squeeze(1)
        pass_logits = self.policy_pass_head(self._masked_mean(features))
        policy_logits = torch.cat([policy_logits, pass_logits], dim=-1)

        value_features = self.value_head(features) * self.valid_mask
        values = self.value_mlp(self._masked_mean(value_features)).squeeze(-1)

        ownership_map = self.ownership_head(features) * self.valid_mask
        ownership = self._from_spatial(ownership_map).squeeze(1)

        score_features = self.score_head(features) * self.valid_mask
        score = self.score_mlp(self._masked_mean(score_features)).squeeze(-1)
        liberty_logits = self._from_spatial(self.liberty_head(features) * self.valid_mask)
        return policy_logits, values, ownership, score, liberty_logits
