from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import ExperimentConfig, action_size_for_side
from .encoder import encode_state, liberty_global_features
from .game import GameState, Stone
from .model import PolicyValueNet


@dataclass
class DeadGroupSuggestion:
    color: str
    vertices: list[int]
    labels: list[str]
    liberties: int
    mean_ownership: float
    confidence: float


@dataclass
class PositionAnalysis:
    winrate: float
    top_actions: list[tuple[int, float]]
    policy: np.ndarray
    ownership: np.ndarray
    predicted_score_margin: float
    suggested_dead_groups: list[DeadGroupSuggestion]


def suggest_dead_groups_from_ownership(
    state: GameState,
    ownership: np.ndarray,
    threshold: float = 0.75,
    max_liberties: int = 2,
) -> list[DeadGroupSuggestion]:
    suggestions: list[DeadGroupSuggestion] = []
    visited: set[int] = set()
    for vertex, stone in enumerate(state.board):
        if stone == Stone.EMPTY or vertex in visited:
            continue
        group, liberties = state.collect_group(vertex)
        visited.update(group)
        if liberties > max_liberties:
            continue
        group_ownership = float(np.mean(ownership[group]))
        if stone == Stone.BLACK and group_ownership <= -threshold:
            suggestions.append(
                DeadGroupSuggestion(
                    color="B",
                    vertices=group,
                    labels=[state.topology.labels.get(index, str(index)) for index in group],
                    liberties=liberties,
                    mean_ownership=group_ownership,
                    confidence=abs(group_ownership),
                )
            )
        elif stone == Stone.WHITE and group_ownership >= threshold:
            suggestions.append(
                DeadGroupSuggestion(
                    color="W",
                    vertices=group,
                    labels=[state.topology.labels.get(index, str(index)) for index in group],
                    liberties=liberties,
                    mean_ownership=group_ownership,
                    confidence=abs(group_ownership),
                )
            )
    suggestions.sort(key=lambda item: item.confidence, reverse=True)
    return suggestions


def analyze_state(model: PolicyValueNet, state: GameState, cfg: ExperimentConfig) -> PositionAnalysis:
    with torch.no_grad():
        planes = torch.from_numpy(encode_state(state, cfg.model.input_history)).unsqueeze(0)
        global_features = torch.from_numpy(liberty_global_features(state)).unsqueeze(0)
        try:
            outputs = model(planes, global_features)
        except TypeError:
            outputs = model(planes)
        logits, value, ownership, score = outputs[:4]
        policy = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    expected_action_size = action_size_for_side(cfg.model.board_side)
    if policy.shape[0] != expected_action_size:
        raise ValueError(f"Policy size mismatch: expected {expected_action_size}, got {policy.shape[0]}")
    perspective = 1.0 if state.to_play == Stone.BLACK else -1.0
    ownership_black_perspective = ownership.squeeze(0).cpu().numpy() * perspective
    score_black_minus_white = float(score.item() * perspective)
    top_indices = np.argsort(policy)[::-1][:8]
    return PositionAnalysis(
        winrate=float((value.item() + 1.0) / 2.0),
        top_actions=[(int(index), float(policy[index])) for index in top_indices],
        policy=policy,
        ownership=ownership_black_perspective,
        predicted_score_margin=score_black_minus_white,
        suggested_dead_groups=suggest_dead_groups_from_ownership(state, ownership_black_perspective),
    )
