from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .config import MCTSConfig
from .encoder import action_size, encode_state, liberty_global_features
from .game import GameState, Move, Stone


@dataclass
class Node:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@dataclass
class SearchResult:
    policy: np.ndarray
    root_value: float
    root_score: float


def softmax_masked(logits: np.ndarray, legal_actions: list[int]) -> np.ndarray:
    masked = np.full_like(logits, -1e9)
    masked[legal_actions] = logits[legal_actions]
    exp = np.exp(masked - np.max(masked))
    exp_sum = np.sum(exp)
    return exp / exp_sum if exp_sum > 0 else np.zeros_like(logits)


class MCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        config: MCTSConfig,
        device: torch.device,
        input_history: int = 2,
        *,
        root_noise_enabled: bool = True,
        exploration_scale: float = 1.0,
        root_prior_shaper: Callable[[GameState, np.ndarray, list[int]], np.ndarray] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.input_history = input_history
        self.root_noise_enabled = root_noise_enabled
        self.exploration_scale = max(0.0, float(exploration_scale))
        self.root_prior_shaper = root_prior_shaper

    def run(self, state: GameState) -> SearchResult:
        root = Node(1.0)
        root_policy, root_value, root_score = self._evaluate(state)
        legal_actions = self._legal_actions(state, root_value, root_score)
        priors = softmax_masked(root_policy, legal_actions)
        priors = self._apply_pass_prior_penalty(state, legal_actions, priors)
        priors = self._apply_root_prior_shaper(state, legal_actions, priors)
        self._expand(root, legal_actions, priors)
        self._add_root_noise(root, legal_actions)

        remaining = self.config.simulations
        batch_size = max(1, int(getattr(self.config, "simulation_batch_size", 1)))
        while remaining > 0:
            pending: list[tuple[list[Node], Node, GameState]] = []
            for _ in range(min(batch_size, remaining)):
                search_state = state.copy()
                node = root
                search_path = [node]
                while node.children:
                    action, node = self._select_child(node)
                    move = Move.pass_turn() if action == search_state.topology.vertex_count else Move.place(action)
                    search_state.apply_move(move)
                    search_path.append(node)
                    if search_state.finished:
                        break

                if search_state.finished and search_state.result is not None:
                    value = search_state.result.value if search_state.to_play == Stone.BLACK else -search_state.result.value
                    self._backpropagate(search_path, value)
                else:
                    self._reserve_path(search_path)
                    pending.append((search_path, node, search_state))
                remaining -= 1

            if not pending:
                continue
            evaluations = self._evaluate_many([item[2] for item in pending])
            for (search_path, node, search_state), (policy_logits, value, score) in zip(pending, evaluations):
                legal_actions = self._legal_actions(search_state, value, score)
                priors = softmax_masked(policy_logits, legal_actions)
                priors = self._apply_pass_prior_penalty(search_state, legal_actions, priors)
                self._expand(node, legal_actions, priors)
                self._complete_reserved_path(search_path, value)

        visits = np.zeros(action_size(state), dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        return SearchResult(
            policy=visits / max(np.sum(visits), 1.0),
            root_value=root_value,
            root_score=root_score,
        )

    def _evaluate(self, state: GameState) -> tuple[np.ndarray, float, float]:
        return self._evaluate_many([state])[0]

    def _evaluate_many(self, states: list[GameState]) -> list[tuple[np.ndarray, float, float]]:
        encoded = torch.from_numpy(np.stack([encode_state(state, self.input_history) for state in states])).to(self.device)
        global_features = torch.from_numpy(np.stack([liberty_global_features(state) for state in states])).to(self.device)
        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        with torch.no_grad():
            try:
                outputs = self.model(encoded, global_features)
            except TypeError:
                outputs = self.model(encoded)
            logits, value, _, score = outputs[:4]
        if was_training:
            self.model.train()
        logits_np = logits.cpu().numpy()
        values_np = value.cpu().numpy()
        scores_np = score.cpu().numpy()
        return [
            (logits_np[index], float(values_np[index]), float(scores_np[index]))
            for index in range(len(states))
        ]

    def _expand(self, node: Node, legal_actions: list[int], priors: np.ndarray) -> None:
        for action in legal_actions:
            if action not in node.children:
                node.children[action] = Node(float(priors[action]))

    def _add_root_noise(self, root: Node, legal_actions: list[int]) -> None:
        if not self.root_noise_enabled or not legal_actions:
            return
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(legal_actions))
        for action, eta in zip(legal_actions, noise):
            child = root.children[action]
            child.prior = child.prior * (1.0 - self.config.root_exploration_fraction) + eta * self.config.root_exploration_fraction

    def _select_child(self, node: Node) -> tuple[int, Node]:
        sqrt_visits = math.sqrt(max(node.visit_count, 1))
        c_puct = self.config.c_puct * self.exploration_scale
        best_action = -1
        best_score = -1e9
        best_child = None
        for action, child in node.children.items():
            ucb = -child.value + c_puct * child.prior * sqrt_visits / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child
        assert best_child is not None
        return best_action, best_child

    def _legal_actions(self, state: GameState, predicted_value: float, predicted_score: float) -> list[int]:
        legal_moves = state.legal_moves()
        legal_actions = [state.topology.vertex_count if move.kind == "pass" else move.index for move in legal_moves]
        return self._apply_consecutive_pass_guard(state, legal_actions, predicted_value, predicted_score)

    def _apply_consecutive_pass_guard(
        self,
        state: GameState,
        legal_actions: list[int],
        predicted_value: float,
        predicted_score: float,
    ) -> list[int]:
        if bool(getattr(self.config, "disable_consecutive_pass_guard", False)):
            return legal_actions
        pass_action = state.topology.vertex_count
        if pass_action not in legal_actions or len(legal_actions) <= 1 or state.consecutive_passes != 1:
            return legal_actions
        if (
            predicted_value < self.config.consecutive_pass_min_value
            or predicted_score < self.config.consecutive_pass_min_score_margin
        ):
            return [action for action in legal_actions if action != pass_action]
        return legal_actions

    def _apply_pass_prior_penalty(self, state: GameState, legal_actions: list[int], priors: np.ndarray) -> np.ndarray:
        pass_action = state.topology.vertex_count
        if pass_action not in legal_actions or len(legal_actions) <= 1:
            return priors
        pass_scale = max(0.0, float(self.config.pass_prior_scale))
        if pass_scale >= 1.0:
            return priors
        adjusted = priors.copy()
        adjusted[pass_action] *= pass_scale
        legal_mass = float(np.sum(adjusted[legal_actions]))
        if legal_mass <= 0.0:
            return priors
        adjusted[legal_actions] /= legal_mass
        return adjusted

    def _apply_root_prior_shaper(self, state: GameState, legal_actions: list[int], priors: np.ndarray) -> np.ndarray:
        if self.root_prior_shaper is None or not legal_actions:
            return priors
        adjusted = np.asarray(self.root_prior_shaper(state, priors.copy(), legal_actions), dtype=np.float32)
        if adjusted.shape != priors.shape:
            raise ValueError("root_prior_shaper returned an invalid prior shape")
        adjusted = np.clip(adjusted, 0.0, None)
        legal_mass = float(np.sum(adjusted[legal_actions]))
        if legal_mass <= 0.0:
            return priors
        adjusted[legal_actions] /= legal_mass
        return adjusted

    def _backpropagate(self, search_path: list[Node], value: float) -> None:
        current = value
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += current
            current = -current

    def _reserve_path(self, search_path: list[Node], virtual_loss: float = 1.0) -> None:
        for node in search_path:
            node.visit_count += 1
            node.value_sum += virtual_loss

    def _complete_reserved_path(self, search_path: list[Node], value: float, virtual_loss: float = 1.0) -> None:
        current = value
        for node in reversed(search_path):
            node.value_sum += current - virtual_loss
            current = -current
