from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass
from typing import Callable

import torch

from .config import ExperimentConfig
from .game import GameConfig, GameState, Move
from .mcts import MCTS
from .model import PolicyValueNet


@dataclass
class EvaluationSummary:
    wins: int
    losses: int
    draws: int

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return self.wins / total if total else 0.0


def play_match(
    black_model: PolicyValueNet,
    white_model: PolicyValueNet,
    cfg: ExperimentConfig,
    simulations_override: int | None = None,
) -> float:
    device = torch.device("cpu")
    simulations = simulations_override if simulations_override is not None else cfg.evaluation.simulations
    if simulations is None or simulations <= 0:
        simulations = cfg.mcts.simulations
    eval_mcts = replace(cfg.mcts, simulations=simulations)
    black_search = MCTS(
        black_model,
        eval_mcts,
        device,
        input_history=cfg.model.input_history,
        root_noise_enabled=False,
        exploration_scale=1.0,
    )
    white_search = MCTS(
        white_model,
        eval_mcts,
        device,
        input_history=cfg.model.input_history,
        root_noise_enabled=False,
        exploration_scale=1.0,
    )
    state = GameState(
        GameConfig(
            side_length=cfg.model.board_side,
            komi=cfg.rules.komi,
            allow_suicide=cfg.rules.allow_suicide,
            opening_no_pass_moves=cfg.rules.opening_no_pass_moves,
            cleanup_dead_stones=cfg.rules.cleanup_dead_stones,
            cleanup_dead_stones_mode=cfg.rules.cleanup_dead_stones_mode,
            cleanup_candidate_max_liberties=cfg.rules.cleanup_candidate_max_liberties,
            cleanup_local_search_depth=cfg.rules.cleanup_local_search_depth,
            cleanup_local_search_nodes=cfg.rules.cleanup_local_search_nodes,
            cleanup_preserve_seki=cfg.rules.cleanup_preserve_seki,
            cleanup_mark_unresolved=cfg.rules.cleanup_mark_unresolved,
        )
    )
    for _ in range(cfg.selfplay.max_moves):
        if state.finished:
            break
        search = black_search if state.to_play == 1 else white_search
        search_result = search.run(state)
        action = int(search_result.policy.argmax())
        move = Move.pass_turn() if action == state.topology.vertex_count else Move.place(action)
        state.apply_move(move)
    forced_end_reason = "max_moves" if len(state.move_history) >= cfg.selfplay.max_moves and not state.finished else (state.end_reason or "score")
    return state.finalize_score(forced_end_reason).value


def evaluate_candidate(
    candidate: PolicyValueNet,
    incumbent: PolicyValueNet,
    cfg: ExperimentConfig,
    games_override: int | None = None,
    simulations_override: int | None = None,
    candidate_starts_black: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EvaluationSummary:
    wins = losses = draws = 0
    total_games = games_override if games_override is not None else cfg.evaluation.games
    for game_index in range(total_games):
        candidate_is_black = candidate_starts_black if game_index % 2 == 0 else not candidate_starts_black
        if candidate_is_black:
            result = play_match(candidate, incumbent, cfg, simulations_override=simulations_override)
        else:
            result = -play_match(incumbent, candidate, cfg, simulations_override=simulations_override)

        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            draws += 1
        if progress_callback is not None:
            progress_callback(game_index + 1, total_games)
    return EvaluationSummary(wins=wins, losses=losses, draws=draws)
