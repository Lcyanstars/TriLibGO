from __future__ import annotations

import random
from itertools import permutations
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig
from .encoder import action_size, encode_state, liberty_global_features, liberty_target_classes
from .game import GameConfig, GameState, Move, Stone
from .mcts import MCTS
from .model import PolicyValueNet


@dataclass
class TrainingSample:
    state_planes: np.ndarray
    policy_target: np.ndarray
    value_target: float
    ownership_target: np.ndarray
    score_target: float
    global_features: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    liberty_target: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    eye_fill_bad_action_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    sample_weight: float = 1.0
    policy_weight: float = 1.0
    terminal_weight: float = 1.0
    captures_by_move: int = 0
    stones_lost_prev_turn: int = 0
    next_move_capture_stones: int = 0
    curriculum_value_bonus: float = 0.0
    curriculum_scale_at_generation: float = 0.0
    is_curriculum_sample: bool = False
    fills_small_true_eye: bool = False


@dataclass
class SelfPlayMoveTrace:
    turn: int
    player: str
    action: int
    move: str
    root_value: float
    root_score_margin: float
    root_score_margin_black_minus_white: float
    score_margin_error_black_minus_white: float
    policy: list[float]
    policy_top: list[tuple[int, float]]
    captures_by_move: int = 0
    stones_lost_prev_turn: int = 0
    immediate_capture_risk_stones: int = 0
    next_move_capture_stones: int = 0
    curriculum_value_bonus: float = 0.0
    is_curriculum_sample: bool = False
    fills_small_true_eye: bool = False
    curriculum_weight: float = 1.0


@dataclass
class SelfPlayGameTrace:
    komi: float
    move_count: int
    black_score: float
    white_score: float
    result_value: float
    winner: str
    first_player_win: bool
    first_pass_turn: int
    total_passes: int
    cleaned_dead_stones: int
    cleanup_rule_resolved_groups: int
    cleanup_local_search_resolved_groups: int
    cleanup_preserved_seki_groups: int
    unresolved_dead_groups: int
    end_reason: str
    avg_abs_predicted_score_error: float
    sample_weight: float
    abnormal_tags: list[str]
    curriculum_scale: float = 0.0
    total_captured_stones: int = 0
    total_prev_turn_losses: int = 0
    total_immediate_capture_risk_stones: int = 0
    total_next_move_capture_stones: int = 0
    total_curriculum_value_bonus: float = 0.0
    true_eye_fills: int = 0
    pure_capture_moves: int = 0
    immediate_capture_risk_rate_moves: int = 0
    next_move_capture_rate_moves: int = 0
    curriculum_sample_moves: int = 0
    positive_curriculum_bonus_moves: int = 0
    negative_curriculum_bonus_moves: int = 0
    true_eye_penalized_moves: int = 0
    final_board: list[int] = field(default_factory=list)
    cleaned_dead_vertices: list[int] = field(default_factory=list)
    moves: list[SelfPlayMoveTrace] = field(default_factory=list)


SelfPlayBatchResult = tuple[list[TrainingSample], list[SelfPlayGameTrace]]


@dataclass
class CurriculumMoveStats:
    captures_by_move: int = 0
    stones_lost_prev_turn: int = 0
    immediate_capture_risk_stones: int = 0
    next_move_capture_stones: int = 0
    fills_small_true_eye: bool = False
    curriculum_value_bonus: float = 0.0
    curriculum_weight: float = 1.0


def sampling_temperature(turn: int, cfg: ExperimentConfig) -> float:
    opening_moves = max(0, cfg.mcts.temperature_opening_moves)
    base_temperature = max(float(cfg.mcts.temperature), 1e-3)
    final_temperature = min(max(float(cfg.selfplay.sampling_final_temperature), 0.0), base_temperature)
    if turn < opening_moves:
        return base_temperature
    decay_moves = max(0, cfg.selfplay.sampling_decay_moves)
    if decay_moves == 0:
        return final_temperature
    progress = min((turn - opening_moves + 1) / decay_moves, 1.0)
    return base_temperature + (final_temperature - base_temperature) * progress


def sample_move_from_policy(policy: np.ndarray, temperature: float) -> int:
    if temperature <= 0.05:
        return int(np.argmax(policy))
    clipped = np.clip(policy, 0.0, 1.0)
    if not np.any(clipped > 0.0):
        return int(np.argmax(policy))
    adjusted = np.zeros_like(clipped)
    positive_mask = clipped > 0.0
    adjusted[positive_mask] = np.exp(np.log(clipped[positive_mask]) / temperature)
    mass = float(np.sum(adjusted))
    if mass <= 0.0:
        return int(np.argmax(policy))
    adjusted /= mass
    return int(np.random.choice(np.arange(policy.shape[0]), p=adjusted))


def choose_komi(cfg: ExperimentConfig) -> float:
    if cfg.rules.selfplay_komi:
        return float(random.choice(cfg.rules.selfplay_komi))
    return float(cfg.rules.komi)


def action_to_label(action: int, state: GameState) -> str:
    if action == action_size(state) - 1:
        return "pass"
    return state.topology.labels.get(action, str(action))


def summarize_policy(policy: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    top_indices = np.argsort(policy)[::-1][:top_k]
    return [(int(index), float(policy[index])) for index in top_indices]


def curriculum_strength(cfg: ExperimentConfig, iteration: int) -> float:
    curriculum = cfg.curriculum
    if not curriculum.enabled:
        return 0.0
    start_iteration = max(1, int(curriculum.start_iteration))
    full_until_iteration = max(start_iteration, int(curriculum.full_strength_until_iteration or start_iteration))
    end_iteration = max(full_until_iteration, int(curriculum.end_iteration or full_until_iteration))
    if iteration < start_iteration or iteration > end_iteration:
        return 0.0
    if iteration <= full_until_iteration:
        return 1.0
    remaining = end_iteration - iteration + 1
    span = max(end_iteration - full_until_iteration, 1)
    return max(0.0, min(float(remaining) / float(span), 1.0))


def _empty_region_maps(state: GameState) -> tuple[dict[int, tuple[int, ...]], dict[int, set[Stone]], dict[int, tuple[int, ...]]]:
    region_points: dict[int, tuple[int, ...]] = {}
    region_borders: dict[int, set[Stone]] = {}
    region_border_vertices: dict[int, tuple[int, ...]] = {}
    for points, borders in state._empty_regions(state.board):
        ordered_points = tuple(sorted(points))
        border_copy = borders.copy()
        border_vertices = sorted(
            {
                neighbor
                for vertex in ordered_points
                for neighbor in state.topology.adjacency[vertex]
                if state.board[neighbor] != Stone.EMPTY
            }
        )
        for vertex in ordered_points:
            region_points[vertex] = ordered_points
            region_borders[vertex] = border_copy
            region_border_vertices[vertex] = tuple(border_vertices)
    return region_points, region_borders, region_border_vertices


def _opponent_can_fill_region(state: GameState, points: tuple[int, ...], owner: Stone) -> bool:
    opponent = Stone.opposite(owner)
    for order in permutations(points):
        test = state.copy()
        fillable = True
        for vertex in order:
            if test.board[vertex] != Stone.EMPTY:
                fillable = False
                break
            test.to_play = opponent
            move = Move.place(vertex)
            if not test.is_legal(move):
                fillable = False
                break
            test.apply_move(move)
        if fillable:
            return True
    return False


def _region_is_single_connected_eye(state: GameState, border_vertices: tuple[int, ...], owner: Stone) -> bool:
    if not border_vertices:
        return False
    owner_vertices = [vertex for vertex in border_vertices if state.board[vertex] == owner]
    if not owner_vertices:
        return False
    group, _ = state.collect_group_with_liberties(owner_vertices[0])
    connected_group = set(group)
    return all(vertex in connected_group for vertex in owner_vertices)


def _small_true_eye_cache(state: GameState) -> dict[int, bool]:
    region_points, region_borders, region_border_vertices = _empty_region_maps(state)
    cache: dict[tuple[tuple[int, ...], int, tuple[int, ...]], bool] = {}
    verdicts: dict[int, bool] = {}
    for vertex, points in region_points.items():
        borders = region_borders[vertex]
        if len(points) > 2 or len(borders) != 1:
            verdicts[vertex] = False
            continue
        owner = next(iter(borders))
        border_vertices = region_border_vertices[vertex]
        key = (points, int(owner), border_vertices)
        if key not in cache:
            cache[key] = _region_is_single_connected_eye(state, border_vertices, owner) and not _opponent_can_fill_region(
                state, points, owner
            )
        verdicts[vertex] = cache[key]
    return verdicts


def _small_true_eye_owner_cache(state: GameState) -> dict[int, Stone]:
    region_points, region_borders, region_border_vertices = _empty_region_maps(state)
    cache: dict[tuple[tuple[int, ...], int, tuple[int, ...]], bool] = {}
    owners: dict[int, Stone] = {}
    for vertex, points in region_points.items():
        borders = region_borders[vertex]
        if len(points) > 2 or len(borders) != 1:
            continue
        owner = next(iter(borders))
        border_vertices = region_border_vertices[vertex]
        key = (points, int(owner), border_vertices)
        if key not in cache:
            cache[key] = _region_is_single_connected_eye(state, border_vertices, owner) and not _opponent_can_fill_region(
                state, points, owner
            )
        if cache[key]:
            owners[vertex] = owner
    return owners


def _curriculum_policy_scale_for_capture(cfg: ExperimentConfig, captures_by_move: int, strength: float) -> float:
    if captures_by_move <= 0 or strength <= 0.0:
        return 1.0
    curriculum = cfg.curriculum
    scale = float(curriculum.selfplay_capture_bonus) + max(captures_by_move - 1, 0) * float(curriculum.selfplay_capture_bonus_per_stone)
    cap = max(1.0, float(curriculum.selfplay_capture_bonus_cap))
    adjusted = min(scale, cap)
    return 1.0 + (adjusted - 1.0) * strength


def _curriculum_policy_scale_for_immediate_capture_risk(cfg: ExperimentConfig, risk_stones: int, strength: float) -> float:
    if risk_stones <= 0 or strength <= 0.0:
        return 1.0
    curriculum = cfg.curriculum
    weight = float(curriculum.sample_immediate_capture_risk_weight) + max(risk_stones - 1, 0) * float(
        curriculum.sample_immediate_capture_risk_per_stone
    )
    cap = max(1.0, float(curriculum.sample_immediate_capture_risk_cap))
    adjusted = min(weight, cap)
    return 1.0 / (1.0 + (adjusted - 1.0) * strength)


def _static_immediate_capture_risk(state: GameState, action: int) -> int:
    move = Move.pass_turn() if action == action_size(state) - 1 else Move.place(action)
    if not state.is_legal(move):
        return 0
    child = state.copy()
    child.apply_move(move)
    max_captures = 0
    for reply in child.legal_moves():
        if reply.kind != "place":
            continue
        max_captures = max(max_captures, child.capture_count_for_move(reply.index))
    return max_captures


def _curriculum_sample_weight(cfg: ExperimentConfig, stats: CurriculumMoveStats, strength: float) -> float:
    if strength <= 0.0:
        return 1.0
    curriculum = cfg.curriculum
    weight = 1.0
    if stats.captures_by_move > 0:
        capture_weight = float(curriculum.sample_capture_weight) + max(stats.captures_by_move - 1, 0) * float(
            curriculum.sample_capture_weight_per_stone
        )
        capture_cap = max(1.0, float(curriculum.sample_capture_weight_cap))
        weight *= 1.0 + (min(capture_weight, capture_cap) - 1.0) * strength
    if stats.next_move_capture_stones > 0:
        risk_weight = float(curriculum.sample_immediate_capture_risk_weight) + max(
            stats.next_move_capture_stones - 1, 0
        ) * float(curriculum.sample_immediate_capture_risk_per_stone)
        risk_cap = max(1.0, float(curriculum.sample_immediate_capture_risk_cap))
        weight *= 1.0 + (min(risk_weight, risk_cap) - 1.0) * strength
    if stats.fills_small_true_eye and stats.captures_by_move == 0:
        penalty_weight = min(1.0, max(0.0, float(curriculum.sample_true_eye_penalty_weight)))
        weight *= 1.0 + (penalty_weight - 1.0) * strength
    return min(max(weight, float(curriculum.sample_weight_min)), float(curriculum.sample_weight_max))


def _curriculum_value_bonus(cfg: ExperimentConfig, stats: CurriculumMoveStats, strength: float) -> float:
    if strength <= 0.0:
        return 0.0
    curriculum = cfg.curriculum
    capture_bonus = 0.0
    if stats.captures_by_move > 0:
        capture_bonus = float(curriculum.value_capture_bonus) + max(stats.captures_by_move - 1, 0) * float(
            curriculum.value_capture_bonus_per_stone
        )
        capture_bonus = min(capture_bonus, float(curriculum.value_capture_bonus_cap))
    next_capture_penalty = 0.0
    if stats.next_move_capture_stones > 0:
        next_capture_penalty = float(curriculum.value_next_move_capture_penalty) + max(
            stats.next_move_capture_stones - 1, 0
        ) * float(curriculum.value_next_move_capture_penalty_per_stone)
        next_capture_penalty = min(next_capture_penalty, float(curriculum.value_next_move_capture_penalty_cap))
    eye_fill_penalty = float(curriculum.value_true_eye_penalty) if stats.fills_small_true_eye and stats.captures_by_move == 0 else 0.0
    return strength * (capture_bonus - next_capture_penalty - eye_fill_penalty)


def _shaped_value_target(base_value: float, curriculum_bonus: float) -> float:
    del curriculum_bonus
    return float(np.clip(base_value, -1.0, 1.0))


def _backfill_next_move_capture_signal(
    previous_stats: CurriculumMoveStats,
    previous_trace: SelfPlayMoveTrace,
    next_move_captures: int,
    cfg: ExperimentConfig,
    strength: float,
) -> None:
    previous_stats.next_move_capture_stones = max(0, int(next_move_captures))
    previous_stats.curriculum_weight = _curriculum_sample_weight(cfg, previous_stats, strength)
    previous_stats.curriculum_value_bonus = _curriculum_value_bonus(cfg, previous_stats, strength)
    previous_trace.next_move_capture_stones = previous_stats.next_move_capture_stones
    previous_trace.curriculum_value_bonus = previous_stats.curriculum_value_bonus
    previous_trace.curriculum_weight = previous_stats.curriculum_weight


def curriculum_move_stats(
    state: GameState,
    action: int,
    cfg: ExperimentConfig,
    strength: float,
    true_eye_cache: dict[int, bool] | None = None,
    true_eye_owner_cache: dict[int, Stone] | None = None,
) -> CurriculumMoveStats:
    stones_lost_prev_turn = int(state.last_move_captures)
    pass_action = action_size(state) - 1
    immediate_capture_risk_stones = _static_immediate_capture_risk(state, action) if strength > 0.0 else 0
    if action == pass_action:
        base_stats = CurriculumMoveStats(
            stones_lost_prev_turn=stones_lost_prev_turn,
            immediate_capture_risk_stones=immediate_capture_risk_stones,
            curriculum_weight=1.0,
        )
        base_stats.curriculum_weight = _curriculum_sample_weight(cfg, base_stats, strength)
        base_stats.curriculum_value_bonus = _curriculum_value_bonus(cfg, base_stats, strength)
        return CurriculumMoveStats(
            stones_lost_prev_turn=stones_lost_prev_turn,
            immediate_capture_risk_stones=immediate_capture_risk_stones,
            curriculum_value_bonus=base_stats.curriculum_value_bonus,
            curriculum_weight=base_stats.curriculum_weight,
        )
    captures_by_move = state.capture_count_for_move(action)
    fills_small_true_eye = False
    if true_eye_cache is None or true_eye_owner_cache is None:
        true_eye_cache = _small_true_eye_cache(state)
        true_eye_owner_cache = _small_true_eye_owner_cache(state)
    if true_eye_cache is not None and true_eye_owner_cache is not None:
        fills_small_true_eye = bool(true_eye_cache.get(action, False)) and true_eye_owner_cache.get(action) == state.to_play
    stats = CurriculumMoveStats(
        captures_by_move=captures_by_move,
        stones_lost_prev_turn=stones_lost_prev_turn,
        immediate_capture_risk_stones=immediate_capture_risk_stones,
        fills_small_true_eye=fills_small_true_eye,
        curriculum_weight=1.0,
    )
    stats.curriculum_weight = _curriculum_sample_weight(cfg, stats, strength)
    stats.curriculum_value_bonus = _curriculum_value_bonus(cfg, stats, strength)
    return CurriculumMoveStats(
        captures_by_move=captures_by_move,
        stones_lost_prev_turn=stones_lost_prev_turn,
        immediate_capture_risk_stones=immediate_capture_risk_stones,
        fills_small_true_eye=fills_small_true_eye,
        curriculum_value_bonus=stats.curriculum_value_bonus,
        curriculum_weight=stats.curriculum_weight,
    )


def eye_fill_bad_action_mask(state: GameState, cfg: ExperimentConfig) -> np.ndarray:
    del cfg
    mask = np.zeros(action_size(state), dtype=np.float32)
    true_eye_cache = _small_true_eye_cache(state)
    true_eye_owner_cache = _small_true_eye_owner_cache(state)
    pass_action = action_size(state) - 1
    for move in state.legal_moves():
        if move.kind != "place":
            continue
        fills_self_true_eye = bool(true_eye_cache.get(move.index, False)) and true_eye_owner_cache.get(move.index) == state.to_play
        if fills_self_true_eye and state.capture_count_for_move(move.index) == 0:
            mask[move.index] = 1.0
    mask[pass_action] = 0.0
    return mask


def apply_curriculum_policy_shaping(
    state: GameState,
    policy: np.ndarray,
    cfg: ExperimentConfig,
    strength: float,
    legal_actions: list[int] | None = None,
) -> tuple[np.ndarray, dict[int, CurriculumMoveStats]]:
    legal_actions = (
        legal_actions
        if legal_actions is not None
        else [action_size(state) - 1 if move.kind == "pass" else move.index for move in state.legal_moves()]
    )
    if not legal_actions:
        return policy, {}
    true_eye_cache = _small_true_eye_cache(state)
    true_eye_owner_cache = _small_true_eye_owner_cache(state)
    pass_action = action_size(state) - 1
    adjusted = policy.copy()
    action_stats: dict[int, CurriculumMoveStats] = {}
    for action in legal_actions:
        stats = curriculum_move_stats(
            state,
            action,
            cfg,
            strength,
            true_eye_cache=true_eye_cache,
            true_eye_owner_cache=true_eye_owner_cache,
        )
        action_stats[action] = stats
        if action == pass_action:
            continue
        if strength <= 0.0:
            continue
        if stats.captures_by_move > 0:
            adjusted[action] *= _curriculum_policy_scale_for_capture(cfg, stats.captures_by_move, strength)
        elif stats.fills_small_true_eye:
            penalty = min(1.0, max(0.0, float(cfg.curriculum.selfplay_true_eye_penalty)))
            adjusted[action] *= 1.0 + (penalty - 1.0) * strength
        if stats.immediate_capture_risk_stones > 0:
            adjusted[action] *= _curriculum_policy_scale_for_immediate_capture_risk(
                cfg, stats.immediate_capture_risk_stones, strength
            )
    if strength <= 0.0:
        return policy, action_stats
    legal_mass = float(np.sum(adjusted[legal_actions]))
    if legal_mass <= 0.0:
        return policy, action_stats
    adjusted[legal_actions] /= legal_mass
    return adjusted, action_stats


def classify_selfplay_game(trace: SelfPlayGameTrace, cfg: ExperimentConfig) -> tuple[list[str], float]:
    tags: list[str] = []
    early_pass_cutoff = max(0, cfg.rules.opening_no_pass_moves + cfg.selfplay.early_pass_extra_moves)
    if trace.first_pass_turn > 0 and trace.first_pass_turn <= early_pass_cutoff:
        tags.append("early_pass")
    if trace.move_count <= max(0, cfg.selfplay.short_game_turn_threshold):
        tags.append("short_game")
    sample_weight = float(cfg.selfplay.abnormal_sample_weight) if tags else 1.0
    return tags, max(0.0, sample_weight)


def terminal_supervision_weight(trace: SelfPlayGameTrace, cfg: ExperimentConfig) -> float:
    weight = 1.0
    training = cfg.training
    if trace.end_reason == "max_moves":
        weight = min(weight, float(training.terminal_weight_on_max_moves))
    if trace.unresolved_dead_groups > 0:
        weight = min(weight, float(training.terminal_weight_on_unresolved_dead_groups))
    if (
        trace.cleaned_dead_stones > int(training.terminal_weight_cleanup_dead_stones_threshold)
        or trace.total_passes > int(training.terminal_weight_total_passes_threshold)
        or abs(trace.black_score - trace.white_score) > float(training.terminal_weight_score_margin_threshold)
    ):
        weight = min(weight, float(training.terminal_weight_on_noisy_cleanup))
    return min(max(weight, 0.0), 1.0)


def generate_selfplay_game(
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
    exploration_scale: float = 1.0,
    iteration: int = 1,
) -> tuple[list[TrainingSample], SelfPlayGameTrace]:
    komi = choose_komi(cfg)
    shaping_scale = curriculum_strength(cfg, iteration)
    is_curriculum_sample = shaping_scale > 0.0
    game = GameState(
        GameConfig(
            side_length=cfg.model.board_side,
            komi=komi,
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
    search = MCTS(
        model,
        cfg.mcts,
        device,
        input_history=cfg.model.input_history,
        root_noise_enabled=True,
        exploration_scale=exploration_scale,
        root_prior_shaper=(
            lambda state, priors, legal_actions: apply_curriculum_policy_shaping(state, priors, cfg, shaping_scale, legal_actions)[0]
        )
        if shaping_scale > 0.0
        else None,
    )
    trajectories: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    trace_moves: list[SelfPlayMoveTrace] = []
    selected_move_stats: list[CurriculumMoveStats] = []

    for turn in range(cfg.selfplay.max_moves):
        if game.finished:
            break
        search_result = search.run(game)
        policy = search_result.policy
        root_value = search_result.root_value
        root_score_margin = search_result.root_score
        root_score_margin_black_minus_white = root_score_margin if game.to_play == Stone.BLACK else -root_score_margin
        trajectories.append(
            (
                encode_state(game, cfg.model.input_history),
                liberty_global_features(game),
                policy.copy(),
                eye_fill_bad_action_mask(game, cfg),
                liberty_target_classes(game),
                int(game.to_play),
            )
        )
        action = sample_move_from_policy(policy, sampling_temperature(turn, cfg))
        move_stats = curriculum_move_stats(game, action, cfg, shaping_scale)
        if selected_move_stats:
            _backfill_next_move_capture_signal(
                selected_move_stats[-1],
                trace_moves[-1],
                move_stats.captures_by_move,
                cfg,
                shaping_scale,
            )
        selected_move_stats.append(move_stats)
        player = "B" if game.to_play == Stone.BLACK else "W"
        trace_moves.append(
            SelfPlayMoveTrace(
                turn=turn + 1,
                player=player,
                action=action,
                move=action_to_label(action, game),
                root_value=float(root_value),
                root_score_margin=float(root_score_margin),
                root_score_margin_black_minus_white=float(root_score_margin_black_minus_white),
                score_margin_error_black_minus_white=0.0,
                policy=policy.astype(float).tolist(),
                policy_top=summarize_policy(policy, cfg.telemetry.top_policy_moves),
                captures_by_move=move_stats.captures_by_move,
                stones_lost_prev_turn=move_stats.stones_lost_prev_turn,
                immediate_capture_risk_stones=move_stats.immediate_capture_risk_stones,
                next_move_capture_stones=move_stats.next_move_capture_stones,
                curriculum_value_bonus=move_stats.curriculum_value_bonus,
                is_curriculum_sample=is_curriculum_sample,
                fills_small_true_eye=move_stats.fills_small_true_eye,
                curriculum_weight=move_stats.curriculum_weight,
            )
        )
        move = Move.pass_turn() if action == action_size(game) - 1 else Move.place(action)
        game.apply_move(move)

    forced_end_reason = "max_moves" if len(game.move_history) >= cfg.selfplay.max_moves and not game.finished else (game.end_reason or "score")
    board_before_cleanup = game.board.copy()
    result = game.finalize_score(forced_end_reason)
    cleaned_dead_vertices = [
        index
        for index, (before, after) in enumerate(zip(board_before_cleanup, game.board))
        if before != Stone.EMPTY and after == Stone.EMPTY
    ]
    final_value = result.value
    ownership = game.ownership_map()
    score_margin = game.score_margin_black_minus_white()
    samples: list[TrainingSample] = []
    for (planes, global_features, policy_target, bad_action_mask, liberty_target, player), move_stats in zip(trajectories, selected_move_stats):
        value = final_value if player == 1 else -final_value
        shaped_value = _shaped_value_target(value, move_stats.curriculum_value_bonus)
        ownership_target = ownership.copy() if player == 1 else -ownership.copy()
        score_target = score_margin if player == 1 else -score_margin
        samples.append(
            TrainingSample(
                state_planes=planes,
                global_features=global_features,
                policy_target=policy_target,
                value_target=shaped_value,
                ownership_target=ownership_target,
                score_target=score_target,
                liberty_target=liberty_target,
                eye_fill_bad_action_mask=bad_action_mask,
                captures_by_move=move_stats.captures_by_move,
                stones_lost_prev_turn=move_stats.stones_lost_prev_turn,
                next_move_capture_stones=move_stats.next_move_capture_stones,
                curriculum_value_bonus=move_stats.curriculum_value_bonus,
                curriculum_scale_at_generation=shaping_scale,
                is_curriculum_sample=is_curriculum_sample,
                fills_small_true_eye=move_stats.fills_small_true_eye,
            )
        )
    winner = "draw"
    if result.black > result.white:
        winner = "B"
    elif result.white > result.black:
        winner = "W"
    pass_turns = [move.turn for move in trace_moves if move.move == "pass"]
    for move in trace_moves:
        move.score_margin_error_black_minus_white = move.root_score_margin_black_minus_white - score_margin
    avg_abs_predicted_score_error = (
        sum(abs(move.score_margin_error_black_minus_white) for move in trace_moves) / len(trace_moves) if trace_moves else 0.0
    )
    trace = SelfPlayGameTrace(
        komi=komi,
        move_count=len(game.move_history),
        black_score=float(result.black),
        white_score=float(result.white),
        result_value=float(result.value),
        winner=winner,
        first_player_win=result.value > 0,
        first_pass_turn=pass_turns[0] if pass_turns else 0,
        total_passes=len(pass_turns),
        cleaned_dead_stones=game.cleaned_dead_stones,
        cleanup_rule_resolved_groups=game.cleanup_rule_resolved_groups,
        cleanup_local_search_resolved_groups=game.cleanup_local_search_resolved_groups,
        cleanup_preserved_seki_groups=game.cleanup_preserved_seki_groups,
        unresolved_dead_groups=game.unresolved_dead_groups,
        end_reason=game.end_reason or forced_end_reason,
        avg_abs_predicted_score_error=float(avg_abs_predicted_score_error),
        sample_weight=1.0,
        abnormal_tags=[],
        curriculum_scale=float(shaping_scale),
        total_captured_stones=sum(move.captures_by_move for move in trace_moves),
        total_prev_turn_losses=sum(move.stones_lost_prev_turn for move in trace_moves),
        total_immediate_capture_risk_stones=sum(move.immediate_capture_risk_stones for move in trace_moves),
        total_next_move_capture_stones=sum(move.next_move_capture_stones for move in trace_moves),
        total_curriculum_value_bonus=float(sum(move.curriculum_value_bonus for move in trace_moves)),
        true_eye_fills=sum(1 for move in trace_moves if move.fills_small_true_eye),
        pure_capture_moves=sum(1 for move in trace_moves if move.captures_by_move > 0),
        immediate_capture_risk_rate_moves=sum(1 for move in trace_moves if move.immediate_capture_risk_stones > 0),
        next_move_capture_rate_moves=sum(1 for move in trace_moves if move.next_move_capture_stones > 0),
        curriculum_sample_moves=sum(1 for move in trace_moves if move.is_curriculum_sample),
        positive_curriculum_bonus_moves=sum(1 for move in trace_moves if move.curriculum_value_bonus > 1e-6),
        negative_curriculum_bonus_moves=sum(1 for move in trace_moves if move.curriculum_value_bonus < -1e-6),
        true_eye_penalized_moves=sum(1 for move in trace_moves if move.fills_small_true_eye and move.captures_by_move == 0),
        final_board=[int(stone) for stone in game.board],
        cleaned_dead_vertices=cleaned_dead_vertices,
        moves=trace_moves,
    )
    abnormal_tags, sample_weight = classify_selfplay_game(trace, cfg)
    terminal_weight_scale = terminal_supervision_weight(trace, cfg)
    trace.abnormal_tags = abnormal_tags
    for sample, move_stats in zip(samples, selected_move_stats):
        sample.sample_weight = sample_weight * move_stats.curriculum_weight
        sample.policy_weight = sample.sample_weight
        sample.terminal_weight = sample.sample_weight * terminal_weight_scale
    trace.sample_weight = sum(sample.sample_weight for sample in samples) / len(samples) if samples else sample_weight
    return samples, trace


def generate_selfplay_batch(
    model_state: dict[str, Any],
    cfg: ExperimentConfig,
    game_count: int,
    seed: int,
    exploration_scale: float = 1.0,
    iteration: int = 1,
) -> SelfPlayBatchResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    cfg.model.action_size = cfg.model.action_size or action_size(GameState(GameConfig(side_length=cfg.model.board_side)))
    model = PolicyValueNet(cfg.model)
    model.load_state_dict(model_state)
    model.eval()

    device = torch.device("cpu")
    samples: list[TrainingSample] = []
    traces: list[SelfPlayGameTrace] = []
    for _ in range(game_count):
        game_samples, trace = generate_selfplay_game(model, cfg, device, exploration_scale=exploration_scale, iteration=iteration)
        samples.extend(game_samples)
        traces.append(trace)
    return samples, traces
