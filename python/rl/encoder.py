from __future__ import annotations

import numpy as np

from .game import GameState, Move, Stone


LIBERTY_CLASS_COUNT = 7


def _encode_board_for_player(board: list[Stone], to_play: Stone) -> tuple[np.ndarray, np.ndarray]:
    current = np.array([1.0 if stone == to_play else 0.0 for stone in board], dtype=np.float32)
    opponent = np.array([1.0 if stone == Stone.opposite(to_play) else 0.0 for stone in board], dtype=np.float32)
    return current, opponent


def _liberty_bucket(liberties: int) -> int:
    if liberties <= 1:
        return 1
    if liberties == 2:
        return 2
    return 3


def liberty_global_features(state: GameState) -> np.ndarray:
    visited: set[int] = set()
    opponent = Stone.opposite(state.to_play)
    counts = [0] * 6
    for vertex, stone in enumerate(state.board):
        if stone == Stone.EMPTY or vertex in visited:
            continue
        group, liberties = state.collect_group_with_liberties(vertex)
        visited.update(group)
        bucket = _liberty_bucket(len(liberties)) - 1
        if stone == state.to_play:
            plane_index = bucket
        elif stone == opponent:
            plane_index = 3 + bucket
        else:
            continue
        counts[plane_index] += 1
    liberty_counts = np.asarray(counts, dtype=np.float32) / max(float(state.topology.vertex_count), 1.0)
    extras = np.asarray(
        [
            1.0 if state.to_play == Stone.BLACK else 0.0,
            min(state.consecutive_passes, 2) / 2.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([liberty_counts, extras], axis=0)


def liberty_target_classes(state: GameState) -> np.ndarray:
    targets = np.zeros(state.topology.vertex_count, dtype=np.int64)
    visited: set[int] = set()
    opponent = Stone.opposite(state.to_play)
    for vertex, stone in enumerate(state.board):
        if stone == Stone.EMPTY or vertex in visited:
            continue
        group, liberties = state.collect_group_with_liberties(vertex)
        visited.update(group)
        bucket = _liberty_bucket(len(liberties))
        if stone == state.to_play:
            target_class = bucket
        elif stone == opponent:
            target_class = 3 + bucket
        else:
            continue
        for group_vertex in group:
            targets[group_vertex] = target_class
    return targets


def encode_state(state: GameState, history: int = 2) -> np.ndarray:
    planes: list[np.ndarray] = []
    current, opponent = _encode_board_for_player(state.board, state.to_play)
    legal = np.zeros(state.topology.vertex_count, dtype=np.float32)
    for move in state.legal_moves():
        if move.kind == "place":
            legal[move.index] = 1.0
    last_move = np.zeros(state.topology.vertex_count, dtype=np.float32)
    if state.move_history:
        move = state.move_history[-1]
        if move.kind == "place":
            last_move[move.index] = 1.0
    planes.extend([current, opponent, legal, last_move])
    historical_boards = state.board_history[:-1] if state.board_history else []
    for offset in range(max(history - 1, 0)):
        board_index = len(historical_boards) - 1 - offset
        if board_index >= 0:
            hist_current, hist_opponent = _encode_board_for_player(historical_boards[board_index], state.to_play)
        else:
            hist_current = np.zeros_like(current)
            hist_opponent = np.zeros_like(current)
        planes.extend([hist_current, hist_opponent])
    return np.stack(planes, axis=0)


def action_size(state: GameState) -> int:
    return state.topology.vertex_count + 1


def move_to_action(move: Move, state: GameState) -> int:
    return state.topology.vertex_count if move.kind == "pass" else move.index
