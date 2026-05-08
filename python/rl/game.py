"""Pure-Python TriLibGo game environment for self-play and evaluation.

This module provides a standalone implementation of the TriLibGo rules
that matches the C++ engine's behavior on move legality, capture resolution,
ko detection, and Chinese area scoring. Used by the RL training pipeline
for board simulation and terminal outcome generation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .topology import BoardTopology


class Stone(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    @staticmethod
    def opposite(stone: "Stone") -> "Stone":
        if stone == Stone.BLACK:
            return Stone.WHITE
        if stone == Stone.WHITE:
            return Stone.BLACK
        return Stone.EMPTY


@dataclass
class GameConfig:
    """Game configuration: board size, komi, cleanup and pass rules."""
    side_length: int = 4
    komi: float = 0.0
    allow_suicide: bool = False
    opening_no_pass_moves: int = 0
    cleanup_dead_stones: bool = True
    cleanup_dead_stones_mode: str = "proof_search"
    cleanup_candidate_max_liberties: int = 4
    cleanup_local_search_depth: int = 10
    cleanup_local_search_nodes: int = 2000
    cleanup_preserve_seki: bool = True
    cleanup_mark_unresolved: bool = True


@dataclass
class Move:
    """A move: either a placement on a vertex or a pass."""
    kind: str
    index: int = -1

    @staticmethod
    def place(index: int) -> "Move":
        return Move("place", index)

    @staticmethod
    def pass_turn() -> "Move":
        return Move("pass", -1)


@dataclass
class Score:
    black: float
    white: float

    @property
    def value(self) -> float:
        if self.black > self.white:
            return 1.0
        if self.white > self.black:
            return -1.0
        return 0.0


@dataclass
class GameState:
    config: GameConfig
    topology: BoardTopology = field(init=False)
    board: list[Stone] = field(init=False)
    to_play: Stone = Stone.BLACK
    consecutive_passes: int = 0
    move_history: list[Move] = field(default_factory=list)
    board_history: list[list[Stone]] = field(default_factory=list)
    hash_history: list[int] = field(default_factory=list)
    finished: bool = False
    result: Score | None = None
    cleaned_dead_stones: int = 0
    cleanup_rule_resolved_groups: int = 0
    cleanup_local_search_resolved_groups: int = 0
    cleanup_preserved_seki_groups: int = 0
    unresolved_dead_groups: int = 0
    end_reason: str = ""
    last_move_captures: int = 0

    def __post_init__(self) -> None:
        self.topology = BoardTopology(self.config.side_length)
        self.board = [Stone.EMPTY] * self.topology.vertex_count
        self.board_history = [self.board.copy()]
        self.hash_history = [self.compute_hash()]

    def copy(self) -> "GameState":
        copied = GameState(self.config)
        copied.board = self.board.copy()
        copied.to_play = self.to_play
        copied.consecutive_passes = self.consecutive_passes
        copied.move_history = self.move_history.copy()
        copied.board_history = [board.copy() for board in self.board_history]
        copied.hash_history = self.hash_history.copy()
        copied.finished = self.finished
        copied.result = self.result
        copied.cleaned_dead_stones = self.cleaned_dead_stones
        copied.cleanup_rule_resolved_groups = self.cleanup_rule_resolved_groups
        copied.cleanup_local_search_resolved_groups = self.cleanup_local_search_resolved_groups
        copied.cleanup_preserved_seki_groups = self.cleanup_preserved_seki_groups
        copied.unresolved_dead_groups = self.unresolved_dead_groups
        copied.end_reason = self.end_reason
        copied.last_move_captures = self.last_move_captures
        return copied

    def compute_hash(self) -> int:
        value = 1469598103934665603
        for stone in self.board:
            value ^= int(stone) + 1
            value *= 1099511628211
        value ^= int(self.to_play) + 11
        value *= 1099511628211
        return value & ((1 << 64) - 1)

    def collect_group(self, start: int) -> tuple[list[int], int]:
        color = self.board[start]
        if color == Stone.EMPTY:
            return [], 0
        queue: deque[int] = deque([start])
        visited = {start}
        stones: list[int] = []
        liberties: set[int] = set()
        while queue:
            current = queue.popleft()
            stones.append(current)
            for neighbor in self.topology.adjacency[current]:
                stone = self.board[neighbor]
                if stone == Stone.EMPTY:
                    liberties.add(neighbor)
                elif stone == color and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return stones, len(liberties)

    def collect_group_with_liberties(self, start: int) -> tuple[list[int], set[int]]:
        return self._collect_group_on_board(self.board, start)

    def legal_moves(self) -> list[Move]:
        if self.finished:
            return []
        moves = [Move.place(v) for v in range(self.topology.vertex_count) if self.is_legal(Move.place(v))]
        if len(self.move_history) >= self.config.opening_no_pass_moves and self.is_legal(Move.pass_turn()):
            moves.append(Move.pass_turn())
        return moves

    def is_legal(self, move: Move) -> bool:
        if self.finished:
            return False
        if move.kind == "pass":
            return True
        if move.index < 0 or move.index >= self.topology.vertex_count or self.board[move.index] != Stone.EMPTY:
            return False
        test = self.copy()
        test._apply_place(move.index)
        _, liberties = test.collect_group(move.index)
        if not self.config.allow_suicide and liberties == 0:
            return False
        if len(self.hash_history) >= 2:
            original_to_play = test.to_play
            test.to_play = Stone.opposite(self.to_play)
            repeats_simple_ko = test.compute_hash() == self.hash_history[-2]
            test.to_play = original_to_play
            if repeats_simple_ko:
                return False
        return True

    def apply_move(self, move: Move) -> None:
        if move.kind == "pass":
            self.move_history.append(move)
            self.consecutive_passes += 1
            self.last_move_captures = 0
            self.to_play = Stone.opposite(self.to_play)
            self.board_history.append(self.board.copy())
            self.hash_history.append(self.compute_hash())
            if self.consecutive_passes >= 2:
                self.finished = True
                self.end_reason = "double_pass"
                self.result = self.score()
            return
        self.last_move_captures = self._apply_place(move.index)
        self.move_history.append(move)
        self.consecutive_passes = 0
        self.to_play = Stone.opposite(self.to_play)
        self.board_history.append(self.board.copy())
        self.hash_history.append(self.compute_hash())

    def capture_count_for_move(self, index: int, player: Stone | None = None) -> int:
        if index < 0 or index >= self.topology.vertex_count or self.board[index] != Stone.EMPTY:
            return 0
        mover = self.to_play if player is None else player
        enemy = Stone.opposite(mover)
        captured = 0
        seen_groups: set[int] = set()
        for neighbor in self.topology.adjacency[index]:
            if self.board[neighbor] != enemy or neighbor in seen_groups:
                continue
            stones, liberties = self.collect_group_with_liberties(neighbor)
            for stone in stones:
                seen_groups.add(stone)
            if liberties == {index}:
                captured += len(stones)
        return captured

    def _apply_place(self, index: int) -> int:
        self.board[index] = self.to_play
        enemy = Stone.opposite(self.to_play)
        removed = 0
        for neighbor in self.topology.adjacency[index]:
            if self.board[neighbor] != enemy:
                continue
            stones, liberties = self.collect_group(neighbor)
            if liberties == 0:
                for stone in stones:
                    self.board[stone] = Stone.EMPTY
                    removed += 1
        return removed
    def _collect_group_on_board(self, board: list[Stone], start: int) -> tuple[list[int], set[int]]:
        color = board[start]
        if color == Stone.EMPTY:
            return [], set()
        queue: deque[int] = deque([start])
        visited = {start}
        stones: list[int] = []
        liberties: set[int] = set()
        while queue:
            current = queue.popleft()
            stones.append(current)
            for neighbor in self.topology.adjacency[current]:
                stone = board[neighbor]
                if stone == Stone.EMPTY:
                    liberties.add(neighbor)
                elif stone == color and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return stones, liberties

    def _empty_regions(self, board: list[Stone]) -> list[tuple[list[int], set[Stone]]]:
        visited = [False] * self.topology.vertex_count
        regions: list[tuple[list[int], set[Stone]]] = []
        for vertex in range(self.topology.vertex_count):
            if board[vertex] != Stone.EMPTY or visited[vertex]:
                continue
            queue: deque[int] = deque([vertex])
            visited[vertex] = True
            region: list[int] = []
            borders: set[Stone] = set()
            while queue:
                current = queue.popleft()
                region.append(current)
                for neighbor in self.topology.adjacency[current]:
                    stone = board[neighbor]
                    if stone == Stone.EMPTY:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                    else:
                        borders.add(stone)
            regions.append((region, borders))
        return regions

    def _remove_obvious_dead_groups(self, board: list[Stone]) -> int:
        to_remove: set[int] = set()
        visited: set[int] = set()
        for vertex, stone in enumerate(board):
            if stone == Stone.EMPTY or vertex in visited:
                continue
            group, liberties = self._collect_group_on_board(board, vertex)
            visited.update(group)
            if len(liberties) <= 1:
                to_remove.update(group)

        for vertex in to_remove:
            board[vertex] = Stone.EMPTY
        return len(to_remove)

    def _territory_score(self, board: list[Stone]) -> Score:
        black = sum(1 for stone in board if stone == Stone.BLACK)
        white = sum(1 for stone in board if stone == Stone.WHITE)
        for region, borders in self._empty_regions(board):
            if len(borders) == 1:
                if Stone.BLACK in borders:
                    black += len(region)
                else:
                    white += len(region)
        return Score(float(black), float(white) + self.config.komi)

    def ownership_map(self) -> np.ndarray:
        ownership = np.zeros(self.topology.vertex_count, dtype=np.float32)
        for index, stone in enumerate(self.board):
            if stone == Stone.BLACK:
                ownership[index] = 1.0
            elif stone == Stone.WHITE:
                ownership[index] = -1.0
        for region, borders in self._empty_regions(self.board):
            if borders == {Stone.BLACK}:
                for vertex in region:
                    ownership[vertex] = 1.0
            elif borders == {Stone.WHITE}:
                for vertex in region:
                    ownership[vertex] = -1.0
        return ownership

    def score_margin_black_minus_white(self) -> float:
        score = self.result if self.result is not None else self.score()
        return float(score.black - score.white)

    def finalize_score(self, end_reason: str = "score") -> Score:
        if not self.finished:
            self.finished = True
            self.end_reason = end_reason
            self.result = None
        elif end_reason and not self.end_reason:
            self.end_reason = end_reason
        if self.result is None:
            self.result = self.score()
        return self.result

    def score(self) -> Score:
        if not self.finished:
            return self._territory_score(self.board)
        board = self.board.copy()
        self.cleaned_dead_stones = 0
        self.cleanup_rule_resolved_groups = 0
        self.cleanup_local_search_resolved_groups = 0
        self.cleanup_preserved_seki_groups = 0
        self.unresolved_dead_groups = 0
        if self.config.cleanup_dead_stones:
            if self.config.cleanup_dead_stones_mode == "proof_search":
                from .endgame import resolve_terminal_board

                resolution = resolve_terminal_board(self)
                board = resolution.board
                self.cleaned_dead_stones = resolution.removed_stones
                self.cleanup_rule_resolved_groups = resolution.rule_resolved_groups
                self.cleanup_local_search_resolved_groups = resolution.local_search_resolved_groups
                self.cleanup_preserved_seki_groups = resolution.preserved_seki_groups
                self.unresolved_dead_groups = resolution.unresolved_dead_groups
            else:
                self.cleaned_dead_stones = self._remove_obvious_dead_groups(board)
            self.board = board
            if not self.board_history:
                self.board_history = [self.board.copy()]
            else:
                self.board_history[-1] = self.board.copy()
        self.result = self._territory_score(self.board)
        return self.result
