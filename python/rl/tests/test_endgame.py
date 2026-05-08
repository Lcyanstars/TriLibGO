from __future__ import annotations

import unittest
from unittest.mock import patch

from python.rl.game import GameConfig, GameState, Stone


class EndgameResolutionTests(unittest.TestCase):
    def test_proof_search_preserves_mixed_border_groups(self) -> None:
        state = GameState(GameConfig(side_length=2, cleanup_dead_stones=True))
        board = [Stone.EMPTY] * state.topology.vertex_count
        board[10] = Stone.BLACK
        board[11] = Stone.WHITE
        state.board = board
        state.board_history = [board.copy()]
        state.finished = True
        state.end_reason = "double_pass"

        score = state.score()

        self.assertEqual(state.cleaned_dead_stones, 0)
        self.assertEqual(state.cleanup_preserved_seki_groups, 2)
        self.assertEqual(state.unresolved_dead_groups, 0)
        self.assertEqual(state.board[10], Stone.BLACK)
        self.assertEqual(state.board[11], Stone.WHITE)
        self.assertGreater(score.black, 0.0)
        self.assertGreater(score.white, 0.0)

    def test_proof_search_preserves_benson_alive_group(self) -> None:
        state = GameState(GameConfig(side_length=2, cleanup_dead_stones=True))
        black_vertices = {0, 1, 2, 5, 6, 7}
        eye_vertices = {3, 4, 8, 9}
        board = [Stone.WHITE] * state.topology.vertex_count
        for vertex in black_vertices:
            board[vertex] = Stone.BLACK
        for vertex in eye_vertices:
            board[vertex] = Stone.EMPTY
        state.board = board
        state.board_history = [board.copy()]
        state.finished = True
        state.end_reason = "double_pass"

        state.score()

        for vertex in black_vertices:
            self.assertEqual(state.board[vertex], Stone.BLACK)
        self.assertEqual(state.cleaned_dead_stones, 14)
        self.assertGreaterEqual(state.cleanup_rule_resolved_groups, 1)
        self.assertEqual(state.unresolved_dead_groups, 0)

    def test_proof_search_shares_budget_across_candidates(self) -> None:
        state = GameState(GameConfig(side_length=2, cleanup_dead_stones=True, cleanup_local_search_nodes=1))
        state.finished = True
        state.end_reason = "double_pass"

        analyze_calls = {"count": 0}
        empty_analysis = type(
            "Analysis",
            (),
            {
                "blocks": (),
                "regions": (),
                "adjacency": tuple(tuple() for _ in range(state.topology.vertex_count)),
                "vertex_to_block": tuple([-1] * state.topology.vertex_count),
                "vertex_to_region": tuple([-1] * state.topology.vertex_count),
            },
        )()

        candidate_analysis = type(
            "Analysis",
            (),
            {
                "blocks": (
                    type("Block", (), {"color": Stone.BLACK, "stones": (0,), "liberties": frozenset({1}), "liberty_regions": frozenset()})(),
                    type("Block", (), {"color": Stone.WHITE, "stones": (2,), "liberties": frozenset({3}), "liberty_regions": frozenset()})(),
                    type("Block", (), {"color": Stone.BLACK, "stones": (4,), "liberties": frozenset({5}), "liberty_regions": frozenset()})(),
                ),
                "regions": (),
                "adjacency": tuple(tuple() for _ in range(state.topology.vertex_count)),
                "vertex_to_block": tuple([-1] * state.topology.vertex_count),
                "vertex_to_region": tuple([-1] * state.topology.vertex_count),
            },
        )()

        def fake_analyze_board(_state: GameState, _board: list[Stone]):
            analyze_calls["count"] += 1
            return candidate_analysis if analyze_calls["count"] == 1 else empty_analysis

        prove_calls: list[tuple[int, int]] = []

        def fake_prove(_state: GameState, _board: list[Stone], _analysis, block_id: int, budget: list[int] | None = None) -> bool:
            assert budget is not None
            prove_calls.append((block_id, budget[0]))
            budget[0] = 0
            return False

        with patch("python.rl.endgame.analyze_board", side_effect=fake_analyze_board), patch(
            "python.rl.endgame._alive_blocks", return_value=set()
        ), patch("python.rl.endgame._candidate_block_ids", side_effect=[[0, 1, 2], []]), patch(
            "python.rl.endgame._ordered_candidate_block_ids", side_effect=lambda _analysis, candidates: candidates
        ), patch("python.rl.endgame._prove_block_dead", side_effect=fake_prove):
            state.score()

        self.assertEqual(prove_calls, [(0, 1)])
        self.assertEqual(state.cleaned_dead_stones, 0)
        self.assertEqual(state.unresolved_dead_groups, 0)


if __name__ == "__main__":
    unittest.main()
