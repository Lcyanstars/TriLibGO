from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.rl.game import GameState, Stone
from python.tools.verify_record import parse_record


def feature_sums(state: GameState, history: int = 2) -> list[float]:
    current = float(sum(1 for stone in state.board if stone == state.to_play))
    opponent = float(sum(1 for stone in state.board if stone == Stone.opposite(state.to_play)))
    legal = float(sum(1 for move in state.legal_moves() if move.kind == "place"))
    to_play = float(state.topology.vertex_count if state.to_play == Stone.BLACK else 0.0)
    last_move = 0.0
    if state.move_history:
        move = state.move_history[-1]
        if move.kind == "place":
            last_move = 1.0
    passes = float(state.topology.vertex_count * min(state.consecutive_passes, 2) / 2.0)
    sums = [current, opponent, legal, to_play, last_move, passes]
    for _ in range(max(history - 1, 0)):
        sums.extend([0.0, 0.0])
    return sums


def main() -> None:
    fixtures = json.loads((ROOT / "records" / "position_fixture_expectations.json").read_text(encoding="utf-8"))
    failures: list[str] = []

    for fixture in fixtures:
        config, moves = parse_record(ROOT / "records" / fixture["record"])
        state = GameState(config)
        for move in moves[: fixture["ply"]]:
            state.apply_move(move)

        legal = state.legal_moves()
        legal_place_indices = [move.index for move in legal if move.kind == "place"]
        actual = {
            "to_play": int(state.to_play),
            "occupied": sum(1 for stone in state.board if int(stone) != 0),
            "legal_count": len(legal),
            "pass_index": state.topology.vertex_count,
            "action_size": state.topology.vertex_count + 1,
            "legal_prefix": legal_place_indices[:12],
            "feature_sums": feature_sums(state),
        }

        for key in ("to_play", "occupied", "legal_count", "pass_index", "action_size", "legal_prefix"):
            if actual[key] != fixture[key]:
                failures.append(f'{fixture["name"]}: expected {key}={fixture[key]}, got {actual[key]}')

        expected_sums = fixture["feature_sums"]
        for index, (lhs, rhs) in enumerate(zip(actual["feature_sums"], expected_sums)):
            if not math.isclose(lhs, rhs, rel_tol=1e-6, abs_tol=1e-6):
                failures.append(f'{fixture["name"]}: feature_sums[{index}] expected {rhs}, got {lhs}')

    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)

    print("All position fixtures verified.")


if __name__ == "__main__":
    main()
