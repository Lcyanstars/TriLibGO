from __future__ import annotations

import json
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


def summarize_position(record_name: str, ply: int) -> dict[str, object]:
    config, moves = parse_record(ROOT / "records" / record_name)
    state = GameState(config)
    for move in moves[:ply]:
        state.apply_move(move)

    legal = state.legal_moves()
    legal_place_indices = [move.index for move in legal if move.kind == "place"]
    return {
        "record": record_name,
        "ply": ply,
        "to_play": int(state.to_play),
        "occupied": sum(1 for stone in state.board if int(stone) != 0),
        "legal_count": len(legal),
        "pass_index": state.topology.vertex_count,
        "action_size": state.topology.vertex_count + 1,
        "legal_prefix": legal_place_indices[:12],
        "feature_sums": feature_sums(state),
    }


def main() -> None:
    input_path = ROOT / "records" / "position_fixtures.json"
    fixtures = json.loads(input_path.read_text(encoding="utf-8"))
    output = [summarize_position(item["record"], int(item["ply"])) for item in fixtures]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
