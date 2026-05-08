from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.tools.verify_record import parse_record
from python.rl.game import GameState


def main() -> None:
    fixtures = json.loads((ROOT / "records" / "fixtures.json").read_text(encoding="utf-8"))
    failures: list[str] = []

    for name, expected in fixtures.items():
        config, moves = parse_record(ROOT / "records" / name)
        state = GameState(config)
        for move in moves:
            if not state.is_legal(move):
                failures.append(f"{name}: illegal move {move}")
                break
            state.apply_move(move)

        score = state.score()
        actual = {
            "side_length": config.side_length,
            "moves": len(moves),
            "black": score.black,
            "white": score.white,
            "value": score.value,
        }
        for key, expected_value in expected.items():
            if actual[key] != expected_value:
                failures.append(f"{name}: expected {key}={expected_value}, got {actual[key]}")

    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)

    print("All fixtures verified.")


if __name__ == "__main__":
    main()
