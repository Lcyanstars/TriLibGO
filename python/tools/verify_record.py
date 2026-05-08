from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.rl.game import GameConfig, GameState, Move
from python.rl.topology import BoardTopology


def parse_record(path: Path) -> tuple[GameConfig, list[Move]]:
    config = GameConfig()
    moves: list[Move] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    moves_text = ""
    for line in lines:
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if key == "side_length":
            config.side_length = int(value)
        elif key == "komi":
            config.komi = float(value)
        elif key == "allow_suicide":
            config.allow_suicide = value == "1"
        elif key == "moves":
            moves_text = value

    topology = BoardTopology(config.side_length)
    label_to_vertex = {label: vertex for vertex, label in topology.labels.items()}
    for token in moves_text.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "pass":
            moves.append(Move.pass_turn())
        else:
            moves.append(Move.place(label_to_vertex[token]))
    return config, moves


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a TriLibGo record in the pure Python environment.")
    parser.add_argument("record", type=Path)
    args = parser.parse_args()

    config, moves = parse_record(args.record)
    state = GameState(config)
    for move in moves:
        if not state.is_legal(move):
            raise SystemExit(f"Illegal move in record: {move}")
        state.apply_move(move)

    score = state.score()
    print(f"side_length={config.side_length}")
    print(f"moves={len(moves)}")
    print(f"black={score.black:.1f}")
    print(f"white={score.white:.1f}")
    print(f"value={score.value:.1f}")


if __name__ == "__main__":
    main()
