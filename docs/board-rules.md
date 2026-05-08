# TriLibGo Rules

TriLibGo is a Go-like game played on the **vertices** of a **hexagonal**
board. Interior stones have three liberties; capture, ko, passing, and scoring
follow standard Go conventions.

## Board

- The board is a hexagonal honeycomb region with `side_length` cells per side.
- Stones are placed on vertices (corners of the hexagons), not in cells.
- For `side_length = N`, there are `3*N*(N+1) + 1` vertices. Example: N=4 → 96
  vertices.
- Interior vertices have exactly 3 adjacent vertices (liberties). Boundary
  vertices have 2.

## Turns

- Black plays first.
- One stone per turn.
- A move is either a **placement** on an empty vertex or a **pass**.
- Suicide (placing a stone with zero liberties that does not capture) is
  disabled by default (`allow_suicide = false`).

## Capture

- Connected stones (chains) share liberties.
- When a move causes an opponent chain to have zero liberties, that chain is
  captured and removed from the board.
- Captured stones are counted for scoring.

## Ko

- Simple ko: a move that would return the board to its **immediately preceding**
  state (same stone positions) is illegal.
- This prevents infinite capture-recapture loops.

## Endgame

- The game ends after **two consecutive passes**.
- Before scoring, an optional dead-stone cleanup pass runs: stones in chains
  with ≤1 liberty are treated as dead and removed. This is a simple heuristic,
  not a full life-death analysis.
- Scoring: Chinese-style area scoring.
  - A player's score = their stones on the board + empty vertices they surround.
  - Komi (compensation points for White) is configurable, default varies by stage.

## Differences from Standard Go

| Aspect | Standard Go | TriLibGo |
|--------|------------|----------|
| Board topology | Rectangular grid | Hexagonal vertices |
| Liberties (interior) | 4 | 3 |
| Play surface | Grid intersections | Hexagon vertices |
| Scoring | Japanese or Chinese | Chinese area scoring |

The hexagonal topology means standard Go patterns (ladders, eyes, life-death
shapes) do not directly translate. The lower liberty count (3 vs. 4) makes
capture dynamics faster and groups more vulnerable.
