from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt


@dataclass(frozen=True)
class VertexPosition:
    x: float
    y: float


class BoardTopology:
    def __init__(self, side_length: int) -> None:
        self.side_length = side_length
        self.positions: list[VertexPosition] = []
        self.adjacency: list[list[int]] = []
        self.labels: dict[int, str] = {}
        self._build()

    def _build(self) -> None:
        radius = self.side_length - 1
        cells: list[tuple[int, int]] = []
        for q in range(-radius, radius + 1):
            r_min = max(-radius, -q - radius)
            r_max = min(radius, -q + radius)
            for r in range(r_min, r_max + 1):
                cells.append((q, r))

        lookup: dict[tuple[int, int], int] = {}
        adjacency_sets: list[set[int]] = []

        def hex_center(q: int, r: int) -> VertexPosition:
            return VertexPosition(sqrt(3.0) * (q + r / 2.0), 1.5 * r)

        def key(pos: VertexPosition) -> tuple[int, int]:
            return (round(pos.x * 1_000_000), round(pos.y * 1_000_000))

        for q, r in cells:
            center = hex_center(q, r)
            ids: list[int] = []
            for i in range(6):
                angle = (60.0 * i - 30.0) * pi / 180.0
                pos = VertexPosition(center.x + cos(angle), center.y + sin(angle))
                k = key(pos)
                if k not in lookup:
                    lookup[k] = len(self.positions)
                    self.positions.append(pos)
                    adjacency_sets.append(set())
                ids.append(lookup[k])
            for i in range(6):
                a = ids[i]
                b = ids[(i + 1) % 6]
                adjacency_sets[a].add(b)
                adjacency_sets[b].add(a)

        self.adjacency = [sorted(list(s)) for s in adjacency_sets]
        order = sorted(range(len(self.positions)), key=lambda i: (self.positions[i].y, self.positions[i].x))
        row = -1
        last_y = None
        col = 0
        for index in order:
            pos = self.positions[index]
            if last_y is None or abs(pos.y - last_y) > 1e-6:
                row += 1
                last_y = pos.y
                col = 0
            self.labels[index] = f"{self._column_label(col)}{row + 1}"
            col += 1

    @staticmethod
    def _column_label(value: int) -> str:
        out = ""
        current = value
        while True:
            out = chr(ord("A") + (current % 26)) + out
            current = current // 26 - 1
            if current < 0:
                return out

    @property
    def vertex_count(self) -> int:
        return len(self.positions)
