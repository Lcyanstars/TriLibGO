from __future__ import annotations

import random
from collections import deque

from .selfplay import TrainingSample


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._data: deque[TrainingSample] = deque(maxlen=capacity)

    def extend(self, samples: list[TrainingSample]) -> None:
        self._data.extend(sample for sample in samples if sample.sample_weight > 0.0)

    def sample(self, batch_size: int) -> list[TrainingSample]:
        return random.sample(list(self._data), min(batch_size, len(self._data)))

    def __len__(self) -> int:
        return len(self._data)

    def summary(self) -> dict[str, float]:
        count = len(self._data)
        if count == 0:
            return {
                "buffer_effective_size": 0.0,
                "buffer_avg_sample_weight": 0.0,
                "buffer_downweighted_rate": 0.0,
                "buffer_next_move_capture_rate": 0.0,
            }
        total_weight = sum(sample.sample_weight for sample in self._data)
        downweighted = sum(1 for sample in self._data if sample.sample_weight < 0.999999)
        next_move_capture_samples = sum(1 for sample in self._data if getattr(sample, "next_move_capture_stones", 0) > 0)
        return {
            "buffer_effective_size": float(total_weight),
            "buffer_avg_sample_weight": float(total_weight / count),
            "buffer_downweighted_rate": float(downweighted / count),
            "buffer_next_move_capture_rate": float(next_move_capture_samples / count),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self._data.maxlen,
            "samples": list(self._data),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ReplayBuffer":
        buffer = cls(int(state["capacity"]))
        buffer.extend(list(state["samples"]))  # type: ignore[arg-type]
        return buffer
