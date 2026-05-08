"""TriLibGo reinforcement learning research stack.

Self-play with PUCT MCTS, policy-value network training, evaluation,
ONNX export, and position analysis. Designed for CPU-only experimentation.

Main entry point: python -m python.rl.train --config <config.json>
"""

from .config import (
    ExperimentConfig,
    ExportConfig,
    MCTSConfig,
    ModelConfig,
    SelfPlayConfig,
    TrainingConfig,
)

