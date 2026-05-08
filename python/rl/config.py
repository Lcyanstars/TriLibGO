"""Experiment configuration: dataclasses, JSON loading, and utility functions.

All training hyperparameters live in ExperimentConfig and its sub-dataclasses.
Configs are loaded from JSON files via load_experiment_config().
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .topology import BoardTopology


@dataclass
class ModelConfig:
    """Network architecture: backbone type, channel counts, and block depth."""
    board_side: int = 4
    input_history: int = 2
    global_feature_count: int = 8
    channels: int = 64
    residual_blocks: int = 6
    policy_head_channels: int = 32
    value_head_channels: int = 32
    ownership_head_channels: int = 32
    score_head_channels: int = 32
    liberty_head_channels: int = 32
    action_size: int = 97


@dataclass
class MCTSConfig:
    """PUCT MCTS parameters: simulation budget, exploration, pass handling, temperature."""
    simulations: int = 96
    simulation_batch_size: int = 1
    c_puct: float = 1.5
    selfplay_exploration_scale: float = 1.25
    selfplay_exploration_decay_iterations: int = 8
    root_dirichlet_alpha: float = 0.20
    root_exploration_fraction: float = 0.25
    pass_prior_scale: float = 0.35
    disable_consecutive_pass_guard: bool = False
    consecutive_pass_min_value: float = 0.0
    consecutive_pass_min_score_margin: float = 0.0
    temperature_opening_moves: int = 12
    temperature: float = 1.0


@dataclass
class SelfPlayConfig:
    """Self-play generation: games per iteration, worker count, replay window, sampling."""
    games_per_iteration: int = 64
    workers: int = 4
    max_moves: int = 256
    replay_window: int = 50000
    iterations: int = 5
    sampling_final_temperature: float = 0.2
    sampling_decay_moves: int = 24
    early_pass_extra_moves: int = 6
    short_game_turn_threshold: int = 20
    abnormal_sample_weight: float = 0.35


@dataclass
class CurriculumConfig:
    """Curriculum learning: capture bonuses, response weighting, eye-fill penalties, metric-driven auto-stop."""
    enabled: bool = False
    start_iteration: int = 1
    full_strength_until_iteration: int = 0
    end_iteration: int = 0
    stop_on_metrics: bool = False
    stop_min_iteration: int = 20
    stop_max_iteration: int = 40
    stop_min_capture_rate: float = 0.12
    stop_max_true_eye_fill_rate: float = 0.04
    stop_max_eye_fill_penalized_rate: float = 0.02
    stop_metric_patience: int = 1
    stop_first_player_win_rate_min: float = 0.45
    stop_first_player_win_rate_max: float = 0.60
    stop_max_abs_score_margin: float = 12.0
    stop_max_moves_rate: float = 0.10
    selfplay_capture_bonus: float = 1.35
    selfplay_capture_bonus_per_stone: float = 0.2
    selfplay_capture_bonus_cap: float = 1.75
    selfplay_true_eye_penalty: float = 0.05
    sample_capture_weight: float = 1.5
    sample_capture_weight_per_stone: float = 0.25
    sample_capture_weight_cap: float = 2.0
    sample_response_weight: float = 1.25
    sample_immediate_capture_risk_weight: float = 1.25
    sample_immediate_capture_risk_per_stone: float = 0.25
    sample_immediate_capture_risk_cap: float = 2.0
    sample_true_eye_penalty_weight: float = 0.1
    value_capture_bonus: float = 0.12
    value_capture_bonus_per_stone: float = 0.04
    value_capture_bonus_cap: float = 0.20
    value_next_move_capture_penalty: float = 0.16
    value_next_move_capture_penalty_per_stone: float = 0.06
    value_next_move_capture_penalty_cap: float = 0.28
    value_true_eye_penalty: float = 0.10
    sample_weight_min: float = 0.05
    sample_weight_max: float = 2.5


@dataclass
class RulesConfig:
    """Game rules: komi, suicide, opening pass ban, dead-stone cleanup, auto komi adjustment."""
    komi: float = 0.0
    allow_suicide: bool = False
    opening_no_pass_moves: int = 12
    cleanup_dead_stones: bool = True
    cleanup_dead_stones_mode: str = "proof_search"
    cleanup_candidate_max_liberties: int = 4
    cleanup_local_search_depth: int = 10
    cleanup_local_search_nodes: int = 2000
    cleanup_preserve_seki: bool = True
    cleanup_mark_unresolved: bool = True
    auto_adjust_komi: bool = False
    komi_adjust_interval: int = 8
    komi_adjust_warmup_iterations: int = 8
    komi_adjust_min_samples: int = 32
    komi_adjust_max_margin: float = 3.0
    komi_adjust_alpha: float = 0.25
    komi_adjust_delta_max: float = 0.5
    selfplay_komi: list[float] = field(default_factory=list)


@dataclass
class TelemetryConfig:
    enabled: bool = True
    save_game_records: bool = True
    show_progress: bool = True
    console_log_mode: str = "summary"
    top_policy_moves: int = 5
    report_recent_games: int = 50
    selfplay_dir: Path = Path("artifacts/selfplay")
    report_path: Path = Path("artifacts/reports/training_report.html")


@dataclass
class TrainingConfig:
    """Training loop: batch size, learning rate schedule, loss weights, checkpointing."""
    batch_size: int = 128
    epochs_per_iteration: int = 2
    steps_per_iteration: int = 0
    min_steps_per_iteration: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    lr_schedule: str = "constant"
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 0
    lr_min_ratio: float = 0.1
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    policy_entropy_weight: float = 0.0
    eye_fill_loss_weight: float = 0.0
    ownership_loss_weight: float = 0.5
    score_loss_weight: float = 0.25
    liberty_loss_weight: float = 0.0
    score_loss_scale: float = 1.0
    terminal_weight_on_max_moves: float = 0.25
    terminal_weight_on_unresolved_dead_groups: float = 0.25
    terminal_weight_on_noisy_cleanup: float = 0.5
    terminal_weight_cleanup_dead_stones_threshold: int = 12
    terminal_weight_total_passes_threshold: int = 6
    terminal_weight_score_margin_threshold: float = 40.0
    checkpoint_keep_recent: int = 3
    checkpoint_keep_every: int = 10
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    replay_dir: Path = Path("artifacts/replay")
    best_dir: Path = Path("artifacts/best")


@dataclass
class EvaluationConfig:
    """Evaluation: promotion match frequency, game count, simulation budget, win-rate threshold."""
    interval: int = 1
    interval_schedule: list[dict[str, int]] = field(default_factory=list)
    games: int = 8
    simulations: int = 0
    promotion_win_rate: float = 0.55
    full_games_every: int = 0
    full_games: int = 0
    full_simulations: int = 0


@dataclass
class ExportConfig:
    onnx_path: Path = Path("artifacts/export/model.onnx")
    opset: int = 17


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration aggregating all sub-configs."""
    name: str = "tiny_cpu_baseline"
    seed: int = 7
    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)


def vertex_count_for_side(board_side: int) -> int:
    return BoardTopology(board_side).vertex_count


def action_size_for_side(board_side: int) -> int:
    return vertex_count_for_side(board_side) + 1


def apply_board_side(config: ExperimentConfig, board_side: int, *, rename_run: bool = True) -> ExperimentConfig:
    board_side = int(board_side)
    if board_side < 2:
        raise ValueError(f"board_side must be >= 2, got {board_side}")
    old_board_side = int(config.model.board_side)
    config.model.board_side = board_side
    config.model.action_size = action_size_for_side(board_side)
    if not rename_run or old_board_side == board_side:
        return config

    old_token = f"side{old_board_side}"
    new_token = f"side{board_side}"
    if old_token in config.name:
        config.name = config.name.replace(old_token, new_token)
    elif f"_{old_board_side}x" not in config.name:
        config.name = f"{config.name}_{new_token}"

    def rewrite_path(path: Path) -> Path:
        raw = path.as_posix()
        if old_token in raw:
            return Path(raw.replace(old_token, new_token))
        if "stage5" in raw and new_token not in raw:
            return Path(raw.replace("stage5", f"stage5_{new_token}"))
        return path

    config.training.checkpoint_dir = rewrite_path(config.training.checkpoint_dir)
    config.training.replay_dir = rewrite_path(config.training.replay_dir)
    config.training.best_dir = rewrite_path(config.training.best_dir)
    config.export.onnx_path = rewrite_path(config.export.onnx_path)
    config.telemetry.selfplay_dir = rewrite_path(config.telemetry.selfplay_dir)
    config.telemetry.report_path = rewrite_path(config.telemetry.report_path)
    return config


def input_planes_for_history(input_history: int) -> int:
    return 2 * input_history + 2


def effective_training_steps(
    buffer_size: int,
    batch_size: int,
    configured_steps: int,
    epochs_per_iteration: int,
    min_steps_per_iteration: int = 0,
) -> int:
    if buffer_size <= 0:
        return 0
    if configured_steps > 0:
        return configured_steps
    batches_per_epoch = max(1, math.ceil(buffer_size / max(batch_size, 1)))
    auto_steps = max(1, max(epochs_per_iteration, 1) * batches_per_epoch)
    return max(auto_steps, max(0, int(min_steps_per_iteration)))


def selfplay_exploration_scale(config: MCTSConfig, iteration: int) -> float:
    max_scale = max(1.0, float(config.selfplay_exploration_scale))
    decay_iterations = max(1, int(config.selfplay_exploration_decay_iterations))
    if decay_iterations == 1:
        return 1.0 if iteration > 1 else max_scale
    progress = min(max(iteration - 1, 0) / max(decay_iterations - 1, 1), 1.0)
    return max_scale + (1.0 - max_scale) * progress


def _apply_updates(obj: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if isinstance(obj, CurriculumConfig) and key == "stop_min_capture_weighted_rate":
            key = "stop_min_capture_rate"
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _apply_updates(current, value)
        else:
            if isinstance(current, Path):
                setattr(obj, key, Path(value))
            else:
                setattr(obj, key, value)
    return obj


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config = ExperimentConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return _apply_updates(config, raw)
