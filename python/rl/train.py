from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .config import (
    ExperimentConfig,
    apply_board_side,
    action_size_for_side,
    effective_training_steps,
    load_experiment_config,
    selfplay_exploration_scale,
)
from .eval import evaluate_candidate
from .export import export_model
from .model import PolicyValueNet
from .replay_buffer import ReplayBuffer
from .selfplay import SelfPlayGameTrace, TrainingSample, curriculum_strength, generate_selfplay_batch, generate_selfplay_game

try:
    from python.tools.render_training_report import load_jsonl, render_html
except ImportError:  # pragma: no cover - optional convenience import
    load_jsonl = None
    render_html = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_iteration(
    model: PolicyValueNet,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer,
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    if len(replay) == 0:
        return {
            "policy_loss": 0.0,
            "policy_entropy": 0.0,
            "entropy_bonus": 0.0,
            "eye_fill_loss": 0.0,
            "eye_fill_bad_prob": 0.0,
            "value_loss": 0.0,
            "ownership_loss": 0.0,
            "score_loss": 0.0,
            "liberty_loss": 0.0,
            "total_loss": 0.0,
            "effective_batch_size": 0.0,
            "avg_batch_weight": 0.0,
            "grad_norm": 0.0,
            "clipped_grad_norm": 0.0,
        }
    batch = replay.sample(cfg.training.batch_size)
    states = torch.tensor(np.stack([sample.state_planes for sample in batch]), dtype=torch.float32, device=device)
    model_global_feature_count = int(getattr(model, "global_feature_count", 8))
    global_feature_count = model_global_feature_count
    global_features_np = []
    for sample in batch:
        features = getattr(sample, "global_features", None)
        if features is None or getattr(features, "shape", (0,))[0] != global_feature_count:
            features = np.zeros(global_feature_count, dtype=np.float32)
        global_features_np.append(features)
    global_features = torch.tensor(np.stack(global_features_np), dtype=torch.float32, device=device)
    policy_targets = torch.tensor(np.stack([sample.policy_target for sample in batch]), dtype=torch.float32, device=device)
    liberty_targets_np = []
    for sample in batch:
        target = getattr(sample, "liberty_target", None)
        if target is None or getattr(target, "shape", (0,))[0] != policy_targets.shape[-1] - 1:
            target = np.zeros(policy_targets.shape[-1] - 1, dtype=np.int64)
        liberty_targets_np.append(target)
    liberty_targets = torch.tensor(np.stack(liberty_targets_np), dtype=torch.long, device=device)
    eye_fill_masks_np = []
    for sample in batch:
        mask = getattr(sample, "eye_fill_bad_action_mask", None)
        if mask is None or getattr(mask, "shape", (0,))[0] != policy_targets.shape[-1]:
            mask = np.zeros(policy_targets.shape[-1], dtype=np.float32)
        eye_fill_masks_np.append(mask)
    eye_fill_bad_masks = torch.tensor(np.stack(eye_fill_masks_np), dtype=torch.float32, device=device)
    value_targets = torch.tensor([sample.value_target for sample in batch], dtype=torch.float32, device=device)
    ownership_targets = torch.tensor(np.stack([sample.ownership_target for sample in batch]), dtype=torch.float32, device=device)
    score_targets = torch.tensor([sample.score_target for sample in batch], dtype=torch.float32, device=device)
    sample_weights = torch.tensor([sample.sample_weight for sample in batch], dtype=torch.float32, device=device)
    policy_weights = torch.tensor([getattr(sample, "policy_weight", sample.sample_weight) for sample in batch], dtype=torch.float32, device=device)
    terminal_weights = torch.tensor([getattr(sample, "terminal_weight", sample.sample_weight) for sample in batch], dtype=torch.float32, device=device)
    batch_weight_sum = float(sample_weights.sum().item())
    policy_weight_sum = torch.clamp(policy_weights.sum(), min=1e-6)
    terminal_weight_sum = torch.clamp(terminal_weights.sum(), min=1e-6)

    model.train()
    try:
        outputs = model(states, global_features)
    except TypeError:
        outputs = model(states)
    policy_logits, values, ownership, score = outputs[:4]
    liberty_logits = outputs[4] if len(outputs) > 4 else None
    log_probs = F.log_softmax(policy_logits, dim=-1)
    probs = log_probs.exp()
    policy_loss = ((-(policy_targets * log_probs).sum(dim=-1)) * policy_weights).sum() / policy_weight_sum
    entropy_mask = policy_targets > 0.0
    has_entropy_actions = entropy_mask.any(dim=-1, keepdim=True)
    entropy_mask = torch.where(has_entropy_actions, entropy_mask, torch.ones_like(entropy_mask))
    masked_logits = policy_logits.masked_fill(~entropy_mask, -1e9)
    masked_log_probs = F.log_softmax(masked_logits, dim=-1)
    masked_probs = masked_log_probs.exp() * entropy_mask
    policy_entropy = ((-(masked_probs * masked_log_probs).sum(dim=-1)) * policy_weights).sum() / policy_weight_sum
    entropy_bonus = max(0.0, float(getattr(cfg.training, "policy_entropy_weight", 0.0))) * policy_entropy
    eye_fill_bad_prob_per_sample = torch.clamp((probs * eye_fill_bad_masks).sum(dim=-1), min=0.0, max=1.0 - 1e-6)
    eye_fill_loss = ((-torch.log1p(-eye_fill_bad_prob_per_sample)) * policy_weights).sum() / policy_weight_sum
    eye_fill_loss_weight = max(0.0, float(getattr(cfg.training, "eye_fill_loss_weight", 0.0)))
    value_loss = (F.mse_loss(values, value_targets, reduction="none") * terminal_weights).sum() / terminal_weight_sum
    ownership_loss = (F.mse_loss(ownership, ownership_targets, reduction="none").mean(dim=-1) * terminal_weights).sum() / terminal_weight_sum
    score_loss_scale = max(float(getattr(cfg.training, "score_loss_scale", 1.0)), 1e-6)
    normalized_score = score / score_loss_scale
    normalized_score_targets = score_targets / score_loss_scale
    score_loss = (F.smooth_l1_loss(normalized_score, normalized_score_targets, reduction="none") * terminal_weights).sum() / terminal_weight_sum
    if liberty_logits is not None:
        liberty_loss_per_vertex = F.cross_entropy(liberty_logits, liberty_targets, reduction="none")
        liberty_loss = (liberty_loss_per_vertex.mean(dim=-1) * sample_weights).sum() / torch.clamp(sample_weights.sum(), min=1e-6)
    else:
        liberty_loss = torch.zeros((), dtype=torch.float32, device=device)
    total_loss = (
        cfg.training.policy_loss_weight * policy_loss
        + cfg.training.value_loss_weight * value_loss
        + cfg.training.ownership_loss_weight * ownership_loss
        + cfg.training.score_loss_weight * score_loss
        + eye_fill_loss_weight * eye_fill_loss
        + max(0.0, float(getattr(cfg.training, "liberty_loss_weight", 0.0))) * liberty_loss
        - entropy_bonus
    )

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm, clipped_grad_norm = clip_gradients(model.parameters(), cfg.training.gradient_clip_norm)
    optimizer.step()
    return {
        "policy_loss": float(policy_loss.item()),
        "policy_entropy": float(policy_entropy.item()),
        "entropy_bonus": float(entropy_bonus.item()),
        "eye_fill_loss": float(eye_fill_loss.item()),
        "eye_fill_bad_prob": float(eye_fill_bad_prob_per_sample.mean().item()),
        "value_loss": float(value_loss.item()),
        "ownership_loss": float(ownership_loss.item()),
        "score_loss": float(score_loss.item()),
        "liberty_loss": float(liberty_loss.item()),
        "total_loss": float(total_loss.item()),
        "effective_batch_size": batch_weight_sum,
        "avg_batch_weight": batch_weight_sum / max(len(batch), 1),
        "grad_norm": grad_norm,
        "clipped_grad_norm": clipped_grad_norm,
    }


def snapshot_path(directory: Path, name: str, iteration: int) -> Path:
    return directory / f"{name}_iter{iteration}.pt"


def compute_gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    stacked = torch.stack([gradient.norm(2) for gradient in gradients])
    return float(stacked.norm(2).item())


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> tuple[float, float]:
    params = [parameter for parameter in parameters if parameter.grad is not None]
    if not params:
        return 0.0, 0.0
    grad_norm = compute_gradient_norm(params)
    if max_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(params, max_norm)
    clipped_grad_norm = compute_gradient_norm(params)
    return grad_norm, clipped_grad_norm


def compute_learning_rate(
    base_learning_rate: float,
    global_step: int,
    warmup_steps: int,
    decay_steps: int,
    min_ratio: float,
    schedule: str,
) -> float:
    schedule_name = schedule.strip().lower()
    if schedule_name not in {"constant", "cosine"}:
        raise ValueError(f"unsupported lr_schedule: {schedule}")
    step_number = max(global_step + 1, 1)
    if warmup_steps > 0 and step_number <= warmup_steps:
        return base_learning_rate * (step_number / warmup_steps)
    if schedule_name == "constant" or decay_steps <= warmup_steps or decay_steps <= 0:
        return base_learning_rate
    progress = min(max(step_number - warmup_steps, 0) / max(decay_steps - warmup_steps, 1), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    ratio = min_ratio + (1.0 - min_ratio) * cosine
    return base_learning_rate * ratio


def set_optimizer_learning_rate(optimizer: optim.Optimizer, learning_rate: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def resolve_lr_decay_steps(cfg: ExperimentConfig) -> int:
    if cfg.training.lr_decay_steps > 0:
        return cfg.training.lr_decay_steps
    if cfg.training.steps_per_iteration > 0:
        return cfg.training.steps_per_iteration * cfg.selfplay.iterations
    if cfg.training.min_steps_per_iteration > 0:
        return cfg.training.min_steps_per_iteration * cfg.selfplay.iterations
    return 0


def summarize_training_steps(step_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not step_metrics:
        return {
            "policy_loss": 0.0,
            "policy_entropy": 0.0,
            "entropy_bonus": 0.0,
            "eye_fill_loss": 0.0,
            "eye_fill_bad_prob": 0.0,
            "value_loss": 0.0,
            "ownership_loss": 0.0,
            "score_loss": 0.0,
            "liberty_loss": 0.0,
            "total_loss": 0.0,
            "effective_batch_size": 0.0,
            "avg_batch_weight": 0.0,
            "grad_norm": 0.0,
            "clipped_grad_norm": 0.0,
            "learning_rate_start": 0.0,
            "learning_rate_end": 0.0,
            "learning_rate": 0.0,
        }
    summary: dict[str, float] = {}
    for field in (
        "policy_loss",
        "policy_entropy",
        "entropy_bonus",
        "eye_fill_loss",
        "eye_fill_bad_prob",
        "value_loss",
        "ownership_loss",
        "score_loss",
        "liberty_loss",
        "total_loss",
        "effective_batch_size",
        "avg_batch_weight",
        "grad_norm",
        "clipped_grad_norm",
    ):
        summary[field] = sum(metrics[field] for metrics in step_metrics) / len(step_metrics)
    summary["learning_rate_start"] = step_metrics[0]["learning_rate"]
    summary["learning_rate_end"] = step_metrics[-1]["learning_rate"]
    summary["learning_rate"] = step_metrics[-1]["learning_rate"]
    return summary


def save_checkpoint(
    path: Path,
    model: PolicyValueNet,
    incumbent: PolicyValueNet,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer,
    cfg: ExperimentConfig,
    metadata: dict[str, object],
    pending_komi_traces: list[SelfPlayGameTrace] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        torch.save(
            {
                "model": model.state_dict(),
                "incumbent": incumbent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "replay": replay.state_dict(),
                "config": cfg,
                "metadata": metadata,
                "pending_komi_traces": list(pending_komi_traces or []),
            },
            temporary_path,
        )
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    temporary_path.replace(path)


def write_metrics_line(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics) + "\n")


def console_log_mode(cfg: ExperimentConfig) -> str:
    return cfg.telemetry.console_log_mode.strip().lower()


def emit_event(cfg: ExperimentConfig, event: str, **payload: object) -> None:
    if console_log_mode(cfg) != "events":
        return
    print(json.dumps({"event": event, **payload}), flush=True)


def print_notice(cfg: ExperimentConfig, message: str) -> None:
    if console_log_mode(cfg) == "events":
        return
    print(message, flush=True)


def _format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60.0)
    remainder = seconds - minutes * 60.0
    return f"{minutes}m{remainder:04.1f}s"


def _format_eval_status(metrics: dict[str, object]) -> str:
    if not bool(metrics.get("eval_ran", False)):
        return "skip"
    wins = int(metrics.get("eval_wins", 0))
    losses = int(metrics.get("eval_losses", 0))
    draws = int(metrics.get("eval_draws", 0))
    return f"{wins}-{losses}-{draws}"


def _format_best_status(metrics: dict[str, object]) -> str:
    if bool(metrics.get("bootstrap_best", False)):
        return "bootstrap"
    if bool(metrics.get("promoted", False)):
        return "promoted"
    if bool(metrics.get("best_updated", False)):
        return "updated"
    return "hold"


def _format_eval_detail(metrics: dict[str, object]) -> str:
    mode = str(metrics.get("eval_mode", "skipped"))
    if not bool(metrics.get("eval_ran", False)):
        return f"{mode} (interval={int(metrics.get('eval_interval', 0))})"
    wins = int(metrics.get("eval_wins", 0))
    losses = int(metrics.get("eval_losses", 0))
    draws = int(metrics.get("eval_draws", 0))
    games = int(metrics.get("eval_games", wins + losses + draws))
    simulations = int(metrics.get("eval_simulations", 0))
    win_rate = float(metrics.get("eval_win_rate", 0.0) or 0.0) * 100.0
    return f"{mode} {wins}-{losses}-{draws} / {games} (wr={win_rate:.1f}%, sims={simulations})"


def _format_komi_detail(metrics: dict[str, object]) -> str:
    before = float(metrics.get("komi_used", 0.0))
    after = float(metrics.get("komi_next", before))
    if not bool(metrics.get("komi_adjustment_checked", False)):
        return f"{before:.2f} -> {after:.2f} (pending)"
    reason = str(metrics.get("komi_adjustment_reason", "unknown"))
    eligible = int(metrics.get("komi_adjustment_eligible_games", 0))
    considered = int(metrics.get("komi_adjustment_considered_games", 0))
    delta = float(metrics.get("komi_adjustment_delta", 0.0))
    return f"{before:.2f} -> {after:.2f} ({reason}, eligible={eligible}/{considered}, delta={delta:+.2f})"


def print_iteration_summary(cfg: ExperimentConfig, total_iterations: int, metrics: dict[str, object]) -> None:
    if console_log_mode(cfg) == "events":
        print(json.dumps(metrics), flush=True)
        return
    iteration = int(metrics["iteration"])
    lines = [
        (
            f"[Iter {iteration}/{total_iterations}] "
            f"selfplay={_format_duration(float(metrics['selfplay_duration_sec']))} "
            f"train={_format_duration(float(metrics['training_duration_sec']))} "
            f"eval={_format_eval_detail(metrics)} "
            f"| best={_format_best_status(metrics)}"
        ),
        (
            f"  Loss     policy={float(metrics['policy_loss']):.3f} "
            f"ent={float(metrics.get('policy_entropy', 0.0)):.3f} "
            f"eye={float(metrics.get('eye_fill_loss', 0.0)):.3f} "
            f"value={float(metrics['value_loss']):.3f} "
            f"ownership={float(metrics['ownership_loss']):.3f} "
            f"score={float(metrics['score_loss']):.3f} "
            f"lib={float(metrics.get('liberty_loss', 0.0)):.3f} "
            f"total={float(metrics['total_loss']):.3f}"
        ),
        (
            f"  Buffer   samples={int(metrics['selfplay_samples'])} "
            f"buffer={int(metrics['buffer_size'])} "
            f"eff={float(metrics['buffer_effective_size']):.1f} "
            f"avg_w={float(metrics['buffer_avg_sample_weight']):.3f} "
            f"down={float(metrics['buffer_downweighted_rate']):.1%} "
            f"steps={int(metrics['training_steps'])} "
            f"lr={float(metrics['learning_rate_start']):.2e}->{float(metrics['learning_rate_end']):.2e}"
        ),
        (
            f"  Games    len={float(metrics['avg_game_length']):.1f} "
            f"pass={float(metrics['avg_first_pass_turn']):.1f}/{float(metrics['avg_total_passes']):.2f} "
            f"margin={float(metrics['avg_score_margin_black_minus_white']):+.2f} "
            f"draw={float(metrics['draw_rate']):.1%} "
            f"first={float(metrics['first_player_win_rate']):.1%} "
            f"max={float(metrics.get('max_moves_rate', 0.0)):.1%} "
            f"abnormal={int(metrics['abnormal_games'])}/{int(metrics['selfplay_games'])} "
            f"pred_err={float(metrics['avg_abs_predicted_score_error']):.2f}"
        ),
        (
            f"  Cleanup  dead={float(metrics['avg_cleaned_dead_stones']):.2f} "
            f"rule={float(metrics['avg_cleanup_rule_resolved_groups']):.2f} "
            f"local={float(metrics['avg_cleanup_local_search_resolved_groups']):.2f} "
            f"seki={float(metrics['avg_cleanup_preserved_seki_groups']):.2f} "
            f"unresolved={float(metrics['avg_unresolved_dead_groups']):.2f} "
            f"| komi={_format_komi_detail(metrics)}"
        ),
        (
            f"  Tactic   curr={float(metrics.get('curriculum_scale', 0.0)):.2f} "
            f"cap={float(metrics.get('avg_captures_by_move', 0.0)):.3f} "
            f"risk={float(metrics.get('avg_immediate_capture_risk_stones', 0.0)):.3f} "
            f"next_cap={float(metrics.get('avg_next_move_capture_stones', 0.0)):.3f} "
            f"eye_fill={float(metrics.get('true_eye_fill_rate', 0.0)):.1%} "
            f"cap_rate={float(metrics.get('pure_capture_rate', 0.0)):.1%} "
            f"risk_rate={float(metrics.get('immediate_capture_risk_rate', 0.0)):.1%} "
            f"next_cap_rate={float(metrics.get('next_move_capture_rate', 0.0)):.1%} "
            f"bonus={float(metrics.get('avg_curriculum_value_bonus', 0.0)):+.3f} "
            f"+b={float(metrics.get('positive_curriculum_bonus_rate', 0.0)):.1%} "
            f"-b={float(metrics.get('negative_curriculum_bonus_rate', 0.0)):.1%} "
            f"pen={float(metrics.get('eye_fill_penalized_rate', 0.0)):.1%}"
        ),
    ]
    print("\n".join(lines), flush=True)


class ProgressBar:
    def __init__(self, label: str, total: int, width: int = 24) -> None:
        self.label = label
        self.total = max(0, total)
        self.width = width
        self.current = 0
        self._active = self.total > 0
        self._complete = False
        if self._active:
            self.update(0)

    def update(self, current: int, total: int | None = None) -> None:
        if not self._active:
            return
        if total is not None and total > 0:
            self.total = total
        self.current = max(0, min(current, self.total))
        filled = self.width if self.total == 0 else int(self.width * self.current / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = 100.0 if self.total == 0 else (100.0 * self.current / self.total)
        print(
            f"\r[{self.label}] [{bar}] {self.current}/{self.total} ({percent:5.1f}%)",
            end="",
            file=sys.stderr,
            flush=True,
        )
        self._complete = self.current >= self.total

    def close(self) -> None:
        if not self._active:
            return
        if not self._complete:
            self.update(self.total)
        print(file=sys.stderr, flush=True)
        self._active = False


def make_progress_bar(cfg: ExperimentConfig, label: str, total: int) -> ProgressBar | None:
    if not cfg.telemetry.show_progress:
        return None
    return ProgressBar(label, total)


class EvaluationPlan:
    def __init__(self, run: bool, mode: str, games: int, simulations: int, interval: int) -> None:
        self.run = run
        self.mode = mode
        self.games = games
        self.simulations = simulations
        self.interval = interval


@dataclass(frozen=True)
class KomiAdjustmentSummary:
    checked: bool
    applied: bool
    reason: str
    considered_games: int
    eligible_games: int
    weighted_margin: float | None
    delta: float
    previous_komi: float
    next_komi: float


def resolved_evaluation_interval(cfg: ExperimentConfig, iteration: int) -> int:
    schedule = list(cfg.evaluation.interval_schedule)
    for entry in schedule:
        until_iteration = int(entry.get("until_iteration", 0))
        if until_iteration > 0 and iteration <= until_iteration:
            return max(1, int(entry.get("interval", cfg.evaluation.interval)))
    return max(1, int(cfg.evaluation.interval))


def evaluation_plan(cfg: ExperimentConfig, iteration: int) -> EvaluationPlan:
    interval = resolved_evaluation_interval(cfg, iteration)
    fast_games = cfg.evaluation.games
    fast_simulations = cfg.evaluation.simulations if cfg.evaluation.simulations > 0 else cfg.mcts.simulations
    full_every = cfg.evaluation.full_games_every
    full_games = cfg.evaluation.full_games if cfg.evaluation.full_games > 0 else fast_games
    full_simulations = cfg.evaluation.full_simulations if cfg.evaluation.full_simulations > 0 else fast_simulations
    if full_every > 0 and iteration % full_every == 0:
        return EvaluationPlan(run=True, mode="full", games=full_games, simulations=full_simulations, interval=interval)
    if iteration % interval == 0:
        return EvaluationPlan(run=True, mode="fast", games=fast_games, simulations=fast_simulations, interval=interval)
    return EvaluationPlan(run=False, mode="skipped", games=0, simulations=0, interval=interval)


def should_adjust_komi(cfg: ExperimentConfig, iteration: int) -> bool:
    if not cfg.rules.auto_adjust_komi:
        return False
    interval = max(1, int(cfg.rules.komi_adjust_interval))
    warmup = max(0, int(cfg.rules.komi_adjust_warmup_iterations))
    return iteration >= warmup and iteration % interval == 0


def eligible_komi_margins(traces: list[SelfPlayGameTrace], cfg: ExperimentConfig) -> list[float]:
    max_margin = max(0.0, float(cfg.rules.komi_adjust_max_margin))
    eligible: list[float] = []
    for trace in traces:
        if trace.end_reason != "double_pass":
            continue
        if trace.unresolved_dead_groups > 0:
            continue
        if "early_pass" in trace.abnormal_tags or "short_game" in trace.abnormal_tags:
            continue
        margin = float(trace.black_score - trace.white_score)
        if max_margin > 0.0 and abs(margin) > max_margin:
            continue
        eligible.append(margin)
    return eligible


def summarize_komi_adjustment(traces: list[SelfPlayGameTrace], cfg: ExperimentConfig, checked: bool = True) -> KomiAdjustmentSummary:
    previous_komi = float(cfg.rules.komi)
    if not checked:
        return KomiAdjustmentSummary(
            checked=False,
            applied=False,
            reason="not_scheduled",
            considered_games=len(traces),
            eligible_games=0,
            weighted_margin=None,
            delta=0.0,
            previous_komi=previous_komi,
            next_komi=previous_komi,
        )

    eligible = eligible_komi_margins(traces, cfg)
    min_samples = max(1, int(cfg.rules.komi_adjust_min_samples))
    if len(eligible) < min_samples:
        return KomiAdjustmentSummary(
            checked=True,
            applied=False,
            reason="insufficient_samples",
            considered_games=len(traces),
            eligible_games=len(eligible),
            weighted_margin=None,
            delta=0.0,
            previous_komi=previous_komi,
            next_komi=previous_komi,
        )

    max_margin = max(float(cfg.rules.komi_adjust_max_margin), 1e-6)
    weights = [max(0.0, 1.0 - abs(margin) / max_margin) for margin in eligible]
    weight_sum = sum(weights)
    if weight_sum <= 1e-9:
        return KomiAdjustmentSummary(
            checked=True,
            applied=False,
            reason="zero_weight",
            considered_games=len(traces),
            eligible_games=len(eligible),
            weighted_margin=None,
            delta=0.0,
            previous_komi=previous_komi,
            next_komi=previous_komi,
        )

    weighted_margin = sum(weight * margin for weight, margin in zip(weights, eligible)) / weight_sum
    raw_delta = float(cfg.rules.komi_adjust_alpha) * weighted_margin
    delta_cap = max(0.0, float(cfg.rules.komi_adjust_delta_max))
    delta = max(-delta_cap, min(raw_delta, delta_cap))
    next_komi = previous_komi + delta
    applied = abs(delta) > 1e-9
    return KomiAdjustmentSummary(
        checked=True,
        applied=applied,
        reason="applied" if applied else "no_change",
        considered_games=len(traces),
        eligible_games=len(eligible),
        weighted_margin=weighted_margin,
        delta=delta,
        previous_komi=previous_komi,
        next_komi=next_komi,
    )


def apply_komi_adjustment(cfg: ExperimentConfig, summary: KomiAdjustmentSummary) -> None:
    if not summary.applied:
        return
    delta = summary.next_komi - summary.previous_komi
    cfg.rules.komi = float(summary.next_komi)
    if cfg.rules.selfplay_komi:
        cfg.rules.selfplay_komi = [float(komi) + delta for komi in cfg.rules.selfplay_komi]


def reset_run_logs(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def checkpoint_iteration(path: Path, run_name: str) -> int | None:
    prefix = f"{run_name}_iter"
    suffix = ".pt"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    raw_value = name[len(prefix) : -len(suffix)]
    if not raw_value.isdigit():
        return None
    return int(raw_value)


def prune_checkpoints(cfg: ExperimentConfig, latest_iteration: int) -> list[Path]:
    keep_recent = max(0, int(getattr(cfg.training, "checkpoint_keep_recent", 0)))
    keep_every = max(0, int(getattr(cfg.training, "checkpoint_keep_every", 0)))
    removed: list[Path] = []
    for path in cfg.training.checkpoint_dir.glob(f"{cfg.name}_iter*.pt"):
        iteration = checkpoint_iteration(path, cfg.name)
        if iteration is None:
            continue
        keep = False
        if iteration <= latest_iteration:
            if keep_recent > 0 and iteration > latest_iteration - keep_recent:
                keep = True
            elif keep_every > 0 and iteration % keep_every == 0:
                keep = True
        if keep:
            continue
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed


def legacy_selfplay_path(cfg: ExperimentConfig) -> Path:
    return cfg.telemetry.selfplay_dir / f"{cfg.name}_selfplay.jsonl"


def iteration_selfplay_path(cfg: ExperimentConfig, iteration: int) -> Path:
    return cfg.telemetry.selfplay_dir / f"{cfg.name}_iter{iteration:04d}.jsonl"


def reset_selfplay_logs(cfg: ExperimentConfig) -> None:
    legacy_path = legacy_selfplay_path(cfg)
    if legacy_path.exists():
        legacy_path.unlink()
    for path in cfg.telemetry.selfplay_dir.glob(f"{cfg.name}_iter*.jsonl"):
        path.unlink(missing_ok=True)


def trim_selfplay_logs(cfg: ExperimentConfig, max_iteration: int) -> int:
    removed = 0
    legacy_path = legacy_selfplay_path(cfg)
    if legacy_path.exists():
        removed += trim_jsonl_to_iteration(legacy_path, max_iteration)
    for path in cfg.telemetry.selfplay_dir.glob(f"{cfg.name}_iter*.jsonl"):
        iteration = checkpoint_iteration(path.with_suffix(".pt"), cfg.name)
        if iteration is not None and iteration > max_iteration:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def load_recent_selfplay_records(cfg: ExperimentConfig) -> list[dict[str, object]]:
    recent_games = max(1, int(getattr(cfg.telemetry, "report_recent_games", 50)))
    records: list[dict[str, object]] = []
    shard_paths = sorted(cfg.telemetry.selfplay_dir.glob(f"{cfg.name}_iter*.jsonl"), reverse=True)
    if shard_paths:
        for path in shard_paths:
            shard_rows = load_jsonl(path) if load_jsonl is not None else []
            records = shard_rows + records
            if len(records) >= recent_games:
                return records[-recent_games:]
        return records[-recent_games:]
    legacy_path = legacy_selfplay_path(cfg)
    if load_jsonl is None:
        return []
    return load_jsonl(legacy_path)[-recent_games:]


def trim_jsonl_to_iteration(path: Path, max_iteration: int) -> int:
    if not path.exists():
        return 0
    kept_lines: list[str] = []
    removed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            kept_lines.append(line)
            continue
        iteration = payload.get("iteration")
        if iteration is None or int(iteration) <= max_iteration:
            kept_lines.append(line)
        else:
            removed += 1
    path.write_text("".join(f"{line}\n" for line in kept_lines), encoding="utf-8")
    return removed


def serialize_game_trace(iteration: int, game_index: int, trace: SelfPlayGameTrace, cfg: ExperimentConfig) -> dict[str, object]:
    return {
        "format": "trilibgo-selfplay-trace-v1",
        "source": "selfplay",
        "iteration": iteration,
        "game_index": game_index,
        "side_length": int(cfg.model.board_side),
        "komi": trace.komi,
        "allow_suicide": bool(cfg.rules.allow_suicide),
        "move_count": trace.move_count,
        "black_score": trace.black_score,
        "white_score": trace.white_score,
        "score_margin_black_minus_white": trace.black_score - trace.white_score,
        "result_value": trace.result_value,
        "winner": trace.winner,
        "first_player_win": trace.first_player_win,
        "first_pass_turn": trace.first_pass_turn,
        "total_passes": trace.total_passes,
        "cleaned_dead_stones": trace.cleaned_dead_stones,
        "cleaned_dead_vertices": list(getattr(trace, "cleaned_dead_vertices", [])),
        "final_board": list(getattr(trace, "final_board", [])),
        "cleanup_rule_resolved_groups": trace.cleanup_rule_resolved_groups,
        "cleanup_local_search_resolved_groups": trace.cleanup_local_search_resolved_groups,
        "cleanup_preserved_seki_groups": trace.cleanup_preserved_seki_groups,
        "unresolved_dead_groups": trace.unresolved_dead_groups,
        "end_reason": trace.end_reason,
        "avg_abs_predicted_score_error": trace.avg_abs_predicted_score_error,
        "sample_weight": trace.sample_weight,
        "abnormal_tags": trace.abnormal_tags,
        "curriculum_scale": trace.curriculum_scale,
        "total_captured_stones": trace.total_captured_stones,
        "total_prev_turn_losses": trace.total_prev_turn_losses,
        "total_immediate_capture_risk_stones": getattr(trace, "total_immediate_capture_risk_stones", 0),
        "total_next_move_capture_stones": trace.total_next_move_capture_stones,
        "total_curriculum_value_bonus": trace.total_curriculum_value_bonus,
        "true_eye_fills": trace.true_eye_fills,
        "pure_capture_moves": trace.pure_capture_moves,
        "immediate_capture_risk_rate_moves": getattr(trace, "immediate_capture_risk_rate_moves", 0),
        "next_move_capture_rate_moves": trace.next_move_capture_rate_moves,
        "curriculum_sample_moves": trace.curriculum_sample_moves,
        "positive_curriculum_bonus_moves": trace.positive_curriculum_bonus_moves,
        "negative_curriculum_bonus_moves": trace.negative_curriculum_bonus_moves,
        "true_eye_penalized_moves": trace.true_eye_penalized_moves,
        "moves": [
            {
                "turn": move.turn,
                "player": move.player,
                "action": move.action,
                "move": move.move,
                "root_value": move.root_value,
                "root_score_margin": move.root_score_margin,
                "root_score_margin_black_minus_white": move.root_score_margin_black_minus_white,
                "score_margin_error_black_minus_white": move.score_margin_error_black_minus_white,
                "policy": move.policy,
                "policy_top": [[action, prob] for action, prob in move.policy_top],
                "captures_by_move": move.captures_by_move,
                "stones_lost_prev_turn": move.stones_lost_prev_turn,
                "immediate_capture_risk_stones": getattr(move, "immediate_capture_risk_stones", 0),
                "next_move_capture_stones": move.next_move_capture_stones,
                "curriculum_value_bonus": move.curriculum_value_bonus,
                "is_curriculum_sample": move.is_curriculum_sample,
                "fills_small_true_eye": move.fills_small_true_eye,
                "curriculum_weight": move.curriculum_weight,
            }
            for move in trace.moves
        ],
    }


def summarize_selfplay_games(traces: list[SelfPlayGameTrace]) -> dict[str, float]:
    if not traces:
        return {
            "selfplay_games": 0.0,
            "first_player_win_rate": 0.0,
            "draw_rate": 0.0,
            "max_moves_rate": 0.0,
            "avg_score_margin_black_minus_white": 0.0,
            "avg_game_length": 0.0,
            "avg_komi": 0.0,
            "avg_first_pass_turn": 0.0,
            "avg_total_passes": 0.0,
            "avg_cleaned_dead_stones": 0.0,
            "avg_cleanup_rule_resolved_groups": 0.0,
            "avg_cleanup_local_search_resolved_groups": 0.0,
            "avg_cleanup_preserved_seki_groups": 0.0,
            "avg_unresolved_dead_groups": 0.0,
            "avg_abs_predicted_score_error": 0.0,
            "avg_sample_weight": 0.0,
            "avg_captures_by_move": 0.0,
            "avg_stones_lost_prev_turn": 0.0,
            "avg_immediate_capture_risk_stones": 0.0,
            "avg_next_move_capture_stones": 0.0,
            "avg_curriculum_value_bonus": 0.0,
            "true_eye_fill_rate": 0.0,
            "pure_capture_rate": 0.0,
            "immediate_capture_risk_rate": 0.0,
            "next_move_capture_rate": 0.0,
            "positive_curriculum_bonus_rate": 0.0,
            "negative_curriculum_bonus_rate": 0.0,
            "eye_fill_penalized_rate": 0.0,
            "curriculum_scale": 0.0,
            "abnormal_games": 0.0,
            "abnormal_game_rate": 0.0,
            "early_pass_games": 0.0,
            "short_games": 0.0,
        }
    first_player_wins = sum(1 for trace in traces if trace.first_player_win)
    draws = sum(1 for trace in traces if trace.winner == "draw")
    abnormal_games = sum(1 for trace in traces if trace.abnormal_tags)
    early_pass_games = sum(1 for trace in traces if "early_pass" in trace.abnormal_tags)
    short_games = sum(1 for trace in traces if "short_game" in trace.abnormal_tags)
    max_moves_games = sum(1 for trace in traces if trace.end_reason == "max_moves")
    score_margin = sum(trace.black_score - trace.white_score for trace in traces)
    avg_len = sum(trace.move_count for trace in traces) / len(traces)
    avg_komi = sum(trace.komi for trace in traces) / len(traces)
    avg_first_pass_turn = sum(trace.first_pass_turn for trace in traces) / len(traces)
    avg_total_passes = sum(trace.total_passes for trace in traces) / len(traces)
    avg_cleaned_dead_stones = sum(trace.cleaned_dead_stones for trace in traces) / len(traces)
    avg_cleanup_rule_resolved_groups = sum(trace.cleanup_rule_resolved_groups for trace in traces) / len(traces)
    avg_cleanup_local_search_resolved_groups = sum(trace.cleanup_local_search_resolved_groups for trace in traces) / len(traces)
    avg_cleanup_preserved_seki_groups = sum(trace.cleanup_preserved_seki_groups for trace in traces) / len(traces)
    avg_unresolved_dead_groups = sum(trace.unresolved_dead_groups for trace in traces) / len(traces)
    avg_sample_weight = sum(trace.sample_weight for trace in traces) / len(traces)
    total_predictions = sum(len(trace.moves) for trace in traces)
    avg_abs_predicted_score_error = (
        sum(trace.avg_abs_predicted_score_error * len(trace.moves) for trace in traces) / total_predictions if total_predictions else 0.0
    )
    avg_captures_by_move = sum(trace.total_captured_stones for trace in traces) / total_predictions if total_predictions else 0.0
    avg_stones_lost_prev_turn = sum(trace.total_prev_turn_losses for trace in traces) / total_predictions if total_predictions else 0.0
    avg_immediate_capture_risk_stones = (
        sum(getattr(trace, "total_immediate_capture_risk_stones", 0) for trace in traces) / total_predictions
        if total_predictions
        else 0.0
    )
    avg_next_move_capture_stones = (
        sum(trace.total_next_move_capture_stones for trace in traces) / total_predictions if total_predictions else 0.0
    )
    avg_curriculum_value_bonus = (
        sum(trace.total_curriculum_value_bonus for trace in traces) / total_predictions if total_predictions else 0.0
    )
    true_eye_fill_rate = sum(trace.true_eye_fills for trace in traces) / total_predictions if total_predictions else 0.0
    pure_capture_rate = sum(trace.pure_capture_moves for trace in traces) / total_predictions if total_predictions else 0.0
    immediate_capture_risk_rate = (
        sum(getattr(trace, "immediate_capture_risk_rate_moves", 0) for trace in traces) / total_predictions
        if total_predictions
        else 0.0
    )
    next_move_capture_rate = (
        sum(trace.next_move_capture_rate_moves for trace in traces) / total_predictions if total_predictions else 0.0
    )
    positive_curriculum_bonus_rate = (
        sum(trace.positive_curriculum_bonus_moves for trace in traces) / total_predictions if total_predictions else 0.0
    )
    negative_curriculum_bonus_rate = (
        sum(trace.negative_curriculum_bonus_moves for trace in traces) / total_predictions if total_predictions else 0.0
    )
    eye_fill_penalized_rate = sum(trace.true_eye_penalized_moves for trace in traces) / total_predictions if total_predictions else 0.0
    avg_curriculum_scale = sum(trace.curriculum_scale for trace in traces) / len(traces)
    avg_margin = score_margin / len(traces)
    return {
        "selfplay_games": float(len(traces)),
        "first_player_win_rate": first_player_wins / len(traces),
        "draw_rate": draws / len(traces),
        "max_moves_rate": max_moves_games / len(traces),
        "avg_score_margin_black_minus_white": avg_margin,
        "avg_game_length": avg_len,
        "avg_komi": avg_komi,
        "avg_first_pass_turn": avg_first_pass_turn,
        "avg_total_passes": avg_total_passes,
        "avg_cleaned_dead_stones": avg_cleaned_dead_stones,
        "avg_cleanup_rule_resolved_groups": avg_cleanup_rule_resolved_groups,
        "avg_cleanup_local_search_resolved_groups": avg_cleanup_local_search_resolved_groups,
        "avg_cleanup_preserved_seki_groups": avg_cleanup_preserved_seki_groups,
        "avg_unresolved_dead_groups": avg_unresolved_dead_groups,
        "avg_abs_predicted_score_error": avg_abs_predicted_score_error,
        "avg_sample_weight": avg_sample_weight,
        "avg_captures_by_move": avg_captures_by_move,
        "avg_stones_lost_prev_turn": avg_stones_lost_prev_turn,
        "avg_immediate_capture_risk_stones": avg_immediate_capture_risk_stones,
        "avg_next_move_capture_stones": avg_next_move_capture_stones,
        "avg_curriculum_value_bonus": avg_curriculum_value_bonus,
        "true_eye_fill_rate": true_eye_fill_rate,
        "pure_capture_rate": pure_capture_rate,
        "immediate_capture_risk_rate": immediate_capture_risk_rate,
        "next_move_capture_rate": next_move_capture_rate,
        "positive_curriculum_bonus_rate": positive_curriculum_bonus_rate,
        "negative_curriculum_bonus_rate": negative_curriculum_bonus_rate,
        "capture_weighted_rate": pure_capture_rate,
        "eye_fill_penalized_rate": eye_fill_penalized_rate,
        "curriculum_scale": avg_curriculum_scale,
        "abnormal_games": float(abnormal_games),
        "abnormal_game_rate": abnormal_games / len(traces),
        "early_pass_games": float(early_pass_games),
        "short_games": float(short_games),
        "max_moves_games": float(max_moves_games),
        "suggested_komi_delta": avg_margin,
    }


def curriculum_metrics_satisfied(metrics: dict[str, object], cfg: ExperimentConfig, iteration: int) -> bool:
    if not bool(cfg.curriculum.stop_on_metrics):
        return False
    min_iteration = max(1, int(getattr(cfg.curriculum, "stop_min_iteration", 1)))
    max_iteration = max(min_iteration, int(getattr(cfg.curriculum, "stop_max_iteration", min_iteration)))
    if iteration < min_iteration or iteration > max_iteration:
        return False
    pure_capture_rate = float(metrics.get("pure_capture_rate", metrics.get("capture_weighted_rate", 0.0)) or 0.0)
    true_eye_fill_rate = float(metrics.get("true_eye_fill_rate", 0.0) or 0.0)
    eye_fill_penalized_rate = float(metrics.get("eye_fill_penalized_rate", 0.0) or 0.0)
    return (
        pure_capture_rate >= float(cfg.curriculum.stop_min_capture_rate)
        and true_eye_fill_rate <= float(cfg.curriculum.stop_max_true_eye_fill_rate)
        and eye_fill_penalized_rate <= float(cfg.curriculum.stop_max_eye_fill_penalized_rate)
    )


def curriculum_should_stop_for_metrics(hit_streak: int, cfg: ExperimentConfig, iteration: int) -> bool:
    if not bool(cfg.curriculum.stop_on_metrics):
        return False
    stop_max_iteration = max(
        int(getattr(cfg.curriculum, "stop_min_iteration", 1)),
        int(getattr(cfg.curriculum, "stop_max_iteration", 1)),
    )
    patience = max(1, int(getattr(cfg.curriculum, "stop_metric_patience", 1)))
    return hit_streak >= patience or iteration >= stop_max_iteration


def clone_model(source: PolicyValueNet, cfg: ExperimentConfig, device: torch.device) -> PolicyValueNet:
    cloned = PolicyValueNet(cfg.model).to(device)
    cloned.load_state_dict(source.state_dict())
    return cloned


def partition_games(total_games: int, workers: int) -> list[int]:
    worker_count = max(1, min(workers, total_games))
    base = total_games // worker_count
    remainder = total_games % worker_count
    return [base + (1 if i < remainder else 0) for i in range(worker_count)]


def generate_selfplay_iteration(
    model: PolicyValueNet,
    cfg: ExperimentConfig,
    device: torch.device,
    iteration: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[TrainingSample], list[SelfPlayGameTrace]]:
    del device
    total_games = cfg.selfplay.games_per_iteration
    if total_games <= 0:
        return [], []

    workers = max(1, cfg.selfplay.workers)
    exploration_scale = selfplay_exploration_scale(cfg.mcts, iteration)
    if workers == 1:
        generated = []
        traces: list[SelfPlayGameTrace] = []
        for game_index in range(total_games):
            samples, trace = generate_selfplay_game(model, cfg, torch.device("cpu"), exploration_scale=exploration_scale, iteration=iteration)
            generated.extend(samples)
            traces.append(trace)
            if progress_callback is not None:
                progress_callback(game_index + 1, total_games)
        return generated, traces

    model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    game_partitions = partition_games(total_games, workers)
    seeds = [cfg.seed + iteration * 10_000 + worker_index * 997 for worker_index in range(len(game_partitions))]

    generated = []
    traces: list[SelfPlayGameTrace] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(game_partitions)) as executor:
        futures = [
            executor.submit(generate_selfplay_batch, model_state, cfg, game_count, seeds[worker_index], exploration_scale, iteration)
            for worker_index, game_count in enumerate(game_partitions)
        ]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            batch_samples, batch_traces = future.result()
            generated.extend(batch_samples)
            traces.extend(batch_traces)
            completed += len(batch_traces)
            if progress_callback is not None:
                progress_callback(completed, total_games)

    return generated, traces


def promote_best(candidate_path: Path, cfg: ExperimentConfig, metrics: dict[str, object]) -> Path:
    cfg.training.best_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = cfg.training.best_dir / f"{cfg.name}_best.pt"
    shutil.copy2(candidate_path, best_checkpoint)
    with (cfg.training.best_dir / f"{cfg.name}_best.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return best_checkpoint


def maybe_export_best(model: PolicyValueNet, cfg: ExperimentConfig) -> str:
    try:
        export_model(model, cfg, cfg.export.onnx_path.as_posix())
        return cfg.export.onnx_path.as_posix()
    except Exception as ex:  # pragma: no cover - depends on optional runtime stack
        return f"export_failed:{ex}"


def best_checkpoint_path(cfg: ExperimentConfig) -> Path:
    return cfg.training.best_dir / f"{cfg.name}_best.pt"


def best_metadata_path(cfg: ExperimentConfig) -> Path:
    return cfg.training.best_dir / f"{cfg.name}_best.json"


def restore_runtime_config_state(saved_cfg: object, cfg: ExperimentConfig) -> None:
    if not isinstance(saved_cfg, ExperimentConfig):
        return
    cfg.curriculum.enabled = bool(saved_cfg.curriculum.enabled)
    cfg.rules.komi = float(saved_cfg.rules.komi)
    cfg.rules.selfplay_komi = [float(komi) for komi in saved_cfg.rules.selfplay_komi]


def load_resume_state(
    resume_path: Path,
    cfg: ExperimentConfig,
    device: torch.device,
) -> tuple[PolicyValueNet, PolicyValueNet, optim.Optimizer, ReplayBuffer, int, int, int, list[SelfPlayGameTrace]]:
    checkpoint = torch.load(resume_path.as_posix(), map_location="cpu", weights_only=False)
    restore_runtime_config_state(checkpoint.get("config"), cfg)
    candidate = PolicyValueNet(cfg.model).to(device)
    candidate.load_state_dict(checkpoint["model"])
    incumbent = PolicyValueNet(cfg.model).to(device)
    if "incumbent" in checkpoint:
        incumbent.load_state_dict(checkpoint["incumbent"])
    else:
        best_checkpoint = cfg.training.best_dir / f"{cfg.name}_best.pt"
        if best_checkpoint.exists():
            best_state = torch.load(best_checkpoint.as_posix(), map_location="cpu", weights_only=False)
            incumbent.load_state_dict(best_state.get("model", checkpoint["model"]))
        else:
            incumbent.load_state_dict(checkpoint["model"])
    optimizer = optim.AdamW(candidate.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    replay = ReplayBuffer.from_state_dict(checkpoint["replay"]) if "replay" in checkpoint else ReplayBuffer(cfg.selfplay.replay_window)
    metadata = checkpoint.get("metadata", {})
    start_iteration = int(metadata.get("iteration", 0)) + 1
    global_train_step = int(metadata.get("global_train_step", 0))
    curriculum_metric_hit_streak = int(metadata.get("curriculum_metric_hit_streak", 0))
    pending_komi_traces = list(checkpoint.get("pending_komi_traces", []))
    return candidate, incumbent, optimizer, replay, start_iteration, global_train_step, curriculum_metric_hit_streak, pending_komi_traces


def run_training(cfg: ExperimentConfig, resume: str | None = None) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cpu")
    cfg.model.action_size = action_size_for_side(cfg.model.board_side)
    lr_decay_steps = resolve_lr_decay_steps(cfg)
    compute_learning_rate(
        cfg.training.learning_rate,
        global_step=0,
        warmup_steps=cfg.training.lr_warmup_steps,
        decay_steps=lr_decay_steps,
        min_ratio=cfg.training.lr_min_ratio,
        schedule=cfg.training.lr_schedule,
    )

    if resume:
        (
            candidate,
            incumbent,
            optimizer,
            replay,
            start_iteration,
            global_train_step,
            curriculum_metric_hit_streak,
            pending_komi_traces,
        ) = load_resume_state(Path(resume), cfg, device)
    else:
        candidate = PolicyValueNet(cfg.model).to(device)
        incumbent = clone_model(candidate, cfg, device)
        optimizer = optim.AdamW(candidate.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        replay = ReplayBuffer(cfg.selfplay.replay_window)
        start_iteration = 1
        global_train_step = 0
        curriculum_metric_hit_streak = 0
        pending_komi_traces = []

    cfg.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.training.best_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.training.checkpoint_dir / f"{cfg.name}_metrics.jsonl"
    if cfg.telemetry.enabled:
        cfg.telemetry.selfplay_dir.mkdir(parents=True, exist_ok=True)
    if resume is None:
        reset_selfplay_logs(cfg)
        reset_run_logs(
            [
                metrics_path,
                cfg.telemetry.report_path,
                best_checkpoint_path(cfg),
                best_metadata_path(cfg),
                cfg.export.onnx_path,
            ]
        )
    else:
        completed_iteration = start_iteration - 1
        trimmed_metrics = trim_jsonl_to_iteration(metrics_path, completed_iteration)
        trimmed_selfplay = trim_selfplay_logs(cfg, completed_iteration)
        pruned_checkpoints = prune_checkpoints(cfg, completed_iteration)
        if cfg.telemetry.report_path.exists():
            cfg.telemetry.report_path.unlink()
        if console_log_mode(cfg) != "events" and (trimmed_metrics > 0 or trimmed_selfplay > 0 or pruned_checkpoints):
            print_notice(
                cfg,
                (
                    f"[Resume] trimmed metrics={trimmed_metrics} selfplay={trimmed_selfplay} "
                    f"pruned_ckpt={len(pruned_checkpoints)} beyond iter {completed_iteration}"
                ),
            )

    if start_iteration > cfg.selfplay.iterations:
        message = "resume checkpoint is already at or beyond the configured iteration budget"
        if console_log_mode(cfg) == "events":
            print(
                json.dumps(
                    {
                        "event": "resume_complete",
                        "message": message,
                        "start_iteration": start_iteration,
                        "configured_iterations": cfg.selfplay.iterations,
                    }
                ),
                flush=True,
            )
        else:
            print_notice(
                cfg,
                f"[Resume] {message} (start={start_iteration}, configured={cfg.selfplay.iterations})",
            )
        return

    if console_log_mode(cfg) != "events":
        run_mode = "resume" if resume else "fresh"
        print_notice(
            cfg,
            (
                f"[Run] name={cfg.name} mode={run_mode} iterations={cfg.selfplay.iterations} "
                f"workers={cfg.selfplay.workers} sims={cfg.mcts.simulations} "
                f"komi={cfg.rules.komi:.2f} progress={'on' if cfg.telemetry.show_progress else 'off'}"
            ),
        )

    for iteration in range(start_iteration, cfg.selfplay.iterations + 1):
        iteration_komi = float(cfg.rules.komi)
        eval_plan = evaluation_plan(cfg, iteration)
        replay_summary = replay.summary()
        emit_event(
            cfg,
            "iteration_start",
            iteration=iteration,
            total_iterations=cfg.selfplay.iterations,
            games_per_iteration=cfg.selfplay.games_per_iteration,
            workers=cfg.selfplay.workers,
            simulations=cfg.mcts.simulations,
            selfplay_exploration_scale=round(selfplay_exploration_scale(cfg.mcts, iteration), 3),
            eval_interval=eval_plan.interval,
            eval_mode=eval_plan.mode,
            eval_games=eval_plan.games,
            eval_simulations=eval_plan.simulations,
            eval_scheduled=eval_plan.run,
            curriculum_scale=round(curriculum_strength(cfg, iteration), 3),
            komi=round(iteration_komi, 3),
            buffer_size=len(replay),
            buffer_effective_size=round(replay_summary["buffer_effective_size"], 3),
            lr_schedule=cfg.training.lr_schedule,
            lr_warmup_steps=cfg.training.lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            global_train_step=global_train_step,
        )

        selfplay_started = time.perf_counter()
        selfplay_progress = make_progress_bar(cfg, "selfplay", cfg.selfplay.games_per_iteration)
        generated, traces = generate_selfplay_iteration(
            candidate,
            cfg,
            device,
            iteration,
            progress_callback=selfplay_progress.update if selfplay_progress is not None else None,
        )
        if selfplay_progress is not None:
            selfplay_progress.close()
        selfplay_duration = time.perf_counter() - selfplay_started
        emit_event(
            cfg,
            "selfplay_complete",
            iteration=iteration,
            games=len(traces),
            samples=len(generated),
            duration_sec=round(selfplay_duration, 3),
            workers=cfg.selfplay.workers,
        )
        selfplay_path = iteration_selfplay_path(cfg, iteration)
        for game_index, trace in enumerate(traces, start=1):
            if cfg.telemetry.enabled and cfg.telemetry.save_game_records:
                write_metrics_line(selfplay_path, serialize_game_trace(iteration, game_index, trace, cfg))
        replay.extend(generated)
        replay_summary = replay.summary()
        pending_komi_traces.extend(traces)

        metrics: dict[str, object] = {}
        training_started = time.perf_counter()
        training_steps = effective_training_steps(
            len(replay),
            cfg.training.batch_size,
            cfg.training.steps_per_iteration,
            cfg.training.epochs_per_iteration,
            cfg.training.min_steps_per_iteration,
        )
        train_progress = make_progress_bar(cfg, "train", training_steps)
        training_step_metrics: list[dict[str, float]] = []
        for epoch_index in range(training_steps):
            learning_rate = compute_learning_rate(
                cfg.training.learning_rate,
                global_step=global_train_step,
                warmup_steps=cfg.training.lr_warmup_steps,
                decay_steps=lr_decay_steps,
                min_ratio=cfg.training.lr_min_ratio,
                schedule=cfg.training.lr_schedule,
            )
            set_optimizer_learning_rate(optimizer, learning_rate)
            step_metrics = train_iteration(candidate, optimizer, replay, cfg, device)
            step_metrics["learning_rate"] = learning_rate
            training_step_metrics.append(step_metrics)
            global_train_step += 1
            if train_progress is not None:
                train_progress.update(epoch_index + 1, training_steps)
            emit_event(
                cfg,
                "train_epoch_complete",
                iteration=iteration,
                epoch=epoch_index + 1,
                epochs=training_steps,
                policy_loss=round(float(step_metrics["policy_loss"]), 6),
                policy_entropy=round(float(step_metrics.get("policy_entropy", 0.0)), 6),
                entropy_bonus=round(float(step_metrics.get("entropy_bonus", 0.0)), 6),
                value_loss=round(float(step_metrics["value_loss"]), 6),
                ownership_loss=round(float(step_metrics["ownership_loss"]), 6),
                score_loss=round(float(step_metrics["score_loss"]), 6),
                liberty_loss=round(float(step_metrics.get("liberty_loss", 0.0)), 6),
                total_loss=round(float(step_metrics["total_loss"]), 6),
                effective_batch_size=round(float(step_metrics["effective_batch_size"]), 3),
                avg_batch_weight=round(float(step_metrics["avg_batch_weight"]), 3),
                grad_norm=round(float(step_metrics["grad_norm"]), 6),
                clipped_grad_norm=round(float(step_metrics["clipped_grad_norm"]), 6),
                learning_rate=round(float(step_metrics["learning_rate"]), 8),
                buffer_size=len(replay),
            )
        if train_progress is not None:
            train_progress.close()
        training_duration = time.perf_counter() - training_started
        metrics = summarize_training_steps(training_step_metrics)

        eval_wins = 0
        eval_losses = 0
        eval_draws = 0
        eval_win_rate: float | None = None
        evaluation_duration = 0.0
        promoted = False
        eval_ran = False
        if eval_plan.run:
            evaluation_started = time.perf_counter()
            evaluation_progress = make_progress_bar(cfg, f"eval:{eval_plan.mode}", eval_plan.games)
            evaluation = evaluate_candidate(
                candidate,
                incumbent,
                cfg,
                games_override=eval_plan.games,
                simulations_override=eval_plan.simulations,
                candidate_starts_black=(iteration % 2 == 1),
                progress_callback=evaluation_progress.update if evaluation_progress is not None else None,
            )
            if evaluation_progress is not None:
                evaluation_progress.close()
            evaluation_duration = time.perf_counter() - evaluation_started
            eval_wins = evaluation.wins
            eval_losses = evaluation.losses
            eval_draws = evaluation.draws
            eval_win_rate = evaluation.win_rate
            promoted = evaluation.win_rate >= cfg.evaluation.promotion_win_rate
            eval_ran = True
            if promoted:
                incumbent.load_state_dict(candidate.state_dict())

        bootstrap_best = resume is None and iteration == start_iteration
        best_updated = promoted or bootstrap_best
        if best_updated:
            incumbent.load_state_dict(candidate.state_dict())

        komi_summary = summarize_komi_adjustment(pending_komi_traces, cfg, checked=should_adjust_komi(cfg, iteration))
        if komi_summary.checked:
            emit_event(
                cfg,
                "komi_adjustment",
                iteration=iteration,
                reason=komi_summary.reason,
                considered_games=komi_summary.considered_games,
                eligible_games=komi_summary.eligible_games,
                weighted_margin=None if komi_summary.weighted_margin is None else round(float(komi_summary.weighted_margin), 6),
                delta=round(float(komi_summary.delta), 6),
                komi_before=round(float(komi_summary.previous_komi), 6),
                komi_after=round(float(komi_summary.next_komi), 6),
                applied=komi_summary.applied,
            )
        apply_komi_adjustment(cfg, komi_summary)
        if komi_summary.checked and komi_summary.reason != "insufficient_samples":
            pending_komi_traces.clear()

        metrics.update(
            {
                "iteration": iteration,
                "komi_used": iteration_komi,
                "komi_next": float(cfg.rules.komi),
                "komi_adjustment_checked": komi_summary.checked,
                "komi_adjustment_applied": komi_summary.applied,
                "komi_adjustment_reason": komi_summary.reason,
                "komi_adjustment_considered_games": komi_summary.considered_games,
                "komi_adjustment_eligible_games": komi_summary.eligible_games,
                "komi_adjustment_weighted_margin": komi_summary.weighted_margin,
                "komi_adjustment_delta": komi_summary.delta,
                "selfplay_workers": cfg.selfplay.workers,
                "buffer_size": len(replay),
                "buffer_effective_size": replay_summary["buffer_effective_size"],
                "buffer_avg_sample_weight": replay_summary["buffer_avg_sample_weight"],
                "buffer_downweighted_rate": replay_summary["buffer_downweighted_rate"],
                "selfplay_samples": len(generated),
                "training_steps": training_steps,
                "global_train_step": global_train_step,
                "eval_interval": eval_plan.interval,
                "eval_mode": eval_plan.mode,
                "eval_ran": eval_ran,
                "eval_games": eval_plan.games,
                "eval_wins": eval_wins,
                "eval_losses": eval_losses,
                "eval_draws": eval_draws,
                "eval_win_rate": eval_win_rate,
                "promoted": promoted,
                "bootstrap_best": bootstrap_best,
                "best_updated": best_updated,
                "selfplay_duration_sec": round(selfplay_duration, 3),
                "training_duration_sec": round(training_duration, 3),
                "evaluation_duration_sec": round(evaluation_duration, 3),
                "eval_simulations": eval_plan.simulations,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        metrics.update(summarize_selfplay_games(traces))
        curriculum_disabled = False
        if bool(cfg.curriculum.enabled) and bool(cfg.curriculum.stop_on_metrics):
            stop_min_iteration = max(1, int(getattr(cfg.curriculum, "stop_min_iteration", 1)))
            stop_max_iteration = max(stop_min_iteration, int(getattr(cfg.curriculum, "stop_max_iteration", stop_min_iteration)))
            metric_ready = curriculum_metrics_satisfied(metrics, cfg, iteration)
            if stop_min_iteration <= iteration <= stop_max_iteration and metric_ready:
                curriculum_metric_hit_streak += 1
            else:
                curriculum_metric_hit_streak = 0
            if curriculum_should_stop_for_metrics(curriculum_metric_hit_streak, cfg, iteration):
                cfg.curriculum.enabled = False
                curriculum_disabled = True
                metric_patience = max(1, int(getattr(cfg.curriculum, "stop_metric_patience", 1)))
                emit_event(
                    cfg,
                    "curriculum_disabled",
                    iteration=iteration,
                    reason="metric_ready" if curriculum_metric_hit_streak >= metric_patience else "stop_max_iteration",
                    metric_hit_streak=curriculum_metric_hit_streak,
                    metric_patience=metric_patience,
                    pure_capture_rate=round(float(metrics.get("pure_capture_rate", 0.0)), 6),
                    next_move_capture_rate=round(float(metrics.get("next_move_capture_rate", 0.0)), 6),
                    true_eye_fill_rate=round(float(metrics.get("true_eye_fill_rate", 0.0)), 6),
                    eye_fill_penalized_rate=round(float(metrics.get("eye_fill_penalized_rate", 0.0)), 6),
                )
        else:
            curriculum_metric_hit_streak = 0
        metrics["curriculum_metric_hit_streak"] = float(curriculum_metric_hit_streak)
        metrics["curriculum_disabled"] = curriculum_disabled
        metrics["curriculum_active_next_iteration"] = bool(cfg.curriculum.enabled)
        write_metrics_line(metrics_path, metrics)

        candidate_path = snapshot_path(cfg.training.checkpoint_dir, cfg.name, iteration)
        prune_checkpoints(cfg, iteration - 1)
        save_checkpoint(candidate_path, candidate, incumbent, optimizer, replay, cfg, metrics, pending_komi_traces)
        prune_checkpoints(cfg, iteration)
        if best_updated:
            best_checkpoint = promote_best(candidate_path, cfg, metrics)
            metrics["best_checkpoint"] = best_checkpoint.as_posix()
            metrics["onnx_export"] = maybe_export_best(candidate, cfg)
            write_metrics_line(metrics_path, {"event": "best_update", **metrics})

        print_iteration_summary(cfg, cfg.selfplay.iterations, metrics)

    if cfg.telemetry.enabled and render_html is not None and cfg.telemetry.save_game_records:
        render_html(
            load_jsonl(metrics_path),
            load_recent_selfplay_records(cfg),
            cfg.telemetry.report_path,
            recent_games=cfg.telemetry.report_recent_games,
        )
        if console_log_mode(cfg) == "events":
            print(json.dumps({"event": "report_written", "path": cfg.telemetry.report_path.as_posix()}), flush=True)
        else:
            print_notice(cfg, f"[Report] {cfg.telemetry.report_path.as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CPU-first TriLibGo policy-value model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("python/rl/configs/tiny_cpu_baseline.json"),
        help="Path to the experiment JSON config.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    parser.add_argument(
        "--board-side",
        type=int,
        default=None,
        help="Override model.board_side and derived action size. Also rewrites stage5 output paths unless disabled.",
    )
    parser.add_argument(
        "--no-board-side-rename",
        action="store_true",
        help="Do not rewrite run name/output paths when --board-side is used.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.board_side is not None:
        apply_board_side(config, args.board_side, rename_run=not args.no_board_side_rename)
    run_training(config, args.resume.as_posix() if args.resume else None)
