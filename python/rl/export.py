from __future__ import annotations

import warnings
from pathlib import Path

import torch

from .config import ExperimentConfig, input_planes_for_history, vertex_count_for_side
from .model import PolicyValueNet


def export_model(model: PolicyValueNet, cfg: ExperimentConfig, output_path: str) -> None:
    export_net = PolicyValueNet(cfg.model)
    export_net.load_state_dict(model.state_dict())
    export_net.eval()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(
        1,
        input_planes_for_history(cfg.model.input_history),
        vertex_count_for_side(cfg.model.board_side),
        dtype=torch.float32,
    )
    dummy_global = torch.zeros(1, cfg.model.global_feature_count, dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using the legacy TorchScript-based ONNX export.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="The feature will be removed.*",
            category=DeprecationWarning,
        )
        torch.onnx.export(
            export_net,
            (dummy, dummy_global),
            output.as_posix(),
            input_names=["planes", "global_features"],
            output_names=["policy_logits", "value", "ownership", "score_margin", "liberty_logits"],
            opset_version=cfg.export.opset,
            dynamo=False,
            dynamic_axes={
                "planes": {0: "batch"},
                "global_features": {0: "batch"},
                "policy_logits": {0: "batch"},
                "value": {0: "batch"},
                "ownership": {0: "batch"},
                "score_margin": {0: "batch"},
                "liberty_logits": {0: "batch"},
            },
        )


def export_model_from_state_dict(model_state: dict[str, torch.Tensor], cfg: ExperimentConfig, output_path: str) -> None:
    model = PolicyValueNet(cfg.model)
    model.load_state_dict(model_state)
    model.eval()
    export_model(model, cfg, output_path)


def export_onnx(checkpoint_path: str, cfg: ExperimentConfig) -> None:
    cfg.model.action_size = vertex_count_for_side(cfg.model.board_side) + 1
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    export_model_from_state_dict(checkpoint["model"], cfg, cfg.export.onnx_path.as_posix())
