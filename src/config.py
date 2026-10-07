"""Centralized configuration loader for the EX-IDA-CBM pipeline.

All hyperparameters live in `configs/*.yaml`; this module exposes a
single dataclass that the training/evaluation scripts consume. Default
values match the values reported in the paper so that running any
script with just `--config configs/phase2.yaml` reproduces the
headline numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    aner_bc_root: str = "data/AneRBC-I"
    chula_rbc_root: str = "data/Chula-RBC-12"
    split_manifest: str = "data/splits/aner_bc_split_42.json"
    crop_size: int = 64
    input_size: int = 224
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42
    stain_reference: str = "data/stain_reference/reference_slide.png"


@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    concept_hidden: tuple[int, int] = (512, 256)
    num_concepts: int = 5
    classifier_hidden: int = 32
    num_classes: int = 2
    dropout: tuple[float, float] = (0.3, 0.2)


@dataclass
class TrainConfig:
    # Phase-1 (Chula-RBC-12 morphology pretraining)
    phase1_epochs: int = 100
    phase1_lr: float = 1e-3
    phase1_batch_size: int = 64
    phase1_warmup_epochs: int = 5

    # Phase-2 (AneRBC-I IDA fine-tuning)
    phase2_epochs: int = 60
    phase2_lr: float = 1e-4
    phase2_backbone_lr_mult: float = 0.1
    phase2_batch_size: int = 64
    phase2_early_stop_patience: int = 10
    phase2_warmup_epochs: int = 5

    # Common
    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    mixup_alpha: float = 1.0
    cutmix_alpha: float = 1.0
    mixup_cutmix_prob: float = 0.5

    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"


@dataclass
class EvalConfig:
    bootstrap_resamples: int = 1000
    ece_bins: int = 15
    severity_thresholds: tuple[float, float, float] = (0.3, 0.5, 0.7)
    mc_dropout_passes: int = 30


@dataclass
class PipelineConfig:
    """Top-level config consumed by every training/eval script."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "results/logs"
    experiment_name: str = "default"


# ------------------------------------------------------------------
#  I/O helpers
# ------------------------------------------------------------------

def _apply_overrides(dc: Any, overrides: dict[str, Any]) -> None:
    """Recursively copy keys from an overrides dict into a dataclass."""
    for k, v in overrides.items():
        if not hasattr(dc, k):
            raise ValueError(f"Unknown config key: {k}")
        current = getattr(dc, k)
        if hasattr(current, "__dataclass_fields__"):
            if not isinstance(v, dict):
                raise ValueError(f"Expected mapping for key {k}, got {type(v)}")
            _apply_overrides(current, v)
        else:
            setattr(dc, k, v)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML config file and merge it on top of the defaults."""
    cfg = PipelineConfig()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    _apply_overrides(cfg, raw)
    return cfg


def dump_config(cfg: PipelineConfig, path: str | Path) -> None:
    """Dump a config back to YAML for provenance tracking."""
    def _to_dict(dc: Any) -> dict[str, Any]:
        if hasattr(dc, "__dataclass_fields__"):
            return {k: _to_dict(getattr(dc, k)) for k in dc.__dataclass_fields__}
        if isinstance(dc, (list, tuple)):
            return list(dc)
        return dc

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_dict(cfg), f, sort_keys=False)
