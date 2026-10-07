"""Dataset wrappers for AneRBC-I (Phase 2 IDA fine-tuning / eval) and
Chula-RBC-12 (Phase 1 same-domain pretraining).

Both datasets are expected to be downloaded manually following the
instructions in README.md. The loaders here accept the on-disk layout
produced by the official releases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ------------------------------------------------------------------
#  AneRBC-I (binary IDA)
# ------------------------------------------------------------------

@dataclass
class AneRBCISample:
    image_path: Path
    label: int  # 0 = Healthy, 1 = IDA


class AneRBCIDataset(Dataset):
    """Primary IDA dataset.

    The on-disk layout is expected to be::

        data/AneRBC-I/
            healthy/*.png
            anemic/*.png

    An optional stratified split manifest (JSON) can be provided so that
    train/val/test partitions are identical across runs.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test", "all"] = "train",
        split_manifest: str | Path | None = None,
        transform: Callable[[np.ndarray], torch.Tensor] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        if split_manifest is not None:
            with open(split_manifest, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if split == "all":
                paths = sum(manifest.values(), [])
            else:
                paths = manifest[split]
            self.samples = [
                AneRBCISample(image_path=self.root / p,
                              label=0 if "healthy" in p.lower() else 1)
                for p in paths
            ]
        else:
            healthy = sorted((self.root / "healthy").glob("*.png"))
            anemic = sorted((self.root / "anemic").glob("*.png"))
            self.samples = (
                [AneRBCISample(image_path=p, label=0) for p in healthy] +
                [AneRBCISample(image_path=p, label=1) for p in anemic]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        s = self.samples[idx]
        image = np.asarray(Image.open(s.image_path).convert("RGB"))
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, s.label


# ------------------------------------------------------------------
#  Chula-RBC-12 (12-class morphology, Phase-1 pretraining only)
# ------------------------------------------------------------------

CHULA_CLASSES = (
    "Normal", "Macrocyte", "Microcyte", "Spherocyte",
    "Target cell", "Stomatocyte", "Ovalocyte", "Teardrop",
    "Burr cell", "Schistocyte", "Hypochromia", "Uncategorized",
)
CHULA_CLASS_TO_IDX = {name: i for i, name in enumerate(CHULA_CLASSES)}


class ChulaRBC12Dataset(Dataset):
    """Phase-1 pretraining corpus.

    Expected layout::

        data/Chula-RBC-12/
            images/*.jpg
            annotations.json   # list of {image, class, bbox}
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "all"] = "train",
        transform: Callable[[np.ndarray], torch.Tensor] | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        ann_file = self.root / "annotations.json"
        if not ann_file.exists():
            raise FileNotFoundError(
                f"Expected Chula-RBC-12 annotations at {ann_file}; see "
                "scripts/build_split_manifests.py for the preprocessing step."
            )
        with open(ann_file, "r", encoding="utf-8") as f:
            all_items = json.load(f)
        self.samples = [a for a in all_items if a.get("split", "train") == split or split == "all"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.samples[idx]
        image = np.asarray(Image.open(self.root / "images" / item["image"]).convert("RGB"))
        # Crop by bbox when present
        if "bbox" in item:
            x0, y0, x1, y1 = item["bbox"]
            image = image[y0:y1, x0:x1]
        label = CHULA_CLASS_TO_IDX[item["class"]]
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, label
