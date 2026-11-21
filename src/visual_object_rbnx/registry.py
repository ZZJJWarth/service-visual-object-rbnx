# SPDX-License-Identifier: MulanPSL-2.0
"""Declarative BlockSpec registry for extensible visual block detection & interaction."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BlockSpec:
    label: str
    display_name: str
    mc_blocks: tuple[str, ...]
    backend: str = "auto"  # "auto" | "classic" | "onnx"
    onnx_class_id: int | None = None
    min_width_m: float = 0.45
    max_width_m: float = 1.40
    min_height_m: float = 0.42
    max_height_m: float = 1.35
    inward_push_m: float = 0.38
    approach_min_m: float = 1.35
    approach_max_m: float = 2.35
    interactable: bool = True
    container: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "display_name": self.display_name,
            "mc_blocks": list(self.mc_blocks),
            "backend": self.backend,
            "onnx_class_id": self.onnx_class_id,
            "min_width_m": self.min_width_m,
            "max_width_m": self.max_width_m,
            "min_height_m": self.min_height_m,
            "max_height_m": self.max_height_m,
            "inward_push_m": self.inward_push_m,
            "approach_min_m": self.approach_min_m,
            "approach_max_m": self.approach_max_m,
            "interactable": self.interactable,
            "container": self.container,
        }


class BlockRegistry:
    """Holds all configured BlockSpecs indexed by label and optional ONNX class ID."""

    def __init__(self, specs: dict[str, BlockSpec] | None = None) -> None:
        self._specs: dict[str, BlockSpec] = dict(specs or {})
        self._by_onnx_id: dict[int, BlockSpec] = {
            sp.onnx_class_id: sp for sp in self._specs.values() if sp.onnx_class_id is not None
        }

    @classmethod
    def load_yaml(cls, path: Path) -> "BlockRegistry":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        blocks = raw.get("blocks") or {}
        if not isinstance(blocks, dict):
            raise ValueError("blocks.yaml must define a 'blocks' mapping")
        specs: dict[str, BlockSpec] = {}
        for label, item in blocks.items():
            if not isinstance(item, dict):
                continue
            key = str(label).strip().lower()
            mc_raw = item.get("mc_blocks") or [f"minecraft:{key}"]
            specs[key] = BlockSpec(
                label=key,
                display_name=str(item.get("display_name") or key),
                mc_blocks=tuple(str(b).strip() for b in mc_raw if str(b).strip()),
                backend=str(item.get("backend") or "auto").strip().lower(),
                onnx_class_id=int(item["onnx_class_id"]) if item.get("onnx_class_id") is not None else None,
                min_width_m=float(item.get("min_width_m", 0.45)),
                max_width_m=float(item.get("max_width_m", 1.40)),
                min_height_m=float(item.get("min_height_m", 0.42)),
                max_height_m=float(item.get("max_height_m", 1.35)),
                inward_push_m=float(item.get("inward_push_m", 0.38)),
                approach_min_m=float(item.get("approach_min_m", 1.35)),
                approach_max_m=float(item.get("approach_max_m", 2.35)),
                interactable=bool(item.get("interactable", True)),
                container=bool(item.get("container", False)),
            )
        return cls(specs)

    def get(self, label: str) -> BlockSpec | None:
        return self._specs.get(str(label or "").strip().lower())

    def by_onnx_id(self, class_id: int) -> BlockSpec | None:
        return self._by_onnx_id.get(int(class_id))

    def labels(self) -> list[str]:
        return sorted(self._specs.keys())

    def all_specs(self) -> list[BlockSpec]:
        return [self._specs[k] for k in sorted(self._specs.keys())]

    def register(self, spec: BlockSpec) -> None:
        self._specs[spec.label] = spec
        if spec.onnx_class_id is not None:
            self._by_onnx_id[spec.onnx_class_id] = spec
