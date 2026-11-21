# SPDX-License-Identifier: MulanPSL-2.0
"""Strict configuration validation for service-visual-object-rbnx."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOPIC_RE = re.compile(r"^/(?:[A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*$")

TOP_LEVEL_KEYS = {
    "rgb_topic",
    "depth_topic",
    "intrinsics_topic",
    "extrinsics_topic",
    "odom_topic",
    "local_cost_grid_topic",
    "cmd_vel_topic",
    "use_sim_time",
    "blocks_config_file",
    "detector_model_path",
    "detect_rate_hz",
    "min_confidence",
    "max_detect_range_m",
    "min_track_hits",
    "webui_enabled",
    "webui_host",
    "webui_port",
}


class ConfigError(RuntimeError):
    """Raised when deployment configuration is invalid."""


@dataclass(frozen=True)
class VisualObjectConfig:
    rgb_topic: str = "/minecraft/camera/rgb"
    depth_topic: str = "/minecraft/camera/depth"
    intrinsics_topic: str = "/minecraft/camera/intrinsics"
    extrinsics_topic: str = "/minecraft/camera/extrinsics"
    odom_topic: str = "/minecraft/chassis/odom"
    local_cost_grid_topic: str = "/robonix/local_mapping/cost_grid"
    cmd_vel_topic: str = "/cmd_vel"
    use_sim_time: bool = True
    blocks_config_file: Path = Path("config/blocks.yaml")
    detector_model_path: str = ""
    detect_rate_hz: float = 4.0
    min_confidence: float = 0.45
    max_detect_range_m: float = 10.0
    min_track_hits: int = 2
    webui_enabled: bool = True
    webui_host: str = "127.0.0.1"
    webui_port: int = 8094


def _num(raw: Any, name: str, lo: float, hi: float) -> float:
    if isinstance(raw, bool):
        raise ConfigError(f"{name} must be a number")
    val = float(raw)
    if not math.isfinite(val) or val < lo or val > hi:
        raise ConfigError(f"{name}={raw!r} must be within [{lo}, {hi}]")
    return val


def _int(raw: Any, name: str, lo: int, hi: int) -> int:
    if isinstance(raw, bool):
        raise ConfigError(f"{name} must be an integer")
    val = int(raw)
    if isinstance(raw, float) and not raw.is_integer():
        raise ConfigError(f"{name} must be an exact integer")
    if val < lo or val > hi:
        raise ConfigError(f"{name}={val} must be within [{lo}, {hi}]")
    return val


def _bool(raw: Any, name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.strip().lower() in ("1", "true", "yes", "on", "0", "false", "no", "off"):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    raise ConfigError(f"{name} must be a boolean")


def _str(raw: Any, name: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return raw.strip()


def _topic(raw: Any, name: str) -> str:
    val = _str(raw, name)
    if not TOPIC_RE.fullmatch(val):
        raise ConfigError(f"{name}={val!r} must be an absolute ROS topic")
    return val


def from_config(config: dict[str, Any], package_root: Path) -> VisualObjectConfig:
    if not isinstance(config, dict):
        raise ConfigError("config must be a mapping")
    unknown = set(config) - TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"unknown config key(s): {sorted(unknown)}")

    raw_blocks = Path(os.path.expanduser(_str(config.get("blocks_config_file", "config/blocks.yaml"), "blocks_config_file")))
    blocks_path = raw_blocks if raw_blocks.is_absolute() else (package_root / raw_blocks).resolve()

    raw_model = str(config.get("detector_model_path", "") or "").strip()
    if raw_model:
        mp = Path(os.path.expanduser(raw_model))
        model_path = str(mp if mp.is_absolute() else (package_root / mp).resolve())
    else:
        model_path = ""

    return VisualObjectConfig(
        rgb_topic=_topic(config.get("rgb_topic", "/minecraft/camera/rgb"), "rgb_topic"),
        depth_topic=_topic(config.get("depth_topic", "/minecraft/camera/depth"), "depth_topic"),
        intrinsics_topic=_topic(config.get("intrinsics_topic", "/minecraft/camera/intrinsics"), "intrinsics_topic"),
        extrinsics_topic=_topic(config.get("extrinsics_topic", "/minecraft/camera/extrinsics"), "extrinsics_topic"),
        odom_topic=_topic(config.get("odom_topic", "/minecraft/chassis/odom"), "odom_topic"),
        local_cost_grid_topic=_topic(
            config.get("local_cost_grid_topic", "/robonix/local_mapping/cost_grid"),
            "local_cost_grid_topic",
        ),
        cmd_vel_topic=_topic(config.get("cmd_vel_topic", "/cmd_vel"), "cmd_vel_topic"),
        use_sim_time=_bool(config.get("use_sim_time", True), "use_sim_time"),
        blocks_config_file=blocks_path,
        detector_model_path=model_path,
        detect_rate_hz=_num(config.get("detect_rate_hz", 4.0), "detect_rate_hz", 0.5, 15.0),
        min_confidence=_num(config.get("min_confidence", 0.45), "min_confidence", 0.1, 0.99),
        max_detect_range_m=_num(config.get("max_detect_range_m", 10.0), "max_detect_range_m", 2.0, 20.0),
        min_track_hits=_int(config.get("min_track_hits", 2), "min_track_hits", 1, 10),
        webui_enabled=_bool(config.get("webui_enabled", True), "webui_enabled"),
        webui_host=_str(config.get("webui_host", "127.0.0.1"), "webui_host"),
        webui_port=_int(config.get("webui_port", 8094), "webui_port", 0, 65535),
    )
