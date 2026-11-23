# SPDX-License-Identifier: MulanPSL-2.0
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from visual_object_rbnx import params
from visual_object_rbnx.approach import LocalCostGrid, aim_with_ray_probes, select_approach_pose
from visual_object_rbnx.detectors import ClassicBlockDetector, DetectorPipeline
from visual_object_rbnx.projection import (
    CameraIntrinsics,
    Observation3D,
    SE3Pose,
    VisualObjectTracker,
    backproject_detection_3d,
    default_base_to_camera,
)
from visual_object_rbnx.registry import BlockRegistry, BlockSpec

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "visual_object_rbnx"


def _make_synthetic_chest_scene(
    *,
    chest_world_x: float = 4.5,
    chest_world_y: float = 0.5,
    chest_world_z: float = 0.44,
    include_wood_wall: bool = True,
) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics, SE3Pose]:
    """Render a realistic 320x240 synthetic RGB + Depth frame with a Minecraft chest
    and an optional large flat oak-plank wall distractor behind/beside it.
    """
    w, h = 320, 240
    fx = fy = 260.0
    cx, cy = w / 2.0, h / 2.0
    intr = CameraIntrinsics(w, h, fx, fy, cx, cy)
    # Camera at origin looking +X (east), eye height z = 1.62m
    T_odom_base = SE3Pose(t=np.array([0.0, 0.0, 1.62]), R=np.eye(3))
    T_odom_cam = T_odom_base.compose(default_base_to_camera())

    # Gray stone background at 8.0 m
    rgb = np.full((h, w, 3), (120, 120, 124), dtype=np.uint8)
    depth = np.full((h, w), 8.0, dtype=np.float32)

    # Optional large flat wooden wall distractor on the left (uniform brown, no dark border/latch, 4m wide!)
    if include_wood_wall:
        rgb[20:220, 5:125] = (175, 130, 75)
        depth[20:220, 5:125] = 4.2

    # Project 0.875m chest front face at x = chest_world_x - 0.4375
    face_x = chest_world_x - 0.4375
    half_w = 0.4375
    z_bot = chest_world_z - 0.4375
    z_top = chest_world_z + 0.4375

    # Compute pixel bounds of the front face
    corners_odom = np.array([
        [face_x, chest_world_y + half_w, z_top],
        [face_x, chest_world_y - half_w, z_top],
        [face_x, chest_world_y + half_w, z_bot],
        [face_x, chest_world_y - half_w, z_bot],
    ])
    R_co = T_odom_cam.R.T
    t_co = -R_co @ T_odom_cam.t
    corners_cam = (corners_odom @ R_co.T) + t_co
    us = intr.cx + intr.fx * corners_cam[:, 0] / corners_cam[:, 2]
    vs = intr.cy + intr.fy * corners_cam[:, 1] / corners_cam[:, 2]
    u1, u2 = max(0, int(round(us.min()))), min(w, int(round(us.max())))
    v1, v2 = max(0, int(round(vs.min()))), min(h, int(round(vs.max())))

    # Paint authentic Minecraft chest front texture: warm golden oak + dark border/seam + silver lock latch
    rgb[v1:v2, u1:u2] = (182, 122, 38)  # warm ochre chest body
    # Dark outer border (2 px)
    rgb[v1 : v1 + 2, u1:u2] = (42, 28, 12)
    rgb[max(v1, v2 - 2) : v2, u1:u2] = (42, 28, 12)
    rgb[v1:v2, u1 : u1 + 2] = (42, 28, 12)
    rgb[v1:v2, max(u1, u2 - 2) : u2] = (42, 28, 12)
    # Dark horizontal lid seam at ~35% height
    seam_y = v1 + int((v2 - v1) * 0.35)
    rgb[seam_y : seam_y + 2, u1:u2] = (35, 22, 10)
    # Silver lock latch in upper center
    lc = (u1 + u2) // 2
    rgb[max(v1, seam_y - 3) : min(v2, seam_y + 5), max(u1, lc - 3) : min(u2, lc + 3)] = (205, 205, 210)
    # Accurate depth on the chest front face
    depth[v1:v2, u1:u2] = float(face_x)
    return rgb, depth, intr, T_odom_cam


def test_registry_and_params_validation():
    cfg = params.from_config({}, ROOT)
    assert cfg.webui_port == 8094
    assert cfg.blocks_config_file.is_file()
    reg = BlockRegistry.load_yaml(cfg.blocks_config_file)
    chest = reg.get("chest")
    assert chest is not None
    assert "minecraft:chest" in chest.mc_blocks
    assert reg.get("furnace") is not None
    assert reg.by_onnx_id(0) == chest
    with pytest.raises(params.ConfigError):
        params.from_config({"bogus_key": 1}, ROOT)


def test_classic_chest_detector_3d_backprojection_and_wood_wall_rejection():
    reg = BlockRegistry.load_yaml(ROOT / "config" / "blocks.yaml")
    pipe = DetectorPipeline(reg, min_confidence=0.45)

    true_x, true_y, true_z = 4.5, -0.5, 0.44
    rgb, depth, intr, T_odom_cam = _make_synthetic_chest_scene(
        chest_world_x=true_x,
        chest_world_y=true_y,
        chest_world_z=true_z,
        include_wood_wall=True,
    )

    dets = pipe.detect(rgb, depth)
    assert len(dets) >= 1
    obs_list = []
    for d in dets:
        spec = reg.get(d.label)
        assert spec is not None
        o = backproject_detection_3d(d, depth, intr, T_odom_cam, spec, max_range_m=10.0)
        if o is not None:
            obs_list.append(o)

    # Wood wall distractor must be rejected; ONLY the true chest survives 3D physical size filtering!
    assert len(obs_list) == 1
    obs = obs_list[0]
    assert obs.label == "chest"
    # Verify inward ray push places estimated 3D center inside exact Minecraft BlockPos (4, 0, 0) (mc_z = floor(-(-0.5)) = 0)!
    assert math.floor(obs.odom_x) == math.floor(true_x)
    assert math.floor(obs.odom_z) == math.floor(true_z)
    assert math.floor(-obs.odom_y) == math.floor(-true_y)
    assert abs(obs.odom_x - true_x) < 0.12
    assert abs(obs.odom_y - true_y) < 0.12

    # Multi-frame tracker confirms after 2 frames
    tracker = VisualObjectTracker(min_hits=2)
    tracker.update(obs_list, now_wall=100.0)
    assert len(tracker.confirmed()) == 0
    tracker.update(obs_list, now_wall=100.3)
    confirmed = tracker.confirmed(label="chest", robot_xy=(0.0, 0.0), robot_z=0.0)
    assert len(confirmed) == 1
    assert confirmed[0].visual_id == "vis_chest_1"
    assert confirmed[0].hits == 2


def test_extensible_custom_block_plugin_registration():
    """Verify registering a new block spec + classic detector plugin works seamlessly."""
    reg = BlockRegistry()
    reg.register(BlockSpec(label="emerald_block", display_name="绿宝石块", mc_blocks=("minecraft:emerald_block",), backend="classic"))

    @ClassicBlockDetector.register("emerald_block")
    def _detect_emerald(rgb, depth):
        return []

    assert "emerald_block" in ClassicBlockDetector.supported_labels()
    pipe = DetectorPipeline(reg)
    assert "emerald_block" in pipe.status()["registered_labels"]


def test_local_costmap_approach_pose_avoids_walls_and_aim_ray_probes():
    # 16m x 16m costmap (0.2m res = 80x80), origin (-8, -8), chest against wall at x=+2.5, y=0.0
    data = np.zeros((80, 80), dtype=np.int8)
    # Wall behind chest: x >= 2.8 (col >= 54) is lethal 100
    data[:, 54:] = 100
    grid = LocalCostGrid(origin_x=-8.0, origin_y=-8.0, resolution=0.2, width=80, height=80, data=data)

    pose = select_approach_pose(grid, target_xy=(2.5, 0.0), robot_xy=(-2.0, 0.0), min_r=1.35, max_r=2.35)
    assert pose is not None
    sx, sy, syaw = pose
    # Must stand in front of the chest (x < 1.5), NOT behind the wall (x > 2.8)!
    assert sx < 1.5
    assert 1.3 <= math.hypot(sx - 2.5, sy - 0.0) <= 2.5
    # Must face toward the chest (+X, yaw near 0)
    assert abs(syaw) < math.radians(35.0)

    # Verify aim_with_ray_probes retries slightly deeper along ray if first voxel misses
    calls = []

    def fake_prim(contract, args, timeout_s):
        calls.append((args["odom_x"], args["odom_y"], args["odom_z"]))
        if len(calls) == 1:
            raise RuntimeError("hit air in front")
        return {"aimed": True, "actual_block": "minecraft:chest", "hit_face": "west"}

    pt, res = aim_with_ray_probes(fake_prim, (4.01, -0.5, 0.5), (1.0, 0.0, 0.0), ("minecraft:chest",))
    assert res["aimed"] is True
    assert len(calls) >= 2


def _obs(x: float, y: float, z: float = 0.44, label: str = "chest") -> Observation3D:
    return Observation3D(
        label=label,
        odom_x=x,
        odom_y=y,
        odom_z=z,
        surface_x=x - 0.38,
        surface_y=y,
        surface_z=z,
        ray_dir=(1.0, 0.0, 0.0),
        distance_m=max(1.0, math.hypot(x, y)),
        width_m=0.88,
        height_m=0.88,
        confidence=0.92,
        bbox=(100, 100, 150, 150),
    )


def test_tracker_same_frame_two_chests_vs_cross_frame_dedup_and_visibility_expiry():
    intr = CameraIntrinsics(320, 240, 260.0, 260.0, 160.0, 120.0)
    # Camera at origin looking +X
    T_look_pos_x = SE3Pose(t=np.array([0.0, 0.0, 1.62]), R=np.eye(3)).compose(default_base_to_camera())
    # Camera at origin looking -X (turned 180 deg away)
    R_180 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    T_look_neg_x = SE3Pose(t=np.array([0.0, 0.0, 1.62]), R=R_180).compose(default_base_to_camera())

    clear_depth = np.full((240, 320), 8.0, dtype=np.float32)
    wall_depth = np.full((240, 320), 1.8, dtype=np.float32)  # solid wall at 1.8m blocking view of 4.5m chest

    # A. Two genuinely separate chests visible in the SAME frame (only 1.05m apart) MUST stay 2 tracks!
    tracker = VisualObjectTracker(min_hits=2, stale_ttl_s=60.0, miss_delete_frames=4)
    tracker.update([_obs(4.5, -0.55), _obs(4.5, 0.50)], depth=clear_depth, intr=intr, T_odom_cam=T_look_pos_x, now_wall=10.0)
    tracker.update([_obs(4.52, -0.53), _obs(4.48, 0.52)], depth=clear_depth, intr=intr, T_odom_cam=T_look_pos_x, now_wall=10.3)
    assert len(tracker.confirmed()) == 2

    # B. Turning away & rescanning the SAME single chest from another angle (shifted by 1.15m) MUST merge into 1!
    t2 = VisualObjectTracker(
        min_hits=2,
        stale_ttl_s=60.0,
        miss_delete_frames=4,
        trust_hold_s=10.0,
        trust_conf_threshold=0.80,
        miss_conf_decay=0.92,
    )
    t2.update([_obs(4.5, 0.0)], depth=clear_depth, intr=intr, T_odom_cam=T_look_pos_x, now_wall=20.0)
    t2.update([_obs(4.55, 0.05)], depth=clear_depth, intr=intr, T_odom_cam=T_look_pos_x, now_wall=20.3)
    assert len(t2.confirmed()) == 1
    assert t2.confirmed()[0].low_conf_since_wall is None
    # Turn 180 deg away: chest leaves FOV -> stays remembered (NOT deleted, confidence untouched!)
    for k in range(10):
        t2.update([], depth=clear_depth, intr=intr, T_odom_cam=T_look_neg_x, now_wall=21.0 + k * 0.3)
    assert len(t2.confirmed()) == 1
    assert t2.confirmed()[0].in_fov is False
    assert t2.confirmed()[0].low_conf_since_wall is None
    # Turn back and re-scan same chest with 1.15m angle/depth shift -> merged into the SAME vis_chest_1!
    t2.update([_obs(5.55, 0.45)], depth=clear_depth, intr=intr, T_odom_cam=T_look_pos_x, now_wall=25.0)
    assert len(t2.confirmed()) == 1
    assert t2.confirmed()[0].visual_id == "vis_chest_1"

    # C. Occluded by a wall at 1.8m while looking toward chest at ~4.8m -> MUST NOT be deleted!
    for k in range(10):
        t2.update([], depth=wall_depth, intr=intr, T_odom_cam=T_look_pos_x, now_wall=26.0 + k * 0.3)
    assert len(t2.confirmed()) == 1
    assert t2.confirmed()[0].low_conf_since_wall is None

    # D. Unobstructed clear view: while confidence >= 0.80, 10s timer does NOT start!
    t2.update([], depth=clear_depth, intr=intr, T_odom_cam=T_look_pos_x, now_wall=30.0)
    assert t2.confirmed()[0].confidence >= 0.80
    assert t2.confirmed()[0].low_conf_since_wall is None
    # Next miss drops confidence below 0.80 (0.92 * 0.92 * 0.92 = 0.779) -> starts 10s trust countdown at t=30.3
    t2.update([], depth=clear_depth, intr=intr, T_odom_cam=T_look_pos_x, now_wall=30.3)
    tr = t2.confirmed()[0]
    assert tr.confidence < 0.80
    assert tr.low_conf_since_wall == pytest.approx(30.3)
    # Within the 10s trust window (e.g. t=39.0, 8.7s elapsed), chest MUST still be trusted & kept!
    for k in range(5):
        t2.update([], depth=clear_depth, intr=intr, T_odom_cam=T_look_pos_x, now_wall=32.0 + k * 1.5)
    assert len(t2.confirmed()) == 1
    # Once >= 10s elapsed since confidence dropped < 0.80 (t=40.4 >= 30.3 + 10.0) -> PRUNED!
    t2.update([], depth=clear_depth, intr=intr, T_odom_cam=T_look_pos_x, now_wall=40.4)
    assert len(t2.confirmed()) == 0

    # E. Out-of-sight TTL expiration
    t3 = VisualObjectTracker(min_hits=2, stale_ttl_s=30.0, miss_delete_frames=4)
    t3.update([_obs(4.5, 0.0)], depth=clear_depth, intr=intr, T_odom_cam=T_look_pos_x, now_wall=100.0)
    t3.update([_obs(4.5, 0.0)], depth=clear_depth, intr=intr, T_odom_cam=T_look_pos_x, now_wall=100.3)
    assert len(t3.confirmed()) == 1
    t3.update([], depth=clear_depth, intr=intr, T_odom_cam=T_look_neg_x, now_wall=135.0)
    assert len(t3.confirmed()) == 0


def test_classic_door_detector_finds_recessed_closed_door_regardless_of_color():
    """Verify pure RGB-D recessed doorway detector detects closed doors across any wood/iron color."""
    from visual_object_rbnx.detectors import detect_minecraft_door

    h, w = 240, 320
    # Plain gray concrete wall at depth 2.5 m with a recessed dark-oak/iron door (width 46px, height 110px) at depth 2.95 m
    rgb = np.full((h, w, 3), (170, 170, 175), dtype=np.uint8)
    depth = np.full((h, w), 2.50, dtype=np.float32)

    dx1, dx2, dy1, dy2 = 138, 184, 58, 168
    depth[dy1:dy2, dx1:dx2] = 2.95
    # Draw crisp vertical wood/iron door panels + handle texture inside the recessed doorway
    rgb[dy1:dy2, dx1:dx2] = (75, 58, 44)
    rgb[dy1 + 10 : dy1 + 48, dx1 + 8 : dx2 - 8] = (120, 96, 70)
    rgb[dy1 + 60 : dy2 - 12, dx1 + 8 : dx2 - 8] = (95, 74, 54)
    rgb[dy1 + 52 : dy1 + 58, dx2 - 12 : dx2 - 6] = (210, 210, 210)

    dets = detect_minecraft_door(rgb, depth)
    assert len(dets) >= 1, "recessed closed door should be detected via RGB-D"
    top = dets[0]
    assert top.label == "door"
    cx, cy = top.center_uv
    assert dx1 <= cx <= dx2 and dy1 <= cy <= dy2

    # Flat wall with NO depth recess must NOT trigger false door detections
    flat_depth = np.full((h, w), 2.50, dtype=np.float32)
    assert detect_minecraft_door(rgb, flat_depth) == []


def test_visual_object_toggle_switch_config_and_filter():
    from visual_object_rbnx import params as vo_params

    cfg = vo_params.from_config(
        {"detection_enabled": False, "enabled_labels": ["door"]},
        ROOT,
    )
    assert cfg.detection_enabled is False
    assert cfg.enabled_labels == ("door",)


def test_no_game_state_references_in_src():
    forbidden = (
        "ground" + "_truth",
        "perception" + "/objects",
        "get" + "_state",
        "SCENE" + "_",
        "robonix/system/scene",
    )
    offenders = []
    for path in sorted(SRC.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pat in forbidden:
            if re.search(pat, text, flags=re.IGNORECASE):
                offenders.append(f"{path.name}: {pat}")
    assert offenders == []
