# SPDX-License-Identifier: MulanPSL-2.0
"""Atlas & ROS2 runtime for service-visual-object-rbnx.

Pure-sensor RGB-D block detection, 360-degree room scanning, local costmap approach,
and container interaction (inspect / withdraw / deposit) decoupled from legacy Scene.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import visual_object_mcp
from robonix_api import Err, Ok, Service

from visual_object_rbnx import params
from visual_object_rbnx.approach import (
    LocalCostGrid,
    aim_with_ray_probes,
    default_nav_rpc,
    default_primitive_mcp,
    select_approach_pose,
)
from visual_object_rbnx.detectors import Detection2D, DetectorPipeline
from visual_object_rbnx.projection import (
    CameraIntrinsics,
    Observation3D,
    SE3Pose,
    TrackedBlock,
    VisualObjectTracker,
    backproject_detection_3d,
    decode_depth_image,
    default_base_to_camera,
    quat_to_rot,
)
from visual_object_rbnx.registry import BlockRegistry
from visual_object_rbnx.webui import VisualObjectWebUI

logging.basicConfig(
    level=os.environ.get("VISUAL_OBJECT_LOG_LEVEL", "INFO").upper(),
    format="[visual_object] %(levelname)s %(message)s",
)
log = logging.getLogger("visual_object_rbnx.service")

CAP_ID = os.environ.get("ROBONIX_CAPABILITY_ID", "visual_object")
NAMESPACE = "robonix/service/visual_object"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

visual_object_service = Service(id=CAP_ID, namespace=NAMESPACE)


def _quat_yaw(q: Any) -> float:
    x = float(getattr(q, "x", 0.0))
    y = float(getattr(q, "y", 0.0))
    z = float(getattr(q, "z", 0.0))
    w = float(getattr(q, "w", 1.0))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class VisualObjectRuntime:
    def __init__(
        self,
        cfg: params.VisualObjectConfig,
        *,
        nav_rpc_fn: Callable[[str, dict[str, Any], float], dict[str, Any]] | None = None,
        primitive_mcp_fn: Callable[[str, dict[str, Any], float], dict[str, Any]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.lock = threading.RLock()
        self.action_lock = threading.Lock()
        self.registry = BlockRegistry.load_yaml(cfg.blocks_config_file)
        self.pipeline = DetectorPipeline(
            self.registry,
            onnx_model_path=cfg.detector_model_path,
            min_confidence=cfg.min_confidence,
        )
        self.tracker = VisualObjectTracker(min_hits=cfg.min_track_hits)
        self._nav_rpc = nav_rpc_fn or default_nav_rpc
        self._prim_mcp = primitive_mcp_fn or default_primitive_mcp

        self._intrinsics: CameraIntrinsics | None = None
        self._T_base_cam: SE3Pose = default_base_to_camera()
        self._T_odom_base: SE3Pose | None = None
        self._odom_xyz_yaw: tuple[float, float, float, float] | None = None
        self._cost_grid: LocalCostGrid | None = None
        self._latest_depth: tuple[np.ndarray, float] | None = None
        self._latest_rgb_preview: bytes | None = None
        self._last_detect_t: float = 0.0
        self._frames_processed: int = 0
        self._last_2d_count: int = 0
        self._last_3d_count: int = 0
        self._busy: bool = False
        self._busy_detail: str = ""
        self._last_inspect: dict[str, Any] | None = None

        self._stop = threading.Event()
        self._ros_node: Any = None
        self._executor: Any = None
        self._ros_thread: threading.Thread | None = None
        self._cmd_vel_pub: Any = None
        self._webui: VisualObjectWebUI | None = None

    def start(self) -> None:
        self._stop.clear()
        self._start_ros()
        if self.cfg.webui_enabled and self.cfg.webui_port > 0:
            self._webui = VisualObjectWebUI(
                self.cfg.webui_host,
                self.cfg.webui_port,
                self._webui_handlers(),
                self._get_preview_jpeg,
            )
            try:
                self._webui.start()
            except OSError as exc:
                log.warning("failed to bind visual_object debug WebUI: %s", exc)
                self._webui = None

    def stop(self) -> None:
        self._stop.set()
        if self._webui is not None:
            self._webui.stop()
            self._webui = None
        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=1.0)
            except Exception:  # noqa: BLE001
                pass
        if self._ros_thread is not None:
            self._ros_thread.join(timeout=2.0)
        if self._ros_node is not None:
            try:
                self._ros_node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass

    def _start_ros(self) -> None:
        import rclpy
        from geometry_msgs.msg import TransformStamped, Twist
        from nav_msgs.msg import OccupancyGrid, Odometry
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        if not rclpy.ok():
            rclpy.init(args=None)
        node = Node(
            "visual_object",
            parameter_overrides=[Parameter("use_sim_time", value=self.cfg.use_sim_time)],
        )
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        node.create_subscription(Odometry, self.cfg.odom_topic, self._on_odom, qos_profile_sensor_data)
        node.create_subscription(OccupancyGrid, self.cfg.local_cost_grid_topic, self._on_cost_grid, latched)
        node.create_subscription(CameraInfo, self.cfg.intrinsics_topic, self._on_camera_info, qos_profile_sensor_data)
        node.create_subscription(TransformStamped, self.cfg.extrinsics_topic, self._on_extrinsics, qos_profile_sensor_data)
        node.create_subscription(Image, self.cfg.depth_topic, self._on_depth, qos_profile_sensor_data)
        node.create_subscription(Image, self.cfg.rgb_topic, self._on_rgb, qos_profile_sensor_data)
        self._cmd_vel_pub = node.create_publisher(Twist, self.cfg.cmd_vel_topic, 10)

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        self._ros_node = node
        self._executor = executor
        self._ros_thread = threading.Thread(target=self._spin_ros, name="visual-object-ros", daemon=True)
        self._ros_thread.start()

    def _spin_ros(self) -> None:
        while not self._stop.is_set() and self._executor is not None:
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception:  # noqa: BLE001
                if self._stop.is_set():
                    break
                log.exception("visual_object ROS2 spin_once error (continuing)")
                time.sleep(0.1)

    # ── pure-sensor callbacks ────────────────────────────────────────────────
    def _on_odom(self, msg: Any) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # /minecraft/chassis/odom publishes minecraft_view_base at eye height (mc_y + 1.62);
        # /minecraft/camera/extrinsics publishes minecraft_view_base -> minecraft_camera_optical_frame (t=0, R(pitch)).
        t = np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)
        R = quat_to_rot(float(q.x), float(q.y), float(q.z), float(q.w))
        yaw = _quat_yaw(q)
        with self.lock:
            self._T_odom_base = SE3Pose(t=t, R=R)
            self._odom_xyz_yaw = (float(p.x), float(p.y), float(p.z), yaw)

    def _on_cost_grid(self, msg: Any) -> None:
        info = msg.info
        w, h = int(info.width), int(info.height)
        if w <= 0 or h <= 0 or len(msg.data) != w * h:
            return
        arr = np.asarray(msg.data, dtype=np.int8).reshape((h, w))
        grid = LocalCostGrid(
            origin_x=float(info.origin.position.x),
            origin_y=float(info.origin.position.y),
            resolution=float(info.resolution),
            width=w,
            height=h,
            data=arr,
        )
        with self.lock:
            self._cost_grid = grid

    def _on_camera_info(self, msg: Any) -> None:
        k = msg.k
        w, h = int(msg.width), int(msg.height)
        if k is None or len(k) < 9 or w <= 0 or h <= 0:
            return
        fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
        if fx > 0 and fy > 0:
            with self.lock:
                self._intrinsics = CameraIntrinsics(w, h, fx, fy, cx, cy)

    def _on_extrinsics(self, msg: Any) -> None:
        tr = msg.transform.translation
        rot = msg.transform.rotation
        t = np.array([float(tr.x), float(tr.y), float(tr.z)], dtype=np.float64)
        R = quat_to_rot(float(rot.x), float(rot.y), float(rot.z), float(rot.w))
        with self.lock:
            self._T_base_cam = SE3Pose(t=t, R=R)

    def _on_depth(self, msg: Any) -> None:
        w, h = int(msg.width), int(msg.height)
        if w <= 0 or h <= 0:
            return
        try:
            arr = decode_depth_image(msg.data, str(msg.encoding or ""), h, w, int(msg.step))
        except Exception:  # noqa: BLE001
            return
        with self.lock:
            self._latest_depth = (arr, time.monotonic())

    def _on_rgb(self, msg: Any) -> None:
        now = time.monotonic()
        if now - self._last_detect_t < (1.0 / max(0.5, self.cfg.detect_rate_hz)):
            return
        self._last_detect_t = now
        w, h = int(msg.width), int(msg.height)
        if w <= 0 or h <= 0:
            return
        enc = str(msg.encoding or "").lower()
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if enc in ("rgb8", "bgr8") and buf.size >= w * h * 3:
            rgb = buf[: w * h * 3].reshape((h, w, 3))
            if enc == "bgr8":
                rgb = rgb[:, :, ::-1]
        elif enc in ("rgba8", "bgra8") and buf.size >= w * h * 4:
            rgb = buf[: w * h * 4].reshape((h, w, 4))[:, :, :3]
            if enc == "bgra8":
                rgb = rgb[:, :, ::-1]
        else:
            return
        self.process_rgbd_frame(rgb)

    def process_rgbd_frame(
        self,
        rgb: np.ndarray,
        *,
        depth_override: np.ndarray | None = None,
    ) -> list[Observation3D]:
        """Run 2D detection + Depth 3D backprojection + multi-frame tracking on one RGB frame."""
        with self.lock:
            intr = self._intrinsics
            T_odom_base = self._T_odom_base
            T_base_cam = self._T_base_cam
            depth = depth_override
            if depth is None and self._latest_depth is not None:
                d_arr, d_t = self._latest_depth
                if (time.monotonic() - d_t) <= 1.5:
                    depth = d_arr

        dets_2d = self.pipeline.detect(rgb, depth)
        obs_3d: list[Observation3D] = []
        if depth is not None and intr is not None and T_odom_base is not None:
            T_odom_cam = T_odom_base.compose(T_base_cam)
            for det in dets_2d:
                spec = self.registry.get(det.label)
                if spec is None:
                    continue
                obs = backproject_detection_3d(
                    det,
                    depth,
                    intr,
                    T_odom_cam,
                    spec,
                    max_range_m=self.cfg.max_detect_range_m,
                )
                if obs is not None:
                    obs_3d.append(obs)
            self.tracker.update(
                obs_3d,
                dets_2d=dets_2d,
                depth=depth,
                intr=intr,
                T_odom_cam=T_odom_cam,
                max_verify_range_m=min(7.5, max(3.0, self.cfg.max_detect_range_m - 1.0)),
            )

        # Render annotated JPEG preview for the 8094 debug UI
        preview = self._render_preview(rgb, dets_2d, obs_3d)
        with self.lock:
            self._frames_processed += 1
            self._last_2d_count = len(dets_2d)
            self._last_3d_count = len(obs_3d)
            if preview is not None:
                self._latest_rgb_preview = preview
        return obs_3d

    @staticmethod
    def _render_preview(
        rgb: np.ndarray,
        dets_2d: list[Detection2D],
        obs_3d: list[Observation3D],
    ) -> bytes | None:
        try:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            obs_by_bbox = {o.bbox: o for o in obs_3d}
            for d in dets_2d:
                box = (d.x1, d.y1, d.x2, d.y2)
                o = obs_by_bbox.get(box)
                if o is not None:
                    color = (60, 220, 90)  # Green: passed 3D physical size check
                    txt = f"{o.label} {int(o.confidence*100)}% {o.distance_m:.1f}m ({o.width_m:.2f}x{o.height_m:.2f}m)"
                else:
                    color = (40, 200, 245)  # Yellow: 2D candidate only
                    txt = f"{d.label}? {int(d.confidence*100)}%"
                cv2.rectangle(bgr, (d.x1, d.y1), (d.x2, d.y2), color, 2)
                cv2.putText(
                    bgr,
                    txt,
                    (d.x1, max(16, d.y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return enc.tobytes() if ok else None
        except Exception:  # noqa: BLE001
            return None

    def _get_preview_jpeg(self) -> bytes | None:
        with self.lock:
            return self._latest_rgb_preview

    # ── high-level actions ───────────────────────────────────────────────────
    def _set_busy(self, busy: bool, detail: str = "") -> None:
        with self.lock:
            self._busy = busy
            self._busy_detail = detail

    def _current_robot_xy_z(self) -> tuple[tuple[float, float] | None, float | None]:
        with self.lock:
            if self._odom_xyz_yaw is None:
                return None, None
            return (self._odom_xyz_yaw[0], self._odom_xyz_yaw[1]), self._odom_xyz_yaw[2]

    def scan_blocks(
        self,
        *,
        label: str = "chest",
        sweep: bool = True,
        max_distance_m: float = 10.0,
        sweep_duration_s: float = 14.5,
    ) -> dict[str, Any]:
        key = str(label or "chest").strip().lower()
        if key and self.registry.get(key) is None:
            return {
                "ok": False,
                "count": 0,
                "objects": [],
                "detail": f"未知方块类别 {key!r}，已注册类别: {self.registry.labels()}",
            }
        with self.action_lock:
            self._set_busy(True, f"正在扫描 {key} (sweep={sweep})...")
            try:
                if sweep and self._cmd_vel_pub is not None:
                    from geometry_msgs.msg import Twist

                    deadline = time.monotonic() + max(3.0, min(25.0, float(sweep_duration_s)))
                    try:
                        while not self._stop.is_set() and time.monotonic() < deadline:
                            tw = Twist()
                            tw.angular.z = 0.46
                            self._cmd_vel_pub.publish(tw)
                            time.sleep(0.15)
                    finally:
                        self._cmd_vel_pub.publish(Twist())
                        time.sleep(0.35)
                else:
                    # Wait briefly for a couple of frames to settle
                    time.sleep(0.6)

                rxy, rz = self._current_robot_xy_z()
                found = self.tracker.confirmed(
                    label=key,
                    robot_xy=rxy,
                    robot_z=rz,
                    max_distance_m=max_distance_m,
                )
                items = [t.to_dict(robot_xy=rxy) for t in found]
                detail = (
                    f"扫描完成，发现 {len(items)} 个 {key}"
                    if items
                    else f"未在 {max_distance_m:.1f}m 内发现已确认的 {key}（请确认视野内有箱子或靠近房间内部）"
                )
                return {"ok": True, "count": len(items), "objects": items, "detail": detail}
            finally:
                self._set_busy(False, "")

    def _resolve_target(self, visual_id: str, label: str = "chest", *, auto_scan: bool = True) -> TrackedBlock | None:
        vid = str(visual_id or "").strip()
        if vid:
            return self.tracker.get(vid)
        rxy, rz = self._current_robot_xy_z()
        cand = self.tracker.confirmed(label=label, robot_xy=rxy, robot_z=rz, max_distance_m=self.cfg.max_detect_range_m)
        if cand:
            return cand[0]
        if auto_scan:
            self.scan_blocks(label=label, sweep=True, max_distance_m=self.cfg.max_detect_range_m)
            rxy, rz = self._current_robot_xy_z()
            cand = self.tracker.confirmed(label=label, robot_xy=rxy, robot_z=rz, max_distance_m=self.cfg.max_detect_range_m)
            if cand:
                return cand[0]
        return None

    def approach_block(self, visual_id: str = "", label: str = "chest") -> dict[str, Any]:
        target = self._resolve_target(visual_id, label=label, auto_scan=True)
        if target is None:
            return {
                "ok": False,
                "visual_id": visual_id,
                "odom_x": 0.0,
                "odom_y": 0.0,
                "odom_z": 0.0,
                "distance_m": 0.0,
                "detail": f"未找到可见的 {label} 目标（请先扫描）",
            }
        spec = self.registry.get(target.label)
        min_r = spec.approach_min_m if spec else 1.35
        max_r = spec.approach_max_m if spec else 2.35

        with self.action_lock:
            self._set_busy(True, f"正在走近 {target.visual_id}...")
            self.tracker.pin(target.visual_id)
            try:
                rxy, _ = self._current_robot_xy_z()
                if rxy is None:
                    return {"ok": False, "visual_id": target.visual_id, "odom_x": target.odom_x, "odom_y": target.odom_y, "odom_z": target.odom_z, "distance_m": 0.0, "detail": "无机器人里程计位姿"}
                cur_d = math.hypot(target.odom_x - rxy[0], target.odom_y - rxy[1])
                if cur_d > max_r + 0.65:
                    with self.lock:
                        grid = self._cost_grid
                    pose = select_approach_pose(grid, (target.odom_x, target.odom_y), rxy, min_r=min_r, max_r=max_r)
                    if pose is None:
                        return {
                            "ok": False,
                            "visual_id": target.visual_id,
                            "odom_x": target.odom_x,
                            "odom_y": target.odom_y,
                            "odom_z": target.odom_z,
                            "distance_m": cur_d,
                            "detail": "局部代价地图中未找到目标附近的可达站位点",
                        }
                    sx, sy, syaw = pose
                    half = syaw / 2.0
                    resp = self._nav_rpc(
                        "navigate",
                        {
                            "goal": {
                                "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "minecraft_odom"},
                                "pose": {
                                    "position": {"x": float(sx), "y": float(sy), "z": 0.0},
                                    "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)},
                                },
                            }
                        },
                        10.0,
                    )
                    rid = str(resp.get("run_id") or "")
                    if not (resp.get("accepted") and rid):
                        return {
                            "ok": False,
                            "visual_id": target.visual_id,
                            "odom_x": target.odom_x,
                            "odom_y": target.odom_y,
                            "odom_z": target.odom_z,
                            "distance_m": cur_d,
                            "detail": f"局部导航拒绝目标: {resp.get('detail', '')}",
                        }
                    deadline = time.monotonic() + 35.0
                    try:
                        while not self._stop.is_set() and time.monotonic() < deadline:
                            st = self._nav_rpc("status", {"run_id": rid}, 5.0)
                            lstate = str(st.get("state") or "RUNNING").upper()
                            if lstate == "SUCCEEDED":
                                break
                            if lstate in ("FAILED", "CANCELED", "CANCELLED"):
                                return {
                                    "ok": False,
                                    "visual_id": target.visual_id,
                                    "odom_x": target.odom_x,
                                    "odom_y": target.odom_y,
                                    "odom_z": target.odom_z,
                                    "distance_m": cur_d,
                                    "detail": f"走近目标失败: {st.get('detail', lstate)}",
                                }
                            time.sleep(0.25)
                    finally:
                        self._nav_rpc("cancel", {"run_id": rid}, 5.0)

                # Brief close-range pause so RGB-D refines 3D center to < 0.05m
                time.sleep(0.55)
                refreshed = self.tracker.get(target.visual_id) or target
                rxy2, _ = self._current_robot_xy_z()
                final_d = (
                    math.hypot(refreshed.odom_x - rxy2[0], refreshed.odom_y - rxy2[1])
                    if rxy2 is not None
                    else refreshed.last_distance_m
                )
                return {
                    "ok": True,
                    "visual_id": refreshed.visual_id,
                    "odom_x": round(refreshed.odom_x, 3),
                    "odom_y": round(refreshed.odom_y, 3),
                    "odom_z": round(refreshed.odom_z, 3),
                    "distance_m": round(final_d, 2),
                    "detail": f"已到达 {refreshed.visual_id} 前 (距离 {final_d:.2f}m)",
                }
            finally:
                self.tracker.unpin(target.visual_id)
                self._set_busy(False, "")

    def _prepare_chest_aim(self, visual_id: str) -> tuple[TrackedBlock, tuple[float, float, float]]:
        app = self.approach_block(visual_id=visual_id, label="chest")
        if not app.get("ok"):
            raise RuntimeError(str(app.get("detail") or "走近箱子失败"))
        vid = str(app["visual_id"])
        self.tracker.pin(vid)
        try:
            target = self.tracker.get(vid)
            if target is None:
                raise RuntimeError("箱子追踪目标丢失")
            spec = self.registry.get("chest")
            expected = spec.mc_blocks if spec else ("minecraft:chest", "minecraft:trapped_chest")
            aimed_pt, _ = aim_with_ray_probes(
                self._prim_mcp,
                (target.odom_x, target.odom_y, target.odom_z),
                target.ray_dir,
                expected,
            )
            return target, aimed_pt
        finally:
            self.tracker.unpin(vid)

    def inspect_chest(self, visual_id: str = "") -> dict[str, Any]:
        try:
            target, (px, py, pz) = self._prepare_chest_aim(visual_id)
            with self.action_lock:
                self._set_busy(True, f"正在开箱检查 {target.visual_id}...")
                res = self._prim_mcp(
                    "robonix/primitive/minecraft/inspect_container_at_point",
                    {"odom_x": px, "odom_y": py, "odom_z": pz},
                    25.0,
                )
            slots = list(res.get("slots") or [])
            non_empty = [s for s in slots if int(s.get("count", 0)) > 0 and s.get("item") != "minecraft:air"]
            summary = ", ".join(f"{s['item']}×{s['count']}" for s in non_empty) or "空箱子"
            out = {
                "ok": True,
                "visual_id": target.visual_id,
                "container_type": str(res.get("container_type") or "minecraft:chest"),
                "slot_count": int(res.get("slot_count") or len(slots)),
                "items": slots,
                "detail": f"已检查 {target.visual_id}：{summary}",
            }
            with self.lock:
                self._last_inspect = out
            return out
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "visual_id": visual_id,
                "container_type": "",
                "slot_count": 0,
                "items": [],
                "detail": f"开箱检查失败: {exc}",
            }
        finally:
            self._set_busy(False, "")

    def withdraw_from_chest(self, visual_id: str, item: str, count: int) -> dict[str, Any]:
        try:
            cnt = int(count)
            if cnt <= 0:
                raise ValueError("count must be >= 1")
            target, (px, py, pz) = self._prepare_chest_aim(visual_id)
            with self.action_lock:
                self._set_busy(True, f"正在从 {target.visual_id} 取出 {cnt}×{item}...")
                res = self._prim_mcp(
                    "robonix/primitive/minecraft/withdraw_item_at_point",
                    {"odom_x": px, "odom_y": py, "odom_z": pz, "item": item.strip(), "count": cnt},
                    25.0,
                )
            return {
                "ok": True,
                "visual_id": target.visual_id,
                "transferred_count": int(res.get("transferred_count", cnt)),
                "player_count_before": int(res.get("player_count_before", 0)),
                "player_count_after": int(res.get("player_count_after", 0)),
                "container_count_before": int(res.get("container_count_before", 0)),
                "container_count_after": int(res.get("container_count_after", 0)),
                "detail": (
                    f"已从 {target.visual_id} 取出 {item}×{cnt} "
                    f"(背包 {res.get('player_count_before')}→{res.get('player_count_after')}, "
                    f"箱子 {res.get('container_count_before')}→{res.get('container_count_after')})"
                ),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "visual_id": visual_id,
                "transferred_count": 0,
                "player_count_before": 0,
                "player_count_after": 0,
                "container_count_before": 0,
                "container_count_after": 0,
                "detail": f"从箱子取物失败: {exc}",
            }
        finally:
            self._set_busy(False, "")

    def deposit_to_chest(self, visual_id: str, item: str, count: int) -> dict[str, Any]:
        try:
            cnt = int(count)
            if cnt <= 0:
                raise ValueError("count must be >= 1")
            target, (px, py, pz) = self._prepare_chest_aim(visual_id)
            with self.action_lock:
                self._set_busy(True, f"正在向 {target.visual_id} 放入 {cnt}×{item}...")
                res = self._prim_mcp(
                    "robonix/primitive/minecraft/deposit_item_at_point",
                    {"odom_x": px, "odom_y": py, "odom_z": pz, "item": item.strip(), "count": cnt},
                    25.0,
                )
            return {
                "ok": True,
                "visual_id": target.visual_id,
                "transferred_count": int(res.get("transferred_count", cnt)),
                "player_count_before": int(res.get("player_count_before", 0)),
                "player_count_after": int(res.get("player_count_after", 0)),
                "container_count_before": int(res.get("container_count_before", 0)),
                "container_count_after": int(res.get("container_count_after", 0)),
                "detail": (
                    f"已向 {target.visual_id} 放入 {item}×{cnt} "
                    f"(背包 {res.get('player_count_before')}→{res.get('player_count_after')}, "
                    f"箱子 {res.get('container_count_before')}→{res.get('container_count_after')})"
                ),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "visual_id": visual_id,
                "transferred_count": 0,
                "player_count_before": 0,
                "player_count_after": 0,
                "container_count_before": 0,
                "container_count_after": 0,
                "detail": f"向箱子放物失败: {exc}",
            }
        finally:
            self._set_busy(False, "")

    # ── Web UI handlers ──────────────────────────────────────────────────────
    def _webui_handlers(self) -> dict[str, Any]:
        return {
            "GET:api/state": self._api_state,
            "POST:api/scan": lambda b: self.scan_blocks(
                label=str(b.get("label") or "chest"),
                sweep=bool(b.get("sweep", True)),
                max_distance_m=float(b.get("max_distance_m", self.cfg.max_detect_range_m)),
            ),
            "POST:api/approach": lambda b: self.approach_block(
                visual_id=str(b.get("visual_id") or ""),
                label=str(b.get("label") or "chest"),
            ),
            "POST:api/inspect": lambda b: self.inspect_chest(visual_id=str(b.get("visual_id") or "")),
            "POST:api/withdraw": lambda b: self.withdraw_from_chest(
                visual_id=str(b.get("visual_id") or ""),
                item=str(b.get("item") or ""),
                count=int(b.get("count", 1)),
            ),
            "POST:api/deposit": lambda b: self.deposit_to_chest(
                visual_id=str(b.get("visual_id") or ""),
                item=str(b.get("item") or ""),
                count=int(b.get("count", 1)),
            ),
            "POST:api/clear": self._api_clear,
        }

    def _api_state(self, _: dict[str, Any]) -> dict[str, Any]:
        rxy, rz = self._current_robot_xy_z()
        with self.lock:
            odom = (
                {
                    "x": round(self._odom_xyz_yaw[0], 3),
                    "y": round(self._odom_xyz_yaw[1], 3),
                    "z": round(self._odom_xyz_yaw[2], 3),
                    "yaw": round(self._odom_xyz_yaw[3], 4),
                }
                if self._odom_xyz_yaw is not None
                else None
            )
            return {
                "ok": True,
                "odom": odom,
                "detector": self.pipeline.status(),
                "specs": [s.to_dict() for s in self.registry.all_specs()],
                "objects": [t.to_dict(robot_xy=rxy) for t in self.tracker.confirmed(robot_xy=rxy, robot_z=rz)],
                "frames_processed": self._frames_processed,
                "last_2d_count": self._last_2d_count,
                "last_3d_count": self._last_3d_count,
                "busy": self._busy,
                "busy_detail": self._busy_detail,
                "last_inspect": self._last_inspect,
            }

    def _api_clear(self, body: dict[str, Any]) -> dict[str, Any]:
        self.tracker.clear(str(body.get("label") or ""))
        with self.lock:
            self._last_inspect = None
        return {"ok": True, "detail": "已清空当前视觉目标记忆表"}


_runtime_lock = threading.Lock()
_runtime: VisualObjectRuntime | None = None


@visual_object_service.mcp("robonix/skill/visual_object/scan")
def mcp_scan(req: visual_object_mcp.ScanBlocks_Request) -> visual_object_mcp.ScanBlocks_Response:
    if _runtime is None:
        return visual_object_mcp.ScanBlocks_Response(ok=False, count=0, objects_json="[]", detail="visual_object inactive")
    res = _runtime.scan_blocks(
        label=(req.label or "chest").strip(),
        sweep=bool(req.sweep),
        max_distance_m=float(req.max_distance_m) if req.max_distance_m > 0 else _runtime.cfg.max_detect_range_m,
    )
    return visual_object_mcp.ScanBlocks_Response(
        ok=bool(res.get("ok")),
        count=int(res.get("count", 0)),
        objects_json=json.dumps(res.get("objects") or [], ensure_ascii=False, separators=(",", ":")),
        detail=str(res.get("detail") or ""),
    )


@visual_object_service.mcp("robonix/skill/visual_object/approach")
def mcp_approach(req: visual_object_mcp.ApproachBlock_Request) -> visual_object_mcp.ApproachBlock_Response:
    if _runtime is None:
        return visual_object_mcp.ApproachBlock_Response(
            ok=False, visual_id="", odom_x=0.0, odom_y=0.0, odom_z=0.0, distance_m=0.0, detail="visual_object inactive"
        )
    res = _runtime.approach_block(visual_id=(req.visual_id or "").strip(), label=(req.label or "chest").strip())
    return visual_object_mcp.ApproachBlock_Response(
        ok=bool(res.get("ok")),
        visual_id=str(res.get("visual_id") or ""),
        odom_x=float(res.get("odom_x", 0.0)),
        odom_y=float(res.get("odom_y", 0.0)),
        odom_z=float(res.get("odom_z", 0.0)),
        distance_m=float(res.get("distance_m", 0.0)),
        detail=str(res.get("detail") or ""),
    )


@visual_object_service.mcp("robonix/skill/visual_object/inspect_chest")
def mcp_inspect_chest(req: visual_object_mcp.InspectChest_Request) -> visual_object_mcp.InspectChest_Response:
    if _runtime is None:
        return visual_object_mcp.InspectChest_Response(
            ok=False, visual_id="", container_type="", slot_count=0, items_json="[]", detail="visual_object inactive"
        )
    res = _runtime.inspect_chest(visual_id=(req.visual_id or "").strip())
    return visual_object_mcp.InspectChest_Response(
        ok=bool(res.get("ok")),
        visual_id=str(res.get("visual_id") or ""),
        container_type=str(res.get("container_type") or ""),
        slot_count=int(res.get("slot_count", 0)),
        items_json=json.dumps(res.get("items") or [], ensure_ascii=False, separators=(",", ":")),
        detail=str(res.get("detail") or ""),
    )


@visual_object_service.mcp("robonix/skill/visual_object/withdraw_from_chest")
def mcp_withdraw(req: visual_object_mcp.WithdrawFromChest_Request) -> visual_object_mcp.WithdrawFromChest_Response:
    if _runtime is None:
        return visual_object_mcp.WithdrawFromChest_Response(
            ok=False, visual_id="", transferred_count=0, player_count_before=0, player_count_after=0,
            container_count_before=0, container_count_after=0, detail="visual_object inactive",
        )
    res = _runtime.withdraw_from_chest(
        visual_id=(req.visual_id or "").strip(),
        item=(req.item or "").strip(),
        count=int(req.count),
    )
    return visual_object_mcp.WithdrawFromChest_Response(
        ok=bool(res.get("ok")),
        visual_id=str(res.get("visual_id") or ""),
        transferred_count=int(res.get("transferred_count", 0)),
        player_count_before=int(res.get("player_count_before", 0)),
        player_count_after=int(res.get("player_count_after", 0)),
        container_count_before=int(res.get("container_count_before", 0)),
        container_count_after=int(res.get("container_count_after", 0)),
        detail=str(res.get("detail") or ""),
    )


@visual_object_service.mcp("robonix/skill/visual_object/deposit_to_chest")
def mcp_deposit(req: visual_object_mcp.DepositToChest_Request) -> visual_object_mcp.DepositToChest_Response:
    if _runtime is None:
        return visual_object_mcp.DepositToChest_Response(
            ok=False, visual_id="", transferred_count=0, player_count_before=0, player_count_after=0,
            container_count_before=0, container_count_after=0, detail="visual_object inactive",
        )
    res = _runtime.deposit_to_chest(
        visual_id=(req.visual_id or "").strip(),
        item=(req.item or "").strip(),
        count=int(req.count),
    )
    return visual_object_mcp.DepositToChest_Response(
        ok=bool(res.get("ok")),
        visual_id=str(res.get("visual_id") or ""),
        transferred_count=int(res.get("transferred_count", 0)),
        player_count_before=int(res.get("player_count_before", 0)),
        player_count_after=int(res.get("player_count_after", 0)),
        container_count_before=int(res.get("container_count_before", 0)),
        container_count_after=int(res.get("container_count_after", 0)),
        detail=str(res.get("detail") or ""),
    )


@visual_object_service.on_init
def on_init(cfg: dict[str, Any]):
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            return Ok()
        try:
            parsed = params.from_config(cfg or {}, PACKAGE_ROOT)
        except Exception as exc:  # noqa: BLE001
            return Err(f"invalid visual_object config: {exc}")
        rt = VisualObjectRuntime(parsed)
        try:
            rt.start()
        except Exception as exc:  # noqa: BLE001
            log.exception("failed to start visual_object runtime")
            rt.stop()
            return Err(f"failed to start visual_object: {exc}")
        _runtime = rt
        log.info("visual_object active (webui=%s:%d)", parsed.webui_host, parsed.webui_port)
        return Ok()


@visual_object_service.on_shutdown
def on_shutdown():
    global _runtime
    with _runtime_lock:
        rt, _runtime = _runtime, None
    if rt is not None:
        rt.stop()
    return Ok()


def main() -> None:
    visual_object_service.run()
