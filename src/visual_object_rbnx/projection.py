# SPDX-License-Identifier: MulanPSL-2.0
"""Pure-sensor Depth 3D backprojection, physical size filtering, and multi-frame object tracking."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from visual_object_rbnx.detectors import Detection2D
from visual_object_rbnx.registry import BlockRegistry, BlockSpec


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class SE3Pose:
    t: np.ndarray  # shape (3,)
    R: np.ndarray  # shape (3, 3)

    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        return (pts @ self.R.T) + self.t

    def transform_point(self, pt: np.ndarray) -> np.ndarray:
        return self.R @ pt + self.t

    def compose(self, other: "SE3Pose") -> "SE3Pose":
        return SE3Pose(t=self.R @ other.t + self.t, R=self.R @ other.R)


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def default_base_to_camera() -> SE3Pose:
    """Default transform from minecraft_view_base to minecraft_camera_optical_frame."""
    # Optical frame: +X right = -Y_base, +Y down = -Z_base, +Z forward = +X_base
    R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    return SE3Pose(t=np.array([0.0, 0.0, 0.0], dtype=np.float64), R=R)


def decode_depth_image(data: bytes | bytearray | memoryview, encoding: str, height: int, width: int, step: int) -> np.ndarray:
    if encoding in ("32FC1", "32FC1_LE"):
        dtype = np.dtype("<f4")
    elif encoding in ("16UC1", "mono16"):
        dtype = np.dtype("<u2")
    else:
        raise ValueError(f"unsupported depth encoding: {encoding}")
    raw = np.frombuffer(data, dtype=np.uint8)
    expected = height * step
    if raw.size < expected:
        raise ValueError("depth payload shorter than height * step")
    rows = raw[:expected].reshape(height, step)
    vals = rows[:, : width * dtype.itemsize].copy().view(dtype).reshape(height, width).astype(np.float32)
    if dtype == np.dtype("<u2"):
        vals /= 1000.0
    return vals


@dataclass(frozen=True)
class Observation3D:
    label: str
    odom_x: float
    odom_y: float
    odom_z: float
    surface_x: float
    surface_y: float
    surface_z: float
    ray_dir: tuple[float, float, float]  # unit ray from camera to block surface in minecraft_odom
    distance_m: float
    width_m: float
    height_m: float
    confidence: float
    bbox: tuple[int, int, int, int]
    backend: str = "classic"


def backproject_detection_3d(
    det: Detection2D,
    depth: np.ndarray,
    intr: CameraIntrinsics,
    T_odom_cam: SE3Pose,
    spec: BlockSpec,
    *,
    max_range_m: float = 10.0,
) -> Observation3D | None:
    """Backproject a 2D detection to a 3D minecraft_odom center and verify physical block dimensions."""
    dh, dw = depth.shape[:2]
    # Scale bbox if depth resolution differs from RGB resolution
    sx = dw / float(max(1, intr.width))
    sy = dh / float(max(1, intr.height))
    x1 = max(0, min(dw - 1, int(round(det.x1 * sx))))
    y1 = max(0, min(dh - 1, int(round(det.y1 * sy))))
    x2 = max(x1 + 1, min(dw, int(round(det.x2 * sx))))
    y2 = max(y1 + 1, min(dh, int(round(det.y2 * sy))))

    patch = depth[y1:y2, x1:x2]
    if patch.size < 9:
        return None

    # Use core pixels (eroded mask or inner 70% box) to ignore background bleed on bbox edges
    if det.mask is not None and det.mask.shape == patch.shape:
        sel_mask = det.mask.astype(bool)
    else:
        ph, pw = patch.shape
        mx = max(1, int(pw * 0.15))
        my = max(1, int(ph * 0.15))
        sel_mask = np.zeros_like(patch, dtype=bool)
        sel_mask[my : max(my + 1, ph - my), mx : max(mx + 1, pw - mx)] = True

    ys_local, xs_local = np.where(sel_mask & np.isfinite(patch) & (patch >= 0.35) & (patch <= max_range_m))
    if ys_local.size < 6:
        # Fallback to full bbox if core mask had too few valid depths
        ys_local, xs_local = np.where(np.isfinite(patch) & (patch >= 0.35) & (patch <= max_range_m))
        if ys_local.size < 6:
            return None

    zs = patch[ys_local, xs_local].astype(np.float64)
    # Robust foreground depth filter: keep pixels within IQR around lower-median (front face)
    z_med = float(np.median(zs))
    keep = np.abs(zs - z_med) <= max(0.35, 0.12 * z_med)
    if np.count_nonzero(keep) < 5:
        keep = np.ones_like(zs, dtype=bool)

    us = (xs_local[keep].astype(np.float64) + x1) / sx
    vs = (ys_local[keep].astype(np.float64) + y1) / sy
    zf = zs[keep]

    cam_x = (us - intr.cx) * zf / intr.fx
    cam_y = (vs - intr.cy) * zf / intr.fy
    pts_cam = np.stack((cam_x, cam_y, zf), axis=1)
    pts_odom = T_odom_cam.transform_points(pts_cam)

    # Estimate physical 3D extent (horizontal span + vertical span) using 5..95 percentiles
    px_lo, px_hi = np.percentile(pts_odom[:, 0], [5, 95])
    py_lo, py_hi = np.percentile(pts_odom[:, 1], [5, 95])
    pz_lo, pz_hi = np.percentile(pts_odom[:, 2], [5, 95])
    horiz_span = math.hypot(float(px_hi - px_lo), float(py_hi - py_lo))
    vert_span = float(pz_hi - pz_lo)

    # Account for slight inner-crop shrinkage (core 70% box -> ~0.75x true span)
    est_w = horiz_span / 0.80 if det.mask is None else horiz_span
    est_h = vert_span / 0.80 if det.mask is None else vert_span

    if est_w < spec.min_width_m or est_w > spec.max_width_m:
        return None
    if est_h < spec.min_height_m or est_h > spec.max_height_m:
        return None

    surf = np.median(pts_odom, axis=0)
    cam_origin = T_odom_cam.t
    vec = surf - cam_origin
    dist = float(np.linalg.norm(vec))
    if dist < 0.35 or dist > max_range_m:
        return None
    ray_u = vec / dist

    # Push inward along camera ray into the physical center of the block so floor() never hits air in front!
    center = surf + ray_u * float(spec.inward_push_m)

    return Observation3D(
        label=det.label,
        odom_x=float(center[0]),
        odom_y=float(center[1]),
        odom_z=float(center[2]),
        surface_x=float(surf[0]),
        surface_y=float(surf[1]),
        surface_z=float(surf[2]),
        ray_dir=(float(ray_u[0]), float(ray_u[1]), float(ray_u[2])),
        distance_m=dist,
        width_m=est_w,
        height_m=est_h,
        confidence=det.confidence,
        bbox=(det.x1, det.y1, det.x2, det.y2),
        backend=det.backend,
    )


@dataclass
class TrackedBlock:
    visual_id: str
    label: str
    odom_x: float
    odom_y: float
    odom_z: float
    surface_x: float
    surface_y: float
    surface_z: float
    ray_dir: tuple[float, float, float]
    confidence: float
    width_m: float
    height_m: float
    hits: int
    first_seen_wall: float
    last_seen_wall: float
    backend: str = "classic"
    last_distance_m: float = 0.0
    in_fov: bool = True
    missed_in_view: int = 0
    low_conf_since_wall: float | None = None

    def to_dict(
        self,
        robot_xy: tuple[float, float] | None = None,
        *,
        trust_hold_s: float = 10.0,
    ) -> dict[str, Any]:
        dist = (
            math.hypot(self.odom_x - robot_xy[0], self.odom_y - robot_xy[1])
            if robot_xy is not None
            else self.last_distance_m
        )
        now = time.time()
        age_s = max(0.0, now - self.last_seen_wall)
        status = "visible" if (self.in_fov and age_s <= 4.0) else "remembered"
        trust_remaining_s: float | None = None
        if self.low_conf_since_wall is not None:
            trust_remaining_s = round(max(0.0, trust_hold_s - (now - self.low_conf_since_wall)), 1)
        return {
            "visual_id": self.visual_id,
            "label": self.label,
            "odom_x": round(self.odom_x, 3),
            "odom_y": round(self.odom_y, 3),
            "odom_z": round(self.odom_z, 3),
            "surface_x": round(self.surface_x, 3),
            "surface_y": round(self.surface_y, 3),
            "surface_z": round(self.surface_z, 3),
            "ray_dir": [round(v, 4) for v in self.ray_dir],
            "distance_m": round(dist, 2),
            "width_m": round(self.width_m, 2),
            "height_m": round(self.height_m, 2),
            "confidence": round(self.confidence, 3),
            "hits": self.hits,
            "backend": self.backend,
            "in_fov": self.in_fov,
            "missed_in_view": self.missed_in_view,
            "trust_remaining_s": trust_remaining_s,
            "status": status,
            "age_s": round(age_s, 2),
        }


class VisualObjectTracker:
    """Multi-frame 3D spatial clustering, same-frame 1-to-1 separation, and visibility-based expiry.

    - Within a single frame: 1-to-1 assignment + co-visibility tracking guarantees two distinct
      blocks visible together in one image remain two separate tracks.
    - Across frames (e.g. turning around and re-scanning the same chest from another angle):
      generous cross-frame association & non-co-visible consolidation merges duplicates.
    - Out of sight: blocks stay remembered until `stale_ttl_s` or leaving the rolling map radius.
    - Back in sight & unobstructed (verified via Depth) but missing for `miss_delete_frames`
      consecutive frames: automatically pruned from the rolling map.
    """

    def __init__(
        self,
        *,
        cluster_radius_m: float = 1.45,
        merge_radius_m: float = 1.25,
        max_dz_m: float = 0.90,
        min_hits: int = 2,
        stale_ttl_s: float = 180.0,
        max_keep_distance_m: float = 32.0,
        miss_delete_frames: int = 6,
        trust_hold_s: float = 10.0,
        trust_conf_threshold: float = 0.80,
        miss_conf_decay: float = 0.94,
    ) -> None:
        self.cluster_radius_m = float(cluster_radius_m)
        self.merge_radius_m = float(merge_radius_m)
        self.max_dz_m = float(max_dz_m)
        self.min_hits = max(1, int(min_hits))
        self.stale_ttl_s = float(stale_ttl_s)
        self.max_keep_distance_m = float(max_keep_distance_m)
        self.miss_delete_frames = max(1, int(miss_delete_frames))
        self.trust_hold_s = max(0.0, float(trust_hold_s))
        self.trust_conf_threshold = float(trust_conf_threshold)
        self.miss_conf_decay = float(miss_conf_decay)
        self._lock = threading.RLock()
        self._tracks: dict[str, TrackedBlock] = {}
        self._co_visible: set[tuple[str, str]] = set()
        self._pinned_ids: set[str] = set()
        self._counters: dict[str, int] = {}

    def pin(self, visual_id: str) -> None:
        vid = str(visual_id or "").strip()
        if vid:
            with self._lock:
                self._pinned_ids.add(vid)

    def unpin(self, visual_id: str) -> None:
        vid = str(visual_id or "").strip()
        if vid:
            with self._lock:
                self._pinned_ids.discard(vid)

    @staticmethod
    def _pair_key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def _delete_track_unlocked(self, vid: str) -> None:
        self._tracks.pop(vid, None)
        self._co_visible = {p for p in self._co_visible if vid not in p}

    def clear(self, label: str = "") -> None:
        with self._lock:
            if not label:
                self._tracks.clear()
                self._co_visible.clear()
                return
            key = label.strip().lower()
            drop = [k for k, v in self._tracks.items() if v.label == key]
            for vid in drop:
                self._delete_track_unlocked(vid)

    def update(
        self,
        obs_list: list[Observation3D],
        *,
        dets_2d: list[Detection2D] | None = None,
        depth: np.ndarray | None = None,
        intr: CameraIntrinsics | None = None,
        T_odom_cam: SE3Pose | None = None,
        max_verify_range_m: float = 7.5,
        now_wall: float | None = None,
    ) -> list[TrackedBlock]:
        now = time.time() if now_wall is None else float(now_wall)
        with self._lock:
            # 1. Prune stale out-of-sight tracks and tracks outside the rolling local map
            cam_xy = (float(T_odom_cam.t[0]), float(T_odom_cam.t[1])) if T_odom_cam is not None else None
            expired: list[str] = []
            for vid, tr in self._tracks.items():
                if vid in self._pinned_ids:
                    continue
                if (now - tr.last_seen_wall) > self.stale_ttl_s:
                    expired.append(vid)
                    continue
                if cam_xy is not None and math.hypot(tr.odom_x - cam_xy[0], tr.odom_y - cam_xy[1]) > self.max_keep_distance_m:
                    expired.append(vid)
            for vid in expired:
                self._delete_track_unlocked(vid)

            # 2. One-to-one greedy bipartite matching within the current frame
            #    (two detections in ONE frame can never collapse into the same track!)
            candidates: list[tuple[float, int, str]] = []
            for oi, obs in enumerate(obs_list):
                for vid, tr in self._tracks.items():
                    if tr.label != obs.label:
                        continue
                    dz = abs(tr.odom_z - obs.odom_z)
                    if dz > self.max_dz_m:
                        continue
                    d_xy = math.hypot(tr.odom_x - obs.odom_x, tr.odom_y - obs.odom_y)
                    if d_xy <= self.cluster_radius_m:
                        candidates.append((d_xy, oi, vid))
            candidates.sort(key=lambda item: item[0])

            used_obs: set[int] = set()
            used_tracks: set[str] = set()
            obs_to_track: dict[int, TrackedBlock] = {}
            for _, oi, vid in candidates:
                if oi in used_obs or vid in used_tracks:
                    continue
                used_obs.add(oi)
                used_tracks.add(vid)
                obs = obs_list[oi]
                tr = self._tracks[vid]
                # Closer observations get higher weight (close-range refinement!)
                alpha = 0.55 if obs.distance_m <= 3.0 else 0.30
                tr.odom_x = (1.0 - alpha) * tr.odom_x + alpha * obs.odom_x
                tr.odom_y = (1.0 - alpha) * tr.odom_y + alpha * obs.odom_y
                tr.odom_z = (1.0 - alpha) * tr.odom_z + alpha * obs.odom_z
                tr.surface_x = (1.0 - alpha) * tr.surface_x + alpha * obs.surface_x
                tr.surface_y = (1.0 - alpha) * tr.surface_y + alpha * obs.surface_y
                tr.surface_z = (1.0 - alpha) * tr.surface_z + alpha * obs.surface_z
                tr.ray_dir = obs.ray_dir
                if obs.confidence >= self.trust_conf_threshold:
                    tr.confidence = max(tr.confidence * 0.9, obs.confidence)
                else:
                    tr.confidence = 0.6 * tr.confidence + 0.4 * obs.confidence
                tr.width_m = (1.0 - alpha) * tr.width_m + alpha * obs.width_m
                tr.height_m = (1.0 - alpha) * tr.height_m + alpha * obs.height_m
                tr.hits += 1
                tr.last_seen_wall = now
                tr.last_distance_m = obs.distance_m
                tr.in_fov = True
                tr.missed_in_view = 0
                if tr.confidence >= self.trust_conf_threshold:
                    tr.low_conf_since_wall = None
                elif tr.low_conf_since_wall is None:
                    tr.low_conf_since_wall = now
                obs_to_track[oi] = tr

            for oi, obs in enumerate(obs_list):
                if oi in obs_to_track:
                    continue
                idx = self._counters.get(obs.label, 0) + 1
                self._counters[obs.label] = idx
                vid = f"vis_{obs.label}_{idx}"
                tr = TrackedBlock(
                    visual_id=vid,
                    label=obs.label,
                    odom_x=obs.odom_x,
                    odom_y=obs.odom_y,
                    odom_z=obs.odom_z,
                    surface_x=obs.surface_x,
                    surface_y=obs.surface_y,
                    surface_z=obs.surface_z,
                    ray_dir=obs.ray_dir,
                    confidence=obs.confidence,
                    width_m=obs.width_m,
                    height_m=obs.height_m,
                    hits=1,
                    first_seen_wall=now,
                    last_seen_wall=now,
                    backend=obs.backend,
                    last_distance_m=obs.distance_m,
                    in_fov=True,
                    missed_in_view=0,
                    low_conf_since_wall=(None if obs.confidence >= self.trust_conf_threshold else now),
                )
                self._tracks[vid] = tr
                obs_to_track[oi] = tr

            matched_tracks = [obs_to_track[i] for i in range(len(obs_list))]
            frame_track_ids = {t.visual_id for t in matched_tracks}

            # 3. Record co-visibility for distinct blocks seen simultaneously in this frame
            if len(frame_track_ids) >= 2:
                ids_sorted = sorted(frame_track_ids)
                for i in range(len(ids_sorted)):
                    for j in range(i + 1, len(ids_sorted)):
                        self._co_visible.add((ids_sorted[i], ids_sorted[j]))

            # 4. Cross-frame consolidation: merge duplicate tracks that drifted close
            #    together and were NEVER seen simultaneously in the same frame.
            self._consolidate_duplicates_unlocked(frame_track_ids)

            # 5. Negative-evidence visibility check: if an unmatched track lies cleanly
            #    inside the camera's unobstructed FOV (verified via live Depth), decay
            #    confidence; only once confidence drops below trust_conf_threshold (80%)
            #    does the 10s trust countdown begin. Remove once countdown expires and
            #    the block is still missing in unobstructed view.
            if depth is not None and intr is not None and T_odom_cam is not None:
                self._apply_visibility_negative_evidence_unlocked(
                    frame_track_ids,
                    dets_2d=dets_2d,
                    depth=depth,
                    intr=intr,
                    T_odom_cam=T_odom_cam,
                    max_verify_range_m=max_verify_range_m,
                    now_wall=now,
                )

            return [t for t in matched_tracks if t.visual_id in self._tracks]

    def _consolidate_duplicates_unlocked(self, frame_track_ids: set[str]) -> None:
        ids = sorted(self._tracks.keys())
        merged_away: set[str] = set()
        for i in range(len(ids)):
            a_id = ids[i]
            if a_id in merged_away or a_id not in self._tracks:
                continue
            for j in range(i + 1, len(ids)):
                b_id = ids[j]
                if b_id in merged_away or b_id not in self._tracks:
                    continue
                if self._pair_key(a_id, b_id) in self._co_visible:
                    continue
                if a_id in frame_track_ids and b_id in frame_track_ids:
                    continue
                ta, tb = self._tracks[a_id], self._tracks[b_id]
                if ta.label != tb.label:
                    continue
                if abs(ta.odom_z - tb.odom_z) > self.max_dz_m:
                    continue
                if math.hypot(ta.odom_x - tb.odom_x, ta.odom_y - tb.odom_y) > self.merge_radius_m:
                    continue
                # Keep stronger track (more hits, tie-break earlier visual_id)
                keeper, donor = (ta, tb) if (ta.hits, -ta.first_seen_wall) >= (tb.hits, -tb.first_seen_wall) else (tb, ta)
                w_keep = max(1, keeper.hits)
                w_don = max(1, donor.hits)
                tot = float(w_keep + w_don)
                keeper.odom_x = (keeper.odom_x * w_keep + donor.odom_x * w_don) / tot
                keeper.odom_y = (keeper.odom_y * w_keep + donor.odom_y * w_don) / tot
                keeper.odom_z = (keeper.odom_z * w_keep + donor.odom_z * w_don) / tot
                keeper.surface_x = (keeper.surface_x * w_keep + donor.surface_x * w_don) / tot
                keeper.surface_y = (keeper.surface_y * w_keep + donor.surface_y * w_don) / tot
                keeper.surface_z = (keeper.surface_z * w_keep + donor.surface_z * w_don) / tot
                keeper.hits += donor.hits
                keeper.confidence = max(keeper.confidence, donor.confidence)
                if donor.last_seen_wall > keeper.last_seen_wall:
                    keeper.last_seen_wall = donor.last_seen_wall
                    keeper.ray_dir = donor.ray_dir
                    keeper.last_distance_m = donor.last_distance_m
                keeper.missed_in_view = min(keeper.missed_in_view, donor.missed_in_view)
                # Transfer co-visibility relations from donor to keeper
                new_pairs: set[tuple[str, str]] = set()
                for pa, pb in self._co_visible:
                    if donor.visual_id in (pa, pb):
                        other = pb if pa == donor.visual_id else pa
                        if other != keeper.visual_id:
                            new_pairs.add(self._pair_key(keeper.visual_id, other))
                self._co_visible.update(new_pairs)
                self._delete_track_unlocked(donor.visual_id)
                merged_away.add(donor.visual_id)

    def _apply_visibility_negative_evidence_unlocked(
        self,
        frame_track_ids: set[str],
        *,
        dets_2d: list[Detection2D] | None = None,
        depth: np.ndarray,
        intr: CameraIntrinsics,
        T_odom_cam: SE3Pose,
        max_verify_range_m: float,
        now_wall: float,
    ) -> None:
        dh, dw = depth.shape[:2]
        if dh <= 0 or dw <= 0:
            return
        sx = dw / float(max(1, intr.width))
        sy = dh / float(max(1, intr.height))
        R_co = T_odom_cam.R.T
        t_co = -R_co @ T_odom_cam.t
        margin_u = 0.14 * intr.width
        margin_v = 0.14 * intr.height
        patch_r = max(3, int(round(min(dw, dh) * 0.03)))

        to_delete: list[str] = []
        for vid, tr in self._tracks.items():
            if vid in frame_track_ids:
                continue
            p_odom = np.array([tr.odom_x, tr.odom_y, tr.odom_z], dtype=np.float64)
            p_cam = R_co @ p_odom + t_co
            z = float(p_cam[2])
            # Don't run negative-evidence deletion when standing right up against the block (< 1.45m)
            # where perspective clipping / oblique viewing makes 2D+3D box sizing unreliable.
            if z < 1.45 or z > max_verify_range_m:
                tr.in_fov = False
                continue
            u = intr.cx + intr.fx * float(p_cam[0]) / z
            v = intr.cy + intr.fy * float(p_cam[1]) / z
            if not (margin_u <= u <= intr.width - margin_u and margin_v <= v <= intr.height - margin_v):
                tr.in_fov = False
                continue

            # Occlusion check using live depth image around projected pixel
            du = int(round(u * sx))
            dv = int(round(v * sy))
            y1 = max(0, dv - patch_r)
            y2 = min(dh, dv + patch_r + 1)
            x1 = max(0, du - patch_r)
            x2 = min(dw, du + patch_r + 1)
            patch = depth[y1:y2, x1:x2]
            valid = patch[np.isfinite(patch) & (patch >= 0.25)]
            if valid.size < 4:
                tr.in_fov = False
                continue
            d_med = float(np.median(valid))
            # If something solid is closer than the expected block -> occluded by wall/obstacle!
            if d_med < z - 0.55:
                tr.in_fov = False
                continue

            # Defense-in-depth: if a 2D candidate of the same label STILL overlaps this exact
            # projected pixel AND depth confirms a solid block is still sitting right at z,
            # do NOT treat this frame as "block disappeared"!
            if dets_2d and abs(d_med - z) <= 0.70:
                pad_uv = max(24.0, 0.04 * min(intr.width, intr.height))
                supported_2d = any(
                    d.label == tr.label
                    and (d.x1 - pad_uv) <= u <= (d.x2 + pad_uv)
                    and (d.y1 - pad_uv) <= v <= (d.y2 + pad_uv)
                    for d in dets_2d
                )
                if supported_2d:
                    tr.in_fov = True
                    tr.missed_in_view = 0
                    tr.last_seen_wall = now_wall
                    if tr.confidence >= self.trust_conf_threshold:
                        tr.low_conf_since_wall = None
                    continue

            # Clear unobstructed line of sight to expected block, yet detector did NOT see it!
            tr.in_fov = True
            tr.missed_in_view += 1

            # Unconfirmed 1-hit transient noise can still be pruned quickly
            if tr.hits < self.min_hits:
                if tr.missed_in_view >= min(2, self.miss_delete_frames):
                    to_delete.append(vid)
                continue

            # Pinned active interaction target is never deleted mid-approach
            if vid in self._pinned_ids:
                continue

            # Gently decay confidence on unobstructed miss; ONLY start the 10s trust
            # countdown once confidence drops below trust_conf_threshold (80%)!
            tr.confidence = max(0.0, tr.confidence * self.miss_conf_decay)
            if tr.confidence >= self.trust_conf_threshold:
                tr.low_conf_since_wall = None
                continue
            if tr.low_conf_since_wall is None:
                tr.low_conf_since_wall = now_wall
            elif (
                (now_wall - tr.low_conf_since_wall) >= self.trust_hold_s
                and tr.missed_in_view >= self.miss_delete_frames
            ):
                to_delete.append(vid)

        for vid in to_delete:
            self._delete_track_unlocked(vid)

    def get(self, visual_id: str) -> TrackedBlock | None:
        with self._lock:
            return self._tracks.get(str(visual_id or "").strip())

    def confirmed(
        self,
        *,
        label: str = "",
        robot_xy: tuple[float, float] | None = None,
        robot_z: float | None = None,
        max_distance_m: float | None = None,
    ) -> list[TrackedBlock]:
        key = str(label or "").strip().lower()
        with self._lock:
            out: list[TrackedBlock] = []
            for tr in self._tracks.values():
                if tr.hits < self.min_hits:
                    continue
                if key and tr.label != key:
                    continue
                if robot_z is not None and abs(tr.odom_z - robot_z) > 2.8:
                    # Different floor!
                    continue
                if robot_xy is not None and max_distance_m is not None:
                    if math.hypot(tr.odom_x - robot_xy[0], tr.odom_y - robot_xy[1]) > max_distance_m:
                        continue
                out.append(tr)
        if robot_xy is not None:
            out.sort(key=lambda t: math.hypot(t.odom_x - robot_xy[0], t.odom_y - robot_xy[1]))
        else:
            out.sort(key=lambda t: t.last_seen_wall, reverse=True)
        return out
