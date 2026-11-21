# SPDX-License-Identifier: MulanPSL-2.0
"""Extensible RGB-D block detector plugins (Classic CV + drop-in ONNX YOLO backend).

100% pure-sensor image/depth processing. Adding a new block type requires either:
  1) Dropping a multi-class ONNX detector model + blocks.yaml entry (zero code); OR
  2) Registering a lightweight classic CV plugin via `@ClassicBlockDetector.register(label)`.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from visual_object_rbnx.registry import BlockRegistry

log = logging.getLogger("visual_object_rbnx.detectors")


@dataclass(frozen=True)
class Detection2D:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    backend: str = "classic"
    mask: np.ndarray | None = None  # optional bool mask cropped to (y2-y1, x2-x1)

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_uv(self) -> tuple[float, float]:
        return 0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2)


class BlockDetector(ABC):
    name: str = "base"

    @abstractmethod
    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        registry: BlockRegistry,
        *,
        target_labels: set[str] | None = None,
    ) -> list[Detection2D]:
        """Return 2D candidate bounding boxes on an HxWx3 RGB uint8 frame."""


ClassicPluginFn = Callable[[np.ndarray, np.ndarray | None], list[Detection2D]]


class ClassicBlockDetector(BlockDetector):
    """Registry of fast, zero-weight visual detectors for distinctive Minecraft blocks."""

    name = "classic"
    _plugins: dict[str, ClassicPluginFn] = {}

    @classmethod
    def register(cls, label: str) -> Callable[[ClassicPluginFn], ClassicPluginFn]:
        key = label.strip().lower()

        def _decorator(fn: ClassicPluginFn) -> ClassicPluginFn:
            cls._plugins[key] = fn
            return fn

        return _decorator

    @classmethod
    def supported_labels(cls) -> set[str]:
        return set(cls._plugins.keys())

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        registry: BlockRegistry,
        *,
        target_labels: set[str] | None = None,
    ) -> list[Detection2D]:
        out: list[Detection2D] = []
        for spec in registry.all_specs():
            if spec.backend not in ("auto", "classic"):
                continue
            if target_labels is not None and spec.label not in target_labels:
                continue
            plugin = self._plugins.get(spec.label)
            if plugin is None:
                continue
            out.extend(plugin(rgb, depth))
        return out


@ClassicBlockDetector.register("chest")
def detect_minecraft_chest(rgb: np.ndarray, depth: np.ndarray | None) -> list[Detection2D]:
    """Detect Minecraft chests (single/double/trapped) using warm ochre HSV + dark border/seam + latch + depth split.

    Rejects plain flat wooden walls/floors via:
    1) Higher ochre saturation + dark horizontal seam/frame border requirement;
    2) Depth-discontinuity splitting (separates chests from background walls/adjacent blocks);
    3) Aspect-ratio & 2D scale sanity (refined further by exact 3D physical size in projection.py).
    """
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        return []
    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        return []

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Minecraft oak chest wood body: rich golden-orange/ochre brown (OpenCV H in [0..179])
    # Covers well-lit indoor torches/lamps as well as shadowed lower faces up close.
    warm_mask = (
        (hsv[:, :, 0] >= 11)
        & (hsv[:, :, 0] <= 26)
        & (hsv[:, :, 1] >= 90)
        & (hsv[:, :, 2] >= 42)
        & (hsv[:, :, 2] <= 245)
    ).astype(np.uint8) * 255

    # Mask out bottom Minecraft HUD/hotbar strip so warm inventory icons never trigger false positives
    hud_y = int(h * 0.84)
    if hud_y < h:
        warm_mask[hud_y:, :] = 0

    # If depth is available, suppress strong depth-jump edges (>0.45m per pixel step) so a chest
    # against a warm wooden background wall does not merge with the wall, without slicing the
    # chest's own angled front/side faces apart!
    has_depth = depth is not None and depth.shape[:2] == (h, w)
    d_clean: np.ndarray | None = None
    if has_depth:
        d_clean = np.where(np.isfinite(depth) & (depth > 0.2) & (depth < 14.0), depth, 0.0).astype(np.float32)
        gx = np.abs(cv2.Sobel(d_clean, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(d_clean, cv2.CV_32F, 0, 1, ksize=3))
        depth_edge = (np.hypot(gx, gy) > 1.8) | (d_clean <= 0.2)
        warm_mask[depth_edge] = 0

    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close_sz = max(5, min(11, int(round(min(h, w) * 0.015)) | 1))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_sz, close_sz))
    cleaned = cv2.morphologyEx(warm_mask, cv2.MORPH_OPEN, k_small)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    # Extract raw warm fragments (lower threshold so up-close chest lid/side panels split by
    # thick black borders are kept and merged into one full chest below).
    min_frag = max(8, int(min(h, w) * 0.018))
    frags: list[dict[str, object]] = []
    for cid in range(1, num_labels):
        fx = int(stats[cid, cv2.CC_STAT_LEFT])
        fy = int(stats[cid, cv2.CC_STAT_TOP])
        fw = int(stats[cid, cv2.CC_STAT_WIDTH])
        fh = int(stats[cid, cv2.CC_STAT_HEIGHT])
        farea = int(stats[cid, cv2.CC_STAT_AREA])
        if fw < min_frag or fh < min_frag or farea < min_frag * min_frag // 2:
            continue
        if fw > w * 0.65 and fh > h * 0.55:
            continue
        med_d: float | None = None
        if d_clean is not None:
            sub_mask = labels[fy : fy + fh, fx : fx + fw] == cid
            sub_d = d_clean[fy : fy + fh, fx : fx + fw][sub_mask]
            sub_d = sub_d[sub_d > 0.2]
            if sub_d.size >= 4:
                med_d = float(np.median(sub_d))
        frags.append(
            {"cid": cid, "x1": fx, "y1": fy, "x2": fx + fw, "y2": fy + fh, "area": farea, "d": med_d}
        )

    # Depth-consistent fragment merging (Union-Find): bridges thick dark chest borders & lid seams
    # at close range (1-3m) while NEVER merging a chest with a wooden wall at a different depth!
    n_frags = len(frags)
    parent = list(range(n_frags))

    def _find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    for i in range(n_frags):
        fi = frags[i]
        wi, hi = int(fi["x2"]) - int(fi["x1"]), int(fi["y2"]) - int(fi["y1"])
        for j in range(i + 1, n_frags):
            fj = frags[j]
            wj, hj = int(fj["x2"]) - int(fj["x1"]), int(fj["y2"]) - int(fj["y1"])
            # Proportional gap bridges thick black rim/seam inside one chest up close,
            # without bridging separate chests far away.
            gap = max(5, min(26, int(round(0.24 * max(min(wi, hi), min(wj, hj), max(wi, hi) * 0.5)))))
            if (
                int(fi["x1"]) - gap > int(fj["x2"])
                or int(fj["x1"]) - gap > int(fi["x2"])
                or int(fi["y1"]) - gap > int(fj["y2"])
                or int(fj["y1"]) - gap > int(fi["y2"])
            ):
                continue
            di, dj = fi["d"], fj["d"]
            if di is not None and dj is not None and abs(float(di) - float(dj)) > 0.75:
                continue
            # Check combined box doesn't become an oversized wall
            mx1 = min(int(fi["x1"]), int(fj["x1"]))
            my1 = min(int(fi["y1"]), int(fj["y1"]))
            mx2 = max(int(fi["x2"]), int(fj["x2"]))
            my2 = max(int(fi["y2"]), int(fj["y2"]))
            if (mx2 - mx1) > w * 0.65 and (my2 - my1) > h * 0.55:
                continue
            ri, rj = _find(i), _find(j)
            if ri != rj:
                parent[rj] = ri

    groups: dict[int, list[dict[str, object]]] = {}
    for i, f in enumerate(frags):
        groups.setdefault(_find(i), []).append(f)

    detections: list[Detection2D] = []
    min_px = max(14, int(min(h, w) * 0.035))

    for grp in groups.values():
        x = min(int(f["x1"]) for f in grp)
        y = min(int(f["y1"]) for f in grp)
        x2 = max(int(f["x2"]) for f in grp)
        y2 = max(int(f["y2"]) for f in grp)
        bw = x2 - x
        bh = y2 - y
        area = sum(int(f["area"]) for f in grp)

        if bw < min_px or bh < min_px:
            continue
        # Reject huge walls/floors covering > 55% of image width & height
        if bw > w * 0.65 and bh > h * 0.55:
            continue
        aspect = bw / max(1.0, float(bh))
        # Single chest (~0.65..1.55) or double chest (~1.55..2.85) in perspective
        if aspect < 0.48 or aspect > 3.15:
            continue
        fill_ratio = area / max(1.0, float(bw * bh))
        if fill_ratio < 0.30:
            continue

        # Expand ROI slightly (10%) to inspect the chest's dark outer border & horizontal lid seam
        pad_x = max(2, int(bw * 0.10))
        pad_y = max(2, int(bh * 0.10))
        rx1, ry1 = max(0, x - pad_x), max(0, y - pad_y)
        rx2, ry2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
        roi_hsv = hsv[ry1:ry2, rx1:rx2]
        roi_v = roi_hsv[:, :, 2]

        # 1) Dark rim / horizontal lid seam (chest texture always has dark pixels V < 75 around border & lid line)
        dark_ratio = float(np.mean(roi_v < 75))
        # Also check mid-upper horizontal band (20%..58% height) for the dark lid-seam & contrast
        band_y1 = y + int(bh * 0.18)
        band_y2 = y + int(bh * 0.58)
        band_v = hsv[band_y1:max(band_y1 + 1, band_y2), x : x + bw, 2]
        seam_dark = float(np.mean(band_v < 80)) if band_v.size else 0.0

        # 2) Metallic silver/gray lock latch near upper-center (present on front face, bonus confidence)
        lx1 = x + int(bw * 0.20)
        lx2 = x + int(bw * 0.80)
        ly1 = y + int(bh * 0.16)
        ly2 = y + int(bh * 0.62)
        latch_roi_s = hsv[ly1:max(ly1 + 1, ly2), lx1:max(lx1 + 1, lx2), 1]
        latch_roi_v = hsv[ly1:max(ly1 + 1, ly2), lx1:max(lx1 + 1, lx2), 2]
        latch_ratio = (
            float(np.mean((latch_roi_s < 65) & (latch_roi_v >= 110) & (latch_roi_v <= 245)))
            if latch_roi_s.size
            else 0.0
        )

        # Require at least subtle dark framing/seam (uniform flat oak planks lack dark chest framing)
        if dark_ratio < 0.03 and seam_dark < 0.02 and latch_ratio < 0.015:
            continue

        # Compute clean confidence in [0.0, 0.98]
        score = 0.34
        score += min(0.24, fill_ratio * 0.32)
        score += min(0.20, dark_ratio * 1.1)
        score += min(0.12, seam_dark * 1.2)
        if latch_ratio >= 0.01:
            score += min(0.16, 0.08 + latch_ratio * 1.2)
        score = max(0.0, min(0.98, score))

        cids = {int(f["cid"]) for f in grp}
        comp_mask = np.isin(labels[y:y2, x:x2], list(cids))
        detections.append(
            Detection2D(
                label="chest",
                x1=x,
                y1=y,
                x2=x2,
                y2=y2,
                confidence=score,
                backend="classic",
                mask=comp_mask,
            )
        )

    return detections


class OnnxBlockDetector(BlockDetector):
    """Drop-in multi-class ONNX object detector (YOLOv8/YOLO11 export compatible).

    Expects standard YOLO ONNX output shape `(1, 4 + num_classes, N)` or `(1, N, 6)`
    with input `(1, 3, H, W)` normalized to `[0, 1]`. Disabled gracefully if no model
    is configured or onnxruntime is unavailable.
    """

    name = "onnx"

    def __init__(self, model_path: str, *, conf_threshold: float = 0.45) -> None:
        self.model_path = str(model_path or "").strip()
        self.conf_threshold = float(conf_threshold)
        self.enabled = False
        self.reason = "disabled by config"
        self._session = None
        self._input_name = ""
        self._in_w = 640
        self._in_h = 640
        self._load()

    def _load(self) -> None:
        if not self.model_path:
            return
        if not os.path.isfile(self.model_path):
            self.reason = f"model file not found: {self.model_path}"
            return
        try:
            import onnxruntime as ort  # type: ignore

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                self.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            shape = inp.shape
            if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                self._in_h, self._in_w = int(shape[2]), int(shape[3])
            self.enabled = True
            self.reason = f"ready ({self._in_w}x{self._in_h})"
            log.info("loaded ONNX block detector: %s (%s)", self.model_path, self.reason)
        except Exception as exc:  # noqa: BLE001
            self.reason = f"onnx load failed: {exc}"
            log.warning("ONNX block detector unavailable: %s", self.reason)

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        registry: BlockRegistry,
        *,
        target_labels: set[str] | None = None,
    ) -> list[Detection2D]:
        if not self.enabled or self._session is None or rgb is None or rgb.ndim != 3:
            return []
        orig_h, orig_w = rgb.shape[:2]
        resized = cv2.resize(rgb, (self._in_w, self._in_h), interpolation=cv2.INTER_LINEAR)
        tensor = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        try:
            raw = np.asarray(self._session.run(None, {self._input_name: tensor})[0], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            log.warning("ONNX inference error (disabling): %s", exc)
            self.enabled = False
            self.reason = f"inference failed: {exc}"
            return []

        sx = orig_w / float(self._in_w)
        sy = orig_h / float(self._in_h)
        out: list[Detection2D] = []

        # Format A: End-to-end exported (1, N, 6) -> [x1, y1, x2, y2, score, cls_id]
        if raw.ndim == 3 and raw.shape[2] == 6:
            rows = raw[0]
            for row in rows:
                conf = float(row[4])
                if conf < self.conf_threshold:
                    continue
                cls_id = int(round(float(row[5])))
                spec = registry.by_onnx_id(cls_id)
                if spec is None or (target_labels is not None and spec.label not in target_labels):
                    continue
                x1 = max(0, min(orig_w - 1, int(round(float(row[0]) * sx))))
                y1 = max(0, min(orig_h - 1, int(round(float(row[1]) * sy))))
                x2 = max(x1 + 1, min(orig_w, int(round(float(row[2]) * sx))))
                y2 = max(y1 + 1, min(orig_h, int(round(float(row[3]) * sy))))
                out.append(Detection2D(spec.label, x1, y1, x2, y2, conf, backend="onnx"))
            return out

        # Format B: Standard YOLOv8/11 (1, 4+C, N)
        arr = np.squeeze(raw, axis=0) if raw.ndim == 3 else raw
        if arr.ndim != 2 or arr.shape[0] < 5:
            return []
        preds = arr.T  # (N, 4+C)
        scores_mat = preds[:, 4:]
        cls_ids = np.argmax(scores_mat, axis=1)
        confs = scores_mat[np.arange(len(preds)), cls_ids]
        keep = confs >= self.conf_threshold
        for box, cid, conf in zip(preds[keep, :4], cls_ids[keep], confs[keep]):
            spec = registry.by_onnx_id(int(cid))
            if spec is None or (target_labels is not None and spec.label not in target_labels):
                continue
            cx, cy, bw, bh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            x1 = max(0, min(orig_w - 1, int(round((cx - bw / 2.0) * sx))))
            y1 = max(0, min(orig_h - 1, int(round((cy - bh / 2.0) * sy))))
            x2 = max(x1 + 1, min(orig_w, int(round((cx + bw / 2.0) * sx))))
            y2 = max(y1 + 1, min(orig_h, int(round((cy + bh / 2.0) * sy))))
            out.append(Detection2D(spec.label, x1, y1, x2, y2, float(conf), backend="onnx"))
        return out


def nms_detections(dets: list[Detection2D], iou_threshold: float = 0.45) -> list[Detection2D]:
    """Per-class Non-Maximum Suppression."""
    if len(dets) <= 1:
        return list(dets)
    ordered = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection2D] = []
    for cand in ordered:
        duplicate = False
        for prev in kept:
            if prev.label != cand.label:
                continue
            ix1 = max(prev.x1, cand.x1)
            iy1 = max(prev.y1, cand.y1)
            ix2 = min(prev.x2, cand.x2)
            iy2 = min(prev.y2, cand.y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter <= 0:
                continue
            union = prev.area + cand.area - inter
            if union > 0 and (inter / union) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


class DetectorPipeline:
    """Runs configured ONNX + Classic block detectors and merges results via NMS."""

    def __init__(
        self,
        registry: BlockRegistry,
        *,
        onnx_model_path: str = "",
        min_confidence: float = 0.45,
    ) -> None:
        self.registry = registry
        self.min_confidence = float(min_confidence)
        self.onnx = OnnxBlockDetector(onnx_model_path, conf_threshold=min_confidence)
        self.classic = ClassicBlockDetector()

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        *,
        target_labels: set[str] | None = None,
    ) -> list[Detection2D]:
        all_dets: list[Detection2D] = []
        if self.onnx.enabled:
            all_dets.extend(self.onnx.detect(rgb, depth, self.registry, target_labels=target_labels))
        # Run classic detectors for labels configured as 'classic' or 'auto' (when ONNX disabled or absent)
        classic_labels: set[str] = set()
        for spec in self.registry.all_specs():
            if target_labels is not None and spec.label not in target_labels:
                continue
            if spec.backend == "classic" or (spec.backend == "auto" and not self.onnx.enabled):
                classic_labels.add(spec.label)
        if classic_labels:
            all_dets.extend(self.classic.detect(rgb, depth, self.registry, target_labels=classic_labels))
        filtered = [d for d in all_dets if d.confidence >= self.min_confidence]
        return nms_detections(filtered)

    def status(self) -> dict[str, object]:
        return {
            "onnx_enabled": self.onnx.enabled,
            "onnx_reason": self.onnx.reason,
            "classic_labels": sorted(self.classic.supported_labels()),
            "registered_labels": self.registry.labels(),
        }
