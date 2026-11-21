# SPDX-License-Identifier: MulanPSL-2.0
"""Pure local-costmap approach pose planner, Nav2 gRPC client, and Minecraft MCP client."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

log = logging.getLogger("visual_object_rbnx.approach")

_NAVIGATION_RPC = {
    "navigate": (
        "robonix/service/navigation/navigate",
        "RobonixServiceNavigationNavigateStub",
        "Navigate",
        "Navigate_Request",
    ),
    "status": (
        "robonix/service/navigation/navigate/status",
        "RobonixServiceNavigationNavigateStatusStub",
        "GetNavigationStatus",
        "GetNavigationStatus_Request",
    ),
    "cancel": (
        "robonix/service/navigation/navigate/cancel",
        "RobonixServiceNavigationNavigateCancelStub",
        "CancelNavigation",
        "CancelNavigation_Request",
    ),
}


@dataclass
class LocalCostGrid:
    origin_x: float
    origin_y: float
    resolution: float
    width: int
    height: int
    data: np.ndarray  # shape (height, width), int8

    def cost_at(self, wx: float, wy: float) -> int | None:
        col = int(math.floor((wx - self.origin_x) / self.resolution))
        row = int(math.floor((wy - self.origin_y) / self.resolution))
        if col < 0 or row < 0 or col >= self.width or row >= self.height:
            return None
        return int(self.data[row, col])

    def has_clear_los_to_block(
        self, sx: float, sy: float, bx: float, by: float, *, stop_before_m: float = 0.72
    ) -> bool:
        """Check no lethal wall lies strictly between standing pose (sx, sy) and block (bx, by)."""
        dx, dy = bx - sx, by - sy
        dist = math.hypot(dx, dy)
        if dist <= stop_before_m:
            return True
        ux, uy = dx / dist, dy / dist
        step = max(0.15, self.resolution * 0.75)
        d = step
        limit = dist - stop_before_m
        while d < limit:
            c = self.cost_at(sx + ux * d, sy + uy * d)
            if c is not None and c >= 90:
                return False
            d += step
        return True


def select_approach_pose(
    cost_grid: LocalCostGrid | None,
    target_xy: tuple[float, float],
    robot_xy: tuple[float, float],
    *,
    min_r: float = 1.35,
    max_r: float = 2.35,
) -> tuple[float, float, float] | None:
    """Pick a collision-free approach pose (sx, sy, yaw_facing_target) from local_mapping cost_grid."""
    bx, by = float(target_xy[0]), float(target_xy[1])
    rx, ry = float(robot_xy[0]), float(robot_xy[1])

    if cost_grid is None:
        # Pure geometric fallback along line from block toward robot
        dx, dy = rx - bx, ry - by
        d = math.hypot(dx, dy)
        r_mid = 0.5 * (min_r + max_r)
        if d < 1e-4:
            sx, sy = bx - r_mid, by
        else:
            sx, sy = bx + (dx / d) * r_mid, by + (dy / d) * r_mid
        return sx, sy, math.atan2(by - sy, bx - sx)

    radii = (min_r + 0.15, 0.5 * (min_r + max_r), max_r - 0.15, max_r + 0.35)
    best: tuple[float, float, float] | None = None
    best_score = float("inf")

    for r in radii:
        for deg in range(0, 360, 15):
            ang = math.radians(deg)
            sx = bx + r * math.cos(ang)
            sy = by + r * math.sin(ang)
            c_center = cost_grid.cost_at(sx, sy)
            if c_center is None or c_center < 0 or c_center >= 40:
                continue
            # Require 3x3 neighborhood clearance so robot footprint fits comfortably
            max_nb = c_center
            blocked = False
            for ox in (-0.35, 0.0, 0.35):
                for oy in (-0.35, 0.0, 0.35):
                    vc = cost_grid.cost_at(sx + ox, sy + oy)
                    if vc is None or vc < 0 or vc >= 65:
                        blocked = True
                        break
                    max_nb = max(max_nb, vc)
                if blocked:
                    break
            if blocked:
                continue
            if not cost_grid.has_clear_los_to_block(sx, sy, bx, by):
                continue
            travel_d = math.hypot(sx - rx, sy - ry)
            ideal_r_pen = abs(r - 1.75) * 1.2
            score = travel_d + ideal_r_pen + max_nb * 0.04
            if score < best_score:
                best_score = score
                best = (sx, sy, math.atan2(by - sy, bx - sx))
    return best


def default_nav_rpc(
    operation: str, payload: dict[str, Any], timeout_s: float = 10.0
) -> dict[str, Any]:
    """Invoke standard robonix/service/navigation/* gRPC capabilities via Atlas."""
    import grpc
    import navigation_pb2  # type: ignore
    import robonix_contracts_pb2_grpc as contracts_grpc  # type: ignore
    from robonix_api import ATLAS, Transport

    contract_id, stub_name, method_name, request_name = _NAVIGATION_RPC[operation]
    channel_ref = None
    try:
        caps = ATLAS.find_capability(contract_id=contract_id, transport=Transport.GRPC)
        if not caps:
            return {"ok": False, "detail": f"no provider for {contract_id}"}
        provider = caps[0]
        channel_ref = ATLAS.connect_capability(
            consumer_id=os.environ.get("ROBONIX_CAPABILITY_ID", "visual_object"),
            provider_id=provider.provider_id,
            contract_id=contract_id,
            transport=Transport.GRPC,
        )
        endpoint = (channel_ref.endpoint or "").strip()
        if not endpoint:
            return {"ok": False, "detail": f"empty endpoint for {contract_id}"}
        with grpc.insecure_channel(endpoint) as ch:
            stub = getattr(contracts_grpc, stub_name)(ch)
            req = getattr(navigation_pb2, request_name)(**payload)
            resp = getattr(stub, method_name)(req, timeout=timeout_s)
        return {f.name: v for f, v in resp.ListFields()}
    except grpc.RpcError as exc:
        return {"ok": False, "detail": f"{contract_id}: {exc.code().name}: {exc.details()}"}
    except Exception as exc:  # noqa: BLE001
        log.exception("visual_object nav rpc %s failed", contract_id)
        return {"ok": False, "detail": str(exc)}
    finally:
        if channel_ref is not None:
            try:
                channel_ref.close()
            except Exception:  # noqa: BLE001
                pass


def default_primitive_mcp(
    contract_id: str,
    arguments: dict[str, Any],
    timeout_s: float = 25.0,
) -> dict[str, Any]:
    """Invoke a Minecraft primitive MCP capability through Atlas + FastMCP StreamableHTTP."""
    from fastmcp import Client
    from robonix_api import ATLAS, Transport

    channel_ref = None
    try:
        caps = ATLAS.find_capability(contract_id=contract_id, transport=Transport.MCP)
        if not caps:
            raise RuntimeError(f"no MCP provider registered for {contract_id}")
        provider = caps[0]
        channel_ref = ATLAS.connect_capability(
            consumer_id=os.environ.get("ROBONIX_CAPABILITY_ID", "visual_object"),
            provider_id=provider.provider_id,
            contract_id=contract_id,
            transport=Transport.MCP,
        )
        endpoint = (channel_ref.endpoint or "").strip()
        if not endpoint:
            raise RuntimeError(f"empty MCP endpoint for {contract_id}")
        leaf = contract_id.rsplit("/", 1)[-1]

        async def _invoke() -> dict[str, Any]:
            async with Client(endpoint, timeout=timeout_s, init_timeout=5.0) as client:
                tools = await client.list_tools()
                tool_name = leaf
                for t in tools:
                    if t.name == leaf or t.name.endswith(f"_{leaf}") or t.name.endswith(f"/{leaf}"):
                        tool_name = t.name
                        break
                result = await client.call_tool(tool_name, arguments, timeout=timeout_s)
            if getattr(result, "is_error", False):
                raise RuntimeError(str(getattr(result, "content", result)))
            data = getattr(result, "structured_content", None)
            if not isinstance(data, dict):
                texts = [getattr(item, "text", "") for item in (getattr(result, "content", None) or [])]
                raw_txt = next((t for t in texts if t), "{}")
                data = json.loads(raw_txt)
            if isinstance(data, dict) and isinstance(data.get("result"), dict):
                return data["result"]
            return dict(data) if isinstance(data, dict) else {"value": data}

        return asyncio.run(_invoke())
    finally:
        if channel_ref is not None:
            try:
                channel_ref.close()
            except Exception:  # noqa: BLE001
                pass


def aim_with_ray_probes(
    primitive_mcp: Callable[[str, dict[str, Any], float], dict[str, Any]],
    odom_xyz: tuple[float, float, float],
    ray_dir: tuple[float, float, float],
    expected_blocks: tuple[str, ...],
    *,
    timeout_s: float = 20.0,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Aim at the sensor-estimated 3D center, probing slight depth offsets along the same visual ray
    if grazing-angle rounding lands on an adjacent voxel. Zero game-truth reads.
    """
    bx, by, bz = odom_xyz
    rx, ry, rz = ray_dir
    probes = (0.0, -0.15, 0.18, -0.30, 0.34)
    seen_voxels: set[tuple[int, int, int]] = set()
    last_err: Exception | None = None

    for delta in probes:
        px, py, pz = bx + rx * delta, by + ry * delta, bz + rz * delta
        vox = (math.floor(px), math.floor(pz), math.floor(-py))
        if vox in seen_voxels:
            continue
        seen_voxels.add(vox)
        for mc_block in expected_blocks:
            try:
                res = primitive_mcp(
                    "robonix/primitive/minecraft/aim_at_odom_point",
                    {
                        "odom_x": px,
                        "odom_y": py,
                        "odom_z": pz,
                        "expected_block": mc_block,
                    },
                    timeout_s,
                )
                if res.get("aimed"):
                    return (px, py, pz), res
            except Exception as exc:  # noqa: BLE001
                last_err = exc
    raise RuntimeError(f"failed to aim at any {expected_blocks} near ({bx:.2f}, {by:.2f}, {bz:.2f}): {last_err}")
