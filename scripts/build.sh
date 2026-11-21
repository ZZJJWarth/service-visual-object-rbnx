#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
set -euo pipefail

PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PKG"
mkdir -p rbnx-build/data

ROS_SETUP="${RBNX_MINECRAFT_ROS_SETUP:-}"
if [[ -z "$ROS_SETUP" ]]; then
    for candidate in "/opt/ros/${ROS_DISTRO:-lyrical}/setup.bash" /opt/ros/lyrical/setup.bash /opt/ros/jazzy/setup.bash /opt/ros/humble/setup.bash; do
        if [[ -f "$candidate" ]]; then ROS_SETUP="$candidate"; break; fi
    done
fi
if [[ -z "$ROS_SETUP" || ! -f "$ROS_SETUP" ]]; then
    echo "[visual-object/build] ROS 2 setup not found; set RBNX_MINECRAFT_ROS_SETUP" >&2
    exit 2
fi
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u

if command -v rbnx >/dev/null 2>&1; then
    FLAGS=(--mcp)
    [[ "${RBNX_BUILD_CLEAN:-}" == "1" ]] && FLAGS+=(--clean)
    echo "[visual-object/build] rbnx codegen ${FLAGS[*]}"
    rbnx codegen -p "$PKG" "${FLAGS[@]}"
else
    echo "[visual-object/build] WARNING: rbnx not in PATH; skipping codegen" >&2
fi

VENV="$PKG/rbnx-build/codegen-venv"
PYBIN="${VISUAL_OBJECT_PYTHON:-$VENV/bin/python3}"
if [[ -z "${VISUAL_OBJECT_PYTHON:-}" ]]; then
    if [[ ! -x "$PYBIN" ]]; then
        echo "[visual-object/build] create runtime venv"
        python3 -m venv --system-site-packages "$VENV"
    fi
    if ! "$PYBIN" -c 'import grpc, google.protobuf, mcp, fastmcp, cv2' >/dev/null 2>&1; then
        echo "[visual-object/build] install runtime dependencies"
        "$PYBIN" -m pip install --no-cache-dir --disable-pip-version-check \
            grpcio==1.80.0 protobuf==6.33.6 'mcp>=1.28.1,<2' 'fastmcp>=3,<4'
    fi
fi
"$PYBIN" - <<'PY'
import importlib
for name in ("grpc", "numpy", "cv2", "yaml", "rclpy", "geometry_msgs", "nav_msgs", "sensor_msgs"):
    importlib.import_module(name)
print("[visual-object/build] runtime imports OK")
PY

touch rbnx-build/.rbnx-built
echo "[visual-object/build] done"
