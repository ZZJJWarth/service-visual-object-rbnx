#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
set -euo pipefail

PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PKG"

VENV="$PKG/rbnx-build/codegen-venv"
if [[ -z "${VISUAL_OBJECT_PYTHON:-}" ]] && { [[ ! -x "$VENV/bin/python3" ]] || ! "$VENV/bin/python3" -c 'import grpc, google.protobuf, mcp, fastmcp, cv2' >/dev/null 2>&1; }; then
    echo "[visual-object/start] runtime venv incomplete; running build"
    bash "$PKG/scripts/build.sh"
fi

ROS_SETUP="${RBNX_MINECRAFT_ROS_SETUP:-}"
if [[ -z "$ROS_SETUP" ]]; then
    for candidate in "/opt/ros/${ROS_DISTRO:-lyrical}/setup.bash" /opt/ros/lyrical/setup.bash /opt/ros/jazzy/setup.bash /opt/ros/humble/setup.bash; do
        if [[ -f "$candidate" ]]; then ROS_SETUP="$candidate"; break; fi
    done
fi
if [[ ! -f "$ROS_SETUP" ]]; then
    echo "[visual-object/start] ROS 2 setup not found; set RBNX_MINECRAFT_ROS_SETUP" >&2
    exit 2
fi
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u

CODEGEN="$PKG/rbnx-build/codegen"
if [[ -d "$CODEGEN/proto_gen" ]]; then
    export PYTHONPATH="$PKG/src:$CODEGEN/proto_gen:$CODEGEN/robonix_mcp_types:${PYTHONPATH:-}"
else
    export PYTHONPATH="$PKG/src:${PYTHONPATH:-}"
fi
if command -v rbnx >/dev/null 2>&1; then
    if API="$(rbnx path robonix-api 2>/dev/null)" && [[ -d "$API" ]]; then
        export PYTHONPATH="$API:$PYTHONPATH"
    fi
fi

export ROBONIX_ATLAS="${ROBONIX_ATLAS:-127.0.0.1:50051}"
export ROBONIX_CAPABILITY_ID="${ROBONIX_CAPABILITY_ID:-visual_object}"
export ROBONIX_PKG_HOST_DIR="$PKG"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

mkdir -p "$PKG/rbnx-build/data"
exec "${VISUAL_OBJECT_PYTHON:-$VENV/bin/python3}" -u -m visual_object_rbnx
