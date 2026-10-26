#!/usr/bin/env bash
# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
# SPDX-License-Identifier: Apache-2.0
#
# Build an RSUSB-backed pyrealsense2 into the active Python environment.
#
#   pixi run -e real build-pyrealsense
#
# WHY THIS EXISTS, instead of just depending on the pyrealsense2 wheel:
#
# Every wheel Intel publishes is built with librealsense's V4L2 backend, which reaches the
# camera through uvcvideo's UVC extension-unit ioctl. On our workstation that path does not
# work -- every call fails with
#
#     set_xu(...). xioctl(UVCIOC_CTRL_QUERY) failed on control 1  Protocol error
#
# on the 1 KB hardware-monitor transfer librealsense uses to read firmware version,
# intrinsics and extrinsics, so the device never opens. Verified against the cp312 wheels
# for 2.56.4, 2.57.7, 2.58.3 and 2.58.4 -- all V4L2, all fail. The wheel also links
# librealsense statically, so LD_LIBRARY_PATH cannot redirect it.
#
# The RSUSB backend (-DFORCE_RSUSB_BACKEND=ON) talks to the device over libusb and never
# touches uvcvideo, and it works. If the PyPI wheel works on your machine
# (`pip install pyrealsense2==2.58.3` and `--backend rs --dry-run` opens the camera), you
# can skip this.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SRC=${LIBREALSENSE_SRC:-$ROOT/.build/librealsense}
TAG=${LIBREALSENSE_TAG:-v2.58.3}
JOBS=${JOBS:-$(nproc)}
PY=$(command -v python)
echo "source     : $SRC ($TAG)"
echo "python     : $PY"
if [ ! -d "$SRC/.git" ]; then
    git clone --depth 1 --branch "$TAG" https://github.com/IntelRealSense/librealsense.git "$SRC"
fi
mkdir -p "$SRC/build"
cd "$SRC/build"
# CMAKE_C_FLAGS_RELEASE is -O2, not the default -O3, on purpose: gcc 13.3 dies with an
# internal compiler error on the bundled sqlite3.c amalgamation at -O3. C++ stays at -O3.
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG" \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE="$PY" \
    -DPython_EXECUTABLE="$PY" \
    -DPython3_EXECUTABLE="$PY" \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DBUILD_UNIT_TESTS=OFF \
    -DBUILD_TOOLS=OFF \
    -DCHECK_FOR_UPDATES=OFF
make -j"$JOBS" pyrealsense2

# Put this build ahead of anything else on sys.path, so a V4L2 pyrealsense2 that some other
# package pulled in can never shadow it. That failure is silent and costs hours.
SITE=$("$PY" -c 'import site; print(site.getsitepackages()[0])')
PTH="$SITE/_pyrealsense2_rsusb.pth"
printf 'import sys; sys.path.insert(0, %s)\n' "\"$SRC/build/Release\"" > "$PTH"
echo "installed path hook: $PTH"
"$PY" - <<'EOF'
import pyrealsense2 as rs
print("import check      :", rs.__file__)
n = len(list(rs.context().query_devices()))
print(f"devices visible   : {n}")
if n == 0:
    print("  (no camera on the bus right now -- a cabling/power question, not a binding one)")
EOF
