# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared backend camera enumeration.

Lists the host's cameras in the SAME index order cv2 will use to open them for
recording and preview, pairing each index with the device's real name and —
where the platform exposes it — a stable hardware ``unique_id``. Both GET
/available-cameras (see lelab/server.py) and the recording start path (see
lelab/record.py) enumerate through here, so a saved camera config can be
re-resolved to the right *physical* device by ``unique_id`` even after a USB
hotplug reshuffles cv2's integer indices.

``unique_id`` is AVFoundation-derived on macOS; on platforms where no stable id
is available (Linux/Windows/generic) entries simply omit it and callers fall
back to the raw index. Degrade, don't gate.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from typing import Any

import cv2

logger = logging.getLogger(__name__)

# Optional Windows-only DirectShow enumeration. Imported here (guarded) rather
# than inside the function so the dependency is declared once at module scope;
# it is absent on macOS/Linux and can fail to import via COM even on Windows.
try:
    from pygrabber.dshow_graph import FilterGraph as _FilterGraph
except Exception:  # ImportError off-Windows, or a COM/DirectShow failure
    _FilterGraph = None


class CameraNotConnectedError(RuntimeError):
    """A camera referenced by ``unique_id`` is not among the connected devices."""


# Runs in a fresh Python — see _avfoundation_cameras_in_cv2_order for why.
# Mirrors OpenCV's macOS enumeration: video + muxed devices sorted by
# uniqueID (cap_avfoundation_mac.mm), so the returned index matches what
# cv2.VideoCapture will open.
_AVF_ENUM_SCRIPT = """
import json, objc
from Foundation import NSBundle
bundle = NSBundle.bundleWithPath_("/System/Library/Frameworks/AVFoundation.framework")
bundle.load()
types = []
for name in (
    "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    "AVCaptureDeviceTypeExternalUnknown",   # macOS < 14
    "AVCaptureDeviceTypeExternal",          # macOS >= 14
    "AVCaptureDeviceTypeContinuityCamera",  # macOS >= 14
    # AVCaptureDeviceTypeDeskViewCamera (Continuity Desk View, macOS >= 13) is
    # deliberately omitted: AVFoundation enumerates it, but OpenCV's index-based
    # cv2.VideoCapture cannot open it ("out device of bound"), so listing it
    # produced permanent /camera-preview 503s and a retry-looping frontend tile.
    # Excluding it here also keeps the reported index aligned with cv2's own
    # ordering, which doesn't count Desk View.
):
    loaded = {}
    try:
        objc.loadBundleVariables(bundle, loaded, [(name, b"@")])
    except objc.error:
        continue
    if loaded.get(name) is not None:
        types.append(loaded[name])
cls = objc.lookUpClass("AVCaptureDeviceDiscoverySession")
devs = []
for mt in ("vide", "muxx"):
    devs.extend(cls.discoverySessionWithDeviceTypes_mediaType_position_(types, mt, 0).devices() or [])
devs.sort(key=lambda d: d.uniqueID())
print(json.dumps([
    {"index": i, "name": str(d.localizedName()), "unique_id": str(d.uniqueID())}
    for i, d in enumerate(devs)
]))
"""


def _avfoundation_cameras_in_cv2_order() -> list[dict[str, Any]]:
    """Enumerate macOS cameras in a fresh Python subprocess.

    AVFoundation's in-process device cache doesn't refresh on USB
    hotplug. Both the deprecated ``+devicesWithMediaType:`` and a
    long-lived ``AVCaptureDeviceDiscoverySession`` go stale, because
    device-connection notifications are delivered via
    ``NSNotificationCenter`` on a thread that needs an active
    ``NSRunLoop`` — uvicorn workers don't run one. A fresh subprocess
    re-initializes AVFoundation, which reads IOKit's live device state
    at startup.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", _AVF_ENUM_SCRIPT],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("AVFoundation enumeration subprocess failed: %s", e)
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        logger.warning("AVFoundation enumeration returned invalid JSON: %s", e)
        return []


def _generic_cv2_cameras(backend) -> list[dict[str, Any]]:
    """Last-resort enumeration: probe cv2 indices with placeholder names."""
    cameras: list[dict[str, Any]] = []
    for i in range(10):
        cap = cv2.VideoCapture(i, backend)
        opened = cap.isOpened()
        cap.release()
        if opened:
            cameras.append({"index": i, "name": f"Camera {i}", "available": True})
    return cameras


def _windows_cameras() -> list[dict[str, Any]]:
    """Enumerate Windows cameras with their real DirectShow names.

    pygrabber lists DirectShow video devices in the same order cv2's DSHOW
    backend indexes them (which recording is pinned to), so the returned index
    matches what ``cv2.VideoCapture(i, CAP_DSHOW)`` opens. The real names let the
    frontend match each index to the browser's ``MediaDeviceInfo.label`` for the
    live preview. Falls back to generic names if pygrabber is unavailable.
    """
    if _FilterGraph is None:
        logger.warning("pygrabber unavailable; using generic camera names")
        return _generic_cv2_cameras(cv2.CAP_DSHOW)
    try:
        names = _FilterGraph().get_input_devices()
    except Exception as e:  # a COM/DirectShow failure at call time
        logger.warning("pygrabber enumeration failed; using generic camera names: %s", e)
        return _generic_cv2_cameras(cv2.CAP_DSHOW)
    return [{"index": i, "name": name, "available": True} for i, name in enumerate(names)]


def _v4l2_camera_name(index: int) -> str | None:
    """Real camera name for /dev/video{index} from sysfs (Linux, no deps)."""
    try:
        with open(f"/sys/class/video4linux/video{index}/name", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _linux_cameras() -> list[dict[str, Any]]:
    """Enumerate Linux cameras, naming each from sysfs (no extra deps)."""
    cameras: list[dict[str, Any]] = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        opened = cap.isOpened()
        cap.release()
        if not opened:
            continue
        cameras.append({"index": i, "name": _v4l2_camera_name(i) or f"Camera {i}", "available": True})
    return cameras


def list_cameras() -> list[dict[str, Any]]:
    """List the host's cameras in cv2 index order.

    Each entry is ``{"index", "name", "available": True}`` and, on macOS, also
    ``"unique_id"`` (AVFoundation's stable device id). Ordering matches the cv2
    backend the recorder is pinned to on each platform, so a returned index
    always opens the named device *at enumeration time* — the point of
    ``unique_id`` is to survive later reshuffles.
    """
    system = platform.system()
    if system == "Darwin":
        cameras = _avfoundation_cameras_in_cv2_order()
        for cam in cameras:
            cam["available"] = True
        return cameras
    if system == "Windows":
        return _windows_cameras()
    if system == "Linux":
        return _linux_cameras()
    return _generic_cv2_cameras(cv2.CAP_ANY)


def find_camera_index(cameras: list[dict[str, Any]], unique_id: str) -> int | None:
    """Return the cv2 index of the enumerated camera with ``unique_id``, or None."""
    for cam in cameras:
        if cam.get("unique_id") and cam["unique_id"] == unique_id:
            return cam["index"]
    return None


def resolve_index(
    unique_id: str,
    enumerated: list[dict[str, Any]] | None = None,
    label: str | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """THE id→current-index resolution: the one place a ``unique_id`` becomes a
    cv2 integer index.

    Both consumers go through here so the semantics can't drift apart:
      * record start (lelab/record.py::_resolve_camera_index) — resolving each
        saved camera entry before any hardware is opened;
      * preview open (lelab/camera_preview.py) — resolving on EVERY fresh
        capture open, so a preview keyed by id can never show whatever device
        slid into a stale integer slot after a hotplug reshuffle.

    ``enumerated`` lets a caller resolving several ids share one enumeration
    (None → enumerate now); the (possibly fresh) list is threaded back out.
    ``label`` names the camera in the error ("wrist", "front") when the caller
    has one. A ``unique_id`` matching no connected device raises the legible
    :class:`CameraNotConnectedError` — never a silent index fallback.
    """
    if enumerated is None:
        enumerated = list_cameras()
    index = find_camera_index(enumerated, unique_id)
    if index is None:
        who = f"Camera '{label}'" if label else "Camera"
        raise CameraNotConnectedError(
            f"{who} (id …{unique_id[-7:]}) is not connected — plug it in or re-select cameras."
        )
    return index, enumerated
