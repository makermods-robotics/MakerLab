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
"""Camera identity → cv2 index translation for THIS process (macOS).

``cv2.VideoCapture(index)`` resolves the index against AVFoundation's
*in-process* device list, which is snapshotted the first time this process
touches AVFoundation and never refreshes — device-connection notifications
are delivered on a thread that needs an active NSRunLoop, which uvicorn
doesn't run. ``/available-cameras`` therefore enumerates in a *fresh
subprocess* (see server._avfoundation_cameras_in_cv2_order), but that yields
indices in the fresh device order, which diverges from this process's order
whenever cameras were plugged/unplugged after startup. Opening by such an
index then silently hits the wrong physical device — e.g. the built-in
webcam instead of a robot camera, poisoning previews AND recordings.

The stable link between the two index spaces is AVFoundation's ``uniqueID``.
:func:`resolve_cv2_index` maps a camera's uniqueID to the index cv2 will
actually open *in this process*, by walking the same in-process device list
cv2 walks (video + muxed devices, uniqueID-sorted — mirrors OpenCV's
cap_avfoundation_mac.mm). A device attached after startup is invisible to
in-process AVFoundation entirely; for that case resolution returns None and
callers must fail loudly (telling the user to restart makerlab) rather than
open whatever now sits at the stale index.
"""

import logging
import platform

logger = logging.getLogger(__name__)

_AVF_DEVICE_TYPE_NAMES = (
    "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    "AVCaptureDeviceTypeExternalUnknown",  # macOS < 14
    "AVCaptureDeviceTypeExternal",  # macOS >= 14
    "AVCaptureDeviceTypeContinuityCamera",  # macOS >= 14
    "AVCaptureDeviceTypeDeskViewCamera",  # macOS >= 13
)


def list_cameras_in_process() -> list[dict] | None:
    """This process's camera list, in cv2 open order.

    Returns ``[{"index", "name", "unique_id"}, ...]`` reflecting the same
    (possibly stale) AVFoundation state cv2.VideoCapture resolves indices
    against, or None when identity can't be established (non-macOS, PyObjC
    unavailable, AVFoundation query failure).
    """
    if platform.system() != "Darwin":
        return None
    try:
        import objc
        from Foundation import NSBundle

        bundle = NSBundle.bundleWithPath_("/System/Library/Frameworks/AVFoundation.framework")
        bundle.load()
        types = []
        for name in _AVF_DEVICE_TYPE_NAMES:
            loaded = {}
            try:
                objc.loadBundleVariables(bundle, loaded, [(name, b"@")])
            except objc.error:
                continue
            if loaded.get(name) is not None:
                types.append(loaded[name])
        cls = objc.lookUpClass("AVCaptureDeviceDiscoverySession")
        devs = []
        for media_type in ("vide", "muxx"):
            session = cls.discoverySessionWithDeviceTypes_mediaType_position_(types, media_type, 0)
            devs.extend(session.devices() or [])
        devs.sort(key=lambda d: d.uniqueID())
        return [
            {"index": i, "name": str(d.localizedName()), "unique_id": str(d.uniqueID())}
            for i, d in enumerate(devs)
        ]
    except Exception as e:
        logger.warning("In-process AVFoundation enumeration failed: %s", e)
        return None


def resolve_cv2_index(unique_id: str | None, fallback_index: int) -> int | None:
    """The index THIS process's cv2 opens for ``unique_id``.

    - No unique_id, or identity unavailable (non-macOS / query failure):
      ``fallback_index`` — legacy trust-the-index behavior.
    - unique_id found in the in-process list: its position there (which can
      differ from the fresh-subprocess enumeration index the caller has).
    - unique_id verifiably absent: None — the device attached after this
      process started; only a restart makes it reachable. Callers must error
      out instead of opening a different physical camera.
    """
    if not unique_id:
        return fallback_index
    cameras = list_cameras_in_process()
    if cameras is None:
        return fallback_index
    for cam in cameras:
        if cam["unique_id"] == unique_id:
            if cam["index"] != fallback_index:
                logger.info(
                    "Camera %s: in-process cv2 index %d differs from enumerated index %d "
                    "(device set changed since makerlab started)",
                    unique_id,
                    cam["index"],
                    fallback_index,
                )
            return cam["index"]
    logger.warning(
        "Camera %s is not visible to this process (attached after makerlab started?) — "
        "restart makerlab to use it.",
        unique_id,
    )
    return None
