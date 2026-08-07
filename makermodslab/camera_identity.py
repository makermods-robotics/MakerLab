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
in-process AVFoundation entirely — and, worse, a camera replugged into the
SAME port keeps its uniqueID, so it stays *present* in the stale list as a
dead device object that opens "successfully" and then never produces a frame
(silently blank previews, reads that can block forever). Resolution returns
None only for the verifiably-absent case; callers must fail loudly (telling
the user to restart MakerMods Lab) rather than open whatever now sits at the
stale index.

Both failure modes disappear while :func:`pump_avfoundation_runloop` runs:
AVFoundation queues its device-cache updates on the main dispatch queue,
which only the MAIN thread's runloop drains, and uvicorn's asyncio loop —
though it runs on the main thread — never pumps it. Pumping briefly on a
timer keeps the in-process snapshot live (hardware-verified 2026-08-07 via
camera_runloop_experiment.py: a background-thread runloop does NOT refresh
the cache; pumping the main thread's does).
"""

import asyncio
import logging
import platform
import threading

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
                    "(device set changed since MakerMods Lab started)",
                    unique_id,
                    cam["index"],
                    fallback_index,
                )
            return cam["index"]
    logger.warning(
        "Camera %s is not visible to this process (attached after MakerMods Lab started?) — "
        "restart MakerMods Lab to use it.",
        unique_id,
    )
    return None


async def pump_avfoundation_runloop(interval_s: float = 0.5, pump_s: float = 0.05) -> None:
    """Keep this process's AVFoundation camera snapshot live (macOS).

    Run as an asyncio background task from server startup. It must live in the
    main thread's event loop — AVFoundation delivers device connect/disconnect
    cache updates via the main dispatch queue, and only the main thread's
    runloop drains that queue (a background thread's runloop verifiably does
    not — see the module docstring). Each cycle blocks the loop for at most
    ``pump_s`` (~50 ms) every ``interval_s``, then yields back to asyncio.

    No-op on non-macOS or when PyObjC is unavailable; exits with a warning if
    scheduled off the main thread; cancels cleanly on shutdown.
    """
    if platform.system() != "Darwin":
        return
    if threading.current_thread() is not threading.main_thread():
        logger.warning(
            "AVFoundation runloop pump scheduled off the main thread — "
            "hotplug refresh disabled (cache updates only drain on the main runloop)"
        )
        return
    try:
        import objc
        from Foundation import NSDate, NSDefaultRunLoopMode, NSRunLoop
    except ImportError:
        logger.warning("PyObjC unavailable — AVFoundation hotplug refresh disabled")
        return
    # First AVFoundation touch from the main thread, so the snapshot exists
    # before the first pump and is anchored where the updates get drained.
    list_cameras_in_process()
    runloop = NSRunLoop.currentRunLoop()
    logger.info("AVFoundation runloop pump running — camera hotplug/replug is live")
    while True:
        with objc.autorelease_pool():
            runloop.runMode_beforeDate_(NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(pump_s))
        await asyncio.sleep(interval_s)
