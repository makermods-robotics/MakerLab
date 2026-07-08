// Pure helpers deciding a newly-added camera's default pixel format (FOURCC).
//
// Background: a recording session failed on a marginal "frame is too old"
// timeout (521ms vs 500ms max) on one of two identical USB cameras — the
// classic uncompressed-YUYV USB-bandwidth signature. MJPG is compressed and
// near-universally supported on UVC (USB) cameras, so defaulting external
// cameras to it keeps two cameras inside the bus budget. Built-in Apple cameras
// (FaceTime / MacBook) aren't bandwidth-constrained and some don't expose MJPG,
// so they stay on auto-detect (undefined). The FOURCC dropdown remains
// user-overridable, and existing saved entries are untouched (no migration).

/** A camera row as surfaced by useAvailableCameras (only `name` is needed). */
export interface CameraLike {
  name?: string;
}

// Apple's built-in / integrated cameras and the non-USB Continuity/Desk-View
// virtual devices, matched on the enumeration display name. These are the
// devices we DON'T force to MJPG.
const BUILT_IN_CAMERA_NAME = /facetime|built[- ]?in|macbook|desk view|continuity/i;

/** True when the enumeration entry looks like a built-in / non-USB camera. */
export function isBuiltInCamera(cam: CameraLike): boolean {
  return BUILT_IN_CAMERA_NAME.test(cam.name ?? "");
}

/**
 * Default FOURCC for a freshly-added camera: "MJPG" for external/USB cameras
 * (bandwidth-safe), undefined (auto-detect) for built-in cameras. Returned
 * value is stored verbatim on the CameraConfig; undefined means the recorder
 * lets OpenCV auto-negotiate the format (record.py: `fourcc or None`).
 */
export function defaultFourccForCamera(cam: CameraLike): string | undefined {
  return isBuiltInCamera(cam) ? undefined : "MJPG";
}
