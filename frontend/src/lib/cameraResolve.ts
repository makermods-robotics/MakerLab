import type { AvailableCamera } from "@/hooks/useAvailableCameras";
import type { CameraConfig } from "@/components/recording/CameraConfiguration";

// A stored camera_index is only a position in cv2's device list — it silently
// rebinds to a different physical camera when devices come and go (a robot cam
// unplugging can leave its index pointing at the laptop's built-in camera).
// These helpers re-anchor a configured camera to its stable unique_id against
// a fresh /available-cameras enumeration before the index is trusted.
//
// Verification is only possible when the record stores a unique_id AND the
// enumeration reports uniqueIds (macOS). Otherwise fall back to legacy
// trust-the-stored-index behavior rather than falsely flagging disconnects.

const canVerify = (cam: CameraConfig, available: AvailableCamera[]) =>
  Boolean(cam.unique_id) && available.some((m) => m.uniqueId);

/** True when `cam` is verifiably attached, or when we can't verify. */
export function isCameraConnected(
  cam: CameraConfig,
  available: AvailableCamera[],
): boolean {
  if (!canVerify(cam, available)) return true;
  return available.some((m) => m.uniqueId === cam.unique_id);
}

/** The cv2 index to open for `cam` right now: the enumerated index of its
 * unique_id when verifiable (stored indices go stale on replug), undefined
 * when the camera is verifiably disconnected, else the stored index. */
export function resolveCameraIndex(
  cam: CameraConfig,
  available: AvailableCamera[],
): number | undefined {
  if (!canVerify(cam, available)) return cam.camera_index;
  return available.find((m) => m.uniqueId === cam.unique_id)?.index;
}
