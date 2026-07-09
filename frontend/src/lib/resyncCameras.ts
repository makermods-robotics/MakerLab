import type { AvailableCamera } from "@/hooks/useAvailableCameras";

/**
 * The minimal shape resyncCameras reads/writes on a seeded camera. Kept
 * structural (not a hard dependency on CameraConfig) so this stays a pure,
 * import-cycle-free helper — CameraConfiguration passes its richer objects and
 * gets them back with the same extra fields preserved (spread).
 */
export interface SeededCamera {
  camera_index?: number;
  unique_id?: string;
  device_id: string;
}

export interface ResyncResult<T> {
  /** The (possibly) updated list; identity-equal to the input when nothing
   * changed, so callers can skip a no-op state write. */
  cameras: T[];
  changed: boolean;
  /** True when at least one seeded camera could NOT be safely bound to a stable
   * unique_id because the only available match was ambiguous — two live cameras
   * share a display name, so a device_id/name match can't prove WHICH physical
   * device it is. Those entries are left untouched; the caller should nudge the
   * user to re-select them (an explicit pick against the live preview is the one
   * unambiguous binding). */
  needsReselect: boolean;
  /** Seeded cameras whose saved unique_id matched NO connected device. On macOS
   * the unique_id is the USB port path, so this happens whenever the camera is
   * unplugged OR moved to a different port (a move mints a new identity). The
   * entries are left untouched — their tiles 503 until the device returns or the
   * user re-selects — but the caller should SAY so, because otherwise the only
   * symptom is a silently retrying tile. Empty when the enumeration itself was
   * empty (a failed probe must not scream "everything is missing"). */
  missing: T[];
}

/**
 * Refresh seeded cameras against the live enumeration WITHOUT ever silently
 * baking in the wrong identity. Two fail-safe rules:
 *
 *  1. An entry that already carries a `unique_id` is refreshed ONLY by matching
 *     that unique_id against the enumeration (unambiguous — it is a stable
 *     hardware id). Its cv2 `camera_index` is updated if the device moved, but
 *     the `unique_id` is NEVER overwritten.
 *
 *     This is the bug fix: the old effect overwrote `unique_id` from a
 *     `device_id` match, and a browser `device_id` is bound to a cv2 index only
 *     by AVFoundation `localizedName` — identical for two same-model cameras. A
 *     re-enumeration that paired the browser ids to cv2 indices in the other
 *     order would then rewrite the entry's `unique_id` to the WRONG camera,
 *     turning a transient index shuffle into a persistent front/wrist identity
 *     flip on disk.
 *
 *  2. A legacy entry with no `unique_id` may be BACKFILLED from a `device_id`
 *     match ONLY when that match is unambiguous — no other enumerated camera
 *     shares the matched device's display name (and the enumeration exposed a
 *     stable id to bind to). When two live cameras share a name, the entry is
 *     left legacy (index-only) and `needsReselect` is set.
 */
export function resyncCameras<T extends SeededCamera>(
  cameras: T[],
  available: AvailableCamera[],
): ResyncResult<T> {
  if (available.length === 0 || cameras.length === 0) {
    return { cameras, changed: false, needsReselect: false, missing: [] };
  }

  // Count display names so we can tell whether a name (hence a device_id match,
  // which is itself name-derived) uniquely identifies a physical device.
  const nameCounts = new Map<string, number>();
  for (const cam of available) {
    nameCounts.set(cam.name, (nameCounts.get(cam.name) ?? 0) + 1);
  }

  let changed = false;
  let needsReselect = false;
  const missing: T[] = [];

  const next = cameras.map((cam) => {
    if (cam.unique_id) {
      // Trust the stable id: re-resolve the current index from it, never rewrite
      // the id. A miss (camera unplugged, or moved to another port — which mints
      // a NEW id) leaves the entry as-is but is REPORTED via `missing`, so the
      // caller can tell the user instead of leaving a silently-503ing tile. The
      // backend still raises a legible "not connected" error at record start.
      const byId = available.find((m) => m.uniqueId === cam.unique_id);
      if (!byId) {
        missing.push(cam);
        return cam;
      }
      if (byId.index !== cam.camera_index) {
        changed = true;
        return { ...cam, camera_index: byId.index };
      }
      return cam;
    }

    // Legacy entry: backfill only on an UNAMBIGUOUS device_id match.
    if (!cam.device_id) return cam;
    const match = available.find((m) => m.deviceId === cam.device_id);
    if (!match) return cam;

    const ambiguous = !match.uniqueId || (nameCounts.get(match.name) ?? 0) > 1;
    if (ambiguous) {
      // Two identical-name cameras (or no stable id on this platform): a
      // device_id match can't prove which physical camera this is. Don't bake in
      // an identity that could be wrong — leave it legacy and nudge a re-select.
      needsReselect = true;
      return cam;
    }

    changed = true;
    return { ...cam, camera_index: match.index, unique_id: match.uniqueId };
  });

  return { cameras: changed ? next : cameras, changed, needsReselect, missing };
}
