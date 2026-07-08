// Retry backoff for a failed backend camera-preview (<img> MJPEG) stream.
// Quick first attempts (a session just released the camera) settling to a slow
// poll (camera unplugged, server down).
export const RETRY_BASE_MS = 2000;
export const RETRY_MAX_MS = 12000;

/**
 * Delay before the next preview retry.
 *
 * While a recording session owns the cameras (the endpoint answered 409), skip
 * the fast escalating retries and poll at the slow cap: hammering can't reclaim
 * the device — the backend latches previews off for the whole session (see
 * camera_preview.block_for_recording) — it only spams 409s and, before that
 * latch existed, could re-acquire the device in the release/open window and
 * starve the recorder. A slow poll still lets the tile recover once recording
 * ends.
 */
export function nextRetryDelayMs(attempt: number, recordingActive: boolean): number {
  if (recordingActive) return RETRY_MAX_MS;
  return Math.min(RETRY_BASE_MS * 2 ** Math.min(attempt, 5), RETRY_MAX_MS);
}
