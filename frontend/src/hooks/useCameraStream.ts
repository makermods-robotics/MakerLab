import { useEffect, useRef, useState } from "react";

/**
 * Attach a live browser camera stream to a `<video>` element by deviceId.
 * Set `paused=true` to release the stream (e.g. so cv2.VideoCapture can claim
 * the camera exclusively). The stream is auto-stopped on unmount.
 *
 * A first getUserMedia attempt can fail two ways that we want to recover from:
 *   - Transiently: the device is briefly held (e.g. right after an enumeration
 *     probe releases it) and reports NotReadableError/AbortError. We back off
 *     and retry the same deviceId a few times.
 *   - Pending an external change: camera permission hasn't been granted yet, or
 *     the device isn't connected. Retrying the same call won't help, but the
 *     browser can fire `devicechange` following a permission grant or a
 *     device-exposure change (timing isn't guaranteed) — so we re-attempt on
 *     that event instead. Without this the hook would otherwise stay stuck on
 *     the error state until a full page reload.
 */
const MAX_RETRIES = 4;
const BASE_DELAY_MS = 300;
// Errors worth an immediate retry. Permission denial (NotAllowedError), missing
// device (NotFoundError) and unsatisfiable constraints (OverconstrainedError)
// won't fix themselves on a retimed retry — those recover via `devicechange`.
const TRANSIENT_ERRORS = new Set(["NotReadableError", "AbortError"]);

export function useCameraStream(deviceId: string, paused: boolean) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [hasError, setHasError] = useState(false);
  // Bumping this forces the stream effect to re-run (a clean retry).
  const [retryKey, setRetryKey] = useState(0);
  // Track the error state for the devicechange handler without re-binding it.
  const hasErrorRef = useRef(false);
  hasErrorRef.current = hasError;

  // A permission grant or hot-plug can prompt the browser to emit
  // `devicechange`; use it to re-attempt a stream that failed pre-permission.
  // Only retry when we're actually in the error state, so an unrelated change
  // (e.g. plugging in a mic) never tears down a healthy stream.
  useEffect(() => {
    const onDeviceChange = () => {
      if (hasErrorRef.current) setRetryKey((k) => k + 1);
    };
    navigator.mediaDevices.addEventListener("devicechange", onDeviceChange);
    return () =>
      navigator.mediaDevices.removeEventListener("devicechange", onDeviceChange);
  }, []);

  useEffect(() => {
    if (paused || !deviceId) {
      if (!deviceId) setHasError(true);
      return;
    }
    let cancelled = false;
    let stream: MediaStream | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    setHasError(false);

    const start = async (attempt: number) => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: { exact: deviceId },
            // Ask for 720p30 so Chromium negotiates MJPEG instead of raw
            // YUYV. On Linux/V4L2 an unconstrained request lands on
            // YUYV 640x480, and one raw stream costs ~150 Mbps of a
            // 480 Mbps USB-2 bus — a third concurrent preview then fails
            // NotReadableError ("could not start video source"). These
            // cameras can't do 720p@30 uncompressed, so this forces the
            // ~10x lighter MJPEG, matching what macOS/AVFoundation picks
            // on its own. `ideal` degrades gracefully on cameras without
            // 720p (no OverconstrainedError).
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30 },
          },
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }
      } catch (err) {
        if (cancelled) return;
        const name = err instanceof DOMException ? err.name : "";
        if (attempt < MAX_RETRIES && TRANSIENT_ERRORS.has(name)) {
          // Exponential backoff: 300ms, 600ms, 1200ms, ...
          retryTimer = setTimeout(
            () => start(attempt + 1),
            BASE_DELAY_MS * 2 ** attempt
          );
        } else {
          setHasError(true);
        }
      }
    };
    start(0);

    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, [deviceId, paused, retryKey]);

  return { videoRef, hasError };
}
