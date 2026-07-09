import React, { useCallback, useEffect, useRef, useState } from "react";
import { VideoOff, RefreshCw } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";

interface BackendCameraSnapshotProps {
  /** Camera address for GET /camera-snapshot/{camera_id}: the stable hardware
   * unique_id (primary), or a purely-numeric string meaning a raw cv2 index —
   * same addressing as BackendCameraStream. */
  cameraId: string;
  className?: string;
  /** How often to refresh the still, in ms. */
  intervalMs?: number;
}

/** Refresh cadence. Each refresh reads the backend's lingering capture (or
 * borrows the device for one frame, funnel-serialized), so the floor is
 * bounded by open cost, not bandwidth; 8s suits passive tiles while still
 * catching a re-aimed or unplugged camera quickly. */
const DEFAULT_INTERVAL_MS = 8000;

/**
 * Polled-still sibling of BackendCameraStream, used by every preview surface.
 * A standing MJPEG stream per tile holds a server-side capture per camera for
 * the tile's whole lifetime and kept losing to session-teardown collisions;
 * stills sidestep that: the server borrows the device (or reads its lingering
 * capture) per poll, so tile count is unlimited and the browser is never a
 * party to camera-session lifecycle. Trade-off is a polled image — pass a
 * ~1000ms interval where responsiveness matters (wave-and-identify pickers,
 * teleop monitoring).
 *
 * Failure UI mirrors BackendCameraStream: on a failed poll, show the
 * endpoint's 409/503 detail on the tile; polling continues, so the tile
 * recovers by itself when the camera returns. The error probe here is cheap
 * (one still, server-cached) — unlike the stream's probe it never opens a
 * second standing capture.
 */
const BackendCameraSnapshot: React.FC<BackendCameraSnapshotProps> = ({
  cameraId,
  className,
  intervalMs = DEFAULT_INTERVAL_MS,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [src, setSrc] = useState<string | null>(null);
  const [reason, setReason] = useState<string | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const poll = useCallback(async () => {
    try {
      const r = await fetchWithHeaders(
        `${baseUrl}/camera-snapshot/${encodeURIComponent(cameraId)}`
      );
      if (!r.ok) {
        const data = await r.json().catch(() => ({}));
        setReason(
          typeof data.detail === "string"
            ? data.detail
            : `snapshot unavailable (${r.status})`
        );
        return;
      }
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = url;
      setSrc(url);
      setReason(null);
    } catch {
      setReason("server unreachable");
    }
  }, [baseUrl, fetchWithHeaders, cameraId]);

  useEffect(() => {
    // New camera id: drop the previous device's image immediately so a stale
    // frame is never shown under the wrong identity.
    setSrc(null);
    setReason(null);
    poll();
    const timer = setInterval(poll, intervalMs);
    return () => {
      clearInterval(timer);
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, [poll, intervalMs]);

  if (reason !== null && src === null) {
    // Never had a frame: show the failure state (polling continues silently).
    return (
      <div
        className={`${className ?? ""} flex flex-col items-center justify-center gap-1 bg-gray-800 text-gray-500`}
      >
        <VideoOff className="w-5 h-5" />
        <span className="text-[10px] leading-tight text-center px-1">{reason}</span>
        <span className="flex items-center gap-1 text-[10px] text-gray-600">
          <RefreshCw className="w-3 h-3" /> retrying…
        </span>
      </div>
    );
  }

  if (src === null) {
    return (
      <div
        className={`${className ?? ""} flex items-center justify-center bg-gray-800 text-gray-600 text-[10px]`}
      >
        loading…
      </div>
    );
  }

  return <img src={src} className={className} alt="Server camera snapshot" />;
};

export default BackendCameraSnapshot;
