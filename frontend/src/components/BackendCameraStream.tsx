import React, { useCallback, useEffect, useRef, useState } from "react";
import { VideoOff, RefreshCw } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { cn } from "@/lib/utils";

/**
 * Which pipe the frames come down. See `useStreamSource` below — this is the
 * whole remote-camera story in one prop.
 */
export type CameraStreamSource =
  /** Config-time preview: the host that owns the camera opens it directly. */
  | "host"
  /** Live teleop/record session: on a remote host, frames arrive over Portal
   *  and are re-served by the leader-bridge on THIS machine. */
  | "session";

interface BackendCameraStreamProps {
  /** cv2 camera index on the server (CameraConfig.camera_index). */
  cameraIndex: number;
  /** Stable device identity (CameraConfig.unique_id). When set, the server
   * re-anchors the index to this physical device before opening — its cv2
   * resolves indices against a startup snapshot that drifts from the fresh
   * enumeration after a replug, so index alone can open the wrong camera. */
  uniqueId?: string;
  /** Defaults to "host" — i.e. exactly the pre-remote behaviour. */
  source?: CameraStreamSource;
  /** The camera's MakerLab name (CameraConfig.name). Required for
   * `source="session"` on a remote host: the leader-bridge keys its Portal
   * video tracks by name, not by the server's cv2 index. */
  cameraName?: string;
  className?: string;
}

// Retry backoff for a failed stream: quick first attempts (a session just
// released the camera), settling to a slow poll (camera unplugged, server
// down). Preview failures are usually TRANSIENT — a recording session holding
// the cameras, a restarting server — so the tile must keep trying instead of
// latching into a dead "Preview failed" state until the modal is reopened.
const RETRY_BASE_MS = 2000;
const RETRY_MAX_MS = 12000;

/**
 * MJPEG `<img>` stream of a camera attached to the *server* machine
 * (GET /camera-preview/{index}), or — during a live session against a remote
 * host — of the Portal feed re-served by this machine's leader-bridge
 * (GET /leader-bridge/camera/{name}). See the `source` prop.
 *
 * Fallback for headless deployments (e.g. a Jetson on the LAN): the browser's
 * getUserMedia only sees the viewing machine's cameras, so a camera plugged
 * into the server has no matching deviceId — this streams it from the backend
 * instead. Owns its whole failure lifecycle: on stream error it asks the
 * endpoint WHY (the 409/503 detail — "recording is using the cameras" etc.),
 * shows that on the tile, and retries with capped backoff; clicking the tile
 * retries immediately. Parents should render it whenever a cameraIndex is
 * known and unmount it to pause/release (unmount cleanup clears the img src
 * so the browser drops the HTTP connection and the server releases the
 * shared capture).
 */
const BackendCameraStream: React.FC<BackendCameraStreamProps> = ({
  cameraIndex,
  uniqueId,
  source = "host",
  cameraName,
  className,
}) => {
  const { baseUrl, localBaseUrl, isRemote, fetchWithHeaders } = useApi();
  const imgRef = useRef<HTMLImageElement>(null);
  const retryTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const attemptRef = useRef(0);
  // Bumping remounts the <img> with a cache-busted URL — a clean retry.
  const [attempt, setAttempt] = useState(0);
  const [down, setDown] = useState(false);
  const [reason, setReason] = useState<string | null>(null);

  // ---- Where the frames come from --------------------------------------
  //
  // Exactly one condition moves the stream off the active host: a LIVE SESSION
  // against a REMOTE host. There, the remote's cameras are already open inside
  // the teleop/record loop, and its /camera-preview would contend with the
  // recorder for the device. The frames the operator needs have instead been
  // tee'd onto Portal and re-served as MJPEG by the leader-bridge running on
  // this laptop (docs/remote-portal/SPEC.md decision 3) — so we read them from
  // localBaseUrl, keyed by camera NAME (Portal's track id), not by cv2 index.
  //
  // Every other case — config-time previews, and everything on a single local
  // machine — keeps hitting {activeHost}/camera-preview/{index} verbatim.
  const viaLeaderBridge = source === "session" && isRemote && !!cameraName;
  const streamBase = viaLeaderBridge ? localBaseUrl : baseUrl;
  const streamPath = viaLeaderBridge
    ? `/leader-bridge/camera/${encodeURIComponent(cameraName as string)}`
    : `/camera-preview/${cameraIndex}`;
  // unique_id only means something to the cv2 preview endpoint.
  const uniqueIdQuery =
    !viaLeaderBridge && uniqueId
      ? `?unique_id=${encodeURIComponent(uniqueId)}`
      : "";
  const streamUrl = `${streamBase}${streamPath}`;

  // A new index, identity, or source is a new stream — forget the previous
  // one's failure state.
  useEffect(() => {
    attemptRef.current = 0;
    setAttempt(0);
    setDown(false);
    setReason(null);
  }, [cameraIndex, uniqueId, streamUrl]);

  useEffect(() => {
    const img = imgRef.current;
    return () => {
      if (retryTimer.current) clearTimeout(retryTimer.current);
      // Detaching an <img> doesn't reliably abort its in-flight request;
      // clearing src does, which is what lets the backend release the camera.
      if (img) img.src = "";
    };
  }, []);

  const scheduleRetry = useCallback(() => {
    if (retryTimer.current) clearTimeout(retryTimer.current);
    const delay = Math.min(
      RETRY_BASE_MS * 2 ** Math.min(attemptRef.current, 5),
      RETRY_MAX_MS
    );
    retryTimer.current = setTimeout(() => {
      attemptRef.current += 1;
      setAttempt((a) => a + 1);
      setDown(false);
    }, delay);
  }, []);

  const handleError = useCallback(() => {
    setDown(true);
    // The <img> can't see WHY the stream died — probe the endpoint once for
    // the status detail so the tile can say "recording is using the cameras"
    // instead of a generic failure. Best-effort: any probe error is itself
    // a reason ("server unreachable").
    fetchWithHeaders(`${streamUrl}${uniqueIdQuery}`, {
      method: "GET",
      headers: { Range: "bytes=0-0" },
    })
      .then(async (r) => {
        if (r.ok) {
          // Endpoint is fine again — the stream just dropped; retry sooner.
          r.body?.cancel?.();
          setReason(null);
          return;
        }
        const data = await r.json().catch(() => ({}));
        setReason(
          typeof data.detail === "string"
            ? data.detail
            : `camera preview unavailable (${r.status})`
        );
      })
      .catch(() =>
        setReason(
          viaLeaderBridge
            ? "leader-bridge unreachable"
            : "server unreachable",
        ),
      )
      .finally(scheduleRetry);
  }, [
    streamUrl,
    viaLeaderBridge,
    fetchWithHeaders,
    uniqueIdQuery,
    scheduleRetry,
  ]);

  const retryNow = useCallback(() => {
    if (retryTimer.current) clearTimeout(retryTimer.current);
    attemptRef.current += 1;
    setAttempt((a) => a + 1);
    setDown(false);
  }, []);

  if (down) {
    return (
      <button
        type="button"
        onClick={retryNow}
        className={cn(
          className,
          "flex cursor-pointer flex-col items-center justify-center gap-1 bg-muted text-muted-foreground"
        )}
        title="Click to retry now"
      >
        <VideoOff className="h-5 w-5" />
        <span className="px-1 text-center text-[10px] leading-tight">
          {reason ?? "Preview failed"}
        </span>
        <span className="flex items-center gap-1 text-[10px] text-muted-foreground/70">
          <RefreshCw className="h-3 w-3" /> retrying...
        </span>
      </button>
    );
  }

  return (
    <img
      key={attempt}
      ref={imgRef}
      src={`${streamUrl}?r=${attempt}${
        !viaLeaderBridge && uniqueId
          ? `&unique_id=${encodeURIComponent(uniqueId)}`
          : ""
      }`}
      onError={handleError}
      className={className}
      alt={viaLeaderBridge ? "Live camera over Portal" : "Server camera preview"}
    />
  );
};

export default BackendCameraStream;
