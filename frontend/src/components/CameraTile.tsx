import React, { useEffect, useRef, useState } from "react";
import { VideoOff } from "lucide-react";
import BackendCameraStream from "@/components/BackendCameraStream";

interface CameraTileProps {
  /** cv2 index on the server — the live backend MJPEG feed at this index, i.e.
   * exactly what the recorder/rollout opens, independent of any browser deviceId
   * match. Undefined (nothing bound) renders the empty placeholder. */
  cameraIndex?: number;
  /** When true, render the paused placeholder instead of the feed. Unmounting
   * BackendCameraStream drops its HTTP connection so the server releases the
   * shared capture — that's how recording/inference start grabs the device. */
  paused?: boolean;
  /** Tile geometry + icon/text sizing. `md` = aspect-[4/3] (config/teleop),
   * `sm` = fixed w-32 h-24 thumbnail (inference). */
  size?: "sm" | "md";
  /** Text shown in the empty (no cameraIndex) placeholder. */
  emptyLabel?: string;
  /** Optional caption rendered under the tile inside a bordered card. */
  label?: string;
}

/**
 * The single camera-preview primitive: renders the backend cv2 MJPEG feed for a
 * camera index, or an idle/paused placeholder. Wraps the one guard
 * (`!paused && cameraIndex !== undefined ? feed : placeholder`) that every
 * surface (recording config, teleop, inference) previously duplicated.
 *
 * BackendCameraStream owns its own failure/retry UI — there's no error latch
 * here. Rendering it whenever a cameraIndex is known and unmounting it to
 * pause/release is what lets the server hand the shared capture to cv2.
 */
const CameraTile: React.FC<CameraTileProps> = ({
  cameraIndex,
  paused = false,
  size = "md",
  emptyLabel = "No camera selected",
  label,
}) => {
  // Only hold a backend stream while the tile is actually on-screen. Each open
  // stream is a server-side cv2 capture + HTTP connection; N configured cameras
  // in the recording config panel would otherwise all stream at once (worst on
  // a headless Jetson). We observe the tile's root and unmount
  // BackendCameraStream when it scrolls out of view — its unmount cleanup clears
  // the <img src>, dropping the HTTP connection so the server releases the
  // shared capture.
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [onScreen, setOnScreen] = useState(false);

  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    // If IntersectionObserver is unavailable (very old browser / SSR), fail
    // open and stream so we never render a permanently blank tile.
    if (typeof IntersectionObserver === "undefined") {
      setOnScreen(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        // Single observed element; the callback also fires once on observe(),
        // so a tile that mounts already visible reports on-screen without any
        // scrolling.
        setOnScreen(entries[0]?.isIntersecting ?? false);
      },
      // Start streaming slightly before the tile fully enters the viewport to
      // avoid a blank flash. Default root (viewport) is correct even inside a
      // scrollable modal, since the dialog scrolls within the viewport.
      { root: null, rootMargin: "200px", threshold: 0 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // paused still forces the stream off regardless of visibility (record-start
  // release / inference submitting must unmount the stream). Effective "show
  // feed" = not paused AND a camera is bound AND the tile is on-screen.
  const showMjpeg = !paused && cameraIndex !== undefined && onScreen;

  const boxClass =
    size === "sm"
      ? "w-32 h-24 bg-gray-800 rounded border border-gray-700 relative overflow-hidden"
      : "aspect-[4/3] bg-gray-800 relative";
  const iconClass = size === "sm" ? "w-5 h-5 mb-1" : "w-8 h-8 mb-2";
  const textClass = size === "sm" ? "text-[10px]" : "text-sm";
  const streamClass =
    size === "sm"
      ? "w-full h-full object-cover bg-black"
      : "w-full h-full object-cover";

  // Placeholder text: a bound-but-off-screen tile shows a subtle idle state
  // (same box dimensions, so no layout jump when it starts/stops streaming).
  const placeholderText = paused
    ? "Preview paused"
    : cameraIndex === undefined
      ? emptyLabel
      : "Scroll into view";

  const tile = (
    <div ref={rootRef} className={boxClass}>
      {showMjpeg ? (
        <BackendCameraStream cameraIndex={cameraIndex} className={streamClass} />
      ) : (
        <div className="w-full h-full flex flex-col items-center justify-center">
          <VideoOff className={`text-gray-500 ${iconClass}`} />
          <span className={`text-gray-500 ${textClass}`}>
            {placeholderText}
          </span>
        </div>
      )}
    </div>
  );

  if (label === undefined) return tile;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
      {tile}
      <div className="p-2 text-sm text-gray-300 truncate border-t border-gray-800">
        {label}
      </div>
    </div>
  );
};

export default CameraTile;
