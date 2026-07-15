import React from "react";
import { VideoOff } from "lucide-react";
import BackendCameraStream from "@/components/BackendCameraStream";
import { Card } from "@/components/ui/card";

interface CameraFeedProps {
  /** cv2 index on the server — the live backend MJPEG feed at this index.
   * Undefined renders the "no camera" state. */
  cameraIndex?: number;
  /** Stable device identity (CameraConfig.unique_id); the server re-anchors
   * the index to this physical device before opening. */
  uniqueId?: string;
  /** Optional caption shown under the feed. */
  label?: string;
  /** Message for the "no camera" state (default "No camera selected") —
   * e.g. "Camera disconnected" when the device is verifiably absent. */
  emptyText?: string;
}

/** Live camera feed: the backend cv2 MJPEG stream for the camera's index.
 * Teleop opens no cv2 cameras (it drives the serial bus), so previewing the
 * server's feed while teleoperating does not contend — and it shows exactly the
 * camera the robot record maps to, with no browser deviceId fuzzy-match. */
const CameraFeed: React.FC<CameraFeedProps> = ({
  cameraIndex,
  uniqueId,
  label,
  emptyText,
}) => {
  // BackendCameraStream owns its own failure/retry UI — no error latch here.
  const showMjpeg = cameraIndex !== undefined;

  return (
    <Card variant="flat" className="overflow-hidden">
      <div className="relative aspect-[4/3] bg-muted">
        {showMjpeg ? (
          <BackendCameraStream
            cameraIndex={cameraIndex}
            uniqueId={uniqueId}
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="flex h-full w-full flex-col items-center justify-center">
            <VideoOff className="mb-2 h-8 w-8 text-muted-foreground" />
            <span className="px-2 text-center text-sm text-muted-foreground">
              {emptyText ?? "No camera selected"}
            </span>
          </div>
        )}
      </div>
      {label && (
        <div className="truncate border-t border-border p-2 font-mono text-sm text-muted-foreground">
          {label}
        </div>
      )}
    </Card>
  );
};

export default CameraFeed;
