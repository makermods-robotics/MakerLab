import React from "react";
import { VideoOff } from "lucide-react";
import { useCameraStream } from "@/hooks/useCameraStream";
import BackendCameraStream from "@/components/BackendCameraStream";
import { Card } from "@/components/ui/card";

interface CameraFeedProps {
  /** Browser deviceId to stream. Empty string renders the "no camera" state. */
  deviceId: string;
  /** cv2 index on the server — MJPEG fallback when there's no deviceId match
   * (headless deployment: the cameras are plugged into the server). */
  cameraIndex?: number;
  /** Optional caption shown under the feed. */
  label?: string;
}

/** Live camera feed: browser getUserMedia when a deviceId match exists,
 * otherwise the backend MJPEG stream for the camera's cv2 index. */
const CameraFeed: React.FC<CameraFeedProps> = ({
  deviceId,
  cameraIndex,
  label,
}) => {
  const { videoRef, hasError } = useCameraStream(deviceId, false);

  const showVideo = deviceId && !hasError;
  // BackendCameraStream owns its own failure/retry UI — no error latch here.
  const showMjpeg = !showVideo && !deviceId && cameraIndex !== undefined;

  return (
    <Card variant="flat" className="overflow-hidden">
      <div className="relative aspect-[4/3] bg-muted">
        {showVideo ? (
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="h-full w-full object-cover"
          />
        ) : showMjpeg ? (
          <BackendCameraStream
            cameraIndex={cameraIndex}
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="flex h-full w-full flex-col items-center justify-center">
            <VideoOff className="mb-2 h-8 w-8 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">
              {deviceId ? "Preview failed" : "No camera selected"}
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
