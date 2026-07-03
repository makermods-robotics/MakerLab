import React from "react";
import { VideoOff } from "lucide-react";
import { useCameraStream } from "@/hooks/useCameraStream";
import BackendCameraStream from "@/components/BackendCameraStream";

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
    <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
      <div className="aspect-[4/3] bg-gray-800 relative">
        {showVideo ? (
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="w-full h-full object-cover"
          />
        ) : showMjpeg ? (
          <BackendCameraStream
            cameraIndex={cameraIndex}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center">
            <VideoOff className="w-8 h-8 text-gray-500 mb-2" />
            <span className="text-gray-500 text-sm">
              {deviceId ? "Preview failed" : "No camera selected"}
            </span>
          </div>
        )}
      </div>
      {label && (
        <div className="p-2 text-sm text-gray-300 truncate border-t border-gray-800">
          {label}
        </div>
      )}
    </div>
  );
};

export default CameraFeed;
