import React, { useEffect, useState } from "react";
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
  const [mjpegError, setMjpegError] = useState(false);
  // A new index is a new stream — forget the previous one's failure.
  useEffect(() => setMjpegError(false), [cameraIndex]);

  const showVideo = deviceId && !hasError;
  const showMjpeg =
    !showVideo && !deviceId && cameraIndex !== undefined && !mjpegError;

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
            onError={() => setMjpegError(true)}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center">
            <VideoOff className="w-8 h-8 text-gray-500 mb-2" />
            <span className="text-gray-500 text-sm">
              {deviceId || mjpegError ? "Preview failed" : "No camera selected"}
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
