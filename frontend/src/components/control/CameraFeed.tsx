import React from "react";
import { VideoOff } from "lucide-react";
import BackendCameraStream from "@/components/BackendCameraStream";

interface CameraFeedProps {
  /** cv2 index on the server — the live backend MJPEG feed at this index.
   * Undefined renders the "no camera" state. */
  cameraIndex?: number;
  /** Optional caption shown under the feed. */
  label?: string;
}

/** Live camera feed: the backend cv2 MJPEG stream for the camera's index.
 * Teleop opens no cv2 cameras (it drives the serial bus), so previewing the
 * server's feed while teleoperating does not contend — and it shows exactly the
 * camera the robot record maps to, with no browser deviceId fuzzy-match. */
const CameraFeed: React.FC<CameraFeedProps> = ({ cameraIndex, label }) => {
  // BackendCameraStream owns its own failure/retry UI — no error latch here.
  const showMjpeg = cameraIndex !== undefined;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
      <div className="aspect-[4/3] bg-gray-800 relative">
        {showMjpeg ? (
          <BackendCameraStream
            cameraIndex={cameraIndex}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center">
            <VideoOff className="w-8 h-8 text-gray-500 mb-2" />
            <span className="text-gray-500 text-sm">No camera selected</span>
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
