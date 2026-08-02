import React from "react";
import { VideoOff } from "lucide-react";
import BackendCameraStream, {
  CameraStreamSource,
} from "@/components/BackendCameraStream";

interface CameraFeedProps {
  /** cv2 index on the server. Undefined renders the "no camera" state. */
  cameraIndex?: number;
  /** Stable device identity, so the server re-anchors the index across replugs. */
  uniqueId?: string;
  /** Optional caption shown under the feed. */
  label?: string;
  /** "session" routes a remote host's feed through this machine's
   * leader-bridge (the Portal video path). Defaults to "host". */
  source?: CameraStreamSource;
  /** The camera's MakerLab name — the leader-bridge's track key. */
  cameraName?: string;
}

/** Live camera feed streamed from the *server* by cv2 index.
 *
 * Previously bound to a browser deviceId via getUserMedia. That deviceId was
 * matched to a cv2 index by localizedName, so two cameras of the same model
 * paired arbitrarily and a feed could be captioned with the other camera's
 * name. Streaming by index removes the ambiguity — the caption and the footage
 * now come from the same identity. Teleoperation opens no cv2 cameras, so
 * these previews never contend with it. */
const CameraFeed: React.FC<CameraFeedProps> = ({
  cameraIndex,
  uniqueId,
  label,
  source,
  cameraName,
}) => {
  return (
    <div className="bg-card rounded-lg border border-border overflow-hidden">
      <div className="aspect-[4/3] bg-muted relative">
        {cameraIndex !== undefined ? (
          <BackendCameraStream
            cameraIndex={cameraIndex}
            uniqueId={uniqueId}
            source={source}
            cameraName={cameraName}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center">
            <VideoOff className="w-8 h-8 text-muted-foreground mb-2" />
            <span className="text-muted-foreground text-sm">
              No camera selected
            </span>
          </div>
        )}
      </div>
      {label && (
        <div className="p-2 text-sm text-muted-foreground truncate border-t border-border">
          {label}
        </div>
      )}
    </div>
  );
};

export default CameraFeed;
