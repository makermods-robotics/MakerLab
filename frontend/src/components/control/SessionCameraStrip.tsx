import React from "react";
import BackendCameraStream from "@/components/BackendCameraStream";
import { useApi } from "@/contexts/ApiContext";
import { useRobots } from "@/hooks/useRobots";
import { cn } from "@/lib/utils";

/**
 * The workspace view during a live session against a REMOTE host.
 *
 * On a remote host the operator is not in the room with the arm, so seeing the
 * cameras is the difference between doing the task and guessing. Those frames
 * cannot come from the remote's `/camera-preview/{index}` — the teleop/record
 * loop is holding those cv2 devices so it can publish them — so they arrive
 * over Portal and are re-served as MJPEG by the leader-bridge on this machine.
 * `BackendCameraStream source="session"` is what picks that path; each track is
 * keyed by the camera's NAME, not its index.
 *
 * Renders NOTHING when the active host is this computer: a local session is
 * unchanged from before remote hosts existed (the operator is sitting next to
 * the arm, and the teleop surface deliberately showed no tiles).
 */
const SessionCameraStrip: React.FC<{ className?: string }> = ({
  className,
}) => {
  const { isRemote, activeHost } = useApi();
  const { selectedRecord } = useRobots();
  const cameras = selectedRecord?.cameras ?? [];

  if (!isRemote || cameras.length === 0) return null;

  return (
    <div className={cn("space-y-1.5", className)}>
      <p className="eyebrow">Live from {activeHost.name}</p>
      <div className="flex flex-wrap gap-2">
        {cameras.map((cam) => (
          <figure
            key={cam.id}
            className="overflow-hidden rounded-lg border border-border bg-muted"
          >
            <BackendCameraStream
              cameraIndex={cam.camera_index ?? 0}
              source="session"
              cameraName={cam.name}
              className="h-32 w-44 object-cover"
            />
            <figcaption className="truncate border-t border-border px-2 py-1 text-[11px] text-muted-foreground">
              {cam.name}
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
};

export default SessionCameraStrip;
