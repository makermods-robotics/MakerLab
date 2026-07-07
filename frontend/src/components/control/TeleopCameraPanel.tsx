import React, { useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useRobots } from "@/hooks/useRobots";
import CameraFeed from "./CameraFeed";

/**
 * Optional live camera panel for the teleoperation page. Off by default so we
 * never call getUserMedia just by landing on the page (same consent pattern as
 * the calibration camera toggle). Teleoperation opens no cv2 cameras, so the
 * browser can stream them directly while the arm runs.
 *
 * A strict mirror of the selected robot's configured cameras: one live feed per
 * camera on the robot record (e.g. "wrist_cam", "webcam"), stacked vertically.
 * If the robot has none configured it shows nothing — teleop never surfaces a
 * device that wasn't deliberately added to the robot.
 */
const TeleopCameraPanel: React.FC = () => {
  const [enabled, setEnabled] = useState(false);
  // Bumped by the retry button to remount the feeds (a fresh getUserMedia
  // attempt) — useful if a camera was unplugged and reconnected.
  const [reloadKey, setReloadKey] = useState(0);
  const { selectedRecord, isLoading: robotsLoading } = useRobots();

  // Feeds come solely from the robot's configured cameras; each carries a stored
  // browser device_id we stream directly. A configured camera whose device is
  // currently absent still shows (name + failed-preview placeholder), so the
  // user can tell it's expected but not detected.
  const configured = selectedRecord?.cameras ?? [];
  const feeds = configured.map((c) => ({
    key: c.id,
    name: c.name,
    deviceId: c.device_id,
    // MJPEG fallback for headless deployments: no browser deviceId match, but
    // the server knows the camera by its cv2 index.
    cameraIndex: c.camera_index,
  }));

  return (
    <Card variant="flat" className="flex h-full flex-col gap-4 p-4">
      <div className="flex items-center justify-between">
        <div>
          <div className="eyebrow">[ Video ]</div>
          <h2 className="mt-2 text-xl">Cameras</h2>
        </div>
        <div className="flex items-center gap-2">
          {enabled && feeds.length > 0 && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => setReloadKey((k) => k + 1)}
              className="h-9 w-9 flex-shrink-0 text-muted-foreground"
              title="Retry camera feeds (e.g. after reconnecting a camera)"
              aria-label="Retry camera feeds"
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          )}
          <Label htmlFor="teleop-camera-toggle" className="text-sm">
            {enabled ? "On" : "Off"}
          </Label>
          <Switch
            id="teleop-camera-toggle"
            checked={enabled}
            onCheckedChange={setEnabled}
          />
        </div>
      </div>

      {enabled ? (
        feeds.length > 0 ? (
          <div className="flex flex-col gap-3 overflow-y-auto">
            {feeds.map((feed) => (
              <CameraFeed
                key={`${feed.key}:${reloadKey}`}
                deviceId={feed.deviceId}
                cameraIndex={feed.cameraIndex}
                label={feed.name}
              />
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            {robotsLoading
              ? "Loading robot..."
              : "No cameras configured for this robot. Add them during calibration to see live feeds here."}
          </p>
        )
      ) : (
        <p className="text-sm text-muted-foreground">
          Turn on to watch your cameras while you teleoperate.
        </p>
      )}
    </Card>
  );
};

export default TeleopCameraPanel;
