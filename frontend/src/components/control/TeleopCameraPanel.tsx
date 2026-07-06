import React, { useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useRobots } from "@/hooks/useRobots";
import CameraFeed from "./CameraFeed";

/**
 * Optional live camera panel for the teleoperation page. Off by default (same
 * consent pattern as the calibration camera toggle). Teleoperation opens no cv2
 * cameras — it drives the serial bus — so previewing the server's camera feeds
 * while the arm runs does not contend for the devices.
 *
 * Each feed is the live backend cv2 MJPEG stream at the camera's index, so it
 * shows exactly the camera the robot record maps to (no browser deviceId
 * fuzzy-match). A strict mirror of the selected robot's configured cameras: one
 * feed per camera on the record (e.g. "wrist_cam", "webcam"), stacked
 * vertically. If the robot has none configured it shows nothing — teleop never
 * surfaces a device that wasn't deliberately added to the robot.
 */
const TeleopCameraPanel: React.FC = () => {
  const [enabled, setEnabled] = useState(false);
  // Bumped by the retry button to remount the feeds (a fresh backend-stream
  // attempt) — useful if a camera was unplugged and reconnected.
  const [reloadKey, setReloadKey] = useState(0);
  const { selectedRecord, isLoading: robotsLoading } = useRobots();

  // Feeds come solely from the robot's configured cameras; each is streamed
  // from the server by its cv2 index. A configured camera the server can't open
  // still shows (name + BackendCameraStream's retry placeholder), so the user
  // can tell it's expected but not detected.
  const configured = selectedRecord?.cameras ?? [];
  const feeds = configured.map((c) => ({
    key: c.id,
    name: c.name,
    cameraIndex: c.camera_index,
  }));

  return (
    <div className="bg-gray-900 rounded-lg p-4 flex flex-col gap-4 h-full">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-medium text-gray-200">Cameras</h2>
        <div className="flex items-center gap-2">
          {enabled && feeds.length > 0 && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => setReloadKey((k) => k + 1)}
              className="h-9 w-9 text-gray-400 hover:text-white flex-shrink-0"
              title="Retry camera feeds (e.g. after reconnecting a camera)"
              aria-label="Retry camera feeds"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          )}
          <Label htmlFor="teleop-camera-toggle" className="text-sm text-gray-400">
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
                cameraIndex={feed.cameraIndex}
                label={feed.name}
              />
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-500">
            {robotsLoading
              ? "Loading robot..."
              : "No cameras configured for this robot. Add them during calibration to see live feeds here."}
          </p>
        )
      ) : (
        <p className="text-sm text-gray-500">
          Turn on to watch your cameras while you teleoperate.
        </p>
      )}
    </div>
  );
};

export default TeleopCameraPanel;
