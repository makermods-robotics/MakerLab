import React, { useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useRobots } from "@/hooks/useRobots";
import { useAvailableCameras } from "@/hooks/useAvailableCameras";
import CameraTile from "@/components/CameraTile";

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
  const {
    cameras: availableCameras,
    isLoading: camerasLoading,
    refresh: refreshCameras,
  } = useAvailableCameras({ enabled, matchBrowserDevices: false });

  // Feeds come solely from the robot's configured cameras; each is streamed
  // from the server by its stable unique_id (the backend re-resolves the cv2
  // index per open, so the feed follows the physical device across USB
  // reshuffles), falling back to the saved cv2 index for legacy entries. A
  // configured camera the server can't open still shows (name +
  // BackendCameraStream's retry placeholder), so the user can tell it's
  // expected but not detected.
  const configured = selectedRecord?.cameras ?? [];
  const feeds = configured.map((c) => {
    const liveById = c.unique_id
      ? availableCameras.find((cam) => cam.uniqueId === c.unique_id)
      : undefined;
    const idProblem = enabled && !camerasLoading && !!c.unique_id && !liveById;
    const legacyFallback = enabled && !camerasLoading && !c.unique_id;
    return {
      key: c.id,
      name: c.name,
      cameraId: idProblem ? undefined : c.unique_id,
      cameraIndex: c.unique_id ? undefined : c.camera_index,
      issue: idProblem
        ? "Saved camera ID is not connected. Re-select and save this camera in calibration."
        : legacyFallback
          ? "No stable camera ID saved; using legacy index fallback. Re-select and save this camera in calibration."
          : null,
      emptyLabel: idProblem ? "Camera ID not found" : "No camera selected",
    };
  });

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
              onClick={() => {
                refreshCameras();
                setReloadKey((k) => k + 1);
              }}
              className="h-9 w-9 text-gray-400 hover:text-white flex-shrink-0"
              title="Retry camera feeds (e.g. after reconnecting a camera)"
              aria-label="Retry camera feeds"
            >
              <RefreshCw className={`w-4 h-4 ${camerasLoading ? "animate-spin" : ""}`} />
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
              <div key={`${feed.key}:${reloadKey}`} className="space-y-1.5">
                {/* ~1s snapshot cadence: operators glance at these while
                    driving, so the slow passive-tile default would mislead;
                    1s polls ride the backend's lingering captures — no
                    standing streams, no extra device opens. */}
                <CameraTile
                  size="md"
                  cameraId={feed.cameraId}
                  cameraIndex={feed.cameraIndex}
                  emptyLabel={feed.emptyLabel}
                  label={feed.name}
                  snapshotIntervalMs={1000}
                />
                {feed.issue && (
                  <p className="rounded border border-amber-700 bg-amber-900/30 px-2 py-1 text-xs text-amber-100">
                    {feed.issue}
                  </p>
                )}
              </div>
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
