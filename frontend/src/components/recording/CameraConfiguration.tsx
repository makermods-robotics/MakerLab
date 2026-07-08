import React, { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Camera, Plus, Trash2, RefreshCw, ChevronRight } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useToast } from "@/hooks/use-toast";
import {
  useAvailableCameras,
  AvailableCamera,
} from "@/hooks/useAvailableCameras";
import { resyncCameras } from "@/lib/resyncCameras";
import { defaultFourccForCamera } from "@/lib/cameraDefaults";
import CameraTile from "@/components/CameraTile";

/** Stable selection/address key for an enumerated camera: the hardware
 * unique_id (primary), falling back to the stringified cv2 index — the same
 * numeric fallback lane the /camera-preview endpoint accepts for
 * platforms/rows without stable ids. */
const cameraKey = (cam: AvailableCamera) => cam.uniqueId ?? String(cam.index);

// Sentinels distinguish "leave unset" (auto-detect / platform default) from an
// explicit choice. Radix Select disallows an empty-string value, so we map these
// to `undefined` on the CameraConfig.
const FOURCC_AUTO = "__auto__";
const BACKEND_DEFAULT = "__default__";
const FOURCC_OPTIONS = ["MJPG", "YUYV", "I420", "NV12", "H264", "MP4V"];
// Mirrors lerobot's Cv2Backends enum names.
const BACKEND_OPTIONS = [
  "ANY",
  "V4L2",
  "DSHOW",
  "PVAPI",
  "ANDROID",
  "AVFOUNDATION",
  "MSMF",
];

export interface CameraConfig {
  id: string;
  name: string;
  type: string;
  camera_index?: number; // cv2 index — what the recorder opens (fallback / back-compat)
  // Stable hardware id (AVFoundation uniqueID on macOS). cv2 indices reshuffle
  // on USB hotplug, and identical camera models can't be told apart by name, so
  // the backend re-resolves the index from this at record start. Absent on
  // platforms that don't expose a stable id — then camera_index is used as-is.
  unique_id?: string;
  device_id: string; // Browser deviceId matched to the cv2 index by AVFoundation localizedName
  width: number;
  height: number;
  fps?: number;
  fourcc?: string; // 4-char OpenCV pixel format (e.g. "MJPG"); undefined = auto-detect
  backend?: string; // Cv2Backends name (e.g. "AVFOUNDATION"); undefined = platform default
}

interface CameraConfigurationProps {
  cameras: CameraConfig[];
  onCamerasChange: (cameras: CameraConfig[]) => void;
  releaseStreamsRef?: React.MutableRefObject<(() => void) | null>; // Ref to expose stream release function
}

const CameraConfiguration: React.FC<CameraConfigurationProps> = ({
  cameras,
  onCamerasChange,
  releaseStreamsRef,
}) => {
  const { toast } = useToast();

  // Recording start pauses the previews via releaseStreamsRef; gate camera
  // enumeration on the same flag so the getUserMedia/devicechange probing fully
  // stops before cv2 opens the devices. Otherwise the enumeration probe can
  // keep index 0 open and starve the recorder (OpenCVCamera(0) actual_fps=5.0).
  const [streamsPaused, setStreamsPaused] = useState(false);

  const {
    cameras: availableCameras,
    isLoading: isLoadingCameras,
    refresh: refreshCameras,
  } = useAvailableCameras({ enabled: !streamsPaused });
  // The dropdown's value: the picked camera's stable key (unique_id, or the
  // stringified index fallback) — id-first, so the selection survives an index
  // reshuffle between pick and add.
  const [selectedCameraKey, setSelectedCameraKey] = useState<string>("");
  const [cameraName, setCameraName] = useState("");
  // True when a seeded camera can't be safely bound to a stable identity because
  // two live cameras share a display name (see resyncCameras rule 2). Drives the
  // "re-select to bind identity" nudge below.
  const [needsReselect, setNeedsReselect] = useState(false);

  // The camera currently picked in the dropdown (not yet added). Drives the
  // immediate live preview shown before the camera is named.
  const selectedCamera = selectedCameraKey
    ? availableCameras.find((cam) => cameraKey(cam) === selectedCameraKey)
    : undefined;

  // cv2's AVFoundation order is uniqueID-sorted, so plugging/unplugging a
  // device between sessions shifts indices. Re-resolve each seeded camera's
  // cv2 index against the live enumeration — but FAIL-SAFE on identity: a
  // unique_id entry is only re-indexed (never rewritten), and a legacy entry is
  // backfilled ONLY on an unambiguous match. Two same-model cameras share a
  // display name, so a device_id/name match can't tell them apart; blindly
  // baking a unique_id in from that match is exactly what flips front/wrist on
  // disk. See lib/resyncCameras.
  useEffect(() => {
    const { cameras: refreshed, changed, needsReselect: reselect } =
      resyncCameras(cameras, availableCameras);
    if (changed) onCamerasChange(refreshed);
    setNeedsReselect(reselect);
    // We deliberately don't depend on `cameras`/`onCamerasChange` to avoid
    // re-running every keystroke in the camera-name input — re-syncing only
    // when the available-cameras list itself changes is sufficient.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableCameras]);

  const addCamera = () => {
    if (!selectedCameraKey || !cameraName.trim()) {
      toast({
        title: "Missing Information",
        description: "Please select a camera and provide a name.",
        variant: "destructive",
      });
      return;
    }

    const selectedCamera = availableCameras.find(
      (cam) => cameraKey(cam) === selectedCameraKey
    );

    if (!selectedCamera) {
      toast({
        title: "Invalid Camera",
        description: "Selected camera is not available.",
        variant: "destructive",
      });
      return;
    }

    // Block duplicates by stable unique_id first (THE camera identity), then
    // browser deviceId, then cv2 index — a stale camera_index in a seeded
    // camera can otherwise let the same physical device sneak in under a
    // different index.
    const isDuplicate = cameras.some(
      (cam) =>
        (selectedCamera.uniqueId && cam.unique_id === selectedCamera.uniqueId) ||
        (selectedCamera.deviceId && cam.device_id === selectedCamera.deviceId) ||
        cam.camera_index === selectedCamera.index,
    );
    if (isDuplicate) {
      toast({
        title: "Camera Already Added",
        description: "This camera is already in the configuration.",
        variant: "destructive",
      });
      return;
    }

    const newCamera: CameraConfig = {
      id: `camera_${Date.now()}`,
      name: cameraName.trim(),
      type: "opencv",
      // unique_id is the canonical identity; camera_index is stored as the
      // legacy/fallback field (the recorder re-resolves from unique_id).
      camera_index: selectedCamera.index,
      unique_id: selectedCamera.uniqueId,
      device_id: selectedCamera.deviceId,
      width: 640,
      height: 480,
      fps: 30,
      // Default USB/external cameras to MJPG (compressed → bandwidth-safe when
      // two identical cameras share a bus); built-in cameras stay auto-detect.
      // User-overridable via the FOURCC dropdown below.
      fourcc: defaultFourccForCamera(selectedCamera),
    };

    onCamerasChange([...cameras, newCamera]);

    setSelectedCameraKey("");
    setCameraName("");

    toast({
      title: "Camera Added",
      description: `${newCamera.name} has been added to the configuration.`,
    });
  };

  const removeCamera = (cameraId: string) => {
    onCamerasChange(cameras.filter((cam) => cam.id !== cameraId));
    toast({
      title: "Camera Removed",
      description: "Camera has been removed from the configuration.",
    });
  };

  const updateCamera = (cameraId: string, updates: Partial<CameraConfig>) => {
    onCamerasChange(
      cameras.map((cam) =>
        cam.id === cameraId ? { ...cam, ...updates } : cam
      )
    );
  };

  // When the recording session is starting, the parent calls
  // releaseStreamsRef.current() to pause every preview: unmounting each backend
  // MJPEG <img> drops its HTTP connection so the server releases the shared
  // capture, letting cv2.VideoCapture grab the camera exclusively for recording.
  // Flipping streamsPaused also disables useAvailableCameras above (which still
  // probes via getUserMedia/enumerateDevices) — see its comment.
  const releaseAllCameraStreams = useCallback(() => {
    setStreamsPaused(true);
  }, []);

  useEffect(() => {
    if (releaseStreamsRef) {
      releaseStreamsRef.current = releaseAllCameraStreams;
    }
  }, [releaseStreamsRef, releaseAllCameraStreams]);


  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">
        Camera Configuration
      </h3>

      {/* Add Camera Section */}
      <div className="bg-gray-800/50 rounded-lg p-4 space-y-4">
        <h4 className="text-md font-medium text-gray-300">Add Camera</h4>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-sm font-medium text-gray-300">
              Available Cameras
            </Label>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => refreshCameras()}
              disabled={isLoadingCameras}
              className="h-6 w-6 text-gray-400 hover:text-white"
              title="Rescan for cameras (e.g. after plugging in a new USB camera)"
              aria-label="Rescan for cameras"
            >
              <RefreshCw
                className={`w-3.5 h-3.5 ${isLoadingCameras ? "animate-spin" : ""}`}
              />
            </Button>
          </div>
          <Select
            value={selectedCameraKey}
            onValueChange={setSelectedCameraKey}
            disabled={isLoadingCameras}
          >
            <SelectTrigger className="bg-gray-800 border-gray-700 text-white">
              <SelectValue
                placeholder={
                  isLoadingCameras ? "Loading cameras..." : "Select camera"
                }
              />
            </SelectTrigger>
            <SelectContent className="bg-gray-800 border-gray-700">
              {availableCameras.map((camera) => {
                // Identity first: unique_id decides "already added"; deviceId
                // and index are fallbacks for rows/entries without stable ids.
                const alreadyAdded = cameras.some(
                  (cam) =>
                    (camera.uniqueId && cam.unique_id === camera.uniqueId) ||
                    (camera.deviceId && cam.device_id === camera.deviceId) ||
                    cam.camera_index === camera.index,
                );
                return (
                  <SelectItem
                    key={cameraKey(camera)}
                    value={cameraKey(camera)}
                    className="text-white hover:bg-gray-700"
                    disabled={!camera.available || alreadyAdded}
                  >
                    <div className="flex flex-col">
                      <span className="font-medium">{camera.name}</span>
                      <span className="text-xs text-gray-400">
                        {camera.uniqueId
                          ? `ID …${camera.uniqueId.slice(-7)}`
                          : `Index ${camera.index}`}
                        {alreadyAdded && " · already added"}
                      </span>
                    </div>
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
        </div>

        {/* Live preview appears as soon as a camera is selected; naming +
            confirmation happens alongside it. This is the recorder's own view
            (the backend cv2 feed, id-addressed so it follows the physical
            device), so what you preview is exactly what records — no browser
            deviceId fuzzy-match. */}
        {selectedCamera && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
              <CameraTile
                size="md"
                cameraId={cameraKey(selectedCamera)}
                paused={streamsPaused}
                emptyLabel="No camera selected"
              />
              <div className="border-t border-gray-700 px-2 py-1.5">
                <span className="text-[11px] text-gray-400 truncate">
                  Recorder's view —{" "}
                  {selectedCamera.uniqueId
                    ? `ID …${selectedCamera.uniqueId.slice(-7)}`
                    : `index ${selectedCamera.index}`}{" "}
                  (what actually records)
                </span>
              </div>
            </div>

            <div className="flex flex-col justify-center gap-4">
              <div className="space-y-2">
                <Label className="text-sm font-medium text-gray-300">
                  Camera Name
                </Label>
                <Input
                  value={cameraName}
                  onChange={(e) => setCameraName(e.target.value)}
                  placeholder="e.g., workspace_cam"
                  className="bg-gray-800 border-gray-700 text-white"
                />
              </div>

              <Button
                onClick={addCamera}
                className="bg-blue-500 hover:bg-blue-600 text-white"
                disabled={!selectedCameraKey || !cameraName.trim()}
              >
                <Plus className="w-4 h-4 mr-2" />
                Add Camera
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Configured Cameras */}
      {cameras.length > 0 && (
        <div className="space-y-4">
          <h4 className="text-md font-medium text-gray-300">
            Configured Cameras ({cameras.length})
          </h4>

          {needsReselect && (
            <div className="rounded-md border border-amber-700 bg-amber-900/40 px-3 py-2 text-xs text-amber-100">
              Two connected cameras share the same name, so a saved camera
              couldn't be automatically matched to a specific device. Re-select
              each camera above (while watching its live preview) to bind it to
              the correct one.
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-4">
            {cameras.map((camera) => (
              <CameraPreview
                key={camera.id}
                camera={camera}
                paused={streamsPaused}
                onRemove={() => removeCamera(camera.id)}
                onUpdate={(updates) => updateCamera(camera.id, updates)}
              />
            ))}
          </div>
        </div>
      )}

      {cameras.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          <Camera className="w-12 h-12 mx-auto mb-4 text-gray-600" />
          <p>No cameras configured. Add a camera to get started.</p>
        </div>
      )}
    </div>
  );
};

interface CameraPreviewProps {
  camera: CameraConfig;
  paused: boolean;
  onRemove: () => void;
  onUpdate: (updates: Partial<CameraConfig>) => void;
}

const CameraPreview: React.FC<CameraPreviewProps> = ({
  camera,
  paused,
  onRemove,
  onUpdate,
}) => {
  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
      {/* id-first: the tile follows the physical device across index
          reshuffles; camera_index is the legacy lane for entries that predate
          unique_id (or platforms without stable ids). */}
      <CameraTile
        size="md"
        cameraId={camera.unique_id}
        cameraIndex={camera.camera_index}
        paused={paused}
        emptyLabel="No camera selected"
      />

      {/* Camera Info */}
      <div className="p-3 space-y-2">
        <div className="flex items-center justify-between">
          <h5 className="font-medium text-white truncate">{camera.name}</h5>
          <Button
            onClick={onRemove}
            size="sm"
            variant="ghost"
            className="text-red-400 hover:text-red-300 hover:bg-red-900/20 p-1"
            aria-label="Remove camera"
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>

        <Collapsible>
          <CollapsibleTrigger className="group flex items-center gap-1.5 text-xs font-medium text-gray-300 hover:text-white transition-colors">
            <ChevronRight className="w-3.5 h-3.5 transition-transform group-data-[state=open]:rotate-90" />
            Configuration
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-2 space-y-2">
            <div className="grid grid-cols-1 gap-2 text-xs text-gray-400">
              <div className="flex items-center gap-2">
                <span className="w-16">Resolution:</span>
                <div className="flex items-center gap-1">
                  <NumberInput
                    value={camera.width}
                    onChange={(v) => {
                      if (v !== undefined) onUpdate({ width: v });
                    }}
                    className="bg-gray-800 border-gray-700 text-white text-xs h-6 px-2 w-16"
                    min="320"
                    max="1920"
                  />
                  <span className="flex items-center">×</span>
                  <NumberInput
                    value={camera.height}
                    onChange={(v) => {
                      if (v !== undefined) onUpdate({ height: v });
                    }}
                    className="bg-gray-800 border-gray-700 text-white text-xs h-6 px-2 w-16"
                    min="240"
                    max="1080"
                  />
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16">FPS:</span>
                <NumberInput
                  value={camera.fps ?? 30}
                  onChange={(v) => {
                    if (v !== undefined) onUpdate({ fps: v });
                  }}
                  className="bg-gray-800 border-gray-700 text-white text-xs h-6 px-2 w-16"
                  min="10"
                  max="60"
                />
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16">FOURCC:</span>
                <Select
                  value={camera.fourcc ?? FOURCC_AUTO}
                  onValueChange={(v) =>
                    onUpdate({ fourcc: v === FOURCC_AUTO ? undefined : v })
                  }
                >
                  <SelectTrigger className="bg-gray-800 border-gray-700 text-white text-xs h-6 px-2 w-28">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-800 border-gray-700">
                    <SelectItem
                      value={FOURCC_AUTO}
                      className="text-white hover:bg-gray-700 text-xs"
                    >
                      Auto
                    </SelectItem>
                    {FOURCC_OPTIONS.map((code) => (
                      <SelectItem
                        key={code}
                        value={code}
                        className="text-white hover:bg-gray-700 text-xs"
                      >
                        {code === "MJPG" ? "MJPG (USB)" : code}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16">Backend:</span>
                <Select
                  value={camera.backend ?? BACKEND_DEFAULT}
                  onValueChange={(v) =>
                    onUpdate({ backend: v === BACKEND_DEFAULT ? undefined : v })
                  }
                >
                  <SelectTrigger className="bg-gray-800 border-gray-700 text-white text-xs h-6 px-2 w-28">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-800 border-gray-700">
                    <SelectItem
                      value={BACKEND_DEFAULT}
                      className="text-white hover:bg-gray-700 text-xs"
                    >
                      Default
                    </SelectItem>
                    {BACKEND_OPTIONS.map((name) => (
                      <SelectItem
                        key={name}
                        value={name}
                        className="text-white hover:bg-gray-700 text-xs"
                      >
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <p className="text-[10px] text-gray-500 leading-tight">
                Overriding the backend can reorder camera indices on macOS.
              </p>
            </div>
            <div className="text-xs text-gray-500">
              Type: {camera.type} |{" "}
              {camera.unique_id
                ? `ID: …${camera.unique_id.slice(-7)}`
                : `Device: ${camera.device_id?.substring(0, 10)}...`}
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>
    </div>
  );
};

export default CameraConfiguration;
