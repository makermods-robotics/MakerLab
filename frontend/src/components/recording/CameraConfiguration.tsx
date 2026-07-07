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
import { Camera, Plus, Trash2, VideoOff, RefreshCw, ChevronRight } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useToast } from "@/hooks/use-toast";
import { useAvailableCameras } from "@/hooks/useAvailableCameras";
import { useCameraStream } from "@/hooks/useCameraStream";
import BackendCameraStream from "@/components/BackendCameraStream";

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
  camera_index?: number; // cv2 index — what the recorder opens
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

  const {
    cameras: availableCameras,
    isLoading: isLoadingCameras,
    refresh: refreshCameras,
  } = useAvailableCameras();
  const [selectedCameraIndex, setSelectedCameraIndex] = useState<string>("");
  const [cameraName, setCameraName] = useState("");

  // The camera currently picked in the dropdown (not yet added). Drives the
  // immediate live preview shown before the camera is named.
  const selectedCamera = selectedCameraIndex
    ? availableCameras.find(
        (cam) => cam.index === parseInt(selectedCameraIndex)
      )
    : undefined;

  // cv2's AVFoundation order is uniqueID-sorted, so plugging/unplugging a
  // device between sessions shifts indices. The browser device_id stays
  // stable per-origin, so use it to refresh each seeded camera's
  // camera_index — otherwise the recorder opens the wrong physical device
  // and the dropdown's "already added" check guards a stale index.
  useEffect(() => {
    if (availableCameras.length === 0 || cameras.length === 0) return;
    let changed = false;
    const refreshed = cameras.map((cam) => {
      if (!cam.device_id) return cam;
      const match = availableCameras.find((m) => m.deviceId === cam.device_id);
      if (match && match.index !== cam.camera_index) {
        changed = true;
        return { ...cam, camera_index: match.index };
      }
      return cam;
    });
    if (changed) onCamerasChange(refreshed);
    // We deliberately don't depend on `cameras`/`onCamerasChange` to avoid
    // re-running every keystroke in the camera-name input — re-syncing only
    // when the available-cameras list itself changes is sufficient.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableCameras]);

  const addCamera = () => {
    if (!selectedCameraIndex || !cameraName.trim()) {
      toast({
        title: "Missing Information",
        description: "Please select a camera and provide a name.",
        variant: "destructive",
      });
      return;
    }

    const cameraIndex = parseInt(selectedCameraIndex);
    const selectedCamera = availableCameras.find(
      (cam) => cam.index === cameraIndex
    );

    if (!selectedCamera) {
      toast({
        title: "Invalid Camera",
        description: "Selected camera is not available.",
        variant: "destructive",
      });
      return;
    }

    // Block duplicates by either cv2 index or browser deviceId — a stale
    // camera_index in a seeded camera can otherwise let the same physical
    // device sneak in under a different index.
    const isDuplicate = cameras.some(
      (cam) =>
        cam.camera_index === selectedCamera.index ||
        (selectedCamera.deviceId && cam.device_id === selectedCamera.deviceId),
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
      camera_index: selectedCamera.index,
      device_id: selectedCamera.deviceId,
      width: 640,
      height: 480,
      fps: 30,
    };

    onCamerasChange([...cameras, newCamera]);

    setSelectedCameraIndex("");
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
  // releaseStreamsRef.current() to make every CameraPreview drop its browser
  // stream so cv2.VideoCapture can grab the camera exclusively.
  const [streamsPaused, setStreamsPaused] = useState(false);
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
      <h3 className="text-lg font-semibold text-foreground border-b border-border pb-2">
        Camera configuration
      </h3>

      {/* Add Camera Section */}
      <div className="rounded-lg border border-border bg-secondary p-4 space-y-4">
        <h4 className="text-md font-medium text-muted-foreground">Add camera</h4>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label>Available cameras</Label>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => refreshCameras()}
              disabled={isLoadingCameras}
              className="h-6 w-6"
              title="Rescan for cameras (e.g. after plugging in a new USB camera)"
              aria-label="Rescan for cameras"
            >
              <RefreshCw
                className={`w-3.5 h-3.5 ${isLoadingCameras ? "animate-spin" : ""}`}
              />
            </Button>
          </div>
          <Select
            value={selectedCameraIndex}
            onValueChange={setSelectedCameraIndex}
            disabled={isLoadingCameras}
          >
            <SelectTrigger>
              <SelectValue
                placeholder={
                  isLoadingCameras ? "Loading cameras..." : "Select camera"
                }
              />
            </SelectTrigger>
            <SelectContent>
              {availableCameras.map((camera) => {
                const alreadyAdded = cameras.some(
                  (cam) =>
                    cam.camera_index === camera.index ||
                    (camera.deviceId && cam.device_id === camera.deviceId),
                );
                return (
                  <SelectItem
                    key={camera.index}
                    value={camera.index.toString()}
                    disabled={!camera.available || alreadyAdded}
                  >
                    <div className="flex flex-col">
                      <span className="font-medium">{camera.name}</span>
                      <span className="font-mono text-xs text-muted-foreground">
                        Index {camera.index}
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
            confirmation happens alongside it. */}
        {selectedCamera && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="rounded-lg border border-border bg-card overflow-hidden">
              <CameraStreamBox
                deviceId={selectedCamera.deviceId}
                cameraIndex={selectedCamera.index}
                paused={streamsPaused}
              />
            </div>

            <div className="flex flex-col justify-center gap-4">
              <div className="space-y-2">
                <Label>Camera name</Label>
                <Input
                  value={cameraName}
                  onChange={(e) => setCameraName(e.target.value)}
                  placeholder="e.g., workspace_cam"
                />
              </div>

              <Button
                onClick={addCamera}
                disabled={!selectedCameraIndex || !cameraName.trim()}
              >
                <Plus className="w-4 h-4 mr-2" />
                Add camera
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Configured Cameras */}
      {cameras.length > 0 && (
        <div className="space-y-4">
          <h4 className="text-md font-medium text-muted-foreground">
            Configured cameras ({cameras.length})
          </h4>

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
        <div className="text-center py-8 text-muted-foreground">
          <Camera className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
          <p>No cameras configured. Add a camera to get started.</p>
        </div>
      )}
    </div>
  );
};

interface CameraStreamBoxProps {
  deviceId: string;
  paused: boolean;
  /** cv2 index on the server — MJPEG fallback when there's no browser
   * deviceId match (headless deployment: cameras plugged into the server). */
  cameraIndex?: number;
}

/** Live preview for a camera. Used both for the pre-add preview (as soon as
 * a camera is picked in the dropdown) and for each configured camera's card.
 * A camera with a browser deviceId match streams via getUserMedia (the hook
 * stops the stream on deviceId change and on unmount); one without a match
 * but with a known cv2 index falls back to the backend MJPEG stream. Pausing
 * (recording start / modal close) unmounts the MJPEG img, whose cleanup
 * clears the src so the HTTP connection drops and the server releases the
 * camera — mirroring the getUserMedia release semantics. */
const CameraStreamBox: React.FC<CameraStreamBoxProps> = ({
  deviceId,
  paused,
  cameraIndex,
}) => {
  const { videoRef, hasError: streamError } = useCameraStream(deviceId, paused);

  const showVideo = !paused && deviceId && !streamError;
  // BackendCameraStream owns its own failure/retry UI — no error latch here.
  const showMjpeg = !paused && !deviceId && cameraIndex !== undefined;
  return (
    <div className="aspect-[4/3] bg-secondary relative">
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
          <VideoOff className="w-8 h-8 text-muted-foreground mb-2" />
          <span className="text-muted-foreground text-sm">
            {paused
              ? "Preview paused"
              : deviceId
              ? "Preview failed"
              : "No browser match"}
          </span>
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
    <div className="bg-card rounded-lg border border-border overflow-hidden">
      <CameraStreamBox
        deviceId={camera.device_id}
        cameraIndex={camera.camera_index}
        paused={paused}
      />

      {/* Camera Info */}
      <div className="p-3 space-y-2">
        <div className="flex items-center justify-between">
          <h5 className="font-medium text-foreground truncate">{camera.name}</h5>
          <Button
            onClick={onRemove}
            size="sm"
            variant="ghost"
            className="text-destructive hover:text-destructive p-1"
            aria-label="Remove camera"
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>

        <Collapsible>
          <CollapsibleTrigger className="group flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors">
            <ChevronRight className="w-3.5 h-3.5 transition-transform group-data-[state=open]:rotate-90" />
            Configuration
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-2 space-y-2">
            <div className="grid grid-cols-1 gap-2 font-mono text-xs text-muted-foreground">
              <div className="flex items-center gap-2">
                <span className="w-16">Resolution:</span>
                <div className="flex items-center gap-1">
                  <NumberInput
                    value={camera.width}
                    onChange={(v) => {
                      if (v !== undefined) onUpdate({ width: v });
                    }}
                    className="text-xs h-6 px-2 w-16"
                    min="320"
                    max="1920"
                  />
                  <span className="flex items-center">×</span>
                  <NumberInput
                    value={camera.height}
                    onChange={(v) => {
                      if (v !== undefined) onUpdate({ height: v });
                    }}
                    className="text-xs h-6 px-2 w-16"
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
                  className="text-xs h-6 px-2 w-16"
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
                  <SelectTrigger className="text-xs h-6 px-2 w-28">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={FOURCC_AUTO} className="text-xs">
                      Auto
                    </SelectItem>
                    {FOURCC_OPTIONS.map((code) => (
                      <SelectItem key={code} value={code} className="text-xs">
                        {code}
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
                  <SelectTrigger className="text-xs h-6 px-2 w-28">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={BACKEND_DEFAULT} className="text-xs">
                      Default
                    </SelectItem>
                    {BACKEND_OPTIONS.map((name) => (
                      <SelectItem key={name} value={name} className="text-xs">
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <p className="text-[10px] text-muted-foreground leading-tight">
                Overriding the backend can reorder camera indices on macOS.
              </p>
            </div>
            <div className="font-mono text-xs text-muted-foreground">
              Type: {camera.type} | Device:{" "}
              {camera.device_id?.substring(0, 10)}...
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>
    </div>
  );
};

export default CameraConfiguration;
