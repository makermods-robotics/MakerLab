import React, { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Plus,
  Radio,
  Square,
} from "lucide-react";
import VisualizerPanel from "@/components/control/VisualizerPanel";
import CameraFeed from "@/components/control/CameraFeed";
import { DatasetLibrary } from "@/components/collect/DatasetLibrary";
import { useTeleopSession } from "@/components/collect/useTeleopSession";
import CameraConfiguration, {
  CameraConfig,
} from "@/components/recording/CameraConfiguration";
import CreateDatasetDialog from "@/components/landing/CreateDatasetDialog";
import DatasetInfoCard from "@/components/landing/DatasetInfoCard";
import MergeDatasetsDialog from "@/components/landing/MergeDatasetsDialog";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge, BadgeDot } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useApi } from "@/contexts/ApiContext";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useToast } from "@/hooks/use-toast";
import { useDatasets } from "@/hooks/useDatasets";
import { useRobots } from "@/hooks/useRobots";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";
import { validateDatasetRepoId } from "@/lib/datasetName";
import { DatasetItem, deleteDataset } from "@/lib/replayApi";
import { cn } from "@/lib/utils";

interface CompletedDatasetState {
  dataset_repo_id?: string;
  repo_id?: string;
  repoId?: string;
  saved_episodes?: number;
  num_episodes?: number;
  episodes?: number;
  recorded_episodes?: number;
}

const getCompletedDataset = (state: unknown): CompletedDatasetState | null => {
  if (!state || typeof state !== "object") return null;
  const value = (state as { completedDataset?: unknown }).completedDataset;
  if (!value) return null;
  if (typeof value === "string") return { dataset_repo_id: value };
  if (typeof value === "object") return value as CompletedDatasetState;
  return null;
};

const getCompletedRepoId = (completed: CompletedDatasetState) =>
  completed.dataset_repo_id ?? completed.repo_id ?? completed.repoId ?? null;

const getCompletedEpisodes = (completed: CompletedDatasetState) =>
  completed.saved_episodes ??
  completed.num_episodes ??
  completed.recorded_episodes ??
  completed.episodes ??
  null;

const Collect: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { toast } = useToast();
  const { auth } = useHfAuth();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { selectedRecord } = useRobots();
  const {
    datasets,
    refresh: refreshDatasets,
  } = useDatasets();
  const { selectedDataset, setSelectedDataset } = useSelectedDataset();
  const teleop = useTeleopSession();

  const [showCreateDatasetDialog, setShowCreateDatasetDialog] = useState(false);
  const [showMergeDialog, setShowMergeDialog] = useState(false);
  const [showCameraDialog, setShowCameraDialog] = useState(false);
  const [datasetLibraryOpen, setDatasetLibraryOpen] = useState(false);
  const [pendingDeleteDataset, setPendingDeleteDataset] =
    useState<DatasetItem | null>(null);

  const [singleTask, setSingleTask] = useState("");
  const [numEpisodes, setNumEpisodes] = useState(5);
  const [episodeTimeS, setEpisodeTimeS] = useState(60);
  const [resetTimeS, setResetTimeS] = useState(15);
  const [streamingEncoding, setStreamingEncoding] = useState(true);
  const [cameras, setCameras] = useState<CameraConfig[]>([]);
  const [streamsPaused, setStreamsPaused] = useState(false);
  const configReleaseStreamsRef = useRef<(() => void) | null>(null);
  const releaseStreamsRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    setCameras(selectedRecord ? [...(selectedRecord.cameras ?? [])] : []);
    setStreamsPaused(false);
  }, [selectedRecord?.name, selectedRecord]);

  useEffect(() => {
    releaseStreamsRef.current = () => {
      setStreamsPaused(true);
      configReleaseStreamsRef.current?.();
    };
  }, []);

  const completedDataset = getCompletedDataset(location.state);
  const completedRepoId = completedDataset
    ? getCompletedRepoId(completedDataset)
    : null;
  const completedEpisodes = completedDataset
    ? getCompletedEpisodes(completedDataset)
    : null;

  const robotBlockReason = !selectedRecord
    ? "Select a robot before collecting data."
    : !selectedRecord.is_clean
      ? `${selectedRecord.name} is missing a calibration.`
      : null;

  const cameraSummary = useMemo(() => {
    if (cameras.length === 0) return "no cameras configured";
    return cameras.map((camera) => camera.name).join(" + ");
  }, [cameras]);

  const statusLine = useMemo(() => {
    if (teleop.status?.last_cleanup_error) {
      return teleop.status.last_cleanup_error;
    }
    if (teleop.status?.releasing) return "returning to rest pose · releasing torque";
    if (teleop.active) return "joints streaming · nothing saved";
    if (teleop.status?.message) return teleop.status.message;
    return "joints idle · leader detached";
  }, [teleop.active, teleop.status]);

  const handleSelectDataset = (repoId: string) => {
    setSelectedDataset(repoId);
    toast({ title: "Dataset selected", description: repoId });
  };

  const handleCreateDataset = (name: string) => {
    const repoId =
      auth.status === "authenticated" ? `${auth.username}/${name}` : name;
    setSelectedDataset(repoId);
    toast({ title: "Dataset selected", description: repoId });
  };

  const confirmDeleteDataset = async () => {
    const item = pendingDeleteDataset;
    if (!item) return;
    setPendingDeleteDataset(null);
    try {
      const res = await deleteDataset(baseUrl, fetchWithHeaders, item.repo_id);
      if (res.success) {
        toast({ title: "Dataset deleted", description: item.repo_id });
        if (selectedDataset === item.repo_id) setSelectedDataset(null);
        refreshDatasets();
      } else {
        toast({
          title: "Delete failed",
          description: res.message ?? "Could not delete the dataset.",
          variant: "destructive",
        });
      }
    } catch (e) {
      toast({
        title: "Delete failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handleStartTeleop = () => {
    if (!selectedRecord || robotBlockReason) return;
    void teleop.startTeleoperation(selectedRecord);
  };

  const handleStartRecording = async () => {
    if (!selectedRecord) {
      toast({
        title: "No robot selected",
        description: "Select or create a robot before recording.",
        variant: "destructive",
      });
      return;
    }
    const robot = selectedRecord;
    if (!robot.is_clean) {
      toast({
        title: "Robot not ready",
        description: `${robot.name} is missing a calibration. Configure it before recording.`,
        variant: "destructive",
      });
      return;
    }
    if (!selectedDataset || !singleTask) {
      toast({
        title: "Missing dataset details",
        description: "Please select a dataset and enter a task description.",
        variant: "destructive",
      });
      return;
    }
    const repoError = validateDatasetRepoId(selectedDataset);
    if (repoError) {
      toast({
        title: "Invalid dataset name",
        description: repoError,
        variant: "destructive",
      });
      return;
    }

    const datasetRepoId = selectedDataset;

    if (cameras.length > 0 && releaseStreamsRef.current) {
      console.log("🔓 Releasing camera streams before starting recording...");
      toast({
        title: "Preparing Camera Resources",
        description: `Releasing ${cameras.length} camera stream(s) for recording...`,
      });
      releaseStreamsRef.current();
      await new Promise((resolve) => setTimeout(resolve, 500));
      console.log("✅ Camera streams released, proceeding with recording...");
      toast({
        title: "Camera Resources Ready",
        description:
          "Camera streams released successfully. Starting recording...",
      });
    }

    const cameraDict = cameras.reduce(
      (acc, cam) => {
        acc[cam.name] = {
          type: cam.type,
          camera_index: cam.camera_index,
          width: cam.width,
          height: cam.height,
          fps: cam.fps,
          ...(cam.fourcc ? { fourcc: cam.fourcc } : {}),
          ...(cam.backend ? { backend: cam.backend } : {}),
        };
        return acc;
      },
      {} as Record<
        string,
        {
          type: string;
          camera_index?: number;
          width: number;
          height: number;
          fps?: number;
          fourcc?: string;
          backend?: string;
        }
      >,
    );

    const recordingConfig = {
      leader_port: robot.leader_port,
      follower_port: robot.follower_port,
      leader_config: robot.leader_config,
      follower_config: robot.follower_config,
      // Bimanual: forward mode + the right arm so the backend records a BiSO pair.
      mode: robot.mode,
      right_leader_port: robot.right_leader_port,
      right_follower_port: robot.right_follower_port,
      right_leader_config: robot.right_leader_config,
      right_follower_config: robot.right_follower_config,
      // Robot name -> BiSO staging base id (bimanual). Names the per-session
      // staging dir; does not affect which calibration drives which arm.
      robot_name: robot.name,
      // Follower torque limit for the session (10-100% of full power).
      motor_power: robot.motor_power ?? 100,
      dataset_repo_id: datasetRepoId,
      single_task: singleTask,
      num_episodes: numEpisodes,
      episode_time_s: episodeTimeS,
      reset_time_s: resetTimeS,
      fps: 30,
      video: true,
      push_to_hub: false,
      resume: false,
      streaming_encoding: streamingEncoding,
      cameras: cameraDict,
    };

    setShowCameraDialog(false);
    navigate("/recording", { state: { recordingConfig } });
  };

  const recordDisabled =
    !!robotBlockReason ||
    !selectedDataset ||
    !singleTask.trim() ||
    numEpisodes < 1 ||
    episodeTimeS < 1 ||
    resetTimeS < 1;

  return (
    <div className="mx-auto w-full max-w-[1440px] px-4 py-6">
      {completedDataset && (
        <div className="mb-4 flex flex-col gap-3 rounded-xl border border-border bg-card p-4 shadow-sm sm:flex-row sm:items-center sm:justify-between">
          <div className="min-w-0">
            <p className="truncate font-medium">
              {completedEpisodes ?? "N"} episodes recorded
              {completedRepoId ? ` · ${completedRepoId}` : ""}
            </p>
            <p className="mt-1 font-mono text-xs text-muted-foreground">
              Recorded just now on {selectedRecord?.name ?? "selected robot"}.
            </p>
          </div>
          <Button
            onClick={() => {
              if (completedRepoId) setSelectedDataset(completedRepoId);
              navigate("/training");
            }}
          >
            Train on this dataset
          </Button>
        </div>
      )}

      <div className="mb-5 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Collect</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Teach by demonstration - practice with teleop, then record episodes.
          </p>
        </div>
        {selectedRecord && (
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={selectedRecord.is_clean ? "ok" : "warn"}>
              <BadgeDot pulse={teleop.active} />
              {selectedRecord.is_clean ? "ready" : "needs calibration"}
            </Badge>
            <Badge variant="secondary">{selectedRecord.mode}</Badge>
            <span className="font-mono text-xs text-muted-foreground">
              {selectedRecord.name}
            </span>
          </div>
        )}
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(320px,1fr)]">
        <div className="min-w-0">
          <div className="overflow-hidden rounded-xl border border-border bg-card shadow-sm">
            <VisualizerPanel
              onGoBack={() => undefined}
              className="min-h-[420px] p-3"
              bimanual={selectedRecord?.mode === "bimanual"}
            />
          </div>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            {(selectedRecord?.cameras ?? []).length > 0 ? (
              (selectedRecord?.cameras ?? []).map((camera) => (
                <CameraFeed
                  key={`${camera.name}-${camera.camera_index ?? "none"}`}
                  cameraIndex={streamsPaused ? undefined : camera.camera_index}
                  label={camera.name}
                />
              ))
            ) : (
              <div className="media-slot flex min-h-40 items-center justify-center rounded-xl border border-dashed border-border bg-card text-sm text-muted-foreground md:col-span-2">
                No cameras configured for this robot.
              </div>
            )}
          </div>
        </div>

        <div className="grid content-start gap-4">
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle>Teleoperate</CardTitle>
              <CardDescription>
                Drive the follower with the leader arm. Nothing is saved.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {robotBlockReason && (
                <p className="flex items-center gap-2 text-sm text-warn">
                  <AlertTriangle className="h-4 w-4" />
                  {robotBlockReason}
                </p>
              )}
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  onClick={handleStartTeleop}
                  disabled={!!robotBlockReason || teleop.active || teleop.starting}
                >
                  <Radio className="mr-2 h-4 w-4" />
                  {teleop.starting ? "Starting..." : "Start teleop"}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => void teleop.stopTeleoperation()}
                  disabled={!teleop.active || teleop.stopping}
                >
                  <Square className="mr-2 h-4 w-4" />
                  {teleop.stopping ? "Stopping..." : "Stop"}
                </Button>
              </div>
              <p className="font-mono text-xs text-muted-foreground">
                {statusLine}
              </p>
            </CardContent>
          </Card>

          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle>Record dataset</CardTitle>
              <CardDescription>
                Episodes go to the selected dataset.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {robotBlockReason && (
                <p className="flex items-center gap-2 text-sm text-warn">
                  <AlertTriangle className="h-4 w-4" />
                  {robotBlockReason}
                </p>
              )}

              <div className="space-y-2">
                <Label>Dataset</Label>
                <div className="flex min-w-0 items-center gap-2">
                  <span
                    className={cn(
                      "min-w-0 flex-1 truncate font-mono text-xs",
                      selectedDataset
                        ? "text-foreground"
                        : "text-muted-foreground",
                    )}
                    title={selectedDataset ?? undefined}
                  >
                    {selectedDataset ?? "No dataset selected"}
                  </span>
                  <Button
                    variant="secondary"
                    size="icon"
                    onClick={() => setShowCreateDatasetDialog(true)}
                    aria-label="Create dataset"
                    title="Create dataset"
                  >
                    <Plus className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setDatasetLibraryOpen(true);
                      requestAnimationFrame(() => {
                        document
                          .getElementById("dataset-library")
                          ?.scrollIntoView({ behavior: "smooth" });
                      });
                    }}
                  >
                    Choose…
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="taskPrompt">Task prompt</Label>
                <textarea
                  id="taskPrompt"
                  value={singleTask}
                  onChange={(e) => setSingleTask(e.target.value)}
                  rows={3}
                  placeholder="Pick up the cube and place it in the tray."
                  className="flex min-h-20 w-full rounded-sm border border-input bg-card px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
                />
              </div>

              <div className="grid grid-cols-3 gap-2">
                <div className="space-y-2">
                  <Label htmlFor="numEpisodes">Episodes</Label>
                  <NumberInput
                    id="numEpisodes"
                    min="1"
                    max="100"
                    value={numEpisodes}
                    onChange={(v) => {
                      if (v !== undefined) setNumEpisodes(v);
                    }}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="episodeTimeS">Episode s</Label>
                  <NumberInput
                    id="episodeTimeS"
                    min="1"
                    value={episodeTimeS}
                    onChange={(v) => {
                      if (v !== undefined) setEpisodeTimeS(v);
                    }}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="resetTimeS">Reset s</Label>
                  <NumberInput
                    id="resetTimeS"
                    min="1"
                    value={resetTimeS}
                    onChange={(v) => {
                      if (v !== undefined) setResetTimeS(v);
                    }}
                  />
                </div>
              </div>

              <div className="flex items-center gap-3 rounded-md border border-border bg-secondary p-3">
                <Camera className="h-4 w-4 shrink-0 text-muted-foreground" />
                <span className="min-w-0 flex-1 truncate font-mono text-xs text-muted-foreground">
                  cameras: {cameraSummary}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowCameraDialog(true)}
                >
                  Configure cameras
                </Button>
              </div>

              <label className="flex items-start gap-3 rounded-md border border-border bg-secondary p-3">
                <Checkbox
                  id="streamingEncoding"
                  checked={streamingEncoding}
                  onCheckedChange={(value) => setStreamingEncoding(value === true)}
                  className="mt-0.5"
                />
                <span className="space-y-1">
                  <span className="block text-sm font-medium">
                    Streaming video encoding
                  </span>
                  <span className="block text-xs text-muted-foreground">
                    Encode frames during capture so each episode saves quickly.
                  </span>
                </span>
              </label>

              {!selectedDataset && (
                <p className="text-sm text-muted-foreground">
                  Select or create a dataset before recording.
                </p>
              )}
              <Button
                className="w-full"
                onClick={() => void handleStartRecording()}
                disabled={recordDisabled}
              >
                <CheckCircle2 className="mr-2 h-4 w-4" />
                Start recording
              </Button>
            </CardContent>
          </Card>

          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle>Dataset</CardTitle>
              <CardDescription>Review, upload, or rename.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {selectedDataset ? (
                <DatasetInfoCard
                  repoId={selectedDataset}
                  onRenamed={(newRepoId) => {
                    setSelectedDataset(newRepoId);
                    refreshDatasets();
                  }}
                />
              ) : (
                <p className="rounded-md border border-border bg-secondary p-3 text-sm text-muted-foreground">
                  Select a dataset to see local details and Hub status.
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      <section id="dataset-library" className="mt-4 scroll-mt-4">
        <DatasetLibrary
          selectedRepoId={selectedDataset}
          onSelect={handleSelectDataset}
          onMerge={() => setShowMergeDialog(true)}
          open={datasetLibraryOpen}
          onOpenChange={setDatasetLibraryOpen}
        />
      </section>

      <Dialog
        open={showCameraDialog}
        onOpenChange={(open) => {
          setShowCameraDialog(open);
          if (!open) configReleaseStreamsRef.current?.();
        }}
      >
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>Configure cameras</DialogTitle>
            <DialogDescription>
              These backend MJPEG previews match the cameras used by recording.
            </DialogDescription>
          </DialogHeader>
          <CameraConfiguration
            cameras={cameras}
            onCamerasChange={setCameras}
            releaseStreamsRef={configReleaseStreamsRef}
          />
        </DialogContent>
      </Dialog>

      <MergeDatasetsDialog
        open={showMergeDialog}
        onOpenChange={setShowMergeDialog}
        datasets={datasets}
        onMerged={refreshDatasets}
      />

      <CreateDatasetDialog
        open={showCreateDatasetDialog}
        onOpenChange={setShowCreateDatasetDialog}
        existingRepoIds={datasets.map((d) => d.repo_id)}
        onCreateNew={handleCreateDataset}
      />

      <AlertDialog
        open={pendingDeleteDataset !== null}
        onOpenChange={(open) => !open && setPendingDeleteDataset(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Delete "{pendingDeleteDataset?.repo_id}"?
            </AlertDialogTitle>
            <AlertDialogDescription>
              This permanently removes the local dataset, including recorded
              episodes and videos. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmDeleteDataset}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default Collect;
