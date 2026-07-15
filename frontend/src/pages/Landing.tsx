import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronsUpDown, GitMerge, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Eyebrow } from "@/components/ui/eyebrow";
import { AppShell } from "@/components/shell/AppShell";
import { useToast } from "@/hooks/use-toast";
import Footer from "@/components/Footer";
import RobotConfigManager from "@/components/landing/RobotConfigManager";
import RecordingModal from "@/components/landing/RecordingModal";
import DatasetPicker from "@/components/landing/DatasetPicker";
import CreateDatasetDialog from "@/components/landing/CreateDatasetDialog";
import DatasetInfoCard from "@/components/landing/DatasetInfoCard";
import MergeDatasetsDialog from "@/components/landing/MergeDatasetsDialog";
import JobsSection from "@/components/jobs/JobsSection";
import { POLICY_TYPE_OPTIONS } from "@/components/training/types";
import {
  fetchPolicyAvailability,
  PolicyAvailability,
} from "@/lib/policyAvailability";

import UsageInstructionsModal from "@/components/landing/UsageInstructionsModal";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useApi } from "@/contexts/ApiContext";
import { useRobots } from "@/hooks/useRobots";
import { useDatasets } from "@/hooks/useDatasets";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";
import { DatasetItem, deleteDataset } from "@/lib/replayApi";
import { CameraConfig } from "@/components/recording/CameraConfiguration";
import { isHostedSpace } from "@/lib/isHostedSpace";
import { validateDatasetName } from "@/lib/datasetName";
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

const ON_SPACE = isHostedSpace();

const Landing = () => {
  const [showUsageModal, setShowUsageModal] = useState(ON_SPACE);
  const { auth } = useHfAuth();
  const { baseUrl, fetchWithHeaders } = useApi();

  const {
    records,
    selectedName,
    selectedRecord,
    availableNames,
    isLoading: isLoadingRobots,
    selectRobot,
    clearSelection,
    createRobot,
    renameRobot,
    deleteRobot,
  } = useRobots();

  const {
    datasets,
    loading: datasetsLoading,
    refresh: refreshDatasets,
  } = useDatasets();
  const [showMergeDialog, setShowMergeDialog] = useState(false);
  const [showCreateDatasetDialog, setShowCreateDatasetDialog] = useState(false);
  const [pendingDeleteDataset, setPendingDeleteDataset] =
    useState<DatasetItem | null>(null);
  const { selectedDataset, setSelectedDataset } = useSelectedDataset();

  // Recording modal state
  const [showRecordingModal, setShowRecordingModal] = useState(false);
  const [datasetName, setDatasetName] = useState("");
  const [singleTask, setSingleTask] = useState("");
  const [numEpisodes, setNumEpisodes] = useState(5);
  const [episodeTimeS, setEpisodeTimeS] = useState(60);
  const [resetTimeS, setResetTimeS] = useState(15);
  const [streamingEncoding, setStreamingEncoding] = useState(true);
  const [cameras, setCameras] = useState<CameraConfig[]>([]);

  const releaseStreamsRef = useRef<(() => void) | null>(null);

  const navigate = useNavigate();
  const { toast } = useToast();

  // Which policy types this backend's lerobot pin can actually train.
  // Buttons stay enabled until the (cached) answer arrives: most types are
  // valid, so briefly optimistic beats greying the whole card on every visit.
  const [policyAvailability, setPolicyAvailability] =
    useState<PolicyAvailability | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetchPolicyAvailability(baseUrl, fetchWithHeaders)
      .then((a) => {
        if (!cancelled) setPolicyAvailability(a);
      })
      .catch(() => {
        // Backend unreachable — leave all buttons enabled; training itself
        // will surface the real error.
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  // Clear camera state and release streams when returning to landing page
  useEffect(() => {
    if (cameras.length > 0) {
      console.log(
        "🧹 Landing page: Cleaning up camera state from previous session",
      );
      if (releaseStreamsRef.current) {
        releaseStreamsRef.current();
      }
      setCameras([]);
    }
  }, []);

  useEffect(() => {
    return () => {
      if (releaseStreamsRef.current) {
        console.log("🧹 Landing page: Cleaning up camera streams on unmount");
        releaseStreamsRef.current();
      }
    };
  }, []);

  const openRecordingModal = () => {
    setCameras(selectedRecord ? [...(selectedRecord.cameras ?? [])] : []);
    setShowRecordingModal(true);
  };

  const handleRecordingModalClose = (open: boolean) => {
    setShowRecordingModal(open);
    if (!open && releaseStreamsRef.current) {
      console.log("🧹 Modal closed: Releasing camera streams");
      releaseStreamsRef.current();
    }
  };

  // Each model-type button is a direct entry into training: the Training page
  // reads `policyType` from router state and preselects it in the config form.
  const handleTrainingClick = (policyType: string) =>
    navigate("/training", { state: { policyType } });

  // Picking a dataset here selects it for training (the single source of truth);
  // Training reads it from the persisted selection.
  const handlePickExisting = (item: DatasetItem) => {
    setSelectedDataset(item.repo_id);
    toast({ title: "Dataset selected", description: item.repo_id });
  };

  const handleOpenCustom = (repoId: string) => {
    setSelectedDataset(repoId);
    toast({ title: "Dataset selected", description: repoId });
  };

  const handleCreateDataset = (name: string) => {
    setDatasetName(name);
    openRecordingModal();
  };

  // Deleting a dataset is destructive and irreversible, so route the picker's
  // trash button through a styled confirm dialog instead of deleting inline.
  const handleDeleteDataset = (item: DatasetItem) => {
    setPendingDeleteDataset(item);
  };

  const confirmDeleteDataset = async () => {
    const item = pendingDeleteDataset;
    if (!item) return;
    setPendingDeleteDataset(null);
    try {
      const res = await deleteDataset(baseUrl, fetchWithHeaders, item.repo_id);
      if (res.success) {
        toast({ title: "Dataset deleted", description: item.repo_id });
        // If the deleted one was selected for training, clear the stale pick.
        if (selectedDataset === item.repo_id) setSelectedDataset("");
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

  const handleStartRecording = async () => {
    if (!selectedRecord) {
      toast({
        title: "No robot selected",
        description: "Select or create a robot on the Landing page first.",
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
    if (!datasetName || !singleTask) {
      toast({
        title: "Missing dataset details",
        description: "Please enter a dataset name and task description.",
        variant: "destructive",
      });
      return;
    }
    const nameError = validateDatasetName(datasetName);
    if (nameError) {
      toast({
        title: "Invalid dataset name",
        description: nameError,
        variant: "destructive",
      });
      return;
    }

    const datasetRepoId =
      auth.status === "authenticated"
        ? `${auth.username}/${datasetName}`
        : datasetName;

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
      // Robot name → BiSO staging base id (bimanual). Names the per-session
      // staging dir; does not affect which calibration drives which arm.
      robot_name: robot.name,
      // Raw follower torque limit for the session (0-1000, default 400).
      max_torque_limit: robot.max_torque_limit ?? 400,
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

    setShowRecordingModal(false);
    navigate("/recording", { state: { recordingConfig } });
  };

  return (
    <AppShell showAuthChip fullBleed>
      <section className="grid-bg border-b border-border">
        <div className="mx-auto max-w-[1440px] px-4 py-10">
          <Eyebrow>[ SO-101 · leader + follower ]</Eyebrow>
          <h1 className="mt-3 text-4xl">Teach, record, and train your robot</h1>
          <p className="mt-2 max-w-[60ch] font-mono text-xs text-muted-foreground">
            calibrate the arms · teleoperate to record demonstrations · train a
            policy — all from the browser
          </p>
        </div>
      </section>

      {/* Scrolls with the page (user preference) — only the slim top bar stays
          sticky; the card row previously pinned itself below it. */}
      <div className="border-b border-border">
        <div className="mx-auto max-w-[1440px] px-4 py-4 grid gap-4 grid-cols-1 lg:grid-cols-[1.2fr_2fr]">
          <RobotConfigManager
            records={records}
            selectedName={selectedName}
            selectedRecord={selectedRecord}
            availableNames={availableNames}
            isLoading={isLoadingRobots}
            selectRobot={selectRobot}
            clearSelection={clearSelection}
            createRobot={createRobot}
            renameRobot={renameRobot}
            deleteRobot={deleteRobot}
          />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <Card variant="flat" className="p-3 flex flex-col gap-2">
              <Eyebrow>[ Dataset ]</Eyebrow>
              <div className="flex items-center gap-2">
                <div className="flex-1 min-w-0">
                  <DatasetPicker
                    datasets={datasets}
                    loading={datasetsLoading}
                    onPickExisting={handlePickExisting}
                    onOpenCustom={handleOpenCustom}
                    onCreateNew={handleCreateDataset}
                    onDelete={handleDeleteDataset}
                    onUploaded={() => refreshDatasets()}
                  >
                    <Button
                      variant="secondary"
                      role="combobox"
                      className="w-full justify-between font-normal"
                    >
                      <span
                        className={`truncate ${selectedDataset ? "text-foreground" : "text-muted-foreground"}`}
                      >
                        {datasetsLoading
                          ? "Loading datasets…"
                          : (selectedDataset ?? "Select or create a dataset…")}
                      </span>
                      <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                    </Button>
                  </DatasetPicker>
                </div>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setShowCreateDatasetDialog(true)}
                  className="h-8 shrink-0"
                >
                  <Plus className="w-3.5 h-3.5 mr-1.5" />
                  New dataset
                </Button>
              </div>
              {selectedDataset && (
                <DatasetInfoCard
                  repoId={selectedDataset}
                  onRenamed={(newRepoId) => {
                    // The renamed dir has a new repo id: repoint the selection
                    // (so the card + training read the new id) and refresh the
                    // picker list so both reflect it without a manual reload.
                    setSelectedDataset(newRepoId);
                    refreshDatasets();
                  }}
                />
              )}
              <button
                type="button"
                onClick={() => setShowMergeDialog(true)}
                className="self-start text-xs text-muted-foreground hover:text-foreground transition-colors inline-flex items-center gap-1"
              >
                <GitMerge className="h-3.5 w-3.5" /> Merge datasets…
              </button>
            </Card>
            <Card variant="flat" className="p-3 flex flex-col gap-2">
              <Eyebrow>[ Models ]</Eyebrow>
              {/* Stable = tested on our hardware (see POLICY_TYPE_OPTIONS).
                  Untested types stay selectable, just visually subdued. */}
              <div className="grid grid-cols-2 gap-2">
                {POLICY_TYPE_OPTIONS.filter((p) => p.stable).map((policy) => {
                  const unavailable =
                    policyAvailability?.[policy.value] === false;
                  return (
                    // Tooltip lives on a wrapper span: the disabled Button
                    // gets pointer-events-none, which would swallow `title`.
                    <span
                      key={policy.value}
                      title={
                        unavailable
                          ? "Not available in this lerobot version"
                          : `Train a ${policy.label} model`
                      }
                    >
                      <Button
                        onClick={() => handleTrainingClick(policy.value)}
                        disabled={!selectedDataset || unavailable}
                        size="sm"
                        className="w-full px-2"
                      >
                        <span className="truncate">{policy.label}</span>
                      </Button>
                    </span>
                  );
                })}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Untested in MakerLab — use at your own risk
              </p>
              <div className="grid grid-cols-3 gap-2">
                {POLICY_TYPE_OPTIONS.filter((p) => !p.stable).map((policy) => {
                  const unavailable =
                    policyAvailability?.[policy.value] === false;
                  return (
                    <span
                      key={policy.value}
                      title={
                        unavailable
                          ? "Not available in this lerobot version"
                          : `Train a ${policy.label} model — untested in MakerLab, use at your own risk`
                      }
                    >
                      <Button
                        onClick={() => handleTrainingClick(policy.value)}
                        disabled={!selectedDataset || unavailable}
                        size="sm"
                        variant="secondary"
                        className="w-full px-2"
                      >
                        <span className="truncate">{policy.label}</span>
                      </Button>
                    </span>
                  );
                })}
              </div>
              {!selectedDataset && (
                <p className="text-xs text-muted-foreground">
                  Select a dataset first.
                </p>
              )}
            </Card>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-[1440px] px-4 py-6">
        <JobsSection />
      </div>

      <Footer />

      <UsageInstructionsModal
        open={showUsageModal}
        onOpenChange={setShowUsageModal}
        dismissible={!ON_SPACE}
      />

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
        onOpenChange={(o) => !o && setPendingDeleteDataset(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Delete "{pendingDeleteDataset?.repo_id}"?
            </AlertDialogTitle>
            <AlertDialogDescription>
              This permanently removes the dataset from local disk — including
              all recorded episodes and videos. You can't undo this.
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

      <RecordingModal
        open={showRecordingModal}
        onOpenChange={handleRecordingModalClose}
        robot={selectedRecord}
        datasetName={datasetName}
        setDatasetName={setDatasetName}
        singleTask={singleTask}
        setSingleTask={setSingleTask}
        numEpisodes={numEpisodes}
        setNumEpisodes={setNumEpisodes}
        episodeTimeS={episodeTimeS}
        setEpisodeTimeS={setEpisodeTimeS}
        resetTimeS={resetTimeS}
        setResetTimeS={setResetTimeS}
        streamingEncoding={streamingEncoding}
        setStreamingEncoding={setStreamingEncoding}
        cameras={cameras}
        setCameras={setCameras}
        onStart={handleStartRecording}
        releaseStreamsRef={releaseStreamsRef}
      />
    </AppShell>
  );
};

export default Landing;
