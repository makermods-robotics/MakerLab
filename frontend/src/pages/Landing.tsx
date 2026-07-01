import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronsUpDown, GitMerge } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import LandingTopBar from "@/components/landing/LandingTopBar";
import Footer from "@/components/Footer";
import RobotConfigManager from "@/components/landing/RobotConfigManager";
import RecordingModal from "@/components/landing/RecordingModal";
import DatasetPicker from "@/components/landing/DatasetPicker";
import MergeDatasetsDialog from "@/components/landing/MergeDatasetsDialog";
import JobsSection from "@/components/jobs/JobsSection";

import UsageInstructionsModal from "@/components/landing/UsageInstructionsModal";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useApi } from "@/contexts/ApiContext";
import { useRobots } from "@/hooks/useRobots";
import { useDatasets } from "@/hooks/useDatasets";
import { DatasetItem, deleteDataset } from "@/lib/replayApi";
import { CameraConfig } from "@/components/recording/CameraConfiguration";
import { isHostedSpace } from "@/lib/isHostedSpace";

const ON_SPACE = isHostedSpace();

const Landing = () => {
  const [showUsageModal, setShowUsageModal] = useState(ON_SPACE);
  const { auth } = useHfAuth();
  const { baseUrl, fetchWithHeaders } = useApi();

  const {
    selectedName,
    selectedRecord,
    availableNames,
    isLoading: isLoadingRobots,
    selectRobot,
    createRobot,
    renameRobot,
    setRobotMode,
    deleteRobot,
  } = useRobots();

  const {
    datasets,
    loading: datasetsLoading,
    refresh: refreshDatasets,
  } = useDatasets();
  const [showMergeDialog, setShowMergeDialog] = useState(false);

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

  const handleTrainingClick = () => navigate("/training");

  const openHubViewer = (repoId: string, isPrivate: boolean) => {
    const spacePath = `/spaces/lerobot/visualize_dataset?path=${encodeURIComponent(`/${repoId}`)}`;
    const target = isPrivate
      ? `https://huggingface.co/login?next=${encodeURIComponent(spacePath)}`
      : `https://huggingface.co${spacePath}`;
    window.open(target, "_blank", "noopener,noreferrer");
  };

  const handlePickExisting = (item: DatasetItem) => {
    if (item.source === "local" || item.source === "both") {
      navigate("/upload", {
        state: {
          datasetInfo: {
            dataset_repo_id: item.repo_id,
            source: item.source,
          },
        },
      });
      return;
    }
    openHubViewer(item.repo_id, item.private);
  };

  const handleOpenCustom = (repoId: string) => {
    // Custom-typed repo IDs are always treated as Hub paths. We don't know
    // privacy, so route through the login redirect to be safe.
    openHubViewer(repoId, true);
  };

  const handleCreateDataset = (name: string) => {
    setDatasetName(name);
    openRecordingModal();
  };

  const handleDeleteDataset = async (item: DatasetItem) => {
    if (
      !window.confirm(
        `Delete "${item.repo_id}"? This removes it from local disk.`,
      )
    )
      return;
    try {
      const res = await deleteDataset(baseUrl, fetchWithHeaders, item.repo_id);
      if (res.success) {
        toast({ title: "Dataset deleted", description: item.repo_id });
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
    <div
      className="min-h-screen bg-black text-white pb-16"
      style={{ ["--lelab-topbar-h" as string]: "48px" }}
    >
      <LandingTopBar />

      <div
        className="sticky z-20 bg-black/95 backdrop-blur supports-[backdrop-filter]:bg-black/70 border-b border-gray-800"
        style={{ top: "var(--lelab-topbar-h)" }}
      >
        <div className="mx-auto max-w-7xl px-4 py-4 grid gap-4 grid-cols-1 lg:grid-cols-[1.2fr_2fr]">
          <RobotConfigManager
            selectedName={selectedName}
            selectedRecord={selectedRecord}
            availableNames={availableNames}
            isLoading={isLoadingRobots}
            selectRobot={selectRobot}
            createRobot={createRobot}
            renameRobot={renameRobot}
            setRobotMode={setRobotMode}
            deleteRobot={deleteRobot}
          />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex flex-col gap-2">
              <h3 className="font-semibold text-lg text-left h-10 flex items-center">
                Dataset
              </h3>
              <DatasetPicker
                datasets={datasets}
                loading={datasetsLoading}
                onPickExisting={handlePickExisting}
                onOpenCustom={handleOpenCustom}
                onCreateNew={handleCreateDataset}
                onDelete={handleDeleteDataset}
              >
                <Button
                  variant="outline"
                  role="combobox"
                  className="w-full justify-between bg-gray-800 border-gray-600 text-white hover:bg-gray-700"
                >
                  <span className="truncate text-gray-300">
                    {datasetsLoading
                      ? "Loading datasets…"
                      : "Select or create a dataset…"}
                  </span>
                  <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                </Button>
              </DatasetPicker>
              <button
                type="button"
                onClick={() => setShowMergeDialog(true)}
                className="self-start text-xs text-gray-400 hover:text-white transition-colors inline-flex items-center gap-1"
              >
                <GitMerge className="h-3.5 w-3.5" /> Merge datasets…
              </button>
            </div>
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex flex-col gap-2">
              <h3 className="font-semibold text-lg text-left h-10 flex items-center">
                Create a model
              </h3>
              <Button
                onClick={handleTrainingClick}
                className="w-full bg-green-500 hover:bg-green-600 text-white"
              >
                Training
              </Button>
            </div>
          </div>
        </div>
      </div>

      <main className="mx-auto max-w-7xl px-4 py-6">
        <JobsSection />
      </main>

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
    </div>
  );
};

export default Landing;
