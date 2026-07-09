import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import LandingTopBar from "@/components/landing/LandingTopBar";
import Footer from "@/components/Footer";
import RobotConfigManager from "@/components/landing/RobotConfigManager";
import RecordingModal from "@/components/landing/RecordingModal";
import DatasetsPanel from "@/components/landing/DatasetsPanel";
import ModelsPanel from "@/components/landing/ModelsPanel";
import JobsSection from "@/components/jobs/JobsSection";

import UsageInstructionsModal from "@/components/landing/UsageInstructionsModal";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useRobots } from "@/hooks/useRobots";
import { DatasetInfo } from "@/lib/replayApi";
import { CameraConfig } from "@/components/recording/CameraConfiguration";
import { isHostedSpace } from "@/lib/isHostedSpace";
import { validateDatasetName } from "@/lib/datasetName";

const ON_SPACE = isHostedSpace();

const Landing = () => {
  const [showUsageModal, setShowUsageModal] = useState(ON_SPACE);
  const { auth } = useHfAuth();

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

  // Recording modal state
  const [showRecordingModal, setShowRecordingModal] = useState(false);
  // Non-null while the modal configures a RESUME session appending episodes to
  // this existing dataset. Carries the /datasets/info summary so the modal can
  // render its compatibility advisory; its repo_id goes to the backend
  // VERBATIM (no username prepend, no name gate, no timestamp — the backend
  // validates and skips stamping on resume). Cleared when the modal closes.
  const [resumeInfo, setResumeInfo] = useState<DatasetInfo | null>(null);
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
    if (!open) {
      // A cancelled resume must not leak into the next plain "Record a
      // dataset" session.
      setResumeInfo(null);
      if (releaseStreamsRef.current) {
        console.log("🧹 Modal closed: Releasing camera streams");
        releaseStreamsRef.current();
      }
    }
  };

  // "Record a dataset" in the Datasets panel: seed the recording form with the
  // new name and open the recording modal (configured here with the selected
  // robot + cameras).
  const handleRecordNew = (name: string) => {
    setDatasetName(name);
    openRecordingModal();
  };

  // "Record more episodes" on a local dataset's info card: open the recording
  // modal in resume mode, seeding the task from the dataset's first task
  // string (user-editable in the modal).
  const handleResumeRecording = (info: DatasetInfo) => {
    setResumeInfo(info);
    if (info.tasks.length > 0 && info.tasks[0].task) {
      setSingleTask(info.tasks[0].task);
    }
    openRecordingModal();
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
    const isResume = resumeInfo !== null;
    if ((!isResume && !datasetName) || !singleTask) {
      toast({
        title: "Missing dataset details",
        description: "Please enter a dataset name and task description.",
        variant: "destructive",
      });
      return;
    }
    // THE ID-SHAPE TRAP: a resume must pass the selected dataset's on-disk
    // repo_id VERBATIM — no username prepend, no bare-name validation gate
    // (the backend validates the full repo id), and the backend skips its
    // timestamp stamping on resume. Only a NEW dataset goes through the
    // name-validate + namespace-prepend path.
    if (!isResume) {
      const nameError = validateDatasetName(datasetName);
      if (nameError) {
        toast({
          title: "Invalid dataset name",
          description: nameError,
          variant: "destructive",
        });
        return;
      }
    }

    const datasetRepoId = isResume
      ? resumeInfo.repo_id
      : auth.status === "authenticated"
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
      resume: isResume,
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

      {/* Scrolls with the page (user preference) — only the slim top bar stays
          sticky; the card row previously pinned itself below it. */}
      <div className="bg-black border-b border-gray-800">
        <div className="mx-auto max-w-7xl px-4 py-4 grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
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
          <DatasetsPanel
            onRecordNew={handleRecordNew}
            onResumeRecording={handleResumeRecording}
          />
          <ModelsPanel />
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

      <RecordingModal
        open={showRecordingModal}
        onOpenChange={handleRecordingModalClose}
        resumeInfo={resumeInfo}
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
