import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import LandingHeader from "@/components/landing/LandingHeader";
import HfAuthBanner from "@/components/landing/HfAuthBanner";
import RobotConfigManager from "@/components/landing/RobotConfigManager";
import ActionList from "@/components/landing/ActionList";
import RecordingModal from "@/components/landing/RecordingModal";

import { Action } from "@/components/landing/types";
import UsageInstructionsModal from "@/components/landing/UsageInstructionsModal";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useRobots } from "@/hooks/useRobots";
import { CameraConfig } from "@/components/recording/CameraConfiguration";
import { isHostedSpace } from "@/lib/isHostedSpace";

const ON_SPACE = isHostedSpace();

const Landing = () => {
  const [showUsageModal, setShowUsageModal] = useState(ON_SPACE);
  const { auth } = useHfAuth();

  const {
    visibleRecords,
    hiddenNames,
    isLoading: isLoadingRobots,
    addToSession,
    removeFromSession,
    createRobot,
    deleteRobot,
  } = useRobots();

  // Recording modal state
  const [showRecordingModal, setShowRecordingModal] = useState(false);
  const [datasetName, setDatasetName] = useState("");
  const [singleTask, setSingleTask] = useState("");
  const [numEpisodes, setNumEpisodes] = useState(5);
  const [cameras, setCameras] = useState<CameraConfig[]>([]);

  const releaseStreamsRef = useRef<(() => void) | null>(null);

  const navigate = useNavigate();
  const { toast } = useToast();

  // Clear camera state and release streams when returning to landing page
  useEffect(() => {
    if (cameras.length > 0) {
      console.log("🧹 Landing page: Cleaning up camera state from previous session");
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

  const handleRecordingClick = () => {
    // Seed the recording's camera list from the lone robot's attached cameras.
    // The user can add more in the modal.
    if (visibleRecords.length === 1) {
      setCameras([...(visibleRecords[0].cameras ?? [])]);
    } else {
      setCameras([]);
    }
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
  const handleReplayDatasetClick = () => navigate("/replay-dataset");
  const handleInferenceClick = () => navigate("/inference");

  const handleStartRecording = async () => {
    if (visibleRecords.length === 0) {
      toast({
        title: "No robot configured",
        description: "Add and configure a robot on the Landing page first.",
        variant: "destructive",
      });
      return;
    }
    if (visibleRecords.length > 1) {
      toast({
        title: "Multiple robots not supported",
        description:
          "Multiple robot configurations are not supported yet. Hide all but one robot tile.",
        variant: "destructive",
      });
      return;
    }
    const robot = visibleRecords[0];
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
      auth.status === "authenticated" ? `${auth.username}/${datasetName}` : datasetName;

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
        description: "Camera streams released successfully. Starting recording...",
      });
    }

    const cameraDict = cameras.reduce((acc, cam) => {
      acc[cam.name] = {
        type: cam.type,
        camera_index: cam.camera_index,
        width: cam.width,
        height: cam.height,
        fps: cam.fps,
      };
      return acc;
    }, {} as Record<string, { type: string; camera_index?: number; width: number; height: number; fps?: number }>);

    const recordingConfig = {
      leader_port: robot.leader_port,
      follower_port: robot.follower_port,
      leader_config: robot.leader_config,
      follower_config: robot.follower_config,
      dataset_repo_id: datasetRepoId,
      single_task: singleTask,
      num_episodes: numEpisodes,
      episode_time_s: 60,
      reset_time_s: 15,
      fps: 30,
      video: true,
      push_to_hub: false,
      resume: false,
      cameras: cameraDict,
    };

    setShowRecordingModal(false);
    navigate("/recording", { state: { recordingConfig } });
  };

  // Calibration and Teleoperation are now per-robot on the tile.
  const actions: Action[] = [
    {
      title: "Record Dataset",
      description: "Record episodes for training data.",
      handler: handleRecordingClick,
      color: "bg-red-500 hover:bg-red-600",
    },
    {
      title: "Replay Dataset",
      description: "Replay and analyze recorded datasets.",
      handler: handleReplayDatasetClick,
      color: "bg-purple-500 hover:bg-purple-600",
    },
    {
      title: "Training",
      description: "Train a model on your datasets.",
      handler: handleTrainingClick,
      color: "bg-green-500 hover:bg-green-600",
      isWorkInProgress: true,
    },
    {
      title: "Inference",
      description: "Run a trained model on the robot arm.",
      handler: handleInferenceClick,
      color: "bg-blue-500 hover:bg-blue-600",
      isWorkInProgress: true,
    },
  ];

  return (
    <div className="min-h-screen bg-black text-white flex flex-col items-center p-4 pt-12 sm:pt-20">
      <div className="w-full max-w-7xl mx-auto px-4 mb-12">
        <HfAuthBanner />
        <LandingHeader />
      </div>

      <div className="p-8 bg-gray-900 rounded-lg shadow-xl w-full max-w-4xl space-y-6 border border-gray-700">
        <RobotConfigManager
          visibleRecords={visibleRecords}
          hiddenNames={hiddenNames}
          isLoading={isLoadingRobots}
          addToSession={addToSession}
          removeFromSession={removeFromSession}
          createRobot={createRobot}
          deleteRobot={deleteRobot}
        />
        <ActionList actions={actions} />
      </div>

      <UsageInstructionsModal
        open={showUsageModal}
        onOpenChange={setShowUsageModal}
        dismissible={!ON_SPACE}
      />

      <RecordingModal
        open={showRecordingModal}
        onOpenChange={handleRecordingModalClose}
        robots={visibleRecords}
        datasetName={datasetName}
        setDatasetName={setDatasetName}
        singleTask={singleTask}
        setSingleTask={setSingleTask}
        numEpisodes={numEpisodes}
        setNumEpisodes={setNumEpisodes}
        cameras={cameras}
        setCameras={setCameras}
        onStart={handleStartRecording}
        releaseStreamsRef={releaseStreamsRef}
      />
    </div>
  );
};

export default Landing;
