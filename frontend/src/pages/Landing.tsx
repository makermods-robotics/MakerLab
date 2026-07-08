import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  ChevronsUpDown,
  CloudDownload,
  GitMerge,
  HardDrive,
  HardDriveDownload,
  Plus,
  Sparkles,
  Video,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useToast } from "@/hooks/use-toast";
import LandingTopBar from "@/components/landing/LandingTopBar";
import Footer from "@/components/Footer";
import RobotConfigManager from "@/components/landing/RobotConfigManager";
import RecordingModal from "@/components/landing/RecordingModal";
import DatasetPicker from "@/components/landing/DatasetPicker";
import CreateDatasetDialog from "@/components/landing/CreateDatasetDialog";
import AddDatasetFromHubDialog from "@/components/landing/AddDatasetFromHubDialog";
import ImportDatasetFromDiskDialog from "@/components/landing/ImportDatasetFromDiskDialog";
import AddModelFromHubDialog from "@/components/landing/AddModelFromHubDialog";
import ImportModelFromDiskDialog from "@/components/landing/ImportModelFromDiskDialog";
import DatasetInfoCard from "@/components/landing/DatasetInfoCard";
import ModelPicker from "@/components/landing/ModelPicker";
import ModelInfoCard from "@/components/landing/ModelInfoCard";
import MergeDatasetsDialog from "@/components/landing/MergeDatasetsDialog";
import ManageCachesDialog from "@/components/landing/ManageCachesDialog";
import JobsSection from "@/components/jobs/JobsSection";

import UsageInstructionsModal from "@/components/landing/UsageInstructionsModal";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useApi } from "@/contexts/ApiContext";
import { useRobots } from "@/hooks/useRobots";
import { useDatasets } from "@/hooks/useDatasets";
import { useModels } from "@/hooks/useModels";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";
import { useSelectedModel } from "@/hooks/useSelectedModel";
import {
  DatasetInfo,
  DatasetItem,
  deleteDataset,
  downloadDataset,
  hideDataset,
  removeCustomDataset,
  saveCustomDataset,
} from "@/lib/replayApi";
import {
  ModelItem,
  deleteModel,
  downloadModel,
  hideModel,
  removeCustomModel,
  saveCustomModel,
} from "@/lib/modelsApi";
import { CameraConfig } from "@/components/recording/CameraConfiguration";
import { isHostedSpace } from "@/lib/isHostedSpace";
import { validateDatasetName } from "@/lib/datasetName";
import { resolveDeleteAction } from "@/lib/deleteSemantics";
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
  const [showManageCachesDialog, setShowManageCachesDialog] = useState(false);
  const [showCreateDatasetDialog, setShowCreateDatasetDialog] = useState(false);
  const [showAddFromHubDialog, setShowAddFromHubDialog] = useState(false);
  const [showImportDatasetDialog, setShowImportDatasetDialog] = useState(false);
  const [pendingDeleteDataset, setPendingDeleteDataset] =
    useState<DatasetItem | null>(null);
  const { selectedDataset, setSelectedDataset } = useSelectedDataset();

  // The DatasetItem for the current selection (if it's in the known list). Used
  // to gate the info card's delete affordance to local-only datasets and to
  // route the right item through the confirm dialog. A custom repo opened by id
  // won't match — no local item, so no card delete, which is correct.
  const selectedDatasetItem =
    datasets.find((d) => d.repo_id === selectedDataset) ?? null;

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

  // Models panel (mirrors the Datasets panel): the merged /models listing plus
  // the home-page model selection, persisted like the dataset selection.
  const {
    models,
    loading: modelsLoading,
    refresh: refreshModels,
  } = useModels();
  const { selectedModel, setSelectedModel } = useSelectedModel();
  const [pendingDeleteModel, setPendingDeleteModel] = useState<ModelItem | null>(
    null,
  );
  const [showAddModelFromHubDialog, setShowAddModelFromHubDialog] =
    useState(false);
  const [showImportModelDialog, setShowImportModelDialog] = useState(false);

  // The ModelItem for the current selection (if it's in the known list). Gates
  // the info card's upload/delete affordances to local-only models and routes
  // the right item through the confirm dialog.
  const selectedModelItem =
    models.find((m) => m.id === selectedModel) ?? null;

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

  // Picking a model selects it as the home-page model (persisted).
  const handlePickModel = (item: ModelItem) => {
    setSelectedModel(item.id);
    toast({ title: "Model selected", description: item.name });
  };

  // Deleting a local model removes its run output dir — destructive, so route
  // the card's trash button through a styled confirm dialog.
  const handleDeleteModel = (item: ModelItem) => {
    setPendingDeleteModel(item);
  };

  // One confirm path for every model delete entry point (picker row + info
  // card). resolveDeleteAction decides the semantics: local file delete
  // (local-only run/checkpoint), local-copy removal ("both" — the hub row
  // stays LISTED and SELECTED), unpin (pinned custom), or hide (own hub-only,
  // persistent hidden-list — the Hub repo is never touched).
  const confirmDeleteModel = async () => {
    const item = pendingDeleteModel;
    if (!item) return;
    setPendingDeleteModel(null);
    const res = resolveDeleteAction("model", item);

    try {
      if (res.action === "unpin") {
        await removeCustomModel(baseUrl, fetchWithHeaders, item.id);
        toast({ title: "Removed from list", description: item.name });
      } else if (res.action === "hide") {
        await hideModel(baseUrl, fetchWithHeaders, item.hf_repo_id ?? item.id);
        toast({ title: "Removed from list", description: item.name });
      } else {
        const r = await deleteModel(baseUrl, fetchWithHeaders, item.id);
        if (!r.deleted) return;
        toast({
          title:
            res.action === "delete-local-copy"
              ? "Local copy removed"
              : "Model deleted",
          description: item.name,
        });
      }
      if (res.clearsSelection) {
        if (selectedModel === item.id) setSelectedModel("");
      } else if (
        selectedModel === item.id &&
        item.hf_repo_id &&
        item.hf_repo_id !== item.id
      ) {
        // both -> hub flip: the local run id vanishes from the listing but the
        // model still exists as its hub repo — repoint the kept selection so
        // the card and picker keep resolving it.
        setSelectedModel(item.hf_repo_id);
      }
      refreshModels();
    } catch (e) {
      toast({
        title: res.action === "delete-local" ? "Delete failed" : "Couldn't remove",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  // "Add a model from Hugging Face": pin + select the typed Hub id (the models
  // twin of handleAddFromHub), and optionally kick off a background download of
  // the checkpoint into the local models dir. The download is fire-and-forget
  // here — the info card for the now-selected model re-attaches to it and shows
  // the "Downloading…" state.
  const handleAddModelFromHub = async (repoId: string, download: boolean) => {
    setSelectedModel(repoId);
    toast({ title: "Model saved", description: repoId });
    try {
      await saveCustomModel(baseUrl, fetchWithHeaders, repoId);
      refreshModels();
    } catch {
      // Non-fatal: the model is still selected for this session.
    }
    if (!download) return;
    try {
      await downloadModel(baseUrl, fetchWithHeaders, repoId);
      toast({ title: "Download started", description: repoId });
    } catch (e) {
      toast({
        title: "Couldn't start download",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  // "Import a model from disk": the dialog copied a checkpoint folder into the
  // local models dir and returns the new id — select it and refresh the picker
  // so it shows under "Local".
  const handleImportedModelFromDisk = (repoId: string) => {
    setSelectedModel(repoId);
    refreshModels();
    toast({ title: "Model imported", description: repoId });
  };

  // Picking a dataset here selects it for training (the single source of truth);
  // Training reads it from the persisted selection.
  const handlePickExisting = (item: DatasetItem) => {
    setSelectedDataset(item.repo_id);
    toast({ title: "Dataset selected", description: item.repo_id });
  };

  // Using a typed-in Hub dataset both selects it AND pins it, so it persists in
  // the picker's Hugging Face list for next time (backend saved_custom). The pin
  // is best-effort — selection still works if the save call fails.
  const handleOpenCustom = async (repoId: string) => {
    setSelectedDataset(repoId);
    toast({ title: "Dataset saved", description: repoId });
    try {
      await saveCustomDataset(baseUrl, fetchWithHeaders, repoId);
      refreshDatasets();
    } catch {
      // Non-fatal: the dataset is still selected for training this session.
    }
  };

  const handleCreateDataset = (name: string) => {
    setDatasetName(name);
    openRecordingModal();
  };

  // "Add a dataset from Hugging Face": pin + select the typed Hub id (reusing
  // handleOpenCustom), and optionally kick off a background download into the
  // local cache. The download is fire-and-forget here — the info card for the
  // now-selected dataset re-attaches to it and shows the "Downloading…" state.
  const handleAddFromHub = async (repoId: string, download: boolean) => {
    await handleOpenCustom(repoId);
    if (!download) return;
    try {
      await downloadDataset(baseUrl, fetchWithHeaders, repoId);
      toast({ title: "Download started", description: repoId });
    } catch (e) {
      toast({
        title: "Couldn't start download",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  // "Import a dataset from disk": the dialog copied a local folder into the
  // cache and returns the new repo id — select it and refresh the picker so it
  // shows under "Local".
  const handleImportedFromDisk = (repoId: string) => {
    setSelectedDataset(repoId);
    refreshDatasets();
    toast({ title: "Dataset imported", description: repoId });
  };

  // Deleting a dataset is destructive and irreversible, so route the picker's
  // trash button through a styled confirm dialog instead of deleting inline.
  const handleDeleteDataset = (item: DatasetItem) => {
    setPendingDeleteDataset(item);
  };

  // One confirm path for every dataset delete entry point (picker row + info
  // card), mirroring confirmDeleteModel. resolveDeleteAction decides the
  // semantics; a "both" first press removes only the local copy — the row
  // stays listed as a hub dataset and the selection is kept (repo ids don't
  // change on the flip, unlike models).
  const confirmDeleteDataset = async () => {
    const item = pendingDeleteDataset;
    if (!item) return;
    setPendingDeleteDataset(null);
    const res = resolveDeleteAction("dataset", item);

    try {
      if (res.action === "unpin") {
        await removeCustomDataset(baseUrl, fetchWithHeaders, item.repo_id);
        toast({ title: "Removed from list", description: item.repo_id });
      } else if (res.action === "hide") {
        await hideDataset(baseUrl, fetchWithHeaders, item.repo_id);
        toast({ title: "Removed from list", description: item.repo_id });
      } else {
        const r = await deleteDataset(baseUrl, fetchWithHeaders, item.repo_id);
        if (!r.success) {
          toast({
            title: "Delete failed",
            description: r.message ?? "Could not delete the dataset.",
            variant: "destructive",
          });
          return;
        }
        toast({
          title:
            res.action === "delete-local-copy"
              ? "Local copy removed"
              : "Dataset deleted",
          description: item.repo_id,
        });
      }
      // Clear the stale pick only when the row fully vanished (hide / unpin /
      // local-only delete) — a both->hub flip keeps the selection.
      if (res.clearsSelection && selectedDataset === item.repo_id) {
        setSelectedDataset("");
      }
      refreshDatasets();
    } catch (e) {
      toast({
        title: res.action === "delete-local" ? "Delete failed" : "Couldn't remove",
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
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex flex-col gap-2">
            <h3 className="font-semibold text-lg text-center h-10 flex items-center justify-center">
              Dataset
            </h3>
            <div className="flex items-center gap-2">
              <div className="flex-1 min-w-0">
                <DatasetPicker
                  datasets={datasets}
                  loading={datasetsLoading}
                  onPickExisting={handlePickExisting}
                  onDeleteItem={handleDeleteDataset}
                >
                  <Button
                    variant="outline"
                    role="combobox"
                    className="w-full justify-between bg-gray-800 border-gray-600 text-white hover:bg-gray-700"
                  >
                    <span
                      className={`truncate ${selectedDataset ? "text-white" : "text-gray-300"}`}
                    >
                      {datasetsLoading
                        ? "Loading datasets…"
                        : (selectedDataset ?? "Select a dataset…")}
                    </span>
                    <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                  </Button>
                </DatasetPicker>
              </div>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 shrink-0 border-gray-600 bg-gray-800 text-white hover:bg-gray-700 hover:text-white"
                  >
                    <Plus className="w-3.5 h-3.5 mr-1.5" />
                    Add dataset
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent
                  align="end"
                  className="w-56 bg-gray-800 border-gray-700 text-white"
                >
                  <DropdownMenuItem
                    onSelect={() => setShowCreateDatasetDialog(true)}
                    className="text-white focus:bg-gray-700 focus:text-white"
                  >
                    <Video className="mr-2 h-4 w-4" />
                    Record a dataset
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onSelect={() => setShowAddFromHubDialog(true)}
                    className="text-white focus:bg-gray-700 focus:text-white"
                  >
                    <CloudDownload className="mr-2 h-4 w-4" />
                    Add from Hugging Face
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onSelect={() => setShowImportDatasetDialog(true)}
                    className="text-white focus:bg-gray-700 focus:text-white"
                  >
                    <HardDriveDownload className="mr-2 h-4 w-4" />
                    Import from disk
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
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
                // Every known row now has delete semantics (local delete /
                // local-copy removal / unpin / hide — see resolveDeleteAction),
                // so the trash shows for any listed selection. Clicking routes
                // through the shared confirm dialog.
                canDelete={!!selectedDatasetItem}
                onDelete={
                  selectedDatasetItem
                    ? () => handleDeleteDataset(selectedDatasetItem)
                    : undefined
                }
                // A Hub-only dataset just got downloaded — refresh the listing so
                // its source flips to "both" and the card gets its local detail.
                onDownloaded={refreshDatasets}
                // "Record more episodes" — opens the recording modal in resume
                // mode with this dataset's info (local datasets only).
                onResume={handleResumeRecording}
              />
            )}
            <div className="flex items-center gap-4">
              <button
                type="button"
                onClick={() => setShowMergeDialog(true)}
                className="text-xs text-gray-400 hover:text-white transition-colors inline-flex items-center gap-1"
              >
                <GitMerge className="h-3.5 w-3.5" /> Merge datasets…
              </button>
              <button
                type="button"
                onClick={() => setShowManageCachesDialog(true)}
                className="text-xs text-gray-400 hover:text-white transition-colors inline-flex items-center gap-1"
              >
                <HardDrive className="h-3.5 w-3.5" /> Manage cached datasets…
              </button>
            </div>
          </div>
          {/* Models browser — mirrors the Dataset panel: a Hub/Local selector
              with a per-model info card, plus an "Add model" chooser (train via
              the dedicated page's policy grid / add from the Hub / import a
              checkpoint from disk). */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex flex-col gap-2">
            <h3 className="font-semibold text-lg text-center h-10 flex items-center justify-center">
              Models
            </h3>
            <div className="flex items-center gap-2">
              <div className="flex-1 min-w-0">
                <ModelPicker
                  models={models}
                  loading={modelsLoading}
                  onPickExisting={handlePickModel}
                  onDeleteItem={handleDeleteModel}
                >
                  <Button
                    variant="outline"
                    role="combobox"
                    className="w-full justify-between bg-gray-800 border-gray-600 text-white hover:bg-gray-700"
                  >
                    <span
                      className={`truncate ${selectedModel ? "text-white" : "text-gray-300"}`}
                    >
                      {modelsLoading
                        ? "Loading models…"
                        : (selectedModelItem?.name ??
                          selectedModel ??
                          "Select a model…")}
                    </span>
                    <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                  </Button>
                </ModelPicker>
              </div>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 shrink-0 border-gray-600 bg-gray-800 text-white hover:bg-gray-700 hover:text-white"
                  >
                    <Plus className="w-3.5 h-3.5 mr-1.5" />
                    Add model
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent
                  align="end"
                  className="w-56 bg-gray-800 border-gray-700 text-white"
                >
                  <DropdownMenuItem
                    onSelect={() => navigate("/create-model")}
                    className="text-white focus:bg-gray-700 focus:text-white"
                  >
                    <Sparkles className="mr-2 h-4 w-4" />
                    Train a model
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onSelect={() => setShowAddModelFromHubDialog(true)}
                    className="text-white focus:bg-gray-700 focus:text-white"
                  >
                    <CloudDownload className="mr-2 h-4 w-4" />
                    Add from Hugging Face
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onSelect={() => setShowImportModelDialog(true)}
                    className="text-white focus:bg-gray-700 focus:text-white"
                  >
                    <HardDriveDownload className="mr-2 h-4 w-4" />
                    Import from disk
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
            {selectedModel && (
              <ModelInfoCard
                id={selectedModel}
                // Upload + Delete are local-only, mirroring the dataset card's
                // gating. A custom/unknown id won't match — no local item, so
                // no local affordances, which is correct.
                isLocal={selectedModelItem?.source === "local"}
                // Every known row now has delete semantics (local delete /
                // local-copy removal / unpin / hide — see resolveDeleteAction),
                // so the trash shows for any listed selection. Clicking routes
                // through the shared confirm dialog.
                canDelete={!!selectedModelItem}
                onDelete={
                  selectedModelItem
                    ? () => handleDeleteModel(selectedModelItem)
                    : undefined
                }
                onUploaded={refreshModels}
                // A Hub-only model just got downloaded — refresh the listing so
                // its source flips to "both" and the card gets its local detail.
                onDownloaded={refreshModels}
              />
            )}
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

      <ManageCachesDialog
        open={showManageCachesDialog}
        onOpenChange={setShowManageCachesDialog}
        datasets={datasets}
        onCleared={refreshDatasets}
      />

      <CreateDatasetDialog
        open={showCreateDatasetDialog}
        onOpenChange={setShowCreateDatasetDialog}
        existingRepoIds={datasets.map((d) => d.repo_id)}
        onCreateNew={handleCreateDataset}
      />

      <AddDatasetFromHubDialog
        open={showAddFromHubDialog}
        onOpenChange={setShowAddFromHubDialog}
        onAdd={handleAddFromHub}
      />

      <ImportDatasetFromDiskDialog
        open={showImportDatasetDialog}
        onOpenChange={setShowImportDatasetDialog}
        onImported={handleImportedFromDisk}
      />

      <AddModelFromHubDialog
        open={showAddModelFromHubDialog}
        onOpenChange={setShowAddModelFromHubDialog}
        onAdd={handleAddModelFromHub}
      />

      <ImportModelFromDiskDialog
        open={showImportModelDialog}
        onOpenChange={setShowImportModelDialog}
        onImported={handleImportedModelFromDisk}
      />

      {/* One delete confirm per kind, shared by the picker rows and the info
          cards, with copy driven by resolveDeleteAction (Delete / Remove local
          copy / Remove-from-list). Rendered at Landing scope so it survives the
          picker popover closing. */}
      {(() => {
        const res = pendingDeleteDataset
          ? resolveDeleteAction("dataset", pendingDeleteDataset)
          : null;
        return (
          <AlertDialog
            open={pendingDeleteDataset !== null}
            onOpenChange={(o) => !o && setPendingDeleteDataset(null)}
          >
            <AlertDialogContent className="bg-gray-900 border-gray-800 text-white">
              <AlertDialogHeader>
                <AlertDialogTitle className="break-words">
                  {res?.titlePrefix} "
                  <span className="break-all">
                    {pendingDeleteDataset?.repo_id}
                  </span>
                  "?
                </AlertDialogTitle>
                <AlertDialogDescription className="text-gray-400">
                  {res?.description}
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel className="border-gray-600 bg-transparent text-gray-200 hover:bg-gray-800 hover:text-white">
                  Cancel
                </AlertDialogCancel>
                <AlertDialogAction
                  onClick={confirmDeleteDataset}
                  className="bg-red-500 hover:bg-red-600 text-white"
                >
                  {res?.confirmLabel}
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        );
      })()}

      {(() => {
        const res = pendingDeleteModel
          ? resolveDeleteAction("model", pendingDeleteModel)
          : null;
        return (
          <AlertDialog
            open={pendingDeleteModel !== null}
            onOpenChange={(o) => !o && setPendingDeleteModel(null)}
          >
            <AlertDialogContent className="bg-gray-900 border-gray-800 text-white">
              <AlertDialogHeader>
                <AlertDialogTitle className="break-words">
                  {res?.titlePrefix} "
                  <span className="break-all">{pendingDeleteModel?.name}</span>
                  "?
                </AlertDialogTitle>
                <AlertDialogDescription className="text-gray-400">
                  {res?.description}
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel className="border-gray-600 bg-transparent text-gray-200 hover:bg-gray-800 hover:text-white">
                  Cancel
                </AlertDialogCancel>
                <AlertDialogAction
                  onClick={confirmDeleteModel}
                  className="bg-red-500 hover:bg-red-600 text-white"
                >
                  {res?.confirmLabel}
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        );
      })()}

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
