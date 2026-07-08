import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ChevronsUpDown,
  CloudDownload,
  HardDriveDownload,
  Plus,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useToast } from "@/hooks/use-toast";
import ModelPicker from "@/components/landing/ModelPicker";
import ModelInfoCard from "@/components/landing/ModelInfoCard";
import ModelLaunchFooter from "@/components/landing/ModelLaunchFooter";
import AddModelFromHubDialog from "@/components/landing/AddModelFromHubDialog";
import ImportModelFromDiskDialog from "@/components/landing/ImportModelFromDiskDialog";
import { useApi } from "@/contexts/ApiContext";
import { useModels } from "@/hooks/useModels";
import { useSelectedModel } from "@/hooks/useSelectedModel";
import { useInferenceLaunch } from "@/hooks/useInferenceLaunch";
import {
  ModelItem,
  deleteModel,
  downloadModel,
  hideModel,
  removeCustomModel,
  saveCustomModel,
} from "@/lib/modelsApi";
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

/**
 * Models browser column — mirrors the Datasets panel: a Hub/Local selector with
 * a per-model info card, an "Add model" chooser (train via the dedicated page's
 * policy grid / add from the Hub / import a checkpoint from disk), and the
 * deploy footer that runs inference via the shared launch machinery. Fully
 * self-contained: it owns the merged /models listing, the home-page model
 * selection, and its delete-confirm + dialogs.
 */
const ModelsPanel = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const navigate = useNavigate();

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

  // Shared inference-launch machinery (the same hook JobsSection consumes):
  // the Models panel footer's "Run inference" drives it, and `modal` renders
  // the one InferenceModal instance for this page section.
  const inferenceLaunch = useInferenceLaunch();

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

  return (
    <>
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
        {/* Deploy footer — mirrors the dataset panel's footer row: pick a
            checkpoint (defaults to latest) and run inference on the
            selected model, via the same launch machinery the Jobs cards
            use (shared useInferenceLaunch). */}
        <ModelLaunchFooter
          model={selectedModelItem}
          onPlay={inferenceLaunch.play}
          onImportSource={inferenceLaunch.importSource}
        />
        {inferenceLaunch.modal}
      </div>

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
    </>
  );
};

export default ModelsPanel;
