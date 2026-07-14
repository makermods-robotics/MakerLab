import { useState } from "react";
import {
  ChevronsUpDown,
  CloudDownload,
  GitMerge,
  HardDrive,
  HardDriveDownload,
  Plus,
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
import DatasetPicker from "@/components/landing/DatasetPicker";
import CreateDatasetDialog from "@/components/landing/CreateDatasetDialog";
import AddDatasetFromHubDialog from "@/components/landing/AddDatasetFromHubDialog";
import ImportDatasetFromDiskDialog from "@/components/landing/ImportDatasetFromDiskDialog";
import DatasetInfoCard from "@/components/landing/DatasetInfoCard";
import MergeDatasetsDialog from "@/components/landing/MergeDatasetsDialog";
import ManageCachesDialog from "@/components/landing/ManageCachesDialog";
import { useApi } from "@/contexts/ApiContext";
import { useDatasets } from "@/hooks/useDatasets";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";
import {
  DatasetInfo,
  DatasetItem,
  deleteDataset,
  downloadDataset,
  hideDataset,
  removeCustomDataset,
  saveCustomDataset,
} from "@/lib/replayApi";
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

interface DatasetsPanelProps {
  // "Record a dataset" → hand a new dataset name up to the page, which seeds the
  // recording form and opens the (page-level) recording modal configured with
  // the selected robot + cameras.
  onRecordNew: (name: string) => void;
  // "Record more episodes" on a local dataset's info card → resume handoff into
  // the page-level recording modal, carrying the dataset's info summary.
  onResumeRecording: (info: DatasetInfo) => void;
}

/**
 * Datasets browser column — the dataset picker + info card + "Add dataset"
 * chooser and its dialogs (record / add-from-Hub / import-from-disk), the shared
 * delete-confirm pipeline, download wiring, and the merge / manage-caches
 * footer. Self-contained except for the recording-modal handoff, which is
 * page-level (see the two callbacks in DatasetsPanelProps).
 */
const DatasetsPanel = ({ onRecordNew, onResumeRecording }: DatasetsPanelProps) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

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

  return (
    <>
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
            onResume={onResumeRecording}
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
        onCreateNew={onRecordNew}
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

      {/* One delete confirm per kind, shared by the picker rows and the info
          cards, with copy driven by resolveDeleteAction (Delete / Remove local
          copy / Remove-from-list). Rendered at panel scope so it survives the
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
    </>
  );
};

export default DatasetsPanel;
