import React, { useState } from "react";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { validateDatasetName } from "@/lib/datasetName";

interface CreateDatasetDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Existing repo ids, used to warn before offering a name that already
   * exists (matched case-insensitively against the bare name). */
  existingRepoIds: string[];
  /** Called with the validated bare dataset name. The parent seeds the
   * recording modal (which applies any namespace/prefix on start). */
  onCreateNew: (name: string) => void;
}

/**
 * Name-only form for creating a new dataset. Uses the same
 * `validateDatasetName` the picker footer used, so the recorder never receives
 * a name it would later reject. On confirm it hands the bare name to the parent
 * (handleCreateDataset), which opens the recording modal.
 */
const CreateDatasetDialog: React.FC<CreateDatasetDialogProps> = ({
  open,
  onOpenChange,
  existingRepoIds,
  onCreateNew,
}) => {
  const [name, setName] = useState("");

  React.useEffect(() => {
    if (open) setName("");
  }, [open]);

  const trimmed = name.trim();
  // Mirrors DatasetPicker: an exact repo_id match (case-insensitive) already
  // exists. The bare name may or may not carry a namespace, so also compare
  // against the trailing segment of each repo id.
  const matchesExisting = existingRepoIds.some((id) => {
    const bare = id.split("/").pop() ?? id;
    return (
      id.toLowerCase() === trimmed.toLowerCase() ||
      bare.toLowerCase() === trimmed.toLowerCase()
    );
  });
  const nameError = trimmed === "" ? null : validateDatasetName(trimmed);
  const canCreate = trimmed !== "" && nameError === null && !matchesExisting;

  const handleConfirm = () => {
    if (!canCreate) return;
    onCreateNew(trimmed);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-gray-800 border-gray-700 text-white sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-white">Create a new dataset</DialogTitle>
          <DialogDescription className="text-gray-400">
            Name the dataset you're about to record. You'll set the task and
            episode count in the next step.
          </DialogDescription>
        </DialogHeader>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleConfirm();
          }}
          className="space-y-4"
        >
          <div>
            <Label htmlFor="new-dataset-name" className="text-gray-300">
              Name
            </Label>
            <Input
              id="new-dataset-name"
              autoFocus
              value={name}
              onChange={(e) =>
                setName(e.target.value.replace(/[^A-Za-z0-9._\-/]/g, "_"))
              }
              placeholder="my_dataset"
              aria-invalid={nameError !== null || matchesExisting}
              className="mt-1 bg-gray-900 border-gray-600 text-white aria-[invalid=true]:border-red-500/70"
            />
            {matchesExisting ? (
              <p className="mt-1 text-xs text-red-400">
                A dataset with this name already exists.
              </p>
            ) : (
              nameError && (
                <p className="mt-1 text-xs text-red-400">{nameError}</p>
              )
            )}
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              className="bg-transparent border-gray-600 text-white hover:bg-gray-700 hover:text-white"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={!canCreate}
              className="bg-green-500 hover:bg-green-600 text-white"
            >
              <Plus className="w-4 h-4 mr-2" /> Create
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default CreateDatasetDialog;
