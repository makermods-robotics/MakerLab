import React, { useEffect, useState } from "react";
import { Download, Plus } from "lucide-react";
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
import { HUB_REPO_ID_RE } from "@/lib/repoId";

interface AddDatasetFromHubDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Add the typed Hub id to the picker (pin + select). When `download` is true
   * the parent also kicks off a background download into the local cache. */
  onAdd: (repoId: string, download: boolean) => void;
}

/**
 * "Add a dataset from Hugging Face" form. Takes a `namespace/name` Hub repo id,
 * validated with the same rule as the backend, and adds it to the picker's
 * Hugging Face list (pin + select via the parent). An optional "Download to
 * this machine now" toggle additionally starts a background download so the
 * dataset is available locally (source flips to "both"); left off, the dataset
 * is listed and training fetches it from the Hub on demand.
 */
const AddDatasetFromHubDialog: React.FC<AddDatasetFromHubDialogProps> = ({
  open,
  onOpenChange,
  onAdd,
}) => {
  const [repoId, setRepoId] = useState("");
  const [download, setDownload] = useState(false);

  useEffect(() => {
    if (open) {
      setRepoId("");
      setDownload(false);
    }
  }, [open]);

  const trimmed = repoId.trim();
  const isValid = HUB_REPO_ID_RE.test(trimmed);
  const showError = trimmed.length > 0 && !isValid;

  const handleConfirm = () => {
    if (!isValid) return;
    onAdd(trimmed, download);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            Add a dataset from Hugging Face
          </DialogTitle>
          <DialogDescription className="text-muted-foreground">
            Enter a Hub dataset id to add it to your list. It appears under
            “Hugging Face” and training fetches it on demand.
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
            <Label htmlFor="hub-dataset-id" className="text-muted-foreground">
              Hub dataset id
            </Label>
            <Input
              id="hub-dataset-id"
              autoFocus
              value={repoId}
              onChange={(e) =>
                setRepoId(e.target.value.replace(/[^A-Za-z0-9._\-/]/g, ""))
              }
              placeholder="org/name"
              aria-invalid={showError}
              className="mt-1 aria-[invalid=true]:border-destructive/70"
            />
            {showError && (
              <p className="mt-1 text-xs text-destructive">
                Enter a Hub dataset id as <span className="font-mono">org/name</span>.
              </p>
            )}
          </div>
          <label className="flex items-start gap-2 text-sm text-muted-foreground cursor-pointer">
            <input
              type="checkbox"
              checked={download}
              onChange={(e) => setDownload(e.target.checked)}
              className="mt-0.5 h-4 w-4 accent-blue-500"
            />
            <span>
              Download to this machine now
              <span className="block text-xs text-muted-foreground">
                Fetches the dataset into the local cache in the background. It can
                be multi-GB.
              </span>
            </span>
          </label>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              className=""
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={!isValid}
              className=""
            >
              {download ? (
                <Download className="w-4 h-4 mr-2" />
              ) : (
                <Plus className="w-4 h-4 mr-2" />
              )}
              {download ? "Add & download" : "Add dataset"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default AddDatasetFromHubDialog;
