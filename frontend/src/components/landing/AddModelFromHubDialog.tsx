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

interface AddModelFromHubDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Add the typed Hub id to the picker (pin + select). When `download` is true
   * the parent also kicks off a background download into the local models dir. */
  onAdd: (repoId: string, download: boolean) => void;
}

/**
 * "Add a model from Hugging Face" form — the models twin of
 * AddDatasetFromHubDialog. Takes a `namespace/name` Hub repo id, validated with
 * the same rule as the backend, and adds it to the picker's Hugging Face list
 * (pin + select via the parent). An optional "Download to this machine now"
 * toggle additionally starts a background download of the checkpoint so
 * inference can run on it offline (source flips to "both"); left off, the model
 * is listed and inference fetches it from the Hub on demand.
 */
const AddModelFromHubDialog: React.FC<AddModelFromHubDialogProps> = ({
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
      <DialogContent className="bg-gray-800 border-gray-700 text-white sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-white">
            Add a model from Hugging Face
          </DialogTitle>
          <DialogDescription className="text-gray-400">
            Enter a Hub model id to add it to your list. It appears under
            “Hugging Face” and inference fetches it on demand.
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
            <Label htmlFor="hub-model-id" className="text-gray-300">
              Hub model id
            </Label>
            <Input
              id="hub-model-id"
              autoFocus
              value={repoId}
              onChange={(e) =>
                setRepoId(e.target.value.replace(/[^A-Za-z0-9._\-/]/g, ""))
              }
              placeholder="org/name"
              aria-invalid={showError}
              className="mt-1 bg-gray-900 border-gray-600 text-white aria-[invalid=true]:border-red-500/70"
            />
            {showError && (
              <p className="mt-1 text-xs text-red-400">
                Enter a Hub model id as <span className="font-mono">org/name</span>.
              </p>
            )}
          </div>
          <label className="flex items-start gap-2 text-sm text-gray-300 cursor-pointer">
            <input
              type="checkbox"
              checked={download}
              onChange={(e) => setDownload(e.target.checked)}
              className="mt-0.5 h-4 w-4 accent-blue-500"
            />
            <span>
              Download to this machine now
              <span className="block text-xs text-gray-500">
                Fetches the checkpoint into the local models cache in the
                background, so inference works offline.
              </span>
            </span>
          </label>
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
              disabled={!isValid}
              className="bg-green-500 hover:bg-green-600 text-white"
            >
              {download ? (
                <Download className="w-4 h-4 mr-2" />
              ) : (
                <Plus className="w-4 h-4 mr-2" />
              )}
              {download ? "Add & download" : "Add model"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default AddModelFromHubDialog;
