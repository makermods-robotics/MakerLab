import React, { useEffect, useState } from "react";
import { HardDriveDownload, Loader2 } from "lucide-react";
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
import { useApi } from "@/contexts/ApiContext";
import { ApiError } from "@/lib/apiClient";
import { importDataset } from "@/lib/replayApi";

interface ImportDatasetFromDiskDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called with the imported dataset's repo id once the copy succeeds, so the
   * parent can select it and refresh the picker. */
  onImported: (repoId: string) => void;
}

/**
 * "Import a dataset from disk" form. Points at a LeRobot dataset folder already
 * on the server machine; the backend COPIES it into the local cache (the source
 * is left intact) and it appears under "Local". An optional name overrides the
 * target repo id (defaults to the source folder's basename). The copy runs
 * synchronously — the dialog shows a spinner until it completes.
 */
const ImportDatasetFromDiskDialog: React.FC<
  ImportDatasetFromDiskDialogProps
> = ({ open, onOpenChange, onImported }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setPath("");
      setName("");
      setError(null);
      setImporting(false);
    }
  }, [open]);

  const trimmedPath = path.trim();

  const handleSubmit = async () => {
    if (!trimmedPath || importing) return;
    setImporting(true);
    setError(null);
    try {
      const res = await importDataset(
        baseUrl,
        fetchWithHeaders,
        trimmedPath,
        name.trim() || undefined,
      );
      onImported(res.repo_id);
      onOpenChange(false);
    } catch (e) {
      setError(
        e instanceof ApiError && e.detail
          ? e.detail
          : e instanceof Error
            ? e.message
            : String(e),
      );
    } finally {
      setImporting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-gray-800 border-gray-700 text-white sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-white">
            Import a dataset from disk
          </DialogTitle>
          <DialogDescription className="text-gray-400">
            Point at a LeRobot dataset folder already on this machine. It's
            copied into your local cache — the original folder is left untouched.
          </DialogDescription>
        </DialogHeader>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void handleSubmit();
          }}
          className="space-y-4"
        >
          <div>
            <Label htmlFor="import-dataset-path" className="text-gray-300">
              Dataset folder path
            </Label>
            <Input
              id="import-dataset-path"
              autoFocus
              value={path}
              onChange={(e) => {
                setPath(e.target.value);
                setError(null);
              }}
              placeholder="/path/to/lerobot_dataset"
              className="mt-1 bg-gray-900 border-gray-600 text-white"
            />
          </div>
          <div>
            <Label htmlFor="import-dataset-name" className="text-gray-300">
              Name (optional)
            </Label>
            <Input
              id="import-dataset-name"
              value={name}
              onChange={(e) =>
                setName(e.target.value.replace(/[^A-Za-z0-9._\-/]/g, "_"))
              }
              placeholder="Defaults to the folder name"
              className="mt-1 bg-gray-900 border-gray-600 text-white"
            />
          </div>
          {error && <p className="text-xs text-red-400">{error}</p>}
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
              disabled={!trimmedPath || importing}
              className="bg-green-500 hover:bg-green-600 text-white"
            >
              {importing ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <HardDriveDownload className="w-4 h-4 mr-2" />
              )}
              {importing ? "Importing…" : "Import"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default ImportDatasetFromDiskDialog;
