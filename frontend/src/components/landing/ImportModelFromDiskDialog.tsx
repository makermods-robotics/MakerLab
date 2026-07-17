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
import { importModelFromDisk } from "@/lib/modelsApi";

interface ImportModelFromDiskDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called with the imported model's id once the copy succeeds, so the parent
   * can select it and refresh the picker. */
  onImported: (repoId: string) => void;
}

/**
 * "Import a model from disk" form — the models twin of
 * ImportDatasetFromDiskDialog. Points at a policy checkpoint folder already on
 * the server machine (a pretrained_model dir with config.json, or a training
 * output with a checkpoints/<step>/pretrained_model tree); the backend COPIES
 * it into the local models dir (the source is left intact) and it appears under
 * "Local". An optional name overrides the target id (defaults to the source
 * folder's basename). The copy runs synchronously — the dialog shows a spinner
 * until it completes.
 */
const ImportModelFromDiskDialog: React.FC<ImportModelFromDiskDialogProps> = ({
  open,
  onOpenChange,
  onImported,
}) => {
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
      const res = await importModelFromDisk(
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
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            Import a model from disk
          </DialogTitle>
          <DialogDescription className="text-muted-foreground">
            Point at a policy checkpoint folder already on this machine. It's
            copied into your local models cache — the original folder is left
            untouched.
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
            <Label htmlFor="import-model-path" className="text-muted-foreground">
              Checkpoint folder path
            </Label>
            <Input
              id="import-model-path"
              autoFocus
              value={path}
              onChange={(e) => {
                setPath(e.target.value);
                setError(null);
              }}
              placeholder="/path/to/pretrained_model"
              className="mt-1"
            />
          </div>
          <div>
            <Label htmlFor="import-model-name" className="text-muted-foreground">
              Name (optional)
            </Label>
            <Input
              id="import-model-name"
              value={name}
              onChange={(e) =>
                setName(e.target.value.replace(/[^A-Za-z0-9._\-/]/g, "_"))
              }
              placeholder="Defaults to the folder name"
              className="mt-1"
            />
          </div>
          {error && <p className="text-xs text-destructive">{error}</p>}
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
              disabled={!trimmedPath || importing}
              className=""
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

export default ImportModelFromDiskDialog;
