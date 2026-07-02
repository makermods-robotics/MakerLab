import React, { useRef, useState } from "react";
import { Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";

interface ImportCalibrationButtonProps {
  /** API device vocabulary: "teleop" (leader) or "robot" (follower). */
  device: "teleop" | "robot";
  /** Called with the saved config name after a successful import. */
  onImported?: (name: string) => void;
}

/**
 * Import a raw lerobot calibration JSON into a side's config library.
 * Reads + parses the file client-side, then POSTs {name, data} to the upload
 * endpoint which shape-validates and never overwrites (409 → rename prompt).
 */
const ImportCalibrationButton: React.FC<ImportCalibrationButtonProps> = ({
  device,
  onImported,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [data, setData] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const pickFile = () => fileInputRef.current?.click();

  const handleFileChosen = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    // Reset the input so re-choosing the same file fires onChange again.
    e.target.value = "";
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      setData(parsed);
      // Default the name to the file's stem; user can edit before importing.
      setName(file.name.replace(/\.json$/i, ""));
      setError(null);
      setOpen(true);
    } catch {
      toast({
        title: "Not a valid JSON file",
        description: "The selected file could not be parsed as JSON.",
        variant: "destructive",
      });
    }
  };

  const submit = async () => {
    const trimmed = name.trim();
    if (!trimmed) {
      setError("Name cannot be empty.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/calibration-configs/${device}/upload`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: trimmed, data }),
        }
      );
      const body = await res.json().catch(() => ({}));
      if (res.ok && body.success) {
        toast({ title: "Calibration imported", description: `Saved as "${body.name}".` });
        setOpen(false);
        onImported?.(body.name);
        return;
      }
      // 409 (collision) and 400 (validation) keep the dialog open with the
      // message so the user can rename / fix and retry.
      setError(body.message || "Import failed.");
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  const sideLabel = device === "teleop" ? "leader" : "follower";

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept=".json,application/json"
        className="hidden"
        onChange={handleFileChosen}
      />
      <Button
        size="icon"
        variant="ghost"
        className="h-7 w-7 text-slate-300 hover:text-white"
        onClick={pickFile}
        aria-label={`Import ${sideLabel} calibration`}
        title={`Import ${sideLabel} calibration`}
      >
        <Upload className="w-4 h-4" />
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="bg-slate-900 border-slate-800 text-white">
          <DialogHeader>
            <DialogTitle>Import {sideLabel} calibration</DialogTitle>
            <DialogDescription className="text-slate-400">
              Saves the uploaded calibration as a new {sideLabel} config. Won't
              overwrite an existing name — pick a different one if it's taken.
            </DialogDescription>
          </DialogHeader>
          <Input
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              setError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void submit();
              }
            }}
            autoFocus
            placeholder="Config name"
            className="bg-slate-800 border-slate-700 text-white"
          />
          {error && <p className="text-sm text-red-400">{error}</p>}
          <DialogFooter className="flex gap-2 justify-end">
            <Button
              variant="outline"
              className="border-slate-600 text-slate-700 dark:text-slate-300"
              onClick={() => setOpen(false)}
            >
              Cancel
            </Button>
            <Button
              className="bg-blue-600 hover:bg-blue-700 text-white"
              disabled={busy || !name.trim()}
              onClick={submit}
            >
              {busy ? "Importing…" : "Import"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default ImportCalibrationButton;
