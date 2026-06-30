import React, { useCallback, useEffect, useState } from "react";
import { Download, Pencil, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
import ImportCalibrationButton from "./ImportCalibrationButton";

interface ConfigEntry {
  name: string;
}

interface CalibrationLibraryProps {
  /** API device vocabulary: "teleop" (leader) or "robot" (follower). */
  device: "teleop" | "robot";
  /** Config name currently assigned to the selected robot (marked "in use"). */
  assignedConfig?: string;
  /** Robot record to reassign when "Use for this robot" is clicked. */
  robotName?: string;
  /**
   * Which record field "Use for this robot" assigns to. Defaults to the
   * primary field for the device (leader_config / follower_config); bimanual
   * right-arm rows pass right_leader_config / right_follower_config.
   */
  configField?: string;
  /**
   * A config already used by the OTHER same-side arm — greyed out here so you
   * can't point two physical arms at one calibration (the proactive guard).
   */
  excludeConfig?: string;
  /** Called after a successful reassignment so the parent can refetch the robot. */
  onAssigned?: () => void | Promise<void>;
}

/**
 * Per-side calibration "library" as a dropdown: pick a saved config, then
 * Download / Delete it, or Import a new one. Both import and download operate on
 * the same named files, so they stay symmetric.
 */
const CalibrationLibrary: React.FC<CalibrationLibraryProps> = ({
  device,
  assignedConfig,
  robotName,
  configField,
  excludeConfig,
  onAssigned,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

  const [configs, setConfigs] = useState<ConfigEntry[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [assigning, setAssigning] = useState(false);
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const [renameError, setRenameError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const res = await fetchWithHeaders(`${baseUrl}/calibration-configs/${device}`);
      const data = await res.json();
      if (data.success) {
        setConfigs((data.configs ?? []).map((c: { name: string }) => ({ name: c.name })));
      }
    } catch {
      // Non-fatal; leave the list as-is.
    }
  }, [baseUrl, fetchWithHeaders, device]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Keep a valid selection: prefer the current pick, then the in-use config,
  // then the first available.
  useEffect(() => {
    setSelected((prev) => {
      if (prev && configs.some((c) => c.name === prev)) return prev;
      if (assignedConfig && configs.some((c) => c.name === assignedConfig)) return assignedConfig;
      return configs[0]?.name ?? null;
    });
  }, [configs, assignedConfig]);

  const download = useCallback(
    async (name: string) => {
      try {
        const res = await fetchWithHeaders(
          `${baseUrl}/calibration-configs/${device}/${encodeURIComponent(name)}/download`
        );
        if (!res.ok) {
          toast({ title: "Download failed", variant: "destructive" });
          return;
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${name}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      } catch (e) {
        toast({ title: "Download failed", description: String(e), variant: "destructive" });
      }
    },
    [baseUrl, fetchWithHeaders, device, toast]
  );

  const confirmDelete = useCallback(async () => {
    const name = pendingDelete;
    if (!name) return;
    setPendingDelete(null);
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/calibration-configs/${device}/${encodeURIComponent(name)}`,
        { method: "DELETE" }
      );
      const data = await res.json().catch(() => ({}));
      if (data.success) {
        toast({ title: "Config deleted", description: `Removed "${name}".` });
        setConfigs((prev) => prev.filter((c) => c.name !== name));
      } else {
        toast({ title: "Delete failed", description: data.message, variant: "destructive" });
      }
    } catch (e) {
      toast({ title: "Delete failed", description: String(e), variant: "destructive" });
    }
  }, [baseUrl, fetchWithHeaders, device, pendingDelete, toast]);

  const assignToRobot = useCallback(async () => {
    if (!selected || !robotName) return;
    setAssigning(true);
    try {
      const field = configField ?? (device === "teleop" ? "leader_config" : "follower_config");
      const res = await fetchWithHeaders(
        `${baseUrl}/robots/${encodeURIComponent(robotName)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ [field]: selected }),
        }
      );
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.status === "success") {
        toast({ title: "Config assigned", description: `"${selected}" is now used for this robot.` });
        await onAssigned?.();
      } else {
        toast({ title: "Assign failed", description: data.message, variant: "destructive" });
      }
    } catch (e) {
      toast({ title: "Assign failed", description: String(e), variant: "destructive" });
    } finally {
      setAssigning(false);
    }
  }, [selected, robotName, device, configField, baseUrl, fetchWithHeaders, toast, onAssigned]);

  const openRename = useCallback(() => {
    if (!selected) return;
    setRenameValue(selected);
    setRenameError(null);
    setRenameOpen(true);
  }, [selected]);

  const renameConfig = useCallback(async () => {
    if (!selected) return;
    const next = renameValue.trim();
    if (!next) {
      setRenameError("Name cannot be empty.");
      return;
    }
    if (next === selected) {
      setRenameOpen(false);
      return;
    }
    setRenaming(true);
    setRenameError(null);
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/calibration-configs/${device}/${encodeURIComponent(selected)}/rename`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ new_name: next }),
        }
      );
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.success) {
        toast({ title: "Config renamed", description: `"${selected}" → "${data.name}".` });
        setRenameOpen(false);
        await refresh();
        setSelected(data.name);
        // A robot referencing this config was repointed server-side; refetch it.
        await onAssigned?.();
      } else {
        // 409/400 keep the dialog open with the message for a retry.
        setRenameError(data.message || "Rename failed.");
      }
    } catch (e) {
      setRenameError(String(e));
    } finally {
      setRenaming(false);
    }
  }, [selected, renameValue, device, baseUrl, fetchWithHeaders, toast, refresh, onAssigned]);

  const empty = configs.length === 0;
  const canAssign =
    !!robotName && !!selected && selected !== assignedConfig;

  return (
    <div className="mt-1 ml-6 space-y-1">
      <div className="flex items-center gap-1">
      <Select
        value={selected ?? ""}
        onValueChange={setSelected}
        disabled={empty}
      >
        <SelectTrigger className="h-8 flex-1 bg-slate-800 border-slate-700 text-white">
          <SelectValue placeholder={empty ? "No saved configs" : "Select a config"} />
        </SelectTrigger>
        <SelectContent className="bg-slate-800 border-slate-700 text-white">
          {configs.map((c) => {
            const usedByOtherArm = !!excludeConfig && c.name === excludeConfig;
            return (
              <SelectItem
                key={c.name}
                value={c.name}
                disabled={usedByOtherArm}
                className="text-white"
              >
                <span className="flex items-center gap-2">
                  {c.name}
                  {c.name === assignedConfig && (
                    <span className="text-[10px] uppercase tracking-wide text-green-400 border border-green-500/40 rounded px-1">
                      in use
                    </span>
                  )}
                  {usedByOtherArm && (
                    <span className="text-[10px] uppercase tracking-wide text-amber-400 border border-amber-500/40 rounded px-1">
                      other arm
                    </span>
                  )}
                </span>
              </SelectItem>
            );
          })}
        </SelectContent>
      </Select>

      <Button
        size="icon"
        variant="ghost"
        className="h-8 w-8 text-slate-300 hover:text-white"
        disabled={!selected}
        onClick={() => selected && download(selected)}
        aria-label="Download selected config"
        title="Download"
      >
        <Download className="w-4 h-4" />
      </Button>
      <Button
        size="icon"
        variant="ghost"
        className="h-8 w-8 text-slate-300 hover:text-white"
        disabled={!selected}
        onClick={openRename}
        aria-label="Rename selected config"
        title="Rename"
      >
        <Pencil className="w-4 h-4" />
      </Button>
      <Button
        size="icon"
        variant="ghost"
        className="h-8 w-8 text-red-400 hover:text-red-300 hover:bg-red-900/20"
        disabled={!selected}
        onClick={() => selected && setPendingDelete(selected)}
        aria-label="Delete selected config"
        title="Delete"
      >
        <Trash2 className="w-4 h-4" />
      </Button>
      <ImportCalibrationButton
        device={device}
        onImported={async (name) => {
          await refresh();
          setSelected(name);
        }}
      />
      </div>

      {canAssign && (
        <Button
          size="sm"
          variant="outline"
          className="w-full h-7 border-blue-500/50 text-blue-300 hover:bg-blue-900/20 hover:text-blue-200"
          disabled={assigning}
          onClick={assignToRobot}
        >
          {assigning ? "Assigning…" : `Use "${selected}" for this robot`}
        </Button>
      )}

      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent className="bg-slate-900 border-slate-800 text-white">
          <DialogHeader>
            <DialogTitle>Rename config</DialogTitle>
            <DialogDescription className="text-slate-400">
              Renames the calibration file. Robots using it are updated
              automatically. Won't overwrite an existing name.
            </DialogDescription>
          </DialogHeader>
          <Input
            value={renameValue}
            onChange={(e) => {
              setRenameValue(e.target.value);
              setRenameError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void renameConfig();
              }
            }}
            autoFocus
            placeholder="New name"
            className="bg-slate-800 border-slate-700 text-white"
          />
          {renameError && <p className="text-sm text-red-400">{renameError}</p>}
          <DialogFooter className="flex gap-2 justify-end">
            <Button
              variant="outline"
              className="border-slate-600 text-slate-300"
              onClick={() => setRenameOpen(false)}
            >
              Cancel
            </Button>
            <Button
              className="bg-blue-600 hover:bg-blue-700 text-white"
              disabled={renaming || !renameValue.trim() || renameValue.trim() === selected}
              onClick={renameConfig}
            >
              {renaming ? "Renaming…" : "Rename"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={pendingDelete !== null} onOpenChange={(o) => !o && setPendingDelete(null)}>
        <DialogContent className="bg-slate-900 border-slate-800 text-white">
          <DialogHeader>
            <DialogTitle>Delete config "{pendingDelete}"?</DialogTitle>
            <DialogDescription className="text-slate-400">
              This deletes the calibration file from disk. Robots referencing it will
              show "Needs configuration". This cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex gap-2 justify-end">
            <Button
              variant="outline"
              className="border-slate-600 text-slate-300"
              onClick={() => setPendingDelete(null)}
            >
              Cancel
            </Button>
            <Button className="bg-red-500 hover:bg-red-600 text-white" onClick={confirmDelete}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default CalibrationLibrary;
