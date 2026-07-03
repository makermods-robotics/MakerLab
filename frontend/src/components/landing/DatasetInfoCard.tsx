import React, { useEffect, useState } from "react";
import {
  AlertTriangle,
  ChevronDown,
  ExternalLink,
  Loader2,
  Pencil,
  Upload as UploadIcon,
} from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
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
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { ApiError } from "@/lib/apiClient";
import { validateDatasetName } from "@/lib/datasetName";
import UploadDatasetDialog from "@/components/landing/UploadDatasetDialog";
import { useDatasetUpload } from "@/hooks/useDatasetUpload";
import {
  DatasetInfo,
  DatasetTask,
  HubStatusValue,
  getDatasetHubStatus,
  getDatasetInfo,
  renameDataset,
} from "@/lib/replayApi";

/** 16723 -> "16.7k", 950 -> "950" */
const formatCount = (n: number): string => {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(n);
};

/** frames ÷ fps, human-formatted: "~9 min", "~45 s", "~1 h 12 min" */
const formatDuration = (frames: number, fps: number | null): string | null => {
  if (!fps || fps <= 0 || frames <= 0) return null;
  const seconds = frames / fps;
  if (seconds < 60) return `~${Math.round(seconds)} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `~${minutes} min`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m > 0 ? `~${h} h ${m} min` : `~${h} h`;
};

const formatBytes = (bytes: number): string => {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
};

const WarningBadge: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => (
  <span className="inline-flex items-center gap-1 rounded border border-red-500/40 bg-red-500/15 px-1.5 py-0.5 text-xs font-medium text-red-400">
    <AlertTriangle className="h-3 w-3 shrink-0" />
    {children}
  </span>
);

const Row: React.FC<{ label: string; children: React.ReactNode }> = ({
  label,
  children,
}) => (
  <div className="flex items-baseline gap-2">
    <span className="w-14 shrink-0 text-gray-500">{label}</span>
    <span className="min-w-0 flex-1 text-gray-300">{children}</span>
  </div>
);

/**
 * Tasks row content. One task renders inline (with its episode count when
 * known); several render as a collapsible disclosure — closed it reads
 * "N tasks", open it lists each task with its episode count right-aligned.
 */
const TaskList: React.FC<{ tasks: DatasetTask[] }> = ({ tasks }) => {
  const [open, setOpen] = useState(false);

  if (tasks.length === 1) {
    const { task, num_episodes } = tasks[0];
    return (
      <span className="flex items-baseline gap-1.5">
        <span className="min-w-0 truncate" title={task}>
          {task}
        </span>
        {num_episodes > 0 && (
          <span className="shrink-0 text-gray-500">· {num_episodes} ep</span>
        )}
      </span>
    );
  }

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex items-center gap-1 text-gray-300 hover:text-gray-100">
        {tasks.length} tasks
        <ChevronDown
          className={`h-3 w-3 text-gray-500 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ul className="mt-1 space-y-0.5">
          {tasks.map(({ task, num_episodes }) => (
            <li key={task} className="flex items-baseline gap-2">
              <span className="min-w-0 flex-1 truncate" title={task}>
                {task}
              </span>
              <span className="shrink-0 text-gray-500">
                {num_episodes} ep
              </span>
            </li>
          ))}
        </ul>
      </CollapsibleContent>
    </Collapsible>
  );
};

/**
 * Hub sync line for the info card: a muted status ("Local only" / "On Hub")
 * plus, when the dataset isn't confirmed on the Hub, an "Upload to Hub" button
 * that opens a confirm popover (private-by-default toggle + optional tags).
 *
 * Status is fetched separately/lazily so it never blocks the card render, and
 * degrades to "unknown" (nothing shown) offline/unauthenticated. The upload
 * runs in the background (see useDatasetUpload): while it's in flight this row
 * shows an "Uploading…" state (which survives navigating away and back), and
 * on completion it flips to "On Hub" and toasts the Hub URL.
 */
const HubSyncRow: React.FC<{ repoId: string }> = ({ repoId }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [status, setStatus] = useState<HubStatusValue>("unknown");
  const [hubUrl, setHubUrl] = useState<string | null>(null);

  const { uploading, start } = useDatasetUpload({
    repoId,
    onDone: (url) => {
      setStatus("on_hub");
      setHubUrl(url);
      toast({
        title: "Uploaded to Hub",
        description: (
          <span>
            {repoId} is now on the Hub.{" "}
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="underline font-medium"
            >
              View dataset
            </a>
          </span>
        ),
      });
    },
    onError: (message, docsUrl) => {
      toast({
        title: "Upload failed",
        description: docsUrl ? (
          <span>
            {message}{" "}
            <a
              href={docsUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="underline font-medium"
            >
              Open setup guide
            </a>
          </span>
        ) : (
          message
        ),
        variant: "destructive",
      });
    },
  });

  useEffect(() => {
    const controller = new AbortController();
    setStatus("unknown");
    setHubUrl(null);
    getDatasetHubStatus(baseUrl, fetchWithHeaders, repoId, controller.signal)
      .then((data) => {
        setStatus(data.status);
        setHubUrl(data.url);
      })
      .catch(() => {
        // Degrade silently to "unknown" — no error spam on the card.
        if (!controller.signal.aborted) setStatus("unknown");
      });
    return () => controller.abort();
  }, [baseUrl, fetchWithHeaders, repoId]);

  if (uploading) {
    return (
      <div className="flex items-center gap-1.5 text-gray-400">
        <Loader2 className="h-3 w-3 animate-spin" />
        <span>Uploading to Hub…</span>
      </div>
    );
  }

  if (status === "on_hub") {
    return (
      <div className="flex items-center gap-1.5 text-gray-500">
        <span>On Hub</span>
        {hubUrl && (
          <a
            href={hubUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-0.5 text-gray-400 hover:text-gray-200"
          >
            <ExternalLink className="h-3 w-3" />
          </a>
        )}
      </div>
    );
  }

  // local_only or unknown: offer upload. For "unknown" we still allow it —
  // the endpoint is a safe upsert and reports auth failures gracefully.
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-gray-500">
        {status === "local_only" ? "Local only" : "Hub status unknown"}
      </span>
      <UploadDatasetDialog repoId={repoId} start={start}>
        <Button
          size="sm"
          variant="outline"
          className="h-6 gap-1 border-teal-500/50 px-2 text-xs text-teal-700 dark:text-teal-300 hover:bg-teal-500/10"
        >
          <UploadIcon className="h-3 w-3" />
          Upload to Hub
        </Button>
      </UploadDatasetDialog>
    </div>
  );
};

/**
 * Rename dialog for a local dataset (mirrors JobCard's rename UI). The namespace
 * prefix is fixed — the user edits only the name segment, shown after a static
 * "namespace/" prefix. A dataset's repo id IS its directory path, so this moves
 * the directory; the Hub copy (if any) keeps its old name, called out below.
 */
const RenameDatasetDialog: React.FC<{
  open: boolean;
  onOpenChange: (open: boolean) => void;
  repoId: string;
  onRenamed: (newRepoId: string) => void;
}> = ({ open, onOpenChange, repoId, onRenamed }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

  const slash = repoId.lastIndexOf("/");
  const namespace = slash >= 0 ? repoId.slice(0, slash) : null;
  const currentName = slash >= 0 ? repoId.slice(slash + 1) : repoId;

  const [value, setValue] = useState(currentName);
  const [error, setError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);

  // Reset the field to the current name whenever the dialog (re)opens.
  useEffect(() => {
    if (open) {
      setValue(currentName);
      setError(null);
    }
  }, [open, currentName]);

  const trimmed = value.trim();
  const validationError = trimmed === "" ? null : validateDatasetName(trimmed);
  const unchanged = trimmed === currentName;

  const doRename = async () => {
    const next = value.trim();
    const nameError = validateDatasetName(next);
    if (nameError) {
      setError(nameError);
      return;
    }
    if (next === currentName) {
      onOpenChange(false);
      return;
    }
    setRenaming(true);
    setError(null);
    try {
      const res = await renameDataset(baseUrl, fetchWithHeaders, repoId, next);
      toast({ title: "Dataset renamed", description: res.repo_id });
      onOpenChange(false);
      onRenamed(res.repo_id);
    } catch (e) {
      setError(
        e instanceof ApiError && e.detail
          ? e.detail
          : e instanceof Error
            ? e.message
            : String(e),
      );
    } finally {
      setRenaming(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-gray-900 border-gray-800 text-white">
        <DialogHeader>
          <DialogTitle>Rename dataset</DialogTitle>
          <DialogDescription className="text-gray-400">
            Renames the local dataset directory. If this dataset has a copy on
            the Hub, the Hub copy keeps its old name.
          </DialogDescription>
        </DialogHeader>
        <div className="flex items-center gap-1">
          {namespace && (
            <span className="shrink-0 font-mono text-sm text-gray-500">
              {namespace}/
            </span>
          )}
          <Input
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              setError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void doRename();
              }
            }}
            autoFocus
            placeholder="New name"
            className="bg-gray-800 border-gray-700 text-white"
          />
        </div>
        {(error ?? validationError) && (
          <p className="text-sm text-red-400">{error ?? validationError}</p>
        )}
        <DialogFooter className="flex gap-2 justify-end">
          <Button
            variant="outline"
            className="border-gray-600 bg-transparent text-gray-200 hover:bg-gray-800 hover:text-white"
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            className="bg-blue-600 hover:bg-blue-700 text-white"
            disabled={
              renaming || trimmed === "" || unchanged || validationError !== null
            }
            onClick={doRename}
          >
            {renaming ? "Renaming…" : "Rename"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

interface DatasetInfoCardProps {
  repoId: string;
  /** Called after a successful rename with the new repo id, so the parent can
   * update the selection and refresh the picker list. */
  onRenamed?: (newRepoId: string) => void;
}

/**
 * Compact always-visible summary of the dataset selected on the home page:
 * episodes/frames/duration, camera names (the load-bearing line for vision
 * training), robot type, task strings, and size on disk. Data comes from the
 * on-demand /datasets/info endpoint, which only covers the local cache.
 */
const DatasetInfoCard: React.FC<DatasetInfoCardProps> = ({
  repoId,
  onRenamed,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [info, setInfo] = useState<DatasetInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<{ notLocal: boolean } | null>(null);
  const [renameOpen, setRenameOpen] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setInfo(null);
    setError(null);
    getDatasetInfo(baseUrl, fetchWithHeaders, repoId, controller.signal)
      .then((data) => {
        setInfo(data);
        setLoading(false);
      })
      .catch((e) => {
        if (controller.signal.aborted) return;
        setError({ notLocal: e instanceof ApiError && e.status === 404 });
        setLoading(false);
      });
    return () => controller.abort();
  }, [baseUrl, fetchWithHeaders, repoId]);

  return (
    <div className="rounded-md border border-gray-700 bg-gray-900/60 px-3 py-2 text-xs">
      {loading && (
        <div className="animate-pulse space-y-2 py-0.5" aria-label="Loading dataset details">
          <div className="h-3 w-3/4 rounded bg-gray-700" />
          <div className="h-3 w-1/2 rounded bg-gray-700" />
          <div className="h-3 w-2/3 rounded bg-gray-700" />
        </div>
      )}

      {!loading && error && (
        <p className="text-gray-500">
          {error.notLocal
            ? "Not in the local cache — details are only available for downloaded datasets."
            : "Couldn't load dataset details."}
        </p>
      )}

      {!loading && info && (
        <div className="space-y-1.5">
          <div className="flex items-start justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2 font-medium text-gray-200">
              <span>
                {info.total_episodes} episode
                {info.total_episodes === 1 ? "" : "s"}
                {" · "}
                {formatCount(info.total_frames)} frames
                {(() => {
                  const d = formatDuration(info.total_frames, info.fps);
                  return d ? ` · ${d}` : "";
                })()}
              </span>
              {info.total_episodes === 0 && (
                <WarningBadge>No episodes recorded</WarningBadge>
              )}
            </div>
            <button
              type="button"
              onClick={() => setRenameOpen(true)}
              aria-label="Rename dataset"
              title="Rename dataset"
              className="-mr-1 -mt-0.5 shrink-0 rounded p-1 text-gray-500 hover:text-gray-200"
            >
              <Pencil className="h-3.5 w-3.5" />
            </button>
          </div>

          <Row label="Cameras">
            {info.cameras.length > 0 ? (
              info.cameras.join(", ")
            ) : (
              <WarningBadge>
                No camera data — unusable for vision training
              </WarningBadge>
            )}
          </Row>

          <Row label="Robot">{info.robot_type ?? "unknown"}</Row>

          {info.tasks.length > 0 && (
            <Row label="Tasks">
              <TaskList tasks={info.tasks} />
            </Row>
          )}

          <Row label="Size">{formatBytes(info.size_bytes)}</Row>

          <div className="mt-1.5 border-t border-gray-800 pt-1.5">
            <HubSyncRow repoId={repoId} />
          </div>

          <RenameDatasetDialog
            open={renameOpen}
            onOpenChange={setRenameOpen}
            repoId={repoId}
            onRenamed={(newRepoId) => onRenamed?.(newRepoId)}
          />
        </div>
      )}
    </div>
  );
};

export default DatasetInfoCard;
