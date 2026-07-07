import React, { useMemo, useState } from "react";
import { ExternalLink, Trash2, Upload as UploadIcon } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Loader2 } from "lucide-react";
import UploadDatasetDialog from "@/components/landing/UploadDatasetDialog";
import { DatasetItem } from "@/lib/replayApi";
import { sortDatasets } from "@/lib/sortDatasets";
import { validateDatasetName } from "@/lib/datasetName";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useToast } from "@/hooks/use-toast";
import { useDatasetUpload } from "@/hooks/useDatasetUpload";

interface DatasetPickerProps {
  datasets: DatasetItem[];
  loading: boolean;
  onPickExisting: (item: DatasetItem) => void;
  onCreateNew: (name: string) => void;
  onOpenCustom: (repoId: string) => void;
  onDelete?: (item: DatasetItem) => void;
  /** Called after a row's Hub upload succeeds so the parent can refresh the
   * list (flips the row's source local -> both, showing the "on Hub" badge). */
  onUploaded?: (item: DatasetItem) => void;
  children: React.ReactNode;
}

const REPO_ID_RE = /^[\w.-]+\/[\w.-]+$/;

/**
 * Per-row "Upload to Hub" control. Owns the background-upload hook for one
 * dataset so the row shows a live "Uploading…" spinner (which survives closing
 * the picker / navigating away and reopening) and toasts on completion, at
 * which point it asks the parent to refresh the list (flips local -> both).
 */
const RowUploadButton: React.FC<{
  repoId: string;
  onUploaded?: () => void;
}> = ({ repoId, onUploaded }) => {
  const { toast } = useToast();
  const { uploading, start } = useDatasetUpload({
    repoId,
    onDone: (url) => {
      onUploaded?.();
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

  if (uploading) {
    return (
      <span
        className="ml-2 flex shrink-0 items-center gap-1 text-xs text-gray-400"
        // Don't let a click on the status count as selecting the row.
        onMouseDown={(e) => {
          e.preventDefault();
          e.stopPropagation();
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <Loader2 className="h-3 w-3 animate-spin" />
        Uploading…
      </span>
    );
  }

  return (
    <UploadDatasetDialog repoId={repoId} start={start}>
      <button
        type="button"
        aria-label={`Upload ${repoId} to Hub`}
        className="ml-2 shrink-0 text-teal-600 hover:text-teal-500 dark:text-teal-400 dark:hover:text-teal-300"
        // Stop cmdk from treating the click as a selection of the row, but
        // don't preventDefault — the wrapping PopoverTrigger skips its
        // toggle when the child's click event is defaultPrevented.
        onMouseDown={(e) => {
          e.preventDefault();
          e.stopPropagation();
        }}
        onClick={(e) => {
          e.stopPropagation();
        }}
      >
        <UploadIcon className="h-3.5 w-3.5" />
      </button>
    </UploadDatasetDialog>
  );
};

const DatasetPicker: React.FC<DatasetPickerProps> = ({
  datasets,
  loading,
  onPickExisting,
  onCreateNew,
  onOpenCustom,
  onDelete,
  onUploaded,
  children,
}) => {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  // Namespace-first alphabetical ordering: datasets under the logged-in HF
  // account's namespace float to the top of each section. Falls back to plain
  // alphabetical when not authenticated / still loading.
  const { auth } = useHfAuth();
  const username = auth.status === "authenticated" ? auth.username : null;

  // Gate each row's "Upload to Hub" button to namespaces the logged-in user can
  // actually push to. A bare repo id (no "/") uploads to the user's own account,
  // which is always writable. Case-insensitive: Hub namespaces aren't
  // case-sensitive for ownership. Hidden entirely when not authenticated — you
  // can't push without a token.
  const writableNamespaces = useMemo(
    () =>
      auth.status === "authenticated"
        ? new Set(auth.writableNamespaces.map((ns) => ns.toLowerCase()))
        : new Set<string>(),
    [auth],
  );
  const canUpload = (repoId: string): boolean => {
    if (auth.status !== "authenticated") return false;
    const ns = repoId.includes("/") ? repoId.split("/")[0] : username;
    return ns != null && writableNamespaces.has(ns.toLowerCase());
  };

  const trimmed = query.trim();
  const matchesExisting = datasets.some(
    (d) => d.repo_id.toLowerCase() === trimmed.toLowerCase(),
  );
  const isRepoId = REPO_ID_RE.test(trimmed);
  // Shared with the backend (validate_dataset_name) so the picker never offers to
  // create a name the recorder will later reject.
  const nameError = validateDatasetName(trimmed);
  const isName = nameError === null;
  const canCreate = trimmed.length > 0 && isName && !matchesExisting;
  const canOpenCustom = isRepoId && !matchesExisting;

  // Strict partition by Hub status (user-decided): Local = not yet on the
  // Hub; Hugging Face = on the Hub, whether or not a local copy also exists.
  // Clearing the local cache of a "both" dataset lives in the "Manage cached
  // datasets" dialog, not inline here.
  const localDatasets = useMemo(
    () => sortDatasets(datasets.filter((d) => d.source === "local"), username),
    [datasets, username],
  );
  const hubDatasets = useMemo(
    () =>
      sortDatasets(
        datasets.filter((d) => d.source === "hub" || d.source === "both"),
        username,
      ),
    [datasets, username],
  );

  const reset = () => {
    setQuery("");
    setOpen(false);
  };

  const handlePick = (item: DatasetItem) => {
    onPickExisting(item);
    reset();
  };

  const handleCreate = () => {
    if (!canCreate) return;
    onCreateNew(trimmed);
    reset();
  };

  const handleOpenCustom = () => {
    if (!canOpenCustom) return;
    onOpenCustom(trimmed);
    reset();
  };

  const renderItem = (d: DatasetItem) => (
    <CommandItem
      key={d.repo_id}
      value={d.repo_id}
      onSelect={() => handlePick(d)}
      className="items-start text-white aria-selected:bg-gray-700"
    >
      <span className="min-w-0 flex-1 break-all">{d.repo_id}</span>
      {d.private && (
        <span className="shrink-0 text-xs text-amber-400">private</span>
      )}
      {/* Upload to Hub — local rows only (a "both" row is already on the Hub).
          Opens the same confirm popover the info card uses; the row shows a
          live "Uploading…" state while the background push runs. */}
      {d.source === "local" && canUpload(d.repo_id) && (
        <RowUploadButton repoId={d.repo_id} onUploaded={() => onUploaded?.(d)} />
      )}
      {/* Trash on local-only rows: deletes the only copy of a not-yet-uploaded
          dataset. "both" rows have no trash here — clearing their local cache
          lives in the "Manage cached datasets" dialog. */}
      {onDelete && d.source === "local" && (
        <button
          type="button"
          aria-label={`Delete ${d.repo_id}`}
          title="Delete dataset"
          className="ml-2 shrink-0 text-gray-500 hover:text-red-400"
          // stop cmdk from treating the click as a selection of the row
          onMouseDown={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onDelete(d);
          }}
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      )}
    </CommandItem>
  );

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>{children}</PopoverTrigger>
      <PopoverContent
        className="w-[320px] p-0 bg-gray-800 border-gray-700 text-white"
        align="end"
      >
        <Command className="bg-gray-800">
          <CommandInput
            placeholder="Search, type a new name, or org/name…"
            value={query}
            onValueChange={(v) =>
              setQuery(v.replace(/[^A-Za-z0-9._\-/]/g, "_"))
            }
            onKeyDown={(e) => {
              if (e.key !== "Enter") return;
              if (canCreate) {
                e.preventDefault();
                handleCreate();
              } else if (canOpenCustom) {
                e.preventDefault();
                handleOpenCustom();
              }
            }}
            className="text-white"
          />
          <CommandList>
            {datasets.length === 0 && !canCreate && !canOpenCustom && (
              <CommandEmpty className="py-4 text-sm text-gray-400 text-center">
                {loading
                  ? "Loading datasets…"
                  : "No datasets yet. Type a name to create one."}
              </CommandEmpty>
            )}
            {hubDatasets.length > 0 && (
              <CommandGroup heading="Hugging Face">
                {hubDatasets.map(renderItem)}
              </CommandGroup>
            )}
            {localDatasets.length > 0 && (
              <CommandGroup heading="Local">
                {localDatasets.map(renderItem)}
              </CommandGroup>
            )}
            {canOpenCustom && (
              <CommandGroup heading="Custom repo">
                <CommandItem
                  value={`__open__${trimmed}`}
                  onSelect={handleOpenCustom}
                  className="text-white aria-selected:bg-gray-700"
                >
                  <ExternalLink className="mr-2 h-4 w-4" />
                  Open &quot;{trimmed}&quot; in viewer
                </CommandItem>
              </CommandGroup>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
};

export default DatasetPicker;
