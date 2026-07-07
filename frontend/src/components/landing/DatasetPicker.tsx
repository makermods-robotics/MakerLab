import React, { useState } from "react";
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
import { validateDatasetName } from "@/lib/datasetName";
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
        className="ml-2 flex shrink-0 items-center gap-1 text-xs text-muted-foreground"
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
        className="ml-2 shrink-0 text-info hover:text-info/80"
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
  // Hub; Hugging Face = on the Hub, whether or not a local copy also exists
  // ("both" rows keep a "local copy" badge and their local-copy trash).
  const localDatasets = datasets.filter((d) => d.source === "local");
  const hubDatasets = datasets.filter(
    (d) => d.source === "hub" || d.source === "both",
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
    >
      <span className="flex-1 truncate">{d.repo_id}</span>
      {/* In the Hugging Face section, "on Hub" is implied by placement — the
          useful signal for a "both" row is that a local working copy exists. */}
      {d.source === "both" && (
        <span className="text-xs text-muted-foreground mr-2">local copy</span>
      )}
      {d.private && <span className="text-xs text-warn">private</span>}
      {/* Upload to Hub — local rows only (a "both" row is already on the Hub).
          Opens the same confirm popover the info card uses; the row shows a
          live "Uploading…" state while the background push runs. */}
      {d.source === "local" && (
        <RowUploadButton repoId={d.repo_id} onUploaded={() => onUploaded?.(d)} />
      )}
      {onDelete && (d.source === "local" || d.source === "both") && (
        <button
          type="button"
          aria-label={`Delete ${d.repo_id}`}
          // On a "both" row (HF section) this deletes only the local copy.
          title={
            d.source === "both"
              ? "Delete local copy — the Hub copy stays"
              : "Delete dataset"
          }
          className="ml-2 shrink-0 text-muted-foreground hover:text-destructive"
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
      <PopoverContent className="w-[320px] p-0" align="end">
        <Command>
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
          />
          <CommandList>
            {datasets.length === 0 && !canCreate && !canOpenCustom && (
              <CommandEmpty className="py-4 text-sm text-muted-foreground text-center">
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
