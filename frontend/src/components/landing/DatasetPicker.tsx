import React, { useState } from "react";
import { Plus, ExternalLink, Trash2 } from "lucide-react";
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
import { DatasetItem } from "@/lib/replayApi";
import { validateDatasetName } from "@/lib/datasetName";

interface DatasetPickerProps {
  datasets: DatasetItem[];
  loading: boolean;
  onPickExisting: (item: DatasetItem) => void;
  onCreateNew: (name: string) => void;
  onOpenCustom: (repoId: string) => void;
  onDelete?: (item: DatasetItem) => void;
  children: React.ReactNode;
}

const REPO_ID_RE = /^[\w.-]+\/[\w.-]+$/;

const DatasetPicker: React.FC<DatasetPickerProps> = ({
  datasets,
  loading,
  onPickExisting,
  onCreateNew,
  onOpenCustom,
  onDelete,
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

  const createDisabled = matchesExisting || (trimmed !== "" && !canCreate);
  const createLabel = matchesExisting
    ? "Already exists"
    : trimmed === ""
      ? "Create new dataset…"
      : canCreate
        ? `Create "${trimmed}"`
        : (nameError ?? "Invalid name");

  const handleFooterCreate = () => {
    if (createDisabled) return;
    onCreateNew(trimmed);
    reset();
  };

  const localDatasets = datasets.filter(
    (d) => d.source === "local" || d.source === "both",
  );
  const hubDatasets = datasets.filter((d) => d.source === "hub");

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
      className="text-white aria-selected:bg-gray-700"
    >
      <span className="flex-1 truncate">{d.repo_id}</span>
      {d.source === "both" && (
        <span className="text-xs text-gray-400 mr-2">on Hub</span>
      )}
      {d.private && <span className="text-xs text-amber-400">private</span>}
      {onDelete && (d.source === "local" || d.source === "both") && (
        <button
          type="button"
          aria-label={`Delete ${d.repo_id}`}
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
            {localDatasets.length > 0 && (
              <CommandGroup heading="Local">
                {localDatasets.map(renderItem)}
              </CommandGroup>
            )}
            {hubDatasets.length > 0 && (
              <CommandGroup heading="Hugging Face">
                {hubDatasets.map(renderItem)}
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
          <button
            type="button"
            onClick={handleFooterCreate}
            disabled={createDisabled}
            className="flex w-full items-center gap-2 border-t border-gray-700 px-3 py-2 text-sm text-white hover:bg-gray-700 disabled:cursor-not-allowed disabled:text-gray-500 disabled:hover:bg-transparent"
          >
            <Plus className="h-4 w-4" />
            {createLabel}
          </button>
        </Command>
      </PopoverContent>
    </Popover>
  );
};

export default DatasetPicker;
