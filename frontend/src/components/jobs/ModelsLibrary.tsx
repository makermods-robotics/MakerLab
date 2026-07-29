import React, { useMemo, useState } from "react";
import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import LibraryToolbar from "@/components/library/LibraryToolbar";
import CappedGrid, { GRID_MIN_H } from "@/components/library/CappedGrid";
import LibraryHeader from "@/components/library/LibraryHeader";
import { SLIDE } from "@/components/studio/panel/primitives";
import { useStudio } from "@/contexts/StudioContext";
import { useLaunchInference } from "@/contexts/InferenceLaunchContext";
import { JobRecord } from "@/lib/jobsApi";
import ModelCard from "./ModelCard";
import HubModelCard from "./HubModelCard";
import ImportModelModal from "./ImportModelModal";
import { useJobsData } from "./JobsDataContext";

interface ModelsLibraryProps {
  /** Run this model (job record + optional checkpoint step) — the Train panel
   * hands it to the shared inference launch flow. */
  onPick: (job: JobRecord, step: number | null) => void;
  /** Optional controlled fold state, so the hosting panel can collapse the
   * library while its own form is open (the Train panel does; omit for the
   * self-managed default). */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

/** Which shelf of the library to look at. The split is the seam the data
 * actually has — the component boundary: "yours" is everything backed by a job
 * record (trained here or imported), which renders a ModelCard with the full
 * affordance set; "hub" is the Hub repos with no local record, which render
 * the thinner HubModelCard. */
type ModelFilter = "all" | "yours" | "hub";

const FILTERS: Array<{ key: ModelFilter; label: string }> = [
  { key: "all", label: "All" },
  { key: "yours", label: "Yours" },
  { key: "hub", label: "On Hub" },
];

/**
 * Model/policy library for the studio Train panel: a search box over finished
 * trainings (a successful run is itself a deployable model), imported models,
 * and uploaded hub repos no job tracks. It sits under Train — training is what
 * produces models, and a model card carries every follow-up action (Run /
 * Resume / Fine-tune / Download). Owns the Import button; rendered even when
 * empty so the entry point is always visible. Card Run actions hand the model
 * up through `onPick`, which the host routes to the shared inference launch.
 *
 * Shaped exactly like the dataset library: title row → one LibraryToolbar
 * (search + a segmented filter) → one CappedGrid, so the two studio columns
 * line their first card row up at the same height. The filter carries the
 * split the two section headers used to. Under "All" the two shelves stay
 * legible without a header: the actionable ones sort first, and each card's
 * own top-left origin chip says which shelf it is on — ModelCard prints
 * Local / Cloud / Imported (+ "from Hub"), HubModelCard prints "Uploaded".
 * Provenance *within* "yours" is not the split, precisely because that chip
 * already puts it on the card's face.
 */
const ModelsLibrary: React.FC<ModelsLibraryProps> = ({
  onPick,
  open,
  onOpenChange,
}) => {
  const { openStudio } = useStudio();
  const {
    importedJobs,
    deployableModels,
    untrackedHubModels,
    ancestorsOf,
    refresh,
    stop,
    remove,
  } = useJobsData();
  // Shared lazy-import (idempotent registration + husk-repo messaging) so an
  // untracked Hub repo resolves to a pseudo-job exactly as everywhere else.
  const { importSource } = useLaunchInference();

  // Uncontrolled by default; a controlled `open` prop wins when the host panel
  // needs to fold the library (Train does while its form is open).
  const [libraryOpenState, setLibraryOpenState] = useState(true);
  const libraryOpen = open ?? libraryOpenState;
  const setLibraryOpen = (next: boolean) => {
    onOpenChange?.(next);
    if (open === undefined) setLibraryOpenState(next);
  };
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [search, setSearch] = useState("");
  // Everything by default, like the dataset library — the row cap keeps the
  // rare untracked-Hub repos behind "Show all" anyway, since they sort last.
  const [filter, setFilter] = useState<ModelFilter>("all");

  const query = search.trim().toLowerCase();
  const matchesQuery = (text: string | null | undefined) =>
    !query || (text ?? "").toLowerCase().includes(query);

  // A finished run is findable by the same handles as an import.
  const visibleTrained = useMemo(
    () =>
      deployableModels.filter(
        (j) =>
          matchesQuery(j.name) ||
          matchesQuery(j.display_name) ||
          matchesQuery(j.hf_repo_id) ||
          matchesQuery(j.output_dir),
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [deployableModels, query],
  );
  // A renamed import is findable by alias, original name, repo id, or path.
  const visibleImported = useMemo(
    () =>
      importedJobs.filter(
        (j) =>
          matchesQuery(j.name) ||
          matchesQuery(j.display_name) ||
          matchesQuery(j.hf_repo_id) ||
          matchesQuery(j.output_dir),
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [importedJobs, query],
  );
  const visibleUploaded = useMemo(
    () => untrackedHubModels.filter((m) => matchesQuery(m.repo_id)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [untrackedHubModels, query],
  );

  // The title row's badge stays the unfiltered total (the dataset library does
  // the same with `libraryDatasets.length`); the filter pills carry no counts
  // of their own, so the per-section counts the sub-headers used to show are
  // gone rather than moved.
  const count =
    deployableModels.length + importedJobs.length + untrackedHubModels.length;

  // Untracked hub model actions: register the repo as an imported pseudo-job
  // first (the proven lazy-import path), then either run it right here or hand
  // it to the Train panel as a fine-tune base.
  const handleHubAction = async (
    repoId: string,
    action: "inference" | "finetune",
  ) => {
    if (action === "finetune") {
      openStudio("train", { train: { baseModelRepoId: repoId } });
      return;
    }
    const record = await importSource(repoId);
    if (!record) return;
    refresh();
    // step null → the checkpoint loader picks the repo's latest.
    onPick(record, null);
  };

  // One grid, two shelves. Each shelf keeps the sort it always had (job
  // records newest-started first, Hub repos newest-modified first) and they
  // are concatenated rather than interleaved, so the actionable models lead
  // and the untracked Hub repos trail — the reading order the sub-headers
  // used to impose, minus the headers.
  const ownedCards =
    filter === "hub"
      ? []
      : [...visibleTrained, ...visibleImported]
          .sort((a, b) => (b.started_at ?? 0) - (a.started_at ?? 0))
          .map((job) => (
            <ModelCard
              key={job.id}
              model={job}
              onStop={stop}
              onDelete={remove}
              onPlay={(j, step) => onPick(j, step)}
              onRenamed={refresh}
              ancestors={ancestorsOf(job)}
            />
          ));
  const hubCards =
    filter === "yours"
      ? []
      : [...visibleUploaded]
          .sort(
            (a, b) =>
              (b.last_modified ? Date.parse(b.last_modified) || 0 : 0) -
              (a.last_modified ? Date.parse(a.last_modified) || 0 : 0),
          )
          .map((model) => (
            <HubModelCard
              key={model.repo_id}
              model={model}
              onDeleted={refresh}
              onAction={handleHubAction}
            />
          ));
  const cards = [...ownedCards, ...hubCards];

  return (
    <Collapsible
      open={libraryOpen}
      onOpenChange={setLibraryOpen}
      className="space-y-3"
    >
      <LibraryHeader
        title="Your models"
        count={count}
        open={libraryOpen}
        actions={
          <Button
            variant="outline"
            size="sm"
            onClick={() => setImportModalOpen(true)}
            className="h-7 shrink-0 gap-1.5 text-xs text-muted-foreground hover:text-foreground"
          >
            <Download className="h-3.5 w-3.5" />
            Import model
          </Button>
        }
      />

      <CollapsibleContent className={SLIDE}>
        {count === 0 ? (
          <div
            className={cn(
              "flex items-center justify-center rounded-md border border-dashed border-border px-4 py-6 text-center text-sm text-muted-foreground",
              GRID_MIN_H,
            )}
          >
            No models yet. Train one, or use Import model to add one from the
            Hub or a local folder.
          </div>
        ) : (
          <div className="space-y-3">
            {/* Search + the two-shelf filter, exactly the dataset library's
                toolbar row — one row above the grid in both columns. */}
            <LibraryToolbar
              query={search}
              onQueryChange={setSearch}
              searchPlaceholder="Search models"
              filters={FILTERS}
              filter={filter}
              onFilterChange={setFilter}
            />
            {cards.length === 0 ? (
              <p
                className={cn(
                  "flex items-center justify-center px-1 py-4 text-center text-sm text-muted-foreground",
                  GRID_MIN_H,
                )}
              >
                No models match.
              </p>
            ) : (
              // One row by default, the rest behind Show all — the datasets
              // column's grid, with the same props it always had here.
              <CappedGrid
                // Model cards render every affordance (Run/Resume/Fine-tune/
                // Download/step-picker) and must not clip — let rows size to
                // content rather than the fixed 16.5rem datasets/jobs use.
                flexHeight
                items={cards}
              />
            )}
          </div>
        )}
      </CollapsibleContent>

      <ImportModelModal
        open={importModalOpen}
        onOpenChange={setImportModalOpen}
        onImported={refresh}
      />
    </Collapsible>
  );
};

export default ModelsLibrary;
