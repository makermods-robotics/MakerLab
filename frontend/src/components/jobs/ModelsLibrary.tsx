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
import { useInferenceLaunch } from "@/hooks/useInferenceLaunch";
import { JobRecord } from "@/lib/jobsApi";
import JobCard from "./JobCard";
import HubModelCard from "./HubModelCard";
import ImportModelModal from "./ImportModelModal";
import { useJobsData } from "./JobsDataContext";

/** How a model got here: everything, imported (local folder or Hub pull), or
 * uploaded Hub repos no job tracks. */
type ModelsFilter = "all" | "imported" | "uploaded";

const FILTERS: Array<{ key: ModelsFilter; label: string }> = [
  { key: "all", label: "All" },
  { key: "imported", label: "Imported" },
  { key: "uploaded", label: "Uploaded" },
];

interface ModelsLibraryProps {
  /** Select this model (job record + optional checkpoint step) as the skill
   * to deploy — wired to the Deploy panel's picker state. */
  onPick: (job: JobRecord, step: number | null) => void;
}

/**
 * Model/policy library for the studio Deploy panel: search + origin filter
 * over a three-up grid of imported models plus uploaded hub repos no job
 * tracks (a model artifact, not a run — so it lives here, not under the Train
 * panel's jobs). Owns the Import button; rendered even when empty so the
 * entry point is always visible. Card Run actions select the model in the
 * Deploy panel instead of opening the legacy modal.
 */
const ModelsLibrary: React.FC<ModelsLibraryProps> = ({ onPick }) => {
  const { openStudio } = useStudio();
  const {
    importedJobs,
    untrackedHubModels,
    refresh,
    stop,
    remove,
  } = useJobsData();
  // Shared lazy-import (idempotent registration + husk-repo messaging) so an
  // untracked Hub repo resolves to a pseudo-job exactly as everywhere else.
  const { importSource } = useInferenceLaunch();

  const [libraryOpen, setLibraryOpen] = useState(true);
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<ModelsFilter>("all");

  const query = search.trim().toLowerCase();
  const matchesQuery = (text: string | null | undefined) =>
    !query || (text ?? "").toLowerCase().includes(query);

  // A renamed import is findable by alias, original name, repo id, or path.
  const visibleImported = useMemo(
    () =>
      filter === "uploaded"
        ? []
        : importedJobs.filter(
            (j) =>
              matchesQuery(j.name) ||
              matchesQuery(j.display_name) ||
              matchesQuery(j.hf_repo_id) ||
              matchesQuery(j.output_dir),
          ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [importedJobs, filter, query],
  );
  const visibleUploaded = useMemo(
    () =>
      filter === "imported"
        ? []
        : untrackedHubModels.filter((m) => matchesQuery(m.repo_id)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [untrackedHubModels, filter, query],
  );

  const count = importedJobs.length + untrackedHubModels.length;
  const visibleCount = visibleImported.length + visibleUploaded.length;

  // Untracked hub model actions: register the repo as an imported pseudo-job
  // first (the proven lazy-import path), then either select it for deployment
  // right here or hand it to the Train panel as a fine-tune base.
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
            <LibraryToolbar
              query={search}
              onQueryChange={setSearch}
              searchPlaceholder="Search models"
              filters={FILTERS}
              filter={filter}
              onFilterChange={setFilter}
            />
            {visibleCount === 0 ? (
              <p
                className={cn(
                  "flex items-center justify-center px-1 py-4 text-center text-sm text-muted-foreground",
                  GRID_MIN_H,
                )}
              >
                No models match.
              </p>
            ) : (
              // Imported and uploaded cards merged newest-first (import time
              // vs Hub last-modified); two rows by default, rest behind
              // Show all.
              <CappedGrid
                items={[
                  ...visibleImported.map((job) => ({
                    time: (job.started_at ?? 0) * 1000,
                    node: (
                      <JobCard
                        key={job.id}
                        job={job}
                        onStop={stop}
                        onDelete={remove}
                        onPlay={(j, step) => onPick(j, step)}
                        onRenamed={refresh}
                      />
                    ),
                  })),
                  ...visibleUploaded.map((model) => ({
                    time: model.last_modified
                      ? Date.parse(model.last_modified) || 0
                      : 0,
                    node: (
                      <HubModelCard
                        key={model.repo_id}
                        model={model}
                        onDeleted={refresh}
                        onAction={handleHubAction}
                      />
                    ),
                  })),
                ]
                  .sort((a, b) => b.time - a.time)
                  .map((e) => e.node)}
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
