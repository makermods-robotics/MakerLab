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
import LibrarySectionHeader from "@/components/library/LibrarySectionHeader";
import { SLIDE } from "@/components/studio/panel/primitives";
import { useStudio } from "@/contexts/StudioContext";
import { useInferenceLaunch } from "@/hooks/useInferenceLaunch";
import { JobRecord } from "@/lib/jobsApi";
import ModelCard from "./ModelCard";
import HubModelCard from "./HubModelCard";
import ImportModelModal from "./ImportModelModal";
import { useJobsData } from "./JobsDataContext";

interface ModelsLibraryProps {
  /** Select this model (job record + optional checkpoint step) as the skill
   * to deploy — the Train panel hands it to the Deploy panel as a prefill. */
  onPick: (job: JobRecord, step: number | null) => void;
  /** Optional controlled fold state, so the hosting panel can collapse the
   * library while its own form is open (the Train panel does; omit for the
   * self-managed default). */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

/**
 * Model/policy library for the studio Train panel: a search box over finished
 * trainings (a successful run is itself a deployable model), imported models,
 * and uploaded hub repos no job tracks. It sits under Train — training is what
 * produces models, and a model card carries every follow-up action (Run /
 * Resume / Fine-tune / Download) — while Deploy stays pick-and-launch. Owns
 * the Import button; rendered even when empty so the entry point is always
 * visible. Card Run actions hand the model to the Deploy panel via `onPick`
 * rather than opening the legacy modal.
 *
 * Two independently collapsible sections, split on the seam the data actually
 * has — the component boundary. "Your models" is everything backed by a job
 * record (trained here or imported), so every one of them renders a ModelCard
 * with the full affordance set; "On Hub, not imported" is the Hub repos with
 * no local record, which render the thinner HubModelCard (its Run/Fine-tune
 * lazily imports the repo first). Provenance is *not* the split: ModelCard
 * already prints Local / Cloud / Imported as an origin chip, so filtering on
 * it would only hide cards by something visible on their faces.
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
  const { importSource } = useInferenceLaunch();

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
  // "Your models" holds everything actionable, so it opens by default; the
  // Hub repos nothing here tracks are the rare bucket and fold away, matching
  // how the run history hides its untracked Hub jobs.
  const [ownedOpen, setOwnedOpen] = useState(true);
  const [hubOpen, setHubOpen] = useState(false);

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

  const count =
    deployableModels.length + importedJobs.length + untrackedHubModels.length;
  const ownedCount = visibleTrained.length + visibleImported.length;
  const visibleCount = ownedCount + visibleUploaded.length;

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
            {/* Search only — the sections carry the split the filter pills
                used to, and provenance is already on every card's face. */}
            <LibraryToolbar
              query={search}
              onQueryChange={setSearch}
              searchPlaceholder="Search models"
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
              // A section with nothing the search matched drops out rather
              // than showing an empty fold; if both drop out, the "No models
              // match" line above stands in for the whole library.
              <div className="space-y-3">
                {ownedCount > 0 ? (
                  <Collapsible
                    open={ownedOpen}
                    onOpenChange={setOwnedOpen}
                    className="space-y-1"
                  >
                    <LibrarySectionHeader
                      title="Imported & trained"
                      count={ownedCount}
                      open={ownedOpen}
                    />
                    <CollapsibleContent className={cn(SLIDE, "space-y-1")}>
                      {/* Trained and imported cards merged newest-first by
                          start/import time; one row by default, rest behind
                          Show all. */}
                      <CappedGrid
                        // Model cards render every affordance (Run/Resume/
                        // Fine-tune/Download/step-picker) and must not clip —
                        // let rows size to content rather than the fixed
                        // 16.5rem datasets/jobs use.
                        flexHeight
                        items={[...visibleTrained, ...visibleImported]
                          .sort(
                            (a, b) => (b.started_at ?? 0) - (a.started_at ?? 0),
                          )
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
                          ))}
                      />
                    </CollapsibleContent>
                  </Collapsible>
                ) : null}

                {visibleUploaded.length > 0 ? (
                  <Collapsible
                    open={hubOpen}
                    onOpenChange={setHubOpen}
                    className="space-y-1"
                  >
                    <LibrarySectionHeader
                      title="On Hub, not imported"
                      count={visibleUploaded.length}
                      open={hubOpen}
                    />
                    <CollapsibleContent className={cn(SLIDE, "space-y-1")}>
                      <CappedGrid
                        flexHeight
                        items={[...visibleUploaded]
                          .sort(
                            (a, b) =>
                              (b.last_modified
                                ? Date.parse(b.last_modified) || 0
                                : 0) -
                              (a.last_modified
                                ? Date.parse(a.last_modified) || 0
                                : 0),
                          )
                          .map((model) => (
                            <HubModelCard
                              key={model.repo_id}
                              model={model}
                              onDeleted={refresh}
                              onAction={handleHubAction}
                            />
                          ))}
                      />
                    </CollapsibleContent>
                  </Collapsible>
                ) : null}
              </div>
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
