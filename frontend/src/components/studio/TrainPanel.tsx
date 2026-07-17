import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Check, Loader2, Play, Plus, X } from "lucide-react";

import { useStudio } from "@/contexts/StudioContext";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useDatasets } from "@/hooks/useDatasets";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import { getModels, ModelItem } from "@/lib/modelsApi";
import { importModel, jobDisplayName } from "@/lib/jobsApi";
import { listJobCheckpoints } from "@/lib/checkpointsApi";
import {
  DatasetItem,
  getDatasetInfo,
  saveCustomDataset,
} from "@/lib/replayApi";
import { HUB_REPO_ID_RE } from "@/lib/repoId";
import TrainingConfigurator, {
  FinetuneSeed,
} from "@/components/training/TrainingConfigurator";
import TrainingJobDialog from "@/components/training/TrainingJobDialog";
import JobsLibrary from "@/components/jobs/JobsLibrary";
import {
  LibrarySection,
  PanelEntryControl,
  PanelHeader,
  SLIDE,
} from "@/components/studio/panel/primitives";

const NONE = "__none__";

/** One search-result row: repo id + (local) episode count, lazily fetched, +
 * Hub marker for remote-only rows. Skips the network for Hub-only rows (a
 * remote meta.json read) — counts are shown "where available". */
const DatasetResultRow: React.FC<{
  item: DatasetItem;
  selected: boolean;
  onPick: () => void;
}> = ({ item, selected, onPick }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [episodes, setEpisodes] = useState<number | null>(null);

  useEffect(() => {
    if (item.source === "hub") return; // avoid a remote fetch just for a count
    let cancelled = false;
    getDatasetInfo(baseUrl, fetchWithHeaders, item.repo_id)
      .then((info) => {
        if (!cancelled) setEpisodes(info.total_episodes);
      })
      .catch(() => {
        /* count is optional — leave it blank */
      });
    return () => {
      cancelled = true;
    };
  }, [item.repo_id, item.source, baseUrl, fetchWithHeaders]);

  return (
    <button
      type="button"
      onClick={onPick}
      className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-muted/50"
    >
      <span className="min-w-0 flex-1 truncate font-mono text-foreground">
        {item.repo_id}
      </span>
      {episodes != null ? (
        <span className="shrink-0 text-xs text-muted-foreground">
          {episodes} ep
        </span>
      ) : null}
      {item.source === "hub" ? (
        <span className="shrink-0 text-xs text-muted-foreground">Hub</span>
      ) : null}
      {selected ? (
        <Check className="h-3.5 w-3.5 shrink-0 text-primary" />
      ) : null}
    </button>
  );
};

/**
 * Studio panel 2 · Train. Mirrors the Collect panel's progressive disclosure:
 * a "Start a new training" button slides the full configuration open in place
 * (base skill → dataset → the shared training configurator), folding the
 * training-jobs library down to its header while the form is open. Policy is
 * chosen inside Run configuration — there is no separate policy grid.
 */
const TrainPanel: React.FC = () => {
  const { trainPrefill, clearTrainPrefill, monitorJobId, closeJobMonitor } =
    useStudio();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { datasets, refresh: refreshDatasets } = useDatasets();
  const { selectedDataset, setSelectedDataset } = useSelectedDataset();

  // The new-training form slides open in place; the jobs library folds to its
  // header while the form is open (still expandable by hand).
  const [formOpen, setFormOpen] = useState(false);
  const [jobsOpen, setJobsOpen] = useState(true);

  // Pinned actions slot above the jobs library. While the form is open the
  // configurator portals its fully-gated Start button here; closed, a
  // disabled stand-in keeps the action visible at the same spot.
  const [actionsEl, setActionsEl] = useState<HTMLDivElement | null>(null);

  const toggleForm = (open: boolean) => {
    setFormOpen(open);
    setJobsOpen(!open);
  };

  // ── Base skill (fine-tune) ────────────────────────────────────────────────
  const [models, setModels] = useState<ModelItem[]>([]);
  const [baseModelId, setBaseModelId] = useState<string>(NONE);
  const [finetuneSeed, setFinetuneSeed] = useState<FinetuneSeed | null>(null);
  const [resolvingBase, setResolvingBase] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getModels(baseUrl, fetchWithHeaders)
      .then((m) => {
        if (!cancelled) setModels(m);
      })
      .catch(() => {
        /* base skill is optional — leave the list empty */
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  // ── Policy ────────────────────────────────────────────────────────────────
  // Chosen inside Run configuration (EssentialsCard's select); a base-skill or
  // prefill choice re-targets it so the fine-tune trains the matching policy.
  const [policyType, setPolicyType] = useState<string>("act");

  // Resolve a base model into a fine-tune seed: fine-tuning needs a concrete
  // job id + checkpoint step (the exact contract Training's finetune flow
  // expects). A Hub-only model is registered as an imported job first (the
  // proven lazy-import path), then its latest checkpoint step seeds the run.
  // Sequence guard: only the LATEST resolution may write state — a slower,
  // older import/checkpoint lookup finishing last must not overwrite a newer
  // base-skill choice.
  const resolveSeqRef = useRef(0);
  const resolveFinetune = useCallback(
    async (opts: {
      jobId?: string;
      repoId?: string;
      name?: string;
      step?: number;
    }) => {
      const seq = ++resolveSeqRef.current;
      const current = () => resolveSeqRef.current === seq;
      setResolvingBase(true);
      try {
        let jobId = opts.jobId;
        let name = opts.name;
        let policy: string | null = null;
        if (!jobId && opts.repoId) {
          const rec = await importModel(baseUrl, fetchWithHeaders, opts.repoId);
          jobId = rec.id;
          name = name ?? jobDisplayName(rec);
          policy = rec.config?.policy_type ?? null;
        }
        if (!jobId || !current()) return;
        const cks = await listJobCheckpoints(baseUrl, fetchWithHeaders, jobId);
        if (!current()) return;
        // A caller-pinned step (the card's dropdown choice) wins when it still
        // exists; otherwise fall back to the latest checkpoint.
        const pinned =
          opts.step != null && cks.some((c) => c.step === opts.step)
            ? opts.step
            : null;
        const latest =
          pinned ?? (cks.length > 0 ? cks[cks.length - 1].step : null);
        if (latest == null) {
          toast({
            title: "No checkpoints in this model",
            description: "It has no saved checkpoint to fine-tune from.",
            variant: "destructive",
          });
          setBaseModelId(NONE);
          setFinetuneSeed(null);
          return;
        }
        setFinetuneSeed({
          jobId,
          step: latest,
          name: name ?? jobId,
          policyType: policy ?? "act",
        });
        if (policy) setPolicyType(policy);
      } catch (e) {
        if (!current()) return;
        toast({
          title: "Couldn't load base model",
          description: e instanceof Error ? e.message : String(e),
          variant: "destructive",
        });
        setBaseModelId(NONE);
        setFinetuneSeed(null);
      } finally {
        if (current()) setResolvingBase(false);
      }
    },
    [baseUrl, fetchWithHeaders, toast],
  );

  const handleBaseModelChange = (value: string) => {
    setBaseModelId(value);
    if (value === NONE) {
      setFinetuneSeed(null);
      return;
    }
    const model = models.find((m) => m.id === value);
    if (!model) {
      setFinetuneSeed(null);
      return;
    }
    // Local runs already have a job id; Hub-only models resolve via import.
    if (model.source === "hub") {
      resolveFinetune({ repoId: model.hf_repo_id ?? model.id, name: model.name });
    } else {
      if (model.policy_type) setPolicyType(model.policy_type);
      resolveFinetune({ jobId: model.id, name: model.name });
    }
  };

  // ── Dataset picker (single-select) ────────────────────────────────────────
  const [selectedId, setSelectedId] = useState<string | null>(
    () => selectedDataset,
  );
  const [query, setQuery] = useState("");

  // Apply a studio prefill (fine-tune base / preselected dataset) once, then
  // clear it so reopening the studio fresh doesn't re-apply a stale one.
  // Local skills arrive as baseJobId (a job registry id), Hub skills as
  // baseModelRepoId — a job id must never be sent through the Hub import path.
  // A prefill is an intent to configure a run, so it slides the form open too.
  useEffect(() => {
    if (!trainPrefill) return;
    if (trainPrefill.datasetRepoId) {
      setSelectedId(trainPrefill.datasetRepoId);
    }
    if (trainPrefill.baseJobId) {
      setBaseModelId(trainPrefill.baseJobId);
      resolveFinetune({
        jobId: trainPrefill.baseJobId,
        step: trainPrefill.baseStep,
        name: trainPrefill.baseName,
      });
    } else if (trainPrefill.baseModelRepoId) {
      setBaseModelId(trainPrefill.baseModelRepoId);
      resolveFinetune({
        repoId: trainPrefill.baseModelRepoId,
        step: trainPrefill.baseStep,
        name: trainPrefill.baseName,
      });
    }
    setFormOpen(true);
    setJobsOpen(false);
    clearTrainPrefill();
  }, [trainPrefill, clearTrainPrefill, resolveFinetune]);

  // Follow the shared selection while mounted: picking a dataset in Collect's
  // library (or the handoff banner) must re-target Train too — the panels are
  // mounted simultaneously, so mount-time seeding isn't enough.
  useEffect(() => {
    if (selectedDataset) setSelectedId(selectedDataset);
  }, [selectedDataset]);

  // Empty keeps the configurator's Start disabled until a dataset is picked.
  const trainingDatasetRepoId = selectedId ?? "";

  // Keep the shared selection (Deploy panel, direct /training route) in step
  // with the dataset chosen here.
  useEffect(() => {
    if (selectedId) setSelectedDataset(selectedId);
  }, [selectedId, setSelectedDataset]);

  // Search-driven picker: results only exist while a query is typed — there is
  // no standing list.
  const trimmedQuery = query.trim();
  const matches = useMemo(() => {
    const q = trimmedQuery.toLowerCase();
    if (!q) return [];
    return datasets.filter((d) => d.repo_id.toLowerCase().includes(q));
  }, [datasets, trimmedQuery]);

  // A well-formed `org/name` id that isn't in the library yet is offered as a
  // public Hub dataset — the affordance that ANY public dataset can be trained
  // on, not just the user's own.
  const hubCandidate = useMemo(() => {
    if (!HUB_REPO_ID_RE.test(trimmedQuery)) return null;
    const q = trimmedQuery.toLowerCase();
    if (datasets.some((d) => d.repo_id.toLowerCase() === q)) return null;
    return trimmedQuery;
  }, [datasets, trimmedQuery]);

  // Picking a result replaces the selection and collapses the results by
  // clearing the query.
  const pickDataset = (repoId: string) => {
    setSelectedId(repoId);
    setQuery("");
  };

  // Selecting a not-yet-listed public Hub id also pins it (best-effort, same
  // path as the library's "Add from Hub") so it persists in dataset lists and
  // training can fetch it on demand.
  const addHubDataset = (repoId: string) => {
    setSelectedId(repoId);
    setQuery("");
    saveCustomDataset(baseUrl, fetchWithHeaders, repoId)
      .then(() => refreshDatasets())
      .catch(() => {
        /* non-fatal: the dataset is still selected for this run */
      });
  };

  return (
    <div className="flex flex-1 flex-col gap-5 p-5">
      <PanelHeader step="2" title="Train" />

      {/* Start a new training — the form slides open in place (no dialog),
          mirroring Collect's "Record new dataset". */}
      <Collapsible open={formOpen} onOpenChange={toggleForm} className="space-y-5">
        <CollapsibleTrigger asChild>
          <PanelEntryControl open={formOpen} dotClassName="bg-emerald-500">
            Start a new training
          </PanelEntryControl>
        </CollapsibleTrigger>
        <CollapsibleContent className={SLIDE}>
          <div className="space-y-6">
            <p className="text-sm leading-relaxed text-muted-foreground">
              Pick the dataset to train on and where the run should execute,
              then start training.
            </p>

            {/* Base skill (optional) → fine-tune */}
            <section className="space-y-3">
              <h3 className="eyebrow">Base skill (optional)</h3>
              <div className="space-y-2">
                <Select value={baseModelId} onValueChange={handleBaseModelChange}>
                  <SelectTrigger className="w-full">
                    {/* A prefilled base (job card's Fine-tune) may not exist as
                        an item in the models listing — render the resolved
                        seed's name so the trigger is never blank (same pattern
                        as the Deploy panel's skill picker). */}
                    {baseModelId !== NONE && finetuneSeed ? (
                      <span className="truncate">{finetuneSeed.name}</span>
                    ) : (
                      <SelectValue placeholder="Train from scratch" />
                    )}
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={NONE}>Train from scratch</SelectItem>
                    {models.map((m) => (
                      <SelectItem key={m.id} value={m.id}>
                        {m.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {resolvingBase ? (
                    <span className="inline-flex items-center gap-1">
                      <Loader2 className="h-3 w-3 animate-spin" /> Loading
                      checkpoints…
                    </span>
                  ) : finetuneSeed ? (
                    "Fine-tunes from this model's latest checkpoint."
                  ) : (
                    "Pick a model to fine-tune from, or train a fresh policy."
                  )}
                </p>
              </div>
            </section>

            {/* Dataset picker — search-driven, no standing list */}
            <section className="space-y-3">
              <h3 className="eyebrow">Dataset</h3>
              {selectedId ? (
                <div className="flex flex-wrap gap-1.5">
                  <span className="inline-flex max-w-full items-center gap-1 rounded-md border border-border bg-muted/40 py-1 pl-2 pr-1 font-mono text-xs text-foreground">
                    <span className="truncate">{selectedId}</span>
                    <button
                      type="button"
                      aria-label={`Remove ${selectedId}`}
                      onClick={() => setSelectedId(null)}
                      className="shrink-0 rounded p-0.5 text-muted-foreground hover:text-foreground"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </span>
                </div>
              ) : null}
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search datasets, or type any public org/name Hub id"
                className="h-8 text-sm"
                aria-label="Search datasets"
              />
              {trimmedQuery ? (
                <div className="max-h-56 divide-y divide-border overflow-auto rounded-md border border-border">
                  {matches.map((d) => (
                    <DatasetResultRow
                      key={d.repo_id}
                      item={d}
                      selected={selectedId === d.repo_id}
                      onPick={() => pickDataset(d.repo_id)}
                    />
                  ))}
                  {hubCandidate ? (
                    <button
                      type="button"
                      onClick={() => addHubDataset(hubCandidate)}
                      className="flex w-full items-start gap-2 px-3 py-2 text-left text-sm hover:bg-muted/50"
                    >
                      <Plus className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate">
                          Use <span className="font-mono">{hubCandidate}</span>{" "}
                          from the Hub
                        </span>
                        <span className="block text-xs text-muted-foreground">
                          Public dataset — training fetches it on demand.
                        </span>
                      </span>
                    </button>
                  ) : null}
                  {matches.length === 0 && !hubCandidate ? (
                    <p className="px-3 py-4 text-sm text-muted-foreground">
                      No matching datasets. Type a full{" "}
                      <span className="font-mono">org/name</span> id to use any
                      public Hugging Face dataset.
                    </p>
                  ) : null}
                </div>
              ) : null}
              {!selectedId ? (
                <p className="text-xs text-muted-foreground">
                  Search to pick a dataset — yours or any public Hugging Face
                  dataset.
                </p>
              ) : null}
            </section>

            {/* Shared configuration form: compute target, run configuration,
                advanced, extras gates, and the Start button with all its
                gating. Policy is chosen inside Run configuration. */}
            <TrainingConfigurator
              key={finetuneSeed?.jobId ?? "fresh"}
              policyType={policyType}
              onPolicyTypeChange={setPolicyType}
              datasetRepoId={trainingDatasetRepoId}
              finetuneSeed={finetuneSeed}
              // Launch opens the monitor dialog over this panel (via
              // openJobMonitor in the configurator); fold the form back so
              // closing the dialog lands on the jobs library, not a stale form.
              onStarted={() => toggleForm(false)}
              actionsContainer={actionsEl}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>

      {/* Start training — pinned directly above the jobs library, level with
          Collect's and Deploy's actions. The configurator portals its real,
          fully-gated button into the slot while the form is open. */}
      <div className="mt-auto pt-2">
        {!formOpen ? (
          <Button disabled className="w-full gap-2">
            <Play className="h-4 w-4" />
            Start training
          </Button>
        ) : null}
        <div ref={setActionsEl} className={formOpen ? undefined : "hidden"} />
      </div>

      {/* Training jobs library — local + online runs as cards, pinned to the
          panel foot like Collect's datasets and Deploy's models. mt-0 keeps it
          glued to the actions slot above, which carries the panel's mt-auto. */}
      <LibrarySection className="mt-0">
        <JobsLibrary open={jobsOpen} onOpenChange={setJobsOpen} />
      </LibrarySection>

      {/* Job monitor as a dialog over the studio (same pattern as Collect's
          RecordingSessionDialog) — closing it lands back on this panel. */}
      {monitorJobId ? (
        <TrainingJobDialog jobId={monitorJobId} onExit={closeJobMonitor} />
      ) : null}
    </div>
  );
};

export default TrainPanel;
