import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useJobsChangedSignal } from "@/hooks/useJobsChangedSignal";
import {
  HubJob,
  HubModel,
  JobProgressSnapshot,
  JobRecord,
  deleteJob,
  dismissHubJob,
  getJob,
  isHubJobActive,
  listHubJobs,
  listJobs,
  stopJob,
} from "@/lib/jobsApi";
import JobCard from "./JobCard";
import HubJobCard from "./HubJobCard";
import HubModelCard from "./HubModelCard";
import InferenceModal from "@/components/landing/InferenceModal";
import ImportModelModal from "./ImportModelModal";
import { useRobots } from "@/hooks/useRobots";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronRight, Download, RefreshCw, Search } from "lucide-react";

const LIMIT = 10;

const isJobActive = (j: JobRecord) =>
  j.state === "running" || j.checkpoint_count > 0;

const JobsSection: React.FC = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

  const [jobs, setJobs] = useState<JobRecord[]>([]);
  // Ancestors referenced via resume_from_job_id but paged out of the list, so a
  // resumed run can still nest its source even when the source is old.
  const [ancestorCache, setAncestorCache] = useState<Record<string, JobRecord>>(
    {},
  );
  const [hubJobs, setHubJobs] = useState<HubJob[]>([]);
  const [hubModels, setHubModels] = useState<HubModel[]>([]);
  const [hubAuthenticated, setHubAuthenticated] = useState(false);
  const [hubJobsPermission, setHubJobsPermission] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hubError, setHubError] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  const { selectedRecord } = useRobots();
  const [inferenceModalOpen, setInferenceModalOpen] = useState(false);
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [inferenceJob, setInferenceJob] = useState<JobRecord | null>(null);
  const [inferenceStep, setInferenceStep] = useState<number | null>(null);

  const refresh = useCallback(async () => {
    // Settle the two fetches independently: a hub failure (network, HF outage,
    // missing scope) must never blank the local jobs, and vice versa.
    const [localRes, hubRes] = await Promise.allSettled([
      listJobs(baseUrl, fetchWithHeaders, LIMIT),
      listHubJobs(baseUrl, fetchWithHeaders),
    ]);
    if (localRes.status === "fulfilled") {
      setJobs(localRes.value);
      setError(null);
    } else {
      const r = localRes.reason;
      setError(r instanceof Error ? r.message : String(r));
    }
    if (hubRes.status === "fulfilled") {
      setHubJobs(hubRes.value.jobs);
      setHubModels(hubRes.value.models);
      setHubAuthenticated(hubRes.value.authenticated);
      setHubJobsPermission(hubRes.value.jobs_permission ?? true);
      setHubError(null);
    } else {
      const r = hubRes.reason;
      setHubError(r instanceof Error ? r.message : String(r));
    }
  }, [baseUrl, fetchWithHeaders]);

  // Initial fetch on mount + refetch when the tab regains focus. Backend
  // pushes a `jobs_changed` WS event on every registry mutation, which
  // covers any change originating on this machine. The focus refresh
  // catches changes originating elsewhere (e.g. a job submitted from
  // another tab or the HF dashboard) without burning the rate limit.
  useEffect(() => {
    refresh();
    const onVisible = () => {
      if (document.visibilityState === "visible") refresh();
    };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", refresh);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("focus", refresh);
    };
  }, [refresh]);

  const applyProgress = useCallback((snapshots: JobProgressSnapshot[]) => {
    if (snapshots.length === 0) return;
    setJobs((prev) => {
      if (prev.length === 0) return prev;
      const byId = new Map(snapshots.map((s) => [s.id, s]));
      let mutated = false;
      const next = prev.map((j) => {
        const s = byId.get(j.id);
        if (!s) return j;
        mutated = true;
        return {
          ...j,
          state: s.state,
          metrics: s.metrics,
          wandb_run_url: s.wandb_run_url,
          checkpoint_count: s.checkpoint_count,
        };
      });
      return mutated ? next : prev;
    });
  }, []);

  useJobsChangedSignal(refresh, applyProgress);

  // Fetch the transitive closure of resume ancestors that aren't in the loaded
  // page (or already cached), so nesting works regardless of how old the source
  // run is. Idempotent: only unseen ids are fetched, so the frequent list
  // refreshes during a run don't re-fetch. Missing/deleted ancestors are
  // skipped, ending the chain.
  useEffect(() => {
    let cancelled = false;
    const loaded = new Set(jobs.map((j) => j.id));
    const queue = jobs
      .map((j) => j.config?.resume_from_job_id)
      .filter(
        (id): id is string => !!id && !loaded.has(id) && !ancestorCache[id],
      );
    if (queue.length === 0) return;
    (async () => {
      const fetched: Record<string, JobRecord> = {};
      const seen = new Set(queue);
      while (queue.length > 0) {
        const id = queue.shift() as string;
        try {
          const rec = await getJob(baseUrl, fetchWithHeaders, id);
          fetched[id] = rec;
          const parent = rec.config?.resume_from_job_id;
          if (
            parent &&
            !loaded.has(parent) &&
            !ancestorCache[parent] &&
            !seen.has(parent)
          ) {
            seen.add(parent);
            queue.push(parent);
          }
        } catch {
          // Ancestor deleted or unreachable — skip; the chain just stops here.
        }
      }
      if (!cancelled && Object.keys(fetched).length > 0) {
        setAncestorCache((prev) => ({ ...prev, ...fetched }));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [jobs, ancestorCache, baseUrl, fetchWithHeaders]);

  const handleStop = async (id: string) => {
    try {
      await stopJob(baseUrl, fetchWithHeaders, id);
      toast({ title: "Job stopping" });
      refresh();
    } catch (e) {
      toast({
        title: "Stop failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handlePlay = (job: JobRecord, step: number) => {
    setInferenceJob(job);
    setInferenceStep(step);
    setInferenceModalOpen(true);
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteJob(baseUrl, fetchWithHeaders, id);
      toast({ title: "Job removed" });
      refresh();
    } catch (e) {
      toast({
        title: "Delete failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  // Untracked hub jobs aren't deletable on the Hub (the Jobs API has no
  // delete), so "remove" is a persisted backend-side dismissal.
  const handleDismissHubJob = async (id: string) => {
    try {
      await dismissHubJob(baseUrl, fetchWithHeaders, id);
      toast({ title: "Job removed from list" });
      refresh();
    } catch (e) {
      toast({
        title: "Remove failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const query = search.trim().toLowerCase();
  const matchesQuery = useCallback(
    (text: string | null | undefined) =>
      !query || (text ?? "").toLowerCase().includes(query),
    [query],
  );

  const filteredJobs = useMemo(
    // Match on the display alias as well as the original name, so a renamed
    // model is findable by either.
    () => jobs.filter((j) => matchesQuery(j.name) || matchesQuery(j.display_name)),
    [jobs, matchesQuery],
  );
  const filteredHubJobs = useMemo(
    () =>
      hubJobs.filter((h) => matchesQuery(h.docker_image ?? h.space_id ?? h.id)),
    [hubJobs, matchesQuery],
  );
  const filteredHubModels = useMemo(
    () => hubModels.filter((m) => matchesQuery(m.repo_id)),
    [hubModels, matchesQuery],
  );

  const localJobs = useMemo(
    () => filteredJobs.filter((j) => j.runner === "local"),
    [filteredJobs],
  );
  const trackedCloudJobs = useMemo(
    () => filteredJobs.filter((j) => j.runner === "hf_cloud"),
    [filteredJobs],
  );
  const importedJobs = useMemo(
    () => filteredJobs.filter((j) => j.runner === "imported"),
    [filteredJobs],
  );
  // Hub jobs already mirrored by a local JobRecord get their richer card via
  // trackedCloudJobs; everything else from the hub gets a plain HubJobCard.
  const trackedHfJobIds = useMemo(
    () =>
      new Set(
        trackedCloudJobs
          .map((j) => j.hf_job_id)
          .filter((id): id is string => !!id),
      ),
    [trackedCloudJobs],
  );
  const untrackedHubJobs = useMemo(
    () => filteredHubJobs.filter((h) => !trackedHfJobIds.has(h.id)),
    [filteredHubJobs, trackedHfJobIds],
  );
  // Hide model repos that map 1-to-1 to a tracked cloud job (those already
  // appear via JobCard); the remainder are past trainings the registry no
  // longer remembers.
  const trackedRepoIds = useMemo(
    () =>
      new Set(
        trackedCloudJobs
          .map((j) => j.hf_repo_id)
          .filter((id): id is string => !!id),
      ),
    [trackedCloudJobs],
  );
  const untrackedHubModels = useMemo(
    () => filteredHubModels.filter((m) => !trackedRepoIds.has(m.repo_id)),
    [filteredHubModels, trackedRepoIds],
  );

  // Resume lineage: job B stores config.resume_from_job_id = A. Hide A (the
  // superseded run) from the top level and nest it under B, so a resumed chain
  // reads as one entry. Lineage is linear — each job resumes from one parent.
  const byId = useMemo(() => {
    const m = new Map(jobs.map((j) => [j.id, j]));
    // Cached ancestors fill in parents paged out of the list (never overriding
    // a loaded record).
    for (const rec of Object.values(ancestorCache)) {
      if (!m.has(rec.id)) m.set(rec.id, rec);
    }
    return m;
  }, [jobs, ancestorCache]);
  const supersededIds = useMemo(() => {
    const s = new Set<string>();
    for (const j of jobs) {
      const parent = j.config?.resume_from_job_id;
      // Only a real successor (running or with its own checkpoints) supersedes
      // its parent — a failed continuation shouldn't hide the source run.
      const legit = j.state === "running" || j.checkpoint_count > 0;
      if (parent && byId.has(parent) && legit) s.add(parent);
    }
    return s;
  }, [jobs, byId]);
  const ancestorsOf = useCallback(
    (job: JobRecord): JobRecord[] => {
      const chain: JobRecord[] = [];
      const seen = new Set<string>([job.id]);
      let cur = byId.get(job.config?.resume_from_job_id ?? "");
      while (cur && !seen.has(cur.id)) {
        chain.push(cur);
        seen.add(cur.id);
        cur = byId.get(cur.config?.resume_from_job_id ?? "");
      }
      return chain;
    },
    [byId],
  );

  // Active = running or has runnable checkpoints. Everything else collapses
  // under UNTRACKED so the eye lands on what's still relevant. Superseded runs
  // are dropped from both — they surface nested under their successor instead.
  const localActive = useMemo(
    () => localJobs.filter((j) => isJobActive(j) && !supersededIds.has(j.id)),
    [localJobs, supersededIds],
  );
  const localUntracked = useMemo(
    () => localJobs.filter((j) => !isJobActive(j) && !supersededIds.has(j.id)),
    [localJobs, supersededIds],
  );
  const trackedCloudActive = useMemo(
    () => trackedCloudJobs.filter(isJobActive),
    [trackedCloudJobs],
  );
  const trackedCloudUntracked = useMemo(
    () => trackedCloudJobs.filter((j) => !isJobActive(j)),
    [trackedCloudJobs],
  );
  const untrackedHubActive = useMemo(
    () => untrackedHubJobs.filter(isHubJobActive),
    [untrackedHubJobs],
  );
  const untrackedHubInactive = useMemo(
    () => untrackedHubJobs.filter((h) => !isHubJobActive(h)),
    [untrackedHubJobs],
  );

  const untrackedCount =
    localUntracked.length +
    trackedCloudUntracked.length +
    untrackedHubInactive.length;

  return (
    <section className="grid grid-cols-1 lg:grid-cols-2 gap-x-8 gap-y-10 items-start">
      {/* Jobs column: local runs, cloud runs, and inactive/untracked leftovers.
          The search box lives here (it primarily filters job text) but keeps
          filtering the models column too, so behavior is unchanged. */}
      <div className="space-y-6">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-lg font-semibold text-white">Jobs</h2>
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-400 pointer-events-none" />
              <Input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search jobs"
                className="h-8 w-48 sm:w-60 pl-8 bg-slate-800/50 border-slate-700 text-sm text-white placeholder:text-slate-500"
                aria-label="Search jobs"
              />
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={refresh}
              className="h-7 w-7 text-slate-400 hover:text-white"
              aria-label="Refresh jobs"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {error ? (
          <p className="text-sm text-red-300">
            Couldn't load local jobs: {error}
          </p>
        ) : null}

        <Collapsible defaultOpen>
          <CollapsibleTrigger className="group flex items-center gap-1.5 text-sm font-semibold uppercase tracking-wide text-slate-400 hover:text-white transition-colors">
            <ChevronRight className="w-3.5 h-3.5 transition-transform group-data-[state=open]:rotate-90" />
            Local jobs ({localActive.length})
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-3">
            {localActive.length === 0 ? (
              <p className="text-sm text-slate-500">
                {query
                  ? "No local jobs match your search."
                  : "No active local jobs. Start one from the Training page."}
              </p>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-1 gap-4">
                {localActive.map((job) => (
                  <JobCard
                    key={job.id}
                    job={job}
                    onStop={handleStop}
                    onDelete={handleDelete}
                    onPlay={handlePlay}
                    onRenamed={refresh}
                    ancestors={ancestorsOf(job)}
                  />
                ))}
              </div>
            )}
          </CollapsibleContent>
        </Collapsible>

        <div className="border-t border-slate-700" />

        <Collapsible defaultOpen>
          <CollapsibleTrigger className="group flex items-center gap-1.5 text-sm font-semibold uppercase tracking-wide text-slate-400 hover:text-white transition-colors">
            <ChevronRight className="w-3.5 h-3.5 transition-transform group-data-[state=open]:rotate-90" />
            Online jobs ({trackedCloudActive.length + untrackedHubActive.length})
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-3">
            {hubError ? (
              <p className="text-sm text-red-300">
                Couldn't load cloud jobs: {hubError}
              </p>
            ) : !hubAuthenticated && trackedCloudJobs.length === 0 ? (
              <p className="text-sm text-slate-500">
                Sign in with Hugging Face to see your cloud jobs.
              </p>
            ) : (
              <div className="space-y-3">
                {hubAuthenticated && !hubJobsPermission ? (
                  <p className="text-sm text-amber-300/80">
                    Your Hugging Face token is missing the{" "}
                    <code className="text-amber-200">job.read</code> permission,
                    so cloud jobs can't be listed. Uploaded models still appear
                    under Models.
                  </p>
                ) : null}
                {trackedCloudActive.length === 0 &&
                untrackedHubActive.length === 0 ? (
                  hubAuthenticated && !hubJobsPermission ? null : (
                    <p className="text-sm text-slate-500">
                      {query
                        ? "No online jobs match your search."
                        : "No active cloud jobs."}
                    </p>
                  )
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-1 gap-4">
                    {trackedCloudActive.map((job) => (
                      <JobCard
                        key={job.id}
                        job={job}
                        onStop={handleStop}
                        onDelete={handleDelete}
                        onPlay={handlePlay}
                        onRenamed={refresh}
                      />
                    ))}
                    {untrackedHubActive.map((job) => (
                      <HubJobCard
                        key={job.id}
                        job={job}
                        onDismiss={handleDismissHubJob}
                      />
                    ))}
                  </div>
                )}
              </div>
            )}
          </CollapsibleContent>
        </Collapsible>

        {untrackedCount > 0 ? (
          <Collapsible>
            <CollapsibleTrigger className="group flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400 hover:text-white transition-colors">
              <ChevronRight className="w-3.5 h-3.5 transition-transform group-data-[state=open]:rotate-90" />
              Untracked ({untrackedCount})
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-3">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-1 gap-4">
                {localUntracked.map((job) => (
                  <JobCard
                    key={job.id}
                    job={job}
                    onStop={handleStop}
                    onDelete={handleDelete}
                    onPlay={handlePlay}
                    onRenamed={refresh}
                    ancestors={ancestorsOf(job)}
                  />
                ))}
                {trackedCloudUntracked.map((job) => (
                  <JobCard
                    key={job.id}
                    job={job}
                    onStop={handleStop}
                    onDelete={handleDelete}
                    onPlay={handlePlay}
                    onRenamed={refresh}
                  />
                ))}
                {untrackedHubInactive.map((job) => (
                  <HubJobCard
                    key={job.id}
                    job={job}
                    onDismiss={handleDismissHubJob}
                  />
                ))}
              </div>
            </CollapsibleContent>
          </Collapsible>
        ) : null}
      </div>

      {/* Models column: imported models plus uploaded hub repos no job tracks
          (a model artifact, not a run — so it doesn't sit under Online jobs).
          Owns the Import button; rendered even when empty so the entry point
          is always visible. */}
      <div className="space-y-6">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-lg font-semibold text-white">
            Models
            {importedJobs.length + untrackedHubModels.length > 0
              ? ` (${importedJobs.length + untrackedHubModels.length})`
              : ""}
          </h2>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setImportModalOpen(true)}
            className="h-8 border-slate-700 bg-slate-800/50 text-slate-200 hover:text-white"
          >
            <Download className="w-3.5 h-3.5 mr-1.5" />
            Import model
          </Button>
        </div>
        {importedJobs.length === 0 && untrackedHubModels.length === 0 ? (
          <p className="text-sm text-slate-500">
            {query
              ? "No models match your search."
              : "No models yet. Use Import model to add one from the Hub or a local folder."}
          </p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-1 gap-4">
            {importedJobs.map((job) => (
              <JobCard
                key={job.id}
                job={job}
                onStop={handleStop}
                onDelete={handleDelete}
                onPlay={handlePlay}
                onRenamed={refresh}
              />
            ))}
            {untrackedHubModels.map((model) => (
              <HubModelCard
                key={model.repo_id}
                model={model}
                onDeleted={refresh}
              />
            ))}
          </div>
        )}
      </div>

      {inferenceJob ? (
        <InferenceModal
          open={inferenceModalOpen}
          onOpenChange={setInferenceModalOpen}
          robot={selectedRecord}
          jobId={inferenceJob.id}
          initialStep={inferenceStep}
        />
      ) : null}

      <ImportModelModal
        open={importModalOpen}
        onOpenChange={setImportModalOpen}
        onImported={refresh}
      />
    </section>
  );
};

export default JobsSection;
