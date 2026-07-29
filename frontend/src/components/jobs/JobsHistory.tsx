import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  Clock,
  ExternalLink,
  FastForward,
  HelpCircle,
  Loader2,
  RefreshCw,
  Trash2,
  XCircle,
} from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import LibraryToolbar from "@/components/library/LibraryToolbar";
import LibrarySectionHeader from "@/components/library/LibrarySectionHeader";
import { SLIDE } from "@/components/studio/panel/primitives";
import { useApi } from "@/contexts/ApiContext";
import { useStudio } from "@/contexts/StudioContext";
import { useToast } from "@/hooks/use-toast";
import { listJobCheckpoints } from "@/lib/checkpointsApi";
import { HubJob, JobRecord, isHubJobActive, jobDisplayName } from "@/lib/jobsApi";
import { useJobsData } from "./JobsDataContext";

/** How many finished rows a section shows before "Show all". Running rows are
 * pinned above this cap — a live run must never be the thing that scrolls out
 * of a collapsed history. */
const HISTORY_CAP = 5;

const jobTime = (j: JobRecord) => (j.started_at ?? 0) * 1000;
const hubTime = (h: HubJob) =>
  h.created_at ? Date.parse(h.created_at) || 0 : 0;

function relativeTime(ms: number): string {
  if (!ms) return "—";
  const diff = Math.max(0, (Date.now() - ms) / 1000);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

interface Presentation {
  label: string;
  color: string;
  Icon: React.ComponentType<{ className?: string }>;
  spin?: boolean;
}

const statePresentation: Record<JobRecord["state"], Presentation> = {
  running: { label: "Running", color: "text-ok", Icon: Loader2, spin: true },
  done: { label: "Done", color: "text-muted-foreground", Icon: CheckCircle2 },
  failed: { label: "Failed", color: "text-destructive", Icon: XCircle },
  interrupted: { label: "Stopped", color: "text-warn", Icon: AlertTriangle },
};

const stagePresentation: Record<string, Presentation> = {
  RUNNING: { label: "Running", color: "text-ok", Icon: Loader2, spin: true },
  QUEUED: { label: "Queued", color: "text-warn", Icon: Clock },
  SCHEDULING: { label: "Starting", color: "text-warn", Icon: Clock },
  COMPLETED: {
    label: "Done",
    color: "text-muted-foreground",
    Icon: CheckCircle2,
  },
  FAILED: { label: "Failed", color: "text-destructive", Icon: XCircle },
  // HF API uses "CANCELED" (single L); accept both spellings.
  CANCELED: { label: "Cancelled", color: "text-warn", Icon: AlertTriangle },
  CANCELLED: { label: "Cancelled", color: "text-warn", Icon: AlertTriangle },
};

// Two-line row. The library drawer is ~320px of usable width and a run name
// runs to ~66 characters, so the name gets a whole line to itself and the
// former columns collapse into one muted meta line beneath it. TITLE_LINE's
// icon (h-3) + gap (1.5) is 1.125rem, which is what META indents by so the
// meta line hangs under the name rather than under the status icon.
const ROW = "block w-full rounded-md px-2 py-1 text-left";
const TITLE_LINE = "flex items-center gap-1.5";
const NAME = "min-w-0 flex-1 truncate text-xs font-medium text-foreground";
const META = "mt-0.5 truncate pl-[1.125rem] font-mono text-[11px]";

/** Footer toggle, styled like the card libraries' "Show all" so the history
 * and the model grid share one footer language. */
const ShowAllToggle: React.FC<{
  expanded: boolean;
  total: number;
  onToggle: () => void;
}> = ({ expanded, total, onToggle }) => (
  <button
    type="button"
    onClick={onToggle}
    className="flex h-[1.875rem] w-full items-center justify-center gap-1 rounded-md border border-dashed border-border text-xs font-medium text-muted-foreground transition-colors hover:border-muted-foreground/40 hover:text-foreground"
  >
    <ChevronDown
      className={cn("h-3.5 w-3.5 transition-transform", expanded && "rotate-180")}
    />
    {expanded ? "Show less" : `Show all ${total}`}
  </button>
);

/** One local run: status icon + name, then Progress · When · Where, plus
 * Resume on a run that died before its target (those never become model cards,
 * so this is the only place a failed run can be continued from). The dataset no
 * longer has a column of its own — it rides along in the row's tooltip.
 * Clicking the row opens the job monitor. */
const LocalRow: React.FC<{
  job: JobRecord;
  onOpen: (id: string) => void;
  onResume: (job: JobRecord) => void;
  resuming: boolean;
}> = ({ job, onOpen, onResume, resuming }) => {
  const present = statePresentation[job.state];
  const Icon = present.Icon;
  const isRunning = job.state === "running";
  const target = job.config?.steps || job.metrics.total_steps || 0;
  const current = job.metrics.current_step;
  const progress =
    target > 0
      ? `${current.toLocaleString()} / ${target.toLocaleString()}`
      : current > 0
        ? current.toLocaleString()
        : "—";
  const progressPct =
    target > 0 ? Math.min(100, (current / target) * 100) : 0;
  const when = relativeTime(
    job.ended_at != null ? job.ended_at * 1000 : jobTime(job),
  );
  const isCloud = job.runner === "hf_cloud";
  const where = isCloud ? (job.hf_flavor ?? "cloud") : "local";
  // Resume is offered on a run that ended early with something to resume from.
  // Mirrors JobCard's gate; the exact checkpoint step is resolved on click.
  const canResume =
    (job.state === "failed" || job.state === "interrupted") &&
    job.checkpoint_count > 0;

  return (
    <div className={cn(isRunning && "rounded-md bg-muted/30")}>
      <div
        role="button"
        tabIndex={0}
        onClick={() => onOpen(job.id)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onOpen(job.id);
          }
        }}
        title={[jobDisplayName(job), job.config?.dataset_repo_id]
          .filter(Boolean)
          .join(" · ")}
        className={cn(
          ROW,
          "cursor-pointer hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        )}
      >
        <div className={TITLE_LINE}>
          <Icon
            className={cn(
              "h-3 w-3 shrink-0",
              present.color,
              present.spin && "animate-spin",
            )}
          />
          {/* The icon alone carries state visually; keep the word for screen
              readers, since the meta line below spends its width on numbers. */}
          <span className="sr-only">{present.label}</span>
          <span className={NAME}>{jobDisplayName(job)}</span>
          {canResume ? (
            <button
              type="button"
              disabled={resuming}
              onClick={(e) => {
                e.stopPropagation();
                onResume(job);
              }}
              aria-label="Resume this run from its last checkpoint"
              title="Resume from the last checkpoint"
              className="flex h-5 w-5 shrink-0 items-center justify-center rounded text-info transition-colors hover:bg-info/10 disabled:opacity-50"
            >
              {resuming ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <FastForward className="h-3 w-3" />
              )}
            </button>
          ) : null}
        </div>
        <p className={cn(META, "text-muted-foreground")}>
          <span className="tabular-nums">{progress}</span>
          {" · "}
          {when}
          {" · "}
          <span title={isCloud ? `Hugging Face cloud · ${where}` : "This machine"}>
            {where}
          </span>
        </p>
      </div>
      {/* Live progress for the pinned running rows — the one thing a collapsed
          history would otherwise lose. */}
      {isRunning ? (
        <div className="mx-2 mb-1 h-1 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full bg-info transition-[width] duration-500"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      ) : null}
    </div>
  );
};

/** One Hub-only job: owner on the title line, then Stage · Flavor · When. A
 * HubJob carries no dataset, step target, or policy, so those fields simply
 * don't exist here rather than rendering a row of em dashes — which is also
 * why the stage keeps its word (QUEUED and SCHEDULING share an icon, and
 * there is no progress figure competing for the meta line). */
const HubRow: React.FC<{
  job: HubJob;
  onDismiss: (id: string) => void;
}> = ({ job, onDismiss }) => {
  const stage = job.status?.stage?.toUpperCase() ?? "";
  const present: Presentation = stagePresentation[stage] ?? {
    label: stage || "Unknown",
    color: "text-muted-foreground",
    Icon: HelpCircle,
  };
  const Icon = present.Icon;
  const title = job.docker_image ?? job.space_id ?? job.id;

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => window.open(job.url, "_blank", "noopener,noreferrer")}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          window.open(job.url, "_blank", "noopener,noreferrer");
        }
      }}
      title={job.status?.message ?? title}
      className={cn(
        ROW,
        "cursor-pointer hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
      )}
    >
      <div className={TITLE_LINE}>
        <Icon
          className={cn(
            "h-3 w-3 shrink-0",
            present.color,
            present.spin && "animate-spin",
          )}
        />
        <span className={NAME}>{job.owner ?? title}</span>
        <ExternalLink className="h-3 w-3 shrink-0 text-muted-foreground" />
        {/* Hub-only jobs can't be deleted on the Hub — "remove" is the
            persisted backend-side dismissal, same call the old card made. */}
        {!isHubJobActive(job) ? (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              if (
                window.confirm(
                  "Remove this job from the list? The job record on Hugging Face is unaffected.",
                )
              )
                onDismiss(job.id);
            }}
            aria-label="Remove job from list"
            title="Remove from list"
            className="flex h-5 w-5 shrink-0 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-muted hover:text-destructive"
          >
            <Trash2 className="h-3 w-3" />
          </button>
        ) : null}
      </div>
      <p className={cn(META, "text-muted-foreground")}>
        <span className={cn("font-medium", present.color)}>{present.label}</span>
        {" · "}
        {job.flavor ?? "—"}
        {" · "}
        {relativeTime(hubTime(job))}
      </p>
    </div>
  );
};

interface JobsHistoryProps {
  /** Fired when a row opens the job monitor, which slides the studio up: the
   * library sheet hosting this tab has to close or the panel comes up behind
   * it (same contract as DatasetDetailDialog's onStudioAction). */
  onStudioAction?: () => void;
}

/**
 * Run history — the "Runs" tab of the "My library" drawer. A training job is an
 * *event*, not an artifact; the artifact (a finished model, with Run / Resume /
 * Fine-tune / Download) is the ModelCard in the studio's Train panel. So runs
 * are listed as two-line rows, not cards: running ones pinned on top with live
 * progress, the five most recent below, the rest behind "Show all". The tab is
 * this list's disclosure, so there is no fold header of its own.
 *
 * Two independently collapsible sections, split on the axis the data actually
 * has: whether THIS install has a record of the run. "Launched from MakerLab"
 * is every run with a local JobRecord — local runs *and* cloud runs started
 * from here, since having launched it is what gives the row a monitor, steps,
 * and Resume. Where the GPU actually was is orthogonal and lives in the Where
 * column (`local` / the cloud flavor); naming the section after the machine
 * would be wrong the moment every tracked run is a cloud run. "Other Hub jobs"
 * is the Hub jobs nothing here tracks — they carry none of that metadata, so
 * each section renders its own column set.
 */
const JobsHistory: React.FC<JobsHistoryProps> = ({ onStudioAction }) => {
  const {
    localJobs,
    trackedCloudJobs,
    untrackedHubJobs,
    hubAuthenticated,
    hubJobsPermission,
    error,
    hubError,
    refresh,
    dismissHub,
  } = useJobsData();
  const { openJobMonitor } = useStudio();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const navigate = useNavigate();

  const [search, setSearch] = useState("");
  // "Launched from MakerLab" holds nearly every run, so it opens by default; the
  // orphan Hub bucket is the rare one and folds away, matching how the studio
  // hides untracked items elsewhere.
  const [localOpen, setLocalOpen] = useState(true);
  const [hubOpen, setHubOpen] = useState(false);
  const [localExpanded, setLocalExpanded] = useState(false);
  const [hubExpanded, setHubExpanded] = useState(false);
  const [resumingId, setResumingId] = useState<string | null>(null);

  const query = search.trim().toLowerCase();
  const matchesQuery = useCallback(
    (text: string | null | undefined) =>
      !query || (text ?? "").toLowerCase().includes(query),
    [query],
  );

  // Every run this machine has a record of — local trainings and the cloud
  // runs MakerLab launched (those have a local record, so they are *not*
  // Hub-only), newest-first.
  const localRows = useMemo(
    () =>
      [...localJobs, ...trackedCloudJobs]
        .filter(
          (j) =>
            matchesQuery(j.name) ||
            matchesQuery(j.display_name) ||
            matchesQuery(j.config?.dataset_repo_id),
        )
        .sort((a, b) => jobTime(b) - jobTime(a)),
    [localJobs, trackedCloudJobs, matchesQuery],
  );
  const hubRows = useMemo(
    () =>
      untrackedHubJobs
        .filter(
          (h) =>
            matchesQuery(h.docker_image) ||
            matchesQuery(h.space_id) ||
            matchesQuery(h.owner) ||
            matchesQuery(h.id),
        )
        .sort((a, b) => hubTime(b) - hubTime(a)),
    [untrackedHubJobs, matchesQuery],
  );

  // Running rows are pinned above the cap; only the finished ones collapse.
  const localRunning = useMemo(
    () => localRows.filter((j) => j.state === "running"),
    [localRows],
  );
  const localRest = useMemo(
    () => localRows.filter((j) => j.state !== "running"),
    [localRows],
  );
  const hubRunning = useMemo(() => hubRows.filter(isHubJobActive), [hubRows]);
  const hubRest = useMemo(
    () => hubRows.filter((h) => !isHubJobActive(h)),
    [hubRows],
  );

  // A section that gains a running job opens itself — otherwise a cloud job
  // starting would go live inside the closed "Other Hub jobs" fold. Deps
  // are the counts, so this fires when a run appears (including on first
  // mount) and not when the user deliberately folds the section afterwards;
  // the header's running tally keeps the run visible in that case.
  const localRunningCount = localRunning.length;
  const hubRunningCount = hubRunning.length;
  useEffect(() => {
    if (localRunningCount > 0) setLocalOpen(true);
  }, [localRunningCount]);
  useEffect(() => {
    if (hubRunningCount > 0) setHubOpen(true);
  }, [hubRunningCount]);

  // The monitor is a dialog over the studio's Train panel, so opening one
  // slides the studio up underneath this sheet — let the host close itself.
  const openRun = useCallback(
    (id: string) => {
      onStudioAction?.();
      openJobMonitor(id);
    },
    [onStudioAction, openJobMonitor],
  );

  // Resume a run that died early: resolve its latest checkpoint, then hand the
  // Training page the same resume state a model/job card would.
  const handleResume = useCallback(
    async (job: JobRecord) => {
      setResumingId(job.id);
      try {
        const cks = await listJobCheckpoints(baseUrl, fetchWithHeaders, job.id);
        const step = cks.length > 0 ? cks[cks.length - 1].step : null;
        if (step == null) {
          toast({
            title: "Nothing to resume from",
            description: "This run saved no checkpoint.",
            variant: "destructive",
          });
          return;
        }
        const target = job.config?.steps ?? 0;
        if (target > 0 && step >= target) {
          toast({
            title: "Nothing to resume",
            description: "This run already reached its step target.",
          });
          return;
        }
        const runner = job.runner === "hf_cloud" ? "hf_cloud" : "local";
        navigate("/training", {
          state: {
            resume: {
              jobId: job.id,
              step,
              name: jobDisplayName(job),
              datasetRepoId: job.config.dataset_repo_id,
              policyType: job.config.policy_type,
              sourceSteps: job.config.steps,
              logFreq: job.config.log_freq,
              saveFreq: job.config.save_freq,
              runner,
              flavor:
                runner === "hf_cloud" ? (job.hf_flavor ?? undefined) : undefined,
            },
          },
        });
      } catch (e) {
        toast({
          title: "Couldn't load checkpoints",
          description: e instanceof Error ? e.message : String(e),
          variant: "destructive",
        });
      } finally {
        setResumingId(null);
      }
    },
    [baseUrl, fetchWithHeaders, navigate, toast],
  );

  const emptyMessage = query
    ? "No runs match your search."
    : "No training runs yet.";

  const localShown = localExpanded
    ? localRest
    : localRest.slice(0, HISTORY_CAP);
  const hubShown = hubExpanded ? hubRest : hubRest.slice(0, HISTORY_CAP);
  // A section with nothing in it (no records, or none the search matched)
  // drops out rather than showing an empty fold; if both drop out, one
  // message stands in for the whole history.
  const nothingToShow = localRows.length === 0 && hubRows.length === 0;

  return (
    <div className="space-y-3">
      {/* Search only — the sections carry the local/Hub split the filter
          pills used to. Refresh sits beside it now that the fold header
          that used to hold it is gone. */}
      <div className="flex items-center gap-2">
        <div className="min-w-0 flex-1">
          <LibraryToolbar
            query={search}
            onQueryChange={setSearch}
            searchPlaceholder="Search runs"
          />
        </div>
        <button
          type="button"
          onClick={refresh}
          aria-label="Refresh run history"
          title="Refresh run history"
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded text-muted-foreground hover:text-foreground"
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </button>
      </div>

      {error ? (
        <p className="text-sm text-destructive">
          Couldn't load local runs: {error}
        </p>
      ) : null}
      {hubError ? (
        <p className="text-sm text-destructive">
          Couldn't load cloud jobs: {hubError}
        </p>
      ) : null}
      {!hubError && !hubAuthenticated && trackedCloudJobs.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          Sign in with Hugging Face to see your cloud jobs.
        </p>
      ) : null}
      {hubAuthenticated && !hubJobsPermission ? (
        <p className="text-sm text-warn">
          Your Hugging Face token is missing the{" "}
          <code className="text-warn">job.read</code> permission, so cloud
          jobs can't be listed.
        </p>
      ) : null}

      {nothingToShow ? (
        <p className="flex items-center justify-center py-6 text-sm text-muted-foreground">
          {emptyMessage}
        </p>
      ) : (
        <div className="space-y-3">
          {localRows.length > 0 ? (
            <Collapsible
              open={localOpen}
              onOpenChange={setLocalOpen}
              className="space-y-1"
            >
              <LibrarySectionHeader
                title="Launched from MakerLab"
                count={localRows.length}
                open={localOpen}
                running={localRunning.length}
              />
              <CollapsibleContent className={cn(SLIDE, "space-y-1")}>
                {[...localRunning, ...localShown].map((job) => (
                  <LocalRow
                    key={job.id}
                    job={job}
                    onOpen={openRun}
                    onResume={handleResume}
                    resuming={resumingId === job.id}
                  />
                ))}
                {localRest.length > HISTORY_CAP ? (
                  <ShowAllToggle
                    expanded={localExpanded}
                    total={localRest.length}
                    onToggle={() => setLocalExpanded(!localExpanded)}
                  />
                ) : null}
              </CollapsibleContent>
            </Collapsible>
          ) : null}

          {hubRows.length > 0 ? (
            <Collapsible
              open={hubOpen}
              onOpenChange={setHubOpen}
              className="space-y-1"
            >
              <LibrarySectionHeader
                title="Other Hub jobs"
                count={hubRows.length}
                open={hubOpen}
                running={hubRunning.length}
              />
              <CollapsibleContent className={cn(SLIDE, "space-y-1")}>
                {[...hubRunning, ...hubShown].map((job) => (
                  <HubRow key={job.id} job={job} onDismiss={dismissHub} />
                ))}
                {hubRest.length > HISTORY_CAP ? (
                  <ShowAllToggle
                    expanded={hubExpanded}
                    total={hubRest.length}
                    onToggle={() => setHubExpanded(!hubExpanded)}
                  />
                ) : null}
              </CollapsibleContent>
            </Collapsible>
          ) : null}
        </div>
      )}
    </div>
  );
};

export default JobsHistory;
