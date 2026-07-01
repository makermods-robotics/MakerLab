import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { JobRecord } from "@/lib/jobsApi";
import {
  Square,
  X,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  XCircle,
  ExternalLink,
  Play,
  FastForward,
  Download,
} from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { JobCheckpoint, listJobCheckpoints } from "@/lib/checkpointsApi";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import PolicyExtraDialog from "@/components/training/PolicyExtraDialog";

interface Props {
  job: JobRecord;
  onStop: (id: string) => void;
  onDelete: (id: string) => void;
  onPlay: (job: JobRecord, step: number) => void;
  // Runs this job was resumed from, nearest-parent first. Rendered nested and
  // hidden from the top-level list so a resumed lineage reads as one entry.
  ancestors?: JobRecord[];
}

function relativeTime(epochSec: number): string {
  const diff = Math.max(0, Date.now() / 1000 - epochSec);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

const statePresentation: Record<
  JobRecord["state"],
  {
    label: string;
    color: string;
    Icon: React.ComponentType<{ className?: string }>;
  }
> = {
  running: { label: "Running", color: "text-green-400", Icon: Loader2 },
  done: { label: "Done", color: "text-slate-400", Icon: CheckCircle2 },
  failed: { label: "Failed", color: "text-red-400", Icon: XCircle },
  interrupted: {
    label: "Interrupted",
    color: "text-amber-400",
    Icon: AlertTriangle,
  },
};

const JobCard: React.FC<Props> = ({
  job,
  onStop,
  onDelete,
  onPlay,
  ancestors = [],
}) => {
  const navigate = useNavigate();
  const { baseUrl, fetchWithHeaders } = useApi();
  const present = statePresentation[job.state];
  const Icon = present.Icon;
  const isRunning = job.state === "running";
  const isImported = job.runner === "imported";
  const importedSource = job.hf_repo_id || job.output_dir;
  const stateLabel = isImported ? "Imported" : present.label;
  const isStarting = isRunning && job.metrics.total_steps === 0;
  const progressPct =
    job.metrics.total_steps > 0
      ? Math.min(
          100,
          (job.metrics.current_step / job.metrics.total_steps) * 100,
        )
      : 0;

  const subtitle = isImported
    ? importedSource
    : isStarting
      ? "starting…"
      : isRunning
        ? `started ${relativeTime(job.started_at)}`
        : job.ended_at != null
          ? `ended ${relativeTime(job.ended_at)}`
          : present.label.toLowerCase();

  // Checkpoints across the resume lineage (this run + the runs it resumed
  // from), each tagged with its owning job so inference/continue route to the
  // right run. Sorted newest-step-first so the current run sits above inherited
  // source checkpoints in the dropdown.
  const [lineageCheckpoints, setLineageCheckpoints] = useState<
    { job: JobRecord; ckpt: JobCheckpoint }[]
  >([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  // Set on a failed run whose policy needs a lerobot extra that's still missing
  // — the likely cause. Offers the same one-click install as the training form.
  const [missingExtra, setMissingExtra] = useState<{
    policyType: string;
    packageName: string;
    installTarget: string;
    installHint: string;
  } | null>(null);
  const [extraDialogOpen, setExtraDialogOpen] = useState(false);

  // Key ancestors by id+count so the frequent list refreshes (which hand us new
  // array refs) don't refetch unless the lineage actually changed.
  const ancestorKey = ancestors
    .map((a) => `${a.id}:${a.checkpoint_count}`)
    .join("|");

  useEffect(() => {
    const lineage = [job, ...ancestors].filter((j) => j.checkpoint_count > 0);
    if (lineage.length === 0) {
      setLineageCheckpoints([]);
      setSelectedStep(null);
      return;
    }
    let cancelled = false;
    Promise.all(
      lineage.map((j) =>
        listJobCheckpoints(baseUrl, fetchWithHeaders, j.id)
          .then((cks) => cks.map((ckpt) => ({ job: j, ckpt })))
          .catch(() => [] as { job: JobRecord; ckpt: JobCheckpoint }[]),
      ),
    ).then((results) => {
      if (cancelled) return;
      const combined = results.flat().sort((a, b) => b.ckpt.step - a.ckpt.step);
      setLineageCheckpoints(combined);
      setSelectedStep((prev) =>
        prev != null && combined.some((c) => c.ckpt.step === prev)
          ? prev
          : (combined[0]?.ckpt.step ?? null),
      );
    });
    return () => {
      cancelled = true;
    };
    // job/ancestors captured via id+count keys above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseUrl, fetchWithHeaders, job.id, job.checkpoint_count, ancestorKey]);

  // A failed local training whose policy needs a lerobot extra that's still not
  // installed almost certainly died on that ImportError — surface the install.
  useEffect(() => {
    const policyType = job.config?.policy_type;
    if (job.state !== "failed" || job.runner !== "local" || !policyType) {
      setMissingExtra(null);
      return;
    }
    let cancelled = false;
    fetchWithHeaders(`${baseUrl}/system/policy-extra/${policyType}`)
      .then((r) => r.json())
      .then(
        (d: {
          policy_type: string;
          needs_extra: boolean;
          available: boolean;
          package: string;
          install_target: string;
          install_hint: string;
        }) => {
          if (cancelled) return;
          setMissingExtra(
            d.needs_extra && !d.available
              ? {
                  policyType: d.policy_type,
                  packageName: d.package,
                  installTarget: d.install_target,
                  installHint: d.install_hint,
                }
              : null,
          );
        },
      )
      .catch(() => {
        if (!cancelled) setMissingExtra(null);
      });
    return () => {
      cancelled = true;
    };
  }, [
    baseUrl,
    fetchWithHeaders,
    job.state,
    job.runner,
    job.config?.policy_type,
  ]);

  const handleAction = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (isRunning) {
      if (window.confirm("Stop this run?")) onStop(job.id);
    } else if (isImported) {
      if (
        window.confirm(
          "Remove this imported model? The source files are left untouched.",
        )
      )
        onDelete(job.id);
    } else if (
      window.confirm("Delete this run? This wipes the output directory.")
    ) {
      onDelete(job.id);
    }
  };

  // The selected checkpoint may belong to this run or an inherited source run;
  // route inference/continue to whichever run owns it.
  const selected =
    lineageCheckpoints.find((c) => c.ckpt.step === selectedStep) ?? null;
  const selectedJob = selected?.job ?? job;
  // Flat list for the dropdown (already newest-first).
  const checkpoints = lineageCheckpoints.map((c) => c.ckpt);

  const handlePlay = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedStep == null) return;
    onPlay(selectedJob, selectedStep);
  };

  // Resume is local-only (lerobot can't resume from the Hub) and needs a saved
  // checkpoint with optimizer/step state — i.e. a finished local training run.
  const canContinue =
    selectedJob.runner === "local" &&
    !isRunning &&
    lineageCheckpoints.length > 0 &&
    selectedStep != null;

  const handleContinue = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedStep == null) return;
    navigate("/training", {
      state: {
        resume: {
          jobId: selectedJob.id,
          step: selectedStep,
          name: selectedJob.name,
          datasetRepoId: selectedJob.config.dataset_repo_id,
          policyType: selectedJob.config.policy_type,
          sourceSteps: selectedJob.config.steps,
        },
      },
    });
  };

  const showProgressBar = isRunning;
  const showInferenceRow =
    lineageCheckpoints.length > 0 && selectedStep != null;

  return (
    <Card
      onClick={() => {
        if (!isImported) navigate(`/training/${job.id}`);
      }}
      className={`bg-slate-800/50 border-slate-700 rounded-xl transition-colors ${
        isImported ? "" : "cursor-pointer hover:border-slate-500"
      }`}
    >
      <CardContent className="p-4 space-y-3">
        <div className="flex items-start justify-between gap-2">
          <div
            className={`flex items-center gap-1.5 text-xs font-semibold ${present.color}`}
          >
            <Icon
              className={`w-3.5 h-3.5 ${isRunning ? "animate-spin" : ""}`}
            />
            {stateLabel}
          </div>
          {job.runner === "hf_cloud" && job.hf_job_url ? (
            <Button
              variant="ghost"
              size="icon"
              asChild
              className="h-7 w-7 text-slate-400 hover:text-white"
              aria-label="Open Hub job page"
            >
              <a
                href={job.hf_job_url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
              >
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </Button>
          ) : (
            <Button
              variant="ghost"
              size="icon"
              onClick={handleAction}
              className="h-7 w-7 text-slate-400 hover:text-white"
              aria-label={isRunning ? "Stop job" : "Delete job"}
            >
              {isRunning ? (
                <Square className="w-3.5 h-3.5" />
              ) : (
                <X className="w-3.5 h-3.5" />
              )}
            </Button>
          )}
        </div>
        <div>
          <div className="text-white font-semibold truncate" title={job.name}>
            {job.name}
          </div>
          {/* Imported subtitles are file paths — truncate the *start* (rtl
              flips the ellipsis to the left) so the more useful tail stays
              visible. The leading LRM keeps the path's first "/" from being
              bidi-reordered to the wrong end. */}
          <div
            className="text-xs text-slate-400 truncate"
            title={subtitle}
            style={
              isImported ? { direction: "rtl", textAlign: "left" } : undefined
            }
          >
            {isImported ? "\u200e" + subtitle : subtitle}
          </div>
        </div>
        {showProgressBar ? (
          <div className="relative h-5 w-full overflow-hidden rounded-md bg-slate-900 border border-slate-700">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-sky-400 transition-[width] duration-500"
              style={{ width: `${progressPct}%` }}
            />
            <div className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-white tabular-nums drop-shadow">
              {isStarting ? "Training starting…" : `${progressPct.toFixed(1)}%`}
            </div>
          </div>
        ) : null}
        {showInferenceRow ? (
          <div className="flex items-center gap-2">
            <CheckpointDropdown
              checkpoints={checkpoints}
              selectedStep={selectedStep}
              onChange={setSelectedStep}
            />
            <Button
              size="icon"
              onClick={handlePlay}
              className="h-8 w-8 bg-green-500 hover:bg-green-600 text-white"
              aria-label="Run inference with this checkpoint"
            >
              <Play className="w-4 h-4" />
            </Button>
            {canContinue ? (
              <Button
                size="sm"
                variant="outline"
                onClick={handleContinue}
                className="h-8 gap-1 border-sky-500/50 text-sky-300 hover:bg-sky-500/10"
                aria-label="Continue training from this checkpoint"
              >
                <FastForward className="w-3.5 h-3.5" /> Continue
              </Button>
            ) : null}
          </div>
        ) : null}
        {missingExtra ? (
          <Button
            size="sm"
            variant="outline"
            onClick={(e) => {
              e.stopPropagation();
              setExtraDialogOpen(true);
            }}
            className="h-8 gap-1.5 border-amber-500/50 text-amber-300 hover:bg-amber-500/10"
          >
            <Download className="w-3.5 h-3.5" /> Install{" "}
            {missingExtra.installTarget}
          </Button>
        ) : null}
      </CardContent>
      {missingExtra ? (
        <PolicyExtraDialog
          open={extraDialogOpen}
          onOpenChange={setExtraDialogOpen}
          policyType={missingExtra.policyType}
          packageName={missingExtra.packageName}
          installTarget={missingExtra.installTarget}
          installHint={missingExtra.installHint}
        />
      ) : null}
    </Card>
  );
};

export default JobCard;
