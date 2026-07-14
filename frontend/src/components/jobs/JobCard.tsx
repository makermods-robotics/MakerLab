import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { JobRecord, jobDisplayName, renameJob } from "@/lib/jobsApi";
import {
  Square,
  Trash2,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  XCircle,
  ExternalLink,
  Pencil,
  Play,
  FastForward,
  Download,
  Sparkles,
  Upload,
} from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { JobCheckpoint, listJobCheckpoints } from "@/lib/checkpointsApi";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import PolicyExtraDialog from "@/components/training/PolicyExtraDialog";

interface Props {
  job: JobRecord;
  onStop: (id: string) => void;
  onDelete: (id: string) => void;
  onPlay: (job: JobRecord, step: number) => void;
  // Called after a successful rename so the parent can refetch the list.
  onRenamed?: () => void;
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
  onRenamed,
  ancestors = [],
}) => {
  const navigate = useNavigate();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const present = statePresentation[job.state];
  const Icon = present.Icon;
  const isRunning = job.state === "running";
  const isImported = job.runner === "imported";
  // A Hub-backed import (vs a local-folder import) — provenance stays visible
  // after an untracked Hub repo is unified into a tracked imported card.
  const isHubImport = isImported && !!job.hf_repo_id;
  // Alias-aware display name; the true identity (run id / hub repo id) stays
  // visible as muted subtext when an alias is set.
  const displayName = jobDisplayName(job);
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

  // Rename dialog (mirrors CalibrationLibrary's rename UI). Sets a display
  // alias only — the run id / output dir / hub repo id never change.
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const [renameError, setRenameError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);

  const openRename = (e: React.MouseEvent) => {
    e.stopPropagation();
    setRenameValue(displayName);
    setRenameError(null);
    setRenameOpen(true);
  };

  const doRename = async () => {
    const next = renameValue.trim();
    if (!next) {
      setRenameError("Name cannot be empty.");
      return;
    }
    if (next === displayName) {
      setRenameOpen(false);
      return;
    }
    setRenaming(true);
    setRenameError(null);
    try {
      await renameJob(baseUrl, fetchWithHeaders, job.id, next);
      toast({
        title: "Model renamed",
        description: `"${displayName}" → "${next}".`,
      });
      setRenameOpen(false);
      onRenamed?.();
    } catch (e) {
      // 400/404 keep the dialog open with the message for a retry.
      setRenameError(e instanceof Error ? e.message : String(e));
    } finally {
      setRenaming(false);
    }
  };

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
    } else if (job.runner === "hf_cloud") {
      // Cloud runs live on the Hub: deleting the record only removes it (and
      // its local logs) from this list — uploaded model repos are untouched.
      if (
        window.confirm(
          "Remove this cloud run from the list? Model repos on the Hub are not deleted.",
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

  // Continue (local resume) needs a saved checkpoint with optimizer/step state
  // on this machine — i.e. a finished local training run.
  const canContinue =
    selectedJob.runner === "local" &&
    !isRunning &&
    lineageCheckpoints.length > 0 &&
    selectedStep != null;

  // Resume (cloud): an HF Job is immutable once ended, so this launches a NEW
  // cloud job that continues from the parent's Hub checkpoint (restoring
  // optimizer + step, unlike Fine-tune). Offered only on a cloud run that ended
  // BEFORE its step target — a failed/interrupted/cancelled run with a saved
  // checkpoint. A `done` run reached its target, so there's nothing to resume.
  const endedBeforeTarget =
    (selectedJob.state === "failed" || selectedJob.state === "interrupted") &&
    (selectedJob.config.steps === 0 ||
      selectedStep == null ||
      selectedStep < selectedJob.config.steps);
  const canResumeCloud =
    selectedJob.runner === "hf_cloud" &&
    !isRunning &&
    lineageCheckpoints.length > 0 &&
    selectedStep != null &&
    endedBeforeTarget;

  const goToResume = (runner: "local" | "hf_cloud") => {
    if (selectedStep == null) return;
    navigate("/training", {
      state: {
        resume: {
          jobId: selectedJob.id,
          step: selectedStep,
          name: jobDisplayName(selectedJob),
          datasetRepoId: selectedJob.config.dataset_repo_id,
          policyType: selectedJob.config.policy_type,
          sourceSteps: selectedJob.config.steps,
          logFreq: selectedJob.config.log_freq,
          saveFreq: selectedJob.config.save_freq,
          runner,
          flavor: runner === "hf_cloud" ? (selectedJob.hf_flavor ?? undefined) : undefined,
        },
      },
    });
  };

  const handleContinue = (e: React.MouseEvent) => {
    e.stopPropagation();
    goToResume("local");
  };

  const handleResumeCloud = (e: React.MouseEvent) => {
    e.stopPropagation();
    goToResume("hf_cloud");
  };

  // Fine-tune: start a FRESH run whose weights are initialized from this
  // (imported) model's checkpoint. Unlike Continue (which needs optimizer/step
  // state and is local-only), fine-tuning works from weights-only imports —
  // which is exactly what imported models are. Gate on an imported source with
  // a selectable checkpoint.
  const canFinetune =
    selectedJob.runner === "imported" &&
    !isRunning &&
    lineageCheckpoints.length > 0 &&
    selectedStep != null;

  const handleFinetune = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedStep == null) return;
    navigate("/training", {
      state: {
        finetune: {
          jobId: selectedJob.id,
          step: selectedStep,
          name: jobDisplayName(selectedJob),
          policyType: selectedJob.config.policy_type,
        },
      },
    });
  };

  // A local checkpoint can be exported as a zip while training continues, so
  // (unlike Continue) this doesn't gate on !isRunning.
  const canDownload =
    selectedJob.runner === "local" &&
    lineageCheckpoints.length > 0 &&
    selectedStep != null;

  const handleDownload = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedStep == null) return;
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/jobs/${selectedJob.id}/checkpoints/${selectedStep}/download`,
      );
      if (!res.ok) {
        toast({ title: "Download failed", variant: "destructive" });
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${jobDisplayName(selectedJob)}_step_${selectedStep}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      toast({
        title: "Download failed",
        description: String(err),
        variant: "destructive",
      });
    }
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
          <div className="flex items-center gap-2">
            <div
              className={`flex items-center gap-1.5 text-xs font-semibold ${present.color}`}
            >
              <Icon
                className={`w-3.5 h-3.5 ${isRunning ? "animate-spin" : ""}`}
              />
              {stateLabel}
            </div>
            {isHubImport ? (
              <div
                className="flex items-center gap-1 text-[11px] font-medium text-sky-400"
                title="Imported from a Hugging Face Hub repo"
              >
                <Upload className="w-3 h-3" />
                from Hub
              </div>
            ) : null}
          </div>
          <div className="flex items-center gap-0.5">
            <Button
              variant="ghost"
              size="icon"
              onClick={openRename}
              className="h-7 w-7 text-slate-400 hover:text-white"
              aria-label="Rename model"
              title="Rename"
            >
              <Pencil className="w-3.5 h-3.5" />
            </Button>
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
            ) : null}
            {/* A running cloud run is steered from its Hub page (the link
                above), so it gets no local action button. Everything else —
                including a FINISHED cloud run — gets stop/delete, so dead
                cloud runs are removable instead of link-only. */}
            {!(job.runner === "hf_cloud" && job.hf_job_url && isRunning) ? (
              <Button
                variant="ghost"
                size="icon"
                onClick={handleAction}
                className={`h-7 w-7 text-slate-400 ${
                  isRunning ? "hover:text-white" : "hover:text-red-400"
                }`}
                aria-label={isRunning ? "Stop job" : "Delete job"}
              >
                {isRunning ? (
                  <Square className="w-3.5 h-3.5" />
                ) : (
                  <Trash2 className="w-3.5 h-3.5" />
                )}
              </Button>
            ) : null}
          </div>
        </div>
        <div>
          <div
            className="text-white font-semibold truncate"
            title={displayName}
          >
            {displayName}
          </div>
          {/* When aliased, keep the true identity visible: the run id for
              trainings (imported models already show their repo id / path in
              the subtitle below). */}
          {!isImported && job.display_name ? (
            <div className="text-[11px] text-slate-500 truncate" title={job.id}>
              {job.id}
            </div>
          ) : null}
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
          <div className="flex flex-wrap items-center gap-2">
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
                className="h-8 gap-1 border-sky-500/50 text-sky-700 dark:text-sky-300 hover:bg-sky-500/10"
                aria-label="Continue training from this checkpoint"
              >
                <FastForward className="w-3.5 h-3.5" /> Continue
              </Button>
            ) : null}
            {canResumeCloud ? (
              <Button
                size="sm"
                variant="outline"
                onClick={handleResumeCloud}
                className="h-8 gap-1 border-sky-500/50 text-sky-700 dark:text-sky-300 hover:bg-sky-500/10"
                aria-label="Resume this cloud run from its last checkpoint"
                title="Resume: launch a new cloud job continuing from this checkpoint"
              >
                <FastForward className="w-3.5 h-3.5" /> Resume
              </Button>
            ) : null}
            {canFinetune ? (
              <Button
                size="sm"
                variant="outline"
                onClick={handleFinetune}
                className="h-8 gap-1 border-violet-500/50 text-violet-700 dark:text-violet-300 hover:bg-violet-500/10"
                aria-label="Fine-tune a new run from this model's weights"
                title="Fine-tune a new run from this model's weights"
              >
                <Sparkles className="w-3.5 h-3.5" /> Fine-tune
              </Button>
            ) : null}
            {canDownload ? (
              <Button
                size="sm"
                variant="outline"
                onClick={handleDownload}
                className="h-8 gap-1 border-slate-500/50 text-slate-700 dark:text-slate-300 hover:bg-slate-500/10"
                aria-label="Download this checkpoint"
                title="Download this checkpoint"
              >
                <Download className="w-3.5 h-3.5" /> Download
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
            className="h-8 gap-1.5 border-amber-500/50 text-amber-700 dark:text-amber-300 hover:bg-amber-500/10"
          >
            <Download className="w-3.5 h-3.5" /> Install{" "}
            {missingExtra.installTarget}
          </Button>
        ) : null}
      </CardContent>
      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent
          className="bg-slate-900 border-slate-800 text-white"
          onClick={(e) => e.stopPropagation()}
        >
          <DialogHeader>
            <DialogTitle>Rename model</DialogTitle>
            <DialogDescription className="text-slate-400">
              Sets a display name only — the underlying{" "}
              {isImported && job.hf_repo_id ? "Hub repo" : "run"} (
              <span className="font-mono text-slate-300">
                {isImported ? importedSource : job.id}
              </span>
              ) is not moved or changed.
            </DialogDescription>
          </DialogHeader>
          <Input
            value={renameValue}
            onChange={(e) => {
              setRenameValue(e.target.value);
              setRenameError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void doRename();
              }
            }}
            autoFocus
            placeholder="New name"
            className="bg-slate-800 border-slate-700 text-white"
          />
          {renameError && <p className="text-sm text-red-400">{renameError}</p>}
          <DialogFooter className="flex gap-2 justify-end">
            <Button
              variant="outline"
              className="border-slate-600 text-slate-700 dark:text-slate-300"
              onClick={() => setRenameOpen(false)}
            >
              Cancel
            </Button>
            <Button
              className="bg-blue-600 hover:bg-blue-700 text-white"
              disabled={
                renaming ||
                !renameValue.trim() ||
                renameValue.trim() === displayName
              }
              onClick={doRename}
            >
              {renaming ? "Renaming…" : "Rename"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
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
