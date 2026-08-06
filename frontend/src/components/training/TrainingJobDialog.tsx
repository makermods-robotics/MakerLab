import React, { useEffect, useRef, useState } from "react";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";

import { TrainingStatus, LogEntry } from "@/components/training/types";
import MonitoringStats from "@/components/training/monitoring/MonitoringStats";
import TrainingLogs from "@/components/training/monitoring/TrainingLogs";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Loader2,
  Play,
  Square,
  Trash2,
  ArrowLeft,
  Upload as UploadIcon,
} from "lucide-react";

import {
  JobRecord,
  getJob,
  getJobLogs,
  getJobLogFile,
  jobDisplayName,
  stopJob,
  deleteJob,
} from "@/lib/jobsApi";
import { JobCheckpoint, listJobCheckpoints } from "@/lib/checkpointsApi";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import { useStudio } from "@/contexts/StudioContext";
import { uploadModel } from "@/lib/modelsApi";
import { ApiError } from "@/lib/apiClient";
import { useCanUpload } from "@/hooks/useCanUpload";
import { useHfAuth } from "@/contexts/HfAuthContext";

const POLL_INTERVAL_MS = 1000;
const MAX_LOG_LINES = 5000;

function jobToStatus(
  job: JobRecord | null,
  isStarting: boolean,
): TrainingStatus {
  // Adapter so MonitoringStats can keep its current prop shape.
  if (!job) {
    return {
      training_active: isStarting,
      current_step: 0,
      total_steps: 0,
      available_controls: {
        stop_training: false,
        pause_training: false,
        resume_training: false,
      },
    };
  }
  return {
    training_active: job.state === "running",
    current_step: job.metrics.current_step,
    total_steps: job.metrics.total_steps,
    current_loss: job.metrics.current_loss ?? undefined,
    current_lr: job.metrics.current_lr ?? undefined,
    grad_norm: job.metrics.grad_norm ?? undefined,
    eta_seconds: job.metrics.eta_seconds ?? undefined,
    available_controls: {
      stop_training: job.state === "running",
      pause_training: false,
      resume_training: false,
    },
  };
}

/**
 * The training-job monitor as a modal dialog over the studio's Train panel —
 * replaces the old /training/:jobId page (the polling/monitor logic is ported
 * verbatim; every navigate-home became `onExit`). Unlike the recording
 * session, dismissing is always safe — training keeps running in the
 * background — so ESC / outside click / the back button all exit to the
 * studio.
 */
const TrainingJobDialog: React.FC<{
  jobId: string;
  /** Called for every exit — closes the dialog, landing back in the studio. */
  onExit: () => void;
}> = ({ jobId, onExit }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

  const { openStudio } = useStudio();
  const canUpload = useCanUpload();
  const { auth } = useHfAuth();
  const username = auth.status === "authenticated" ? auth.username : null;
  const [job, setJob] = useState<JobRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [repoId, setRepoId] = useState("");
  const [uploading, setUploading] = useState(false);
  const [publishOpen, setPublishOpen] = useState(false);

  // Seed logs from the persistent on-disk file once on mount, so closing and
  // reopening the dialog (or coming in fresh on a finished/interrupted job)
  // shows the full log history. Polling /logs continues from this point — the
  // backend drains the live queue in the same /log-file call so we don't
  // double-display lines that were buffered when we landed.
  useEffect(() => {
    let cancelled = false;
    getJobLogFile(baseUrl, fetchWithHeaders, jobId)
      .then((seeded) => {
        if (!cancelled && seeded.length > 0) setLogs(seeded);
      })
      .catch(() => {
        // 404 or transient — fall through; live polling will fill in.
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders, jobId]);

  // Read latest job state from a ref so the polling intervals below stay
  // stable instead of tearing down/rebuilding on every state transition.
  const jobStateRef = useRef(job?.state);
  jobStateRef.current = job?.state;

  // Poll checkpoints — every 5s while the job is running.
  useEffect(() => {
    let cancelled = false;
    const tick = () => {
      listJobCheckpoints(baseUrl, fetchWithHeaders, jobId)
        .then((cks) => {
          if (cancelled) return;
          setCheckpoints(cks);
          if (cks.length > 0) {
            const latest = cks[cks.length - 1].step;
            setSelectedStep((prev) =>
              prev != null && cks.some((c) => c.step === prev) ? prev : latest,
            );
          } else {
            setSelectedStep(null);
          }
        })
        .catch(() => {
          if (!cancelled) {
            setCheckpoints([]);
            setSelectedStep(null);
          }
        });
    };
    tick();
    const id = setInterval(() => {
      if (cancelled) return;
      if (jobStateRef.current && jobStateRef.current !== "running") return;
      tick();
    }, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [baseUrl, fetchWithHeaders, jobId]);

  // Poll the job + its logs while running. Caps log lines to avoid unbounded growth.
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const next = await getJob(baseUrl, fetchWithHeaders, jobId);
        if (cancelled) return;
        setJob(next);
        if (next.state === "running") {
          const newLogs = await getJobLogs(baseUrl, fetchWithHeaders, jobId);
          if (!cancelled && newLogs.length > 0) {
            setLogs((prev) => {
              const merged = [...prev, ...newLogs];
              return merged.length > MAX_LOG_LINES
                ? merged.slice(merged.length - MAX_LOG_LINES)
                : merged;
            });
          }
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    };
    tick();
    const id = setInterval(() => {
      if (cancelled) return;
      if (jobStateRef.current && jobStateRef.current !== "running") return;
      tick();
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [baseUrl, fetchWithHeaders, jobId]);

  // Auto-scroll the log panel as new lines arrive.
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, "0")}:${minutes
      .toString()
      .padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const getProgressPercentage = () => {
    if (!job || job.metrics.total_steps === 0) return 0;
    return (job.metrics.current_step / job.metrics.total_steps) * 100;
  };

  const handleStop = async () => {
    if (!job) return;
    if (!window.confirm("Stop this run?")) return;
    try {
      const next = await stopJob(baseUrl, fetchWithHeaders, job.id);
      setJob(next);
      toast({ title: "Stopping…" });
    } catch (e) {
      toast({
        title: "Stop failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handleDelete = async () => {
    if (!job) return;
    if (!window.confirm("Delete this run? This wipes the output directory."))
      return;
    try {
      await deleteJob(baseUrl, fetchWithHeaders, job.id);
      toast({ title: "Job removed" });
      onExit();
    } catch (e) {
      toast({
        title: "Delete failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handleUpload = async () => {
    if (!job) return;
    setUploading(true);
    try {
      const res = await uploadModel(
        baseUrl,
        fetchWithHeaders,
        job.id,
        repoId.trim() || undefined,
      );
      setPublishOpen(false);
      toast({
        title: "Uploaded to Hub",
        description: (
          <span>
            {res.repo_id} is now on the Hub.{" "}
            <a
              href={res.url}
              target="_blank"
              rel="noopener noreferrer"
              className="underline font-medium"
            >
              View model
            </a>
          </span>
        ),
      });
      try {
        const next = await getJob(baseUrl, fetchWithHeaders, job.id);
        setJob(next);
      } catch {
        // The upload itself already succeeded and the user already has
        // their success toast + link — a refetch blip here shouldn't look
        // like a failed upload. The row may linger until the next poll
        // tick or a manual reopen picks up the real (now-uploaded) state.
      }
    } catch (e) {
      toast({
        title: "Upload failed",
        description:
          e instanceof ApiError && e.detail
            ? e.detail
            : e instanceof Error
              ? e.message
              : String(e),
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };

  const isRunning = job?.state === "running";

  const backButton = (
    <Button
      variant="ghost"
      size="sm"
      onClick={onExit}
      className="-ml-2 shrink-0 text-muted-foreground hover:text-foreground"
    >
      <ArrowLeft className="mr-1.5 h-4 w-4" /> Skill studio
    </Button>
  );

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onExit();
      }}
    >
      <DialogContent
        hideClose
        className="max-h-[92vh] w-[min(96vw,72rem)] max-w-none gap-0 overflow-y-auto p-6"
        aria-describedby={undefined}
      >
        <DialogTitle className="sr-only">Training job status</DialogTitle>

        {error && !job ? (
          <div className="space-y-4">
            {backButton}
            <p className="text-sm text-destructive">
              Couldn't load job {jobId}: {error}
            </p>
          </div>
        ) : !job ? (
          <div className="space-y-4">
            {backButton}
            <div className="flex items-center justify-center py-20 text-muted-foreground">
              <Loader2 className="mr-3 h-6 w-6 animate-spin" /> Loading job…
            </div>
          </div>
        ) : (
          <div className="space-y-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="flex min-w-0 items-start gap-2">
                {backButton}
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <h2 className="truncate text-base">
                      {jobDisplayName(job)}
                    </h2>
                    {job.runner === "hf_cloud" ? (
                      <span className="rounded border border-border bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                        HF · {job.hf_flavor ?? "cloud"}
                      </span>
                    ) : (
                      <span className="rounded border border-border bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                        Local
                      </span>
                    )}
                    {job.hf_repo_id && (
                      <a
                        href={`https://huggingface.co/${job.hf_repo_id}`}
                        target="_blank"
                        rel="noreferrer"
                        className="text-xs text-primary hover:underline"
                      >
                        View on Hub ↗
                      </a>
                    )}
                    {job.wandb_run_url && (
                      <a
                        href={job.wandb_run_url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-xs text-primary hover:underline"
                      >
                        View on W&B ↗
                      </a>
                    )}
                  </div>
                  {/* When aliased, keep the immutable run id visible as subtext. */}
                  {job.display_name ? (
                    <p className="font-mono text-[11px] text-muted-foreground">
                      {job.id}
                    </p>
                  ) : null}
                  <p className="text-xs text-muted-foreground">
                    {job.state}
                    {job.error_message ? ` — ${job.error_message}` : ""}
                  </p>
                </div>
              </div>
              {isRunning ? (
                <Button onClick={handleStop} variant="destructive" size="sm">
                  <Square className="mr-2 h-4 w-4" /> Stop
                </Button>
              ) : (
                <Button
                  onClick={handleDelete}
                  variant="ghost"
                  size="sm"
                  className="text-muted-foreground hover:text-foreground"
                >
                  <Trash2 className="mr-2 h-4 w-4" /> Delete
                </Button>
              )}
            </div>

            <MonitoringStats
              jobId={jobId}
              trainingStatus={jobToStatus(job, false)}
              getProgressPercentage={getProgressPercentage}
              formatTime={formatTime}
            />
            <div className="flex items-center gap-3 rounded-md border border-border bg-card p-4">
              <span className="eyebrow">Run inference</span>
              {checkpoints.length === 0 ? (
                <span className="text-xs text-muted-foreground">
                  No checkpoints yet — wait for the first save.
                </span>
              ) : (
                <>
                  <CheckpointDropdown
                    checkpoints={checkpoints}
                    selectedStep={selectedStep}
                    onChange={setSelectedStep}
                  />
                  <Button
                    onClick={() => {
                      // Land on the Deploy panel with this job + checkpoint
                      // prefilled (DeployPanel consumes the prefill) instead
                      // of stacking the legacy InferenceModal over the dialog.
                      openStudio("deploy", {
                        deploy: {
                          source: "job",
                          id: jobId,
                          step: selectedStep ?? undefined,
                        },
                      });
                      onExit();
                    }}
                    disabled={selectedStep == null}
                    size="sm"
                  >
                    <Play className="mr-2 h-4 w-4" />
                    Run on robot
                  </Button>
                </>
              )}
            </div>
            {!isRunning && checkpoints.length > 0 && !job.hf_repo_id && canUpload && (
              <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-card p-4">
                <div className="min-w-0">
                  <span className="eyebrow">Publish to Hub</span>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    Share this run's final checkpoint as a public model on the
                    Hub.
                  </p>
                </div>
                <Popover open={publishOpen} onOpenChange={setPublishOpen}>
                  <PopoverTrigger asChild>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={uploading}
                      className="h-8 shrink-0 gap-1.5 border-teal-500/50 text-teal-700 hover:bg-teal-500/10 dark:text-teal-300"
                    >
                      {uploading ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <UploadIcon className="h-3.5 w-3.5" />
                      )}
                      {uploading ? "Uploading…" : "Upload to Hub"}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent align="end" className="w-72 text-xs">
                    <div className="space-y-3">
                      <div className="space-y-1">
                        <Label
                          htmlFor={`publish-repo-id-${jobId}`}
                          className="font-normal text-muted-foreground"
                        >
                          Repo name (optional)
                        </Label>
                        <Input
                          id={`publish-repo-id-${jobId}`}
                          value={repoId}
                          onChange={(e) => setRepoId(e.target.value)}
                          placeholder={
                            username ? `${username}/${jobId}` : "namespace/repo-name"
                          }
                          className="h-7 text-xs"
                          disabled={uploading}
                        />
                        <p className="leading-snug text-muted-foreground">
                          Leave blank to publish under your namespace as{" "}
                          <span className="font-mono">
                            {username ? `${username}/${jobId}` : jobId}
                          </span>
                          .
                        </p>
                      </div>
                      <Button
                        size="sm"
                        onClick={handleUpload}
                        disabled={uploading}
                        className="h-7 w-full gap-1 text-xs"
                      >
                        {uploading ? (
                          <>
                            <Loader2 className="h-3 w-3 animate-spin" />
                            Uploading…
                          </>
                        ) : (
                          <>
                            <UploadIcon className="h-3 w-3" />
                            Confirm & upload
                          </>
                        )}
                      </Button>
                    </div>
                  </PopoverContent>
                </Popover>
              </div>
            )}
            <TrainingLogs logs={logs} logContainerRef={logContainerRef} />
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default TrainingJobDialog;
