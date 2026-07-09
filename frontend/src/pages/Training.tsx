import React, { useEffect, useRef, useState } from "react";
import { Navigate, useNavigate, useParams } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";

import { TrainingStatus, LogEntry } from "@/components/training/types";
import MonitoringStats from "@/components/training/monitoring/MonitoringStats";
import TrainingLogs from "@/components/training/monitoring/TrainingLogs";

import { AppShell } from "@/components/shell/AppShell";
import { StatusPill, type SessionPhase } from "@/components/ui/status-pill";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Loader2, Play, Square, Trash2 } from "lucide-react";

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
import InferenceModal from "@/components/landing/InferenceModal";
import { useRobots } from "@/hooks/useRobots";

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

const MonitoringMode: React.FC<{ jobId: string }> = ({ jobId }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const navigate = useNavigate();

  const { selectedRecord } = useRobots();
  const [job, setJob] = useState<JobRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [inferenceModalOpen, setInferenceModalOpen] = useState(false);

  // Seed logs from the persistent on-disk file once on mount, so navigating
  // away and back (or coming in fresh on a finished/interrupted job) shows
  // the full log history. Polling /logs continues from this point — the
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
      navigate("/");
    } catch (e) {
      toast({
        title: "Delete failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  if (error && !job) {
    return (
      <AppShell back={{ to: "/", label: "jobs" }}>
        <p className="text-destructive">
          Couldn't load job {jobId}: {error}
        </p>
      </AppShell>
    );
  }

  if (!job) {
    return (
      <AppShell back={{ to: "/", label: "jobs" }}>
        <div className="flex items-center justify-center py-24 text-muted-foreground">
          <Loader2 className="mr-3 h-6 w-6 animate-spin" /> Loading job…
        </div>
      </AppShell>
    );
  }

  const isRunning = job.state === "running";
  const statusPhase: SessionPhase = isRunning
    ? "running"
    : job.state === "done"
      ? "idle"
      : "setup";

  return (
    <AppShell
      back={{ to: "/", label: "jobs" }}
      status={<StatusPill phase={statusPhase} label={job.state} />}
      actions={
        isRunning ? (
          <Button onClick={handleStop} variant="destructive" size="sm">
            <Square className="mr-2 h-4 w-4" /> Stop
          </Button>
        ) : (
          <Button onClick={handleDelete} variant="ghost" size="sm">
            <Trash2 className="mr-2 h-4 w-4" /> Delete
          </Button>
        )
      }
    >
      <div className="space-y-6">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="font-mono text-xl font-bold text-foreground">
              {jobDisplayName(job)}
            </h1>
            {job.runner === "hf_cloud" ? (
              <Badge variant="warn">HF · {job.hf_flavor ?? "cloud"}</Badge>
            ) : (
              <Badge variant="outline">Local</Badge>
            )}
            {job.runner === "hf_cloud" &&
              job.hf_repo_id &&
              job.state === "done" && (
                <a
                  href={`https://huggingface.co/${job.hf_repo_id}`}
                  target="_blank"
                  rel="noreferrer"
                  className="font-mono text-xs text-info underline underline-offset-2"
                >
                  View on Hub ↗
                </a>
              )}
            {job.wandb_run_url && (
              <a
                href={job.wandb_run_url}
                target="_blank"
                rel="noreferrer"
                className="font-mono text-xs text-info underline underline-offset-2"
              >
                View on W&B ↗
              </a>
            )}
          </div>
          {/* When aliased, keep the immutable run id visible as subtext. */}
          {job.display_name ? (
            <p className="mt-1 font-mono text-[11px] text-muted-foreground">
              {job.id}
            </p>
          ) : null}
          <p className="mt-1 font-mono text-xs text-muted-foreground">
            {job.state}
            {job.error_message ? ` — ${job.error_message}` : ""}
          </p>
        </div>

        <MonitoringStats
          jobId={jobId}
          trainingStatus={jobToStatus(job, false)}
          getProgressPercentage={getProgressPercentage}
          formatTime={formatTime}
        />
        <Card variant="flat" className="flex items-center gap-3 p-4">
          <span className="font-display text-sm font-semibold text-foreground">
            Run inference
          </span>
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
                onClick={() => setInferenceModalOpen(true)}
                disabled={selectedStep == null}
              >
                <Play className="mr-2 h-4 w-4" />
                Run on robot
              </Button>
            </>
          )}
        </Card>
        <InferenceModal
          open={inferenceModalOpen}
          onOpenChange={setInferenceModalOpen}
          robot={selectedRecord}
          jobId={jobId}
          initialStep={selectedStep}
        />
        <TrainingLogs logs={logs} logContainerRef={logContainerRef} />
      </div>
    </AppShell>
  );
};

const Training: React.FC = () => {
  const { jobId } = useParams<{ jobId?: string }>();
  // /training without a job id is the Train & Deploy stage page (TrainDeploy);
  // this component only ever monitors a specific job.
  return jobId ? <MonitoringMode jobId={jobId} /> : <Navigate to="/training" replace />;
};

export default Training;
