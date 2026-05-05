import React, { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useHfAuth } from "@/contexts/HfAuthContext";

import { TrainingConfig, TrainingStatus, LogEntry } from "@/components/training/types";
import TrainingHeader from "@/components/training/TrainingHeader";
import ConfigurationTab from "@/components/training/ConfigurationTab";
import MonitoringStats from "@/components/training/monitoring/MonitoringStats";
import TrainingLogs from "@/components/training/monitoring/TrainingLogs";
import TrainingExtraGate from "@/components/training/TrainingExtraGate";
import HfAuthBanner from "@/components/landing/HfAuthBanner";

import { Button } from "@/components/ui/button";
import { Loader2, Play, Square, Trash2, ArrowLeft } from "lucide-react";

import { DatasetItem, listDatasets } from "@/lib/replayApi";
import {
  JobRecord,
  TrainingRequest,
  getJob,
  getJobLogs,
  getJobLogFile,
  listJobs,
  startTrainingJob,
  stopJob,
  deleteJob,
  listRunnerHardware,
  RunnerFlavor,
} from "@/lib/jobsApi";

const POLL_INTERVAL_MS = 1000;

function jobToStatus(job: JobRecord | null, isStarting: boolean): TrainingStatus {
  // Adapter so MonitoringStats can keep its current prop shape.
  if (!job) {
    return {
      training_active: isStarting,
      current_step: 0,
      total_steps: 0,
      available_controls: { stop_training: false, pause_training: false, resume_training: false },
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

function configToRequest(c: TrainingConfig): TrainingRequest {
  // The backend's TrainingRequest has more optional fields; the form covers
  // the user-meaningful subset.
  return {
    target: c.target,
    dataset_repo_id: c.dataset_repo_id,
    policy_type: c.policy_type,
    steps: c.steps,
    batch_size: c.batch_size,
    seed: c.seed,
    num_workers: c.num_workers,
    log_freq: c.log_freq,
    save_freq: c.save_freq,
    save_checkpoint: c.save_checkpoint,
    resume: c.resume,
    wandb_enable: c.wandb_enable,
    wandb_project: c.wandb_project,
    wandb_entity: c.wandb_entity,
    wandb_notes: c.wandb_notes,
    wandb_mode: c.wandb_mode,
    wandb_disable_artifact: c.wandb_disable_artifact,
    policy_device: c.policy_device,
    policy_use_amp: c.policy_use_amp,
    optimizer_type: c.optimizer_type,
    optimizer_lr: c.optimizer_lr,
    optimizer_weight_decay: c.optimizer_weight_decay,
    optimizer_grad_clip_norm: c.optimizer_grad_clip_norm,
    use_policy_training_preset: c.use_policy_training_preset,
  };
}

const ConfigurationMode: React.FC = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { auth } = useHfAuth();
  const { toast } = useToast();
  const navigate = useNavigate();

  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    target: { runner: "local" },
    dataset_repo_id: "",
    policy_type: "act",
    steps: 10000,
    batch_size: 8,
    seed: 1000,
    num_workers: 4,
    log_freq: 250,
    save_freq: 1000,
    save_checkpoint: true,
    resume: false,
    wandb_enable: false,
    wandb_mode: "online",
    wandb_disable_artifact: false,
    policy_device: "cuda",
    policy_use_amp: false,
    optimizer_type: "adam",
    use_policy_training_preset: true,
  });

  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [trainingExtraAvailable, setTrainingExtraAvailable] = useState<boolean | null>(null);
  const [trainingExtraInstallHint, setTrainingExtraInstallHint] = useState<string>("pip install accelerate");
  const [runningJobExists, setRunningJobExists] = useState<boolean>(false);
  const [isStarting, setIsStarting] = useState(false);
  const [authenticated, setAuthenticated] = useState<boolean>(false);
  const [flavors, setFlavors] = useState<RunnerFlavor[]>([]);
  const [hardwareLoading, setHardwareLoading] = useState(true);

  useEffect(() => {
    setDatasetsLoading(true);
    listDatasets(baseUrl, fetchWithHeaders)
      .then(setDatasets)
      .catch(() => setDatasets([]))
      .finally(() => setDatasetsLoading(false));
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    fetchWithHeaders(`${baseUrl}/system/training-extra`)
      .then((r) => r.json())
      .then((data: { available: boolean; install_hint: string }) => {
        setTrainingExtraAvailable(data.available);
        setTrainingExtraInstallHint(data.install_hint);
      })
      .catch(() => setTrainingExtraAvailable(true));
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    listJobs(baseUrl, fetchWithHeaders, 1)
      .then((j) => setRunningJobExists(j.some((r) => r.state === "running")))
      .catch(() => setRunningJobExists(false));
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    // Re-fetches when auth status flips (e.g. user pastes a token in
    // HfAuthBanner) so flavors unlock without a page reload.
    setHardwareLoading(true);
    listRunnerHardware(baseUrl, fetchWithHeaders)
      .then((data) => {
        setAuthenticated(data.authenticated);
        setFlavors(data.flavors);
      })
      .catch(() => {
        setAuthenticated(false);
        setFlavors([]);
      })
      .finally(() => setHardwareLoading(false));
  }, [baseUrl, fetchWithHeaders, auth.status]);

  const updateConfig = <T extends keyof TrainingConfig>(key: T, value: TrainingConfig[T]) => {
    setTrainingConfig((prev) => ({ ...prev, [key]: value }));
  };

  const handleStart = async () => {
    if (!trainingConfig.dataset_repo_id.trim()) {
      toast({ title: "Error", description: "Dataset repository ID is required", variant: "destructive" });
      return;
    }
    setIsStarting(true);
    try {
      const job = await startTrainingJob(baseUrl, fetchWithHeaders, configToRequest(trainingConfig));
      toast({ title: "Training Started", description: job.name });
      navigate(`/training/${job.id}`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      toast({ title: "Error", description: msg, variant: "destructive" });
      // If the failure was the 409 case, refresh our running-job knowledge.
      listJobs(baseUrl, fetchWithHeaders, 1)
        .then((j) => setRunningJobExists(j.some((r) => r.state === "running")))
        .catch(() => {});
    } finally {
      setIsStarting(false);
    }
  };

  if (trainingExtraAvailable === null) {
    return (
      <div className="min-h-screen bg-slate-900 text-white p-4">
        <div className="max-w-7xl mx-auto">
          <TrainingHeader />
          <div className="flex items-center justify-center py-24 text-slate-400">
            <Loader2 className="w-6 h-6 animate-spin mr-3" />
            Checking training environment…
          </div>
        </div>
      </div>
    );
  }

  if (trainingExtraAvailable === false) {
    return (
      <div className="min-h-screen bg-slate-900 text-white p-4">
        <div className="max-w-7xl mx-auto">
          <TrainingHeader />
          <TrainingExtraGate installHint={trainingExtraInstallHint} />
        </div>
      </div>
    );
  }

  const targetRequiresAuth = trainingConfig.target.runner === "hf_cloud";
  const targetMissingFlavor =
    trainingConfig.target.runner === "hf_cloud" && !trainingConfig.target.flavor;
  const startDisabled =
    isStarting ||
    !trainingConfig.dataset_repo_id.trim() ||
    runningJobExists ||
    (targetRequiresAuth && !authenticated) ||
    targetMissingFlavor;
  const startTooltip = runningJobExists
    ? "Another training is already running"
    : targetRequiresAuth && !authenticated
    ? "Log in to Hugging Face to use cloud compute"
    : targetMissingFlavor
    ? "Select a hardware flavor"
    : undefined;

  return (
    <div className="min-h-screen bg-slate-900 text-white p-4">
      <div className="max-w-7xl mx-auto">
        <TrainingHeader />
        <HfAuthBanner />
        <ConfigurationTab
          config={trainingConfig}
          updateConfig={updateConfig}
          datasets={datasets}
          datasetsLoading={datasetsLoading}
          authenticated={authenticated}
          flavors={flavors}
          hardwareLoading={hardwareLoading}
        />
        <div className="max-w-3xl mx-auto mt-6 flex justify-end">
          <Button
            onClick={handleStart}
            disabled={startDisabled}
            title={startTooltip}
            size="lg"
            className="bg-green-500 hover:bg-green-600 text-white font-semibold px-6"
          >
            {isStarting ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" /> Starting…
              </>
            ) : (
              <>
                <Play className="w-5 h-5 mr-2" /> Start Training
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
};

const MonitoringMode: React.FC<{ jobId: string }> = ({ jobId }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const navigate = useNavigate();

  const [job, setJob] = useState<JobRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const logContainerRef = useRef<HTMLDivElement>(null);

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

  // Poll the job + its logs while running.
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
            setLogs((prev) => [...prev, ...newLogs]);
          }
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    };
    tick();
    const id = setInterval(() => {
      if (!cancelled) {
        if (job?.state && job.state !== "running") return; // pause polling once finished
        tick();
      }
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [baseUrl, fetchWithHeaders, jobId, job?.state]);

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
    if (!window.confirm("Delete this run? This wipes the output directory.")) return;
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
      <div className="min-h-screen bg-slate-900 text-white p-4">
        <div className="max-w-7xl mx-auto space-y-4">
          <Button variant="ghost" onClick={() => navigate("/")} className="text-slate-400">
            <ArrowLeft className="w-4 h-4 mr-2" /> Back to Jobs
          </Button>
          <p className="text-red-300">Couldn't load job {jobId}: {error}</p>
        </div>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="min-h-screen bg-slate-900 text-white p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-center py-24 text-slate-400">
          <Loader2 className="w-6 h-6 animate-spin mr-3" /> Loading job…
        </div>
      </div>
    );
  }

  const isRunning = job.state === "running";

  return (
    <div className="min-h-screen bg-slate-900 text-white p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" onClick={() => navigate("/")} className="text-slate-400 hover:text-white">
              <ArrowLeft className="w-4 h-4 mr-2" /> Jobs
            </Button>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-semibold text-white">{job.name}</h1>
                {job.runner === "hf_cloud" ? (
                  <span className="text-xs px-2 py-0.5 rounded bg-amber-900/40 text-amber-200 border border-amber-700">
                    HF · {job.hf_flavor ?? "cloud"}
                  </span>
                ) : (
                  <span className="text-xs px-2 py-0.5 rounded bg-slate-700 text-slate-200 border border-slate-600">
                    Local
                  </span>
                )}
                {job.runner === "hf_cloud" && job.hf_repo_id && job.state === "done" && (
                  <a
                    href={`https://huggingface.co/${job.hf_repo_id}`}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-amber-300 hover:text-amber-200 underline"
                  >
                    View on Hub ↗
                  </a>
                )}
              </div>
              <p className="text-xs text-slate-400">
                {job.state}
                {job.error_message ? ` — ${job.error_message}` : ""}
              </p>
            </div>
          </div>
          {isRunning ? (
            <Button onClick={handleStop} className="bg-red-500 hover:bg-red-600 text-white">
              <Square className="w-4 h-4 mr-2" /> Stop
            </Button>
          ) : (
            <Button onClick={handleDelete} variant="ghost" className="text-slate-400 hover:text-white">
              <Trash2 className="w-4 h-4 mr-2" /> Delete
            </Button>
          )}
        </div>

        <MonitoringStats
          trainingStatus={jobToStatus(job, false)}
          getProgressPercentage={getProgressPercentage}
          formatTime={formatTime}
        />
        <TrainingLogs logs={logs} logContainerRef={logContainerRef} />
      </div>
    </div>
  );
};

const Training: React.FC = () => {
  const { jobId } = useParams<{ jobId?: string }>();
  return jobId ? <MonitoringMode jobId={jobId} /> : <ConfigurationMode />;
};

export default Training;
