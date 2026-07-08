import React, { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useHfAuth } from "@/contexts/HfAuthContext";

import {
  TrainingConfig,
  TrainingStatus,
  LogEntry,
} from "@/components/training/types";
import TrainingHeader from "@/components/training/TrainingHeader";
import ConfigurationTab from "@/components/training/ConfigurationTab";
import MonitoringStats from "@/components/training/monitoring/MonitoringStats";
import TrainingLogs from "@/components/training/monitoring/TrainingLogs";
import TrainingExtraGate from "@/components/training/TrainingExtraGate";
import PolicyExtraDialog from "@/components/training/PolicyExtraDialog";
import HfAuthBanner from "@/components/landing/HfAuthBanner";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";
import { Loader2, Play, Square, Trash2, ArrowLeft } from "lucide-react";

import { useSelectedDataset } from "@/hooks/useSelectedDataset";
import {
  JobRecord,
  TrainingRequest,
  getJob,
  getJobLogs,
  getJobLogFile,
  jobDisplayName,
  listJobs,
  startTrainingJob,
  stopJob,
  deleteJob,
  listRunnerHardware,
  RunnerFlavor,
} from "@/lib/jobsApi";
import { JobCheckpoint, listJobCheckpoints } from "@/lib/checkpointsApi";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import InferenceModal from "@/components/landing/InferenceModal";
import { useRobots } from "@/hooks/useRobots";
import { useDatasets } from "@/hooks/useDatasets";
import { useDatasetUpload } from "@/hooks/useDatasetUpload";
import { getDatasetInfo } from "@/lib/replayApi";
import LocalDatasetCloudNotice from "@/components/training/config/LocalDatasetCloudNotice";

const POLL_INTERVAL_MS = 1000;
const MAX_LOG_LINES = 5000;

// The policy type is chosen before landing here (home-page model buttons, or
// inherited by the Continue/Fine-tune flows) and arrives via router state.
// Router state is lost on a hard refresh / direct visit, so mirror the last
// resolved type in sessionStorage and fall back to it — the frozen Policy
// display then survives a refresh instead of silently reverting to "act".
const POLICY_TYPE_STORAGE_KEY = "lelab.training.policyType";

function readStoredPolicyType(): string | null {
  try {
    return sessionStorage.getItem(POLICY_TYPE_STORAGE_KEY);
  } catch {
    return null;
  }
}

// Passed via router state by the "Continue" button on a completed local job,
// or the "Resume" button on a cloud run that ended before its step target.
type ResumeSource = {
  jobId: string;
  step: number | null; // null ⇒ resume from the latest checkpoint
  name: string;
  datasetRepoId: string;
  policyType: string;
  sourceSteps: number; // the source run's configured total, for a sane prefill
  logFreq?: number; // the source run's log cadence, to preserve on resume
  saveFreq?: number; // the source run's checkpoint cadence, to preserve on resume
  // Cloud resume: the parent run's runner + flavor. A local Continue omits
  // these (runner defaults to "local"). When "hf_cloud", the launched run
  // targets the same flavor and continues into the parent's Hub output repo.
  runner?: "local" | "hf_cloud";
  flavor?: string;
};

// Passed via router state by the "Fine-tune" button on an imported model. A
// fine-tune is a FRESH run (fresh optimizer, step 0) whose weights are
// initialized from the source checkpoint — distinct from resume.
type FinetuneSource = {
  jobId: string;
  step: number | null; // null ⇒ latest checkpoint of the source
  name: string;
  policyType: string;
};

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

function configToRequest(c: TrainingConfig): TrainingRequest {
  // The backend's TrainingRequest has more optional fields; the form covers
  // the user-meaningful subset.
  return {
    target: c.target,
    dataset_repo_id: c.dataset_repo_id,
    policy_type: c.policy_type,
    job_name: c.job_name,
    steps: c.steps,
    batch_size: c.batch_size,
    seed: c.seed,
    num_workers: c.num_workers,
    log_freq: c.log_freq,
    save_freq: c.save_freq,
    save_checkpoint: c.save_checkpoint,
    resume: c.resume,
    resume_from_job_id: c.resume_from_job_id,
    resume_from_step: c.resume_from_step,
    finetune_from_job_id: c.finetune_from_job_id,
    finetune_from_step: c.finetune_from_step,
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
  const location = useLocation();
  const navState = location.state as {
    resume?: ResumeSource;
    finetune?: FinetuneSource;
    // Set by the landing page's per-model-type "Create a model" buttons so
    // the chosen policy arrives preselected. Direct visits have no state and
    // fall back to the default.
    policyType?: string;
  } | null;
  const resumeSource = navState?.resume ?? null;
  const finetuneSource = navState?.finetune ?? null;
  const preselectedPolicyType = navState?.policyType ?? null;
  // Dataset is chosen on the home page (single source of truth); a resume
  // inherits the source run's dataset. A fine-tune trains on a NEW dataset
  // (the imported model has none), so it uses the home-page selection like a
  // normal fresh run.
  const { selectedDataset } = useSelectedDataset();
  const prefilledDatasetRepoId =
    resumeSource?.datasetRepoId ?? selectedDataset ?? "";

  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    // A cloud resume inherits the parent run's target so the continuation runs
    // on the same flavor and pushes into the same Hub repo; everything else
    // defaults to a fresh local run.
    target:
      resumeSource?.runner === "hf_cloud"
        ? { runner: "hf_cloud", flavor: resumeSource.flavor }
        : { runner: "local" },
    dataset_repo_id: prefilledDatasetRepoId,
    policy_type:
      resumeSource?.policyType ??
      finetuneSource?.policyType ??
      preselectedPolicyType ??
      readStoredPolicyType() ??
      "act",
    job_name: "",
    // On resume, everything but steps is inherited from the checkpoint's
    // train_config.json; prefill steps above the source's total so the
    // continuation actually trains further. Fine-tune is a fresh run, so it
    // uses the normal fresh default.
    steps: resumeSource ? resumeSource.sourceSteps * 2 : 10000,
    batch_size: 8,
    seed: 1000,
    num_workers: 4,
    log_freq: resumeSource?.logFreq ?? 50,
    save_freq: resumeSource?.saveFreq ?? 1000,
    save_checkpoint: true,
    // Fine-tune is NOT a resume — it's a fresh run whose weights are seeded from
    // the source checkpoint. resume stays false; the backend resolves
    // finetune_from_* into --policy.pretrained_path.
    resume: !!resumeSource,
    resume_from_job_id: resumeSource?.jobId,
    resume_from_step: resumeSource?.step ?? undefined,
    finetune_from_job_id: finetuneSource?.jobId,
    finetune_from_step: finetuneSource?.step ?? undefined,
    wandb_enable: false,
    wandb_mode: "online",
    wandb_disable_artifact: false,
    policy_device: "auto",
    policy_use_amp: false,
    optimizer_type: "adam",
    use_policy_training_preset: true,
  });

  // Mirror the resolved (frozen) policy type so a refresh — which drops
  // router state — restores the same choice via readStoredPolicyType().
  useEffect(() => {
    try {
      sessionStorage.setItem(
        POLICY_TYPE_STORAGE_KEY,
        trainingConfig.policy_type,
      );
    } catch {
      // storage unavailable (private mode) — refresh falls back to default
    }
  }, [trainingConfig.policy_type]);

  const [trainingExtraAvailable, setTrainingExtraAvailable] = useState<
    boolean | null
  >(null);
  const [trainingExtraInstallHint, setTrainingExtraInstallHint] =
    useState<string>("pip install accelerate");
  const [localJobRunning, setLocalJobRunning] = useState<boolean>(false);
  const [isStarting, setIsStarting] = useState(false);
  const [policyExtra, setPolicyExtra] = useState<{
    policyType: string;
    packageName: string;
    installTarget: string;
    installHint: string;
  } | null>(null);
  const [authenticated, setAuthenticated] = useState<boolean>(false);
  const [flavors, setFlavors] = useState<RunnerFlavor[]>([]);
  const [hardwareLoading, setHardwareLoading] = useState(true);
  // HF_HUB_OFFLINE on the backend: Hub writes (incl. dataset upload) disabled.
  const [offline, setOffline] = useState<boolean>(false);

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
    // Only the local lock matters for the Start button; cloud jobs can stack.
    // Pull a generous slice so a running local isn't masked by newer cloud
    // jobs in the started_at-desc ordering.
    listJobs(baseUrl, fetchWithHeaders, 200)
      .then((j) =>
        setLocalJobRunning(
          j.some((r) => r.runner === "local" && r.state === "running"),
        ),
      )
      .catch(() => setLocalJobRunning(false));
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    // Re-fetches when auth status flips (e.g. user pastes a token in
    // HfAuthBanner) so flavors unlock without a page reload.
    setHardwareLoading(true);
    listRunnerHardware(baseUrl, fetchWithHeaders)
      .then((data) => {
        setAuthenticated(data.authenticated);
        setFlavors(data.flavors);
        setOffline(!!data.offline);
      })
      .catch(() => {
        setAuthenticated(false);
        setFlavors([]);
        setOffline(false);
      })
      .finally(() => setHardwareLoading(false));
  }, [baseUrl, fetchWithHeaders, auth.status]);

  const updateConfig = <T extends keyof TrainingConfig>(
    key: T,
    value: TrainingConfig[T],
  ) => {
    setTrainingConfig((prev) => ({ ...prev, [key]: value }));
  };

  // Cloud training runs from the Hub, so a dataset that only exists in this
  // machine's local cache must be uploaded first. We read the chosen dataset's
  // `source` from the /datasets listing (already carries it) and, for a
  // local-only + cloud combo, chain an upload before the job launches.
  const { datasets } = useDatasets();
  const datasetRepoId = trainingConfig.dataset_repo_id.trim();
  const isCloud = trainingConfig.target.runner === "hf_cloud";
  const selectedDatasetItem = datasets.find((d) => d.repo_id === datasetRepoId);
  // Only "local" needs uploading; "both"/"hub" already exist on the Hub, and an
  // unknown item (listing not yet loaded / dataset typed by hand) is left alone
  // — the backend preflight is the belt-and-braces catch for that case.
  const datasetLocalOnly = selectedDatasetItem?.source === "local";
  const needsUpload = isCloud && datasetLocalOnly;

  // Approximate on-disk size for the notice (cheap detail endpoint; local only).
  const [datasetSizeBytes, setDatasetSizeBytes] = useState<number | null>(null);
  useEffect(() => {
    if (!needsUpload || !datasetRepoId) {
      setDatasetSizeBytes(null);
      return;
    }
    let cancelled = false;
    getDatasetInfo(baseUrl, fetchWithHeaders, datasetRepoId)
      .then((info) => {
        if (!cancelled) setDatasetSizeBytes(info.size_bytes ?? null);
      })
      .catch(() => {
        if (!cancelled) setDatasetSizeBytes(null);
      });
    return () => {
      cancelled = true;
    };
  }, [needsUpload, datasetRepoId, baseUrl, fetchWithHeaders]);

  const [uploadError, setUploadError] = useState<string | null>(null);

  // The actual job launch, factored out so it can run either directly (dataset
  // already on the Hub) or as the upload's success continuation.
  const launchJob = useCallback(async () => {
    setIsStarting(true);
    try {
      const job = await startTrainingJob(
        baseUrl,
        fetchWithHeaders,
        configToRequest(trainingConfig),
      );
      toast({ title: "Training Started", description: job.name });
      navigate(`/training/${job.id}`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      toast({ title: "Error", description: msg, variant: "destructive" });
      // If the failure was the 409 case, refresh our running-job knowledge.
      listJobs(baseUrl, fetchWithHeaders, 200)
        .then((j) =>
          setLocalJobRunning(
            j.some((r) => r.runner === "local" && r.state === "running"),
          ),
        )
        .catch(() => {});
    } finally {
      setIsStarting(false);
    }
  }, [baseUrl, fetchWithHeaders, trainingConfig, toast, navigate]);

  // Latest launchJob without re-subscribing the upload hook every render.
  const launchJobRef = useRef(launchJob);
  launchJobRef.current = launchJob;

  const { uploading, start: startUpload } = useDatasetUpload({
    repoId: datasetRepoId,
    onDone: () => {
      // Upload finished: launch the cloud job exactly as a Hub dataset would.
      setUploadError(null);
      launchJobRef.current();
    },
    onError: (message) => {
      // Upload failed: stop cleanly, surface the error, launch nothing (no
      // orphan output repo — the job was never submitted).
      setUploadError(message);
      setIsStarting(false);
      toast({
        title: "Upload failed",
        description: message,
        variant: "destructive",
      });
    },
  });

  const handleStart = async () => {
    if (!datasetRepoId) {
      toast({
        title: "Error",
        description: "Dataset repository ID is required",
        variant: "destructive",
      });
      return;
    }

    // Pre-flight: smolvla/pi0/diffusion need an optional package installed
    // locally. Catch it here with a one-click installer instead of a buried
    // ImportError after the job has already started. Cloud jobs run in their
    // own environment, so the local package is irrelevant — skip the check.
    if (trainingConfig.target.runner === "local") {
      try {
        const r = await fetchWithHeaders(
          `${baseUrl}/system/policy-extra/${trainingConfig.policy_type}`,
        );
        if (r.ok) {
          const extra = await r.json();
          if (extra.needs_extra && !extra.available) {
            setPolicyExtra({
              policyType: trainingConfig.policy_type,
              packageName: extra.package,
              installTarget: extra.install_target,
              installHint: extra.install_hint,
            });
            return;
          }
        }
      } catch {
        // Check failed (offline / older backend) — fall through and let the
        // job report any problem itself.
      }
    }

    // Cloud run on a local-only dataset: upload first, then launch on success.
    // The Start button is disabled while `uploading`, so we never reach here
    // with an in-flight upload for this repo (the hook re-attaches on mount and
    // will fire onDone/onError for one already running). On success the hook
    // seeds a running status and its poll fires onDone → launchJob; a refusal
    // (409 / dataset busy) is a hard stop surfaced in the notice.
    if (needsUpload) {
      setUploadError(null);
      setIsStarting(true);
      const err = await startUpload([], false /* public: LeLab uploads are public by default */);
      if (err) {
        setUploadError(err);
        setIsStarting(false);
        toast({
          title: "Upload failed",
          description: err,
          variant: "destructive",
        });
      }
      return;
    }

    await launchJob();
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
    trainingConfig.target.runner === "hf_cloud" &&
    !trainingConfig.target.flavor;
  const localBlocked =
    trainingConfig.target.runner === "local" && localJobRunning;
  // When resuming, total steps must be strictly above the checkpoint's step:
  // equal trains nothing and lerobot requires --steps above the resumed step.
  const resumeStepError =
    resumeSource != null &&
    resumeSource.step != null &&
    trainingConfig.steps <= resumeSource.step
      ? `Total steps must be greater than the checkpoint's step (${resumeSource.step.toLocaleString()}).`
      : null;
  // A local-only dataset on a cloud run is uploadable — unless the backend is
  // in offline mode, in which case uploads are impossible and Start is a hard
  // block. (needsUpload is already gated on isCloud.)
  const uploadBlockedOffline = needsUpload && offline;
  const startDisabled =
    isStarting ||
    uploading ||
    !datasetRepoId ||
    localBlocked ||
    (targetRequiresAuth && !authenticated) ||
    targetMissingFlavor ||
    uploadBlockedOffline ||
    resumeStepError != null;
  const startTooltip = localBlocked
    ? "Another local training is already running"
    : targetRequiresAuth && !authenticated
      ? "Log in to Hugging Face to use cloud compute"
      : targetMissingFlavor
        ? "Select a hardware flavor"
        : uploadBlockedOffline
          ? "Offline mode is on — the dataset can't be uploaded to the Hub"
          : undefined;

  return (
    <div className="min-h-screen bg-slate-900 text-white p-4">
      <div className="max-w-7xl mx-auto">
        <TrainingHeader />
        <HfAuthBanner />
        {resumeSource ? (
          <div className="max-w-3xl mx-auto mb-4 rounded-lg border border-sky-500/40 bg-sky-500/10 p-4 text-sm text-sky-100">
            <div className="font-semibold">
              Continuing “{resumeSource.name}”
              {resumeSource.step != null
                ? ` from step ${resumeSource.step.toLocaleString()}`
                : " from its latest checkpoint"}
            </div>
            <p className="mt-1 text-sky-200/80">
              The dataset, policy, batch size, and optimizer are inherited from
              the checkpoint — only <span className="font-medium">Steps</span>{" "}
              applies here. Set it above the resumed step to train further
              (prefilled to {trainingConfig.steps.toLocaleString()}).
            </p>
          </div>
        ) : null}
        {finetuneSource ? (
          <div className="max-w-3xl mx-auto mb-4 rounded-lg border border-violet-500/40 bg-violet-500/10 p-4 text-sm text-violet-100">
            <div className="font-semibold">
              Fine-tuning from “{finetuneSource.name}”
              {finetuneSource.step != null
                ? ` (step ${finetuneSource.step.toLocaleString()})`
                : " (latest checkpoint)"}
            </div>
            <p className="mt-1 text-violet-200/80">
              This starts a{" "}
              <span className="font-medium">fresh run</span> (new optimizer, from
              step 0) with the policy weights initialized from that model. Pick a{" "}
              <span className="font-medium">dataset</span> to train on and set
              your training parameters as usual.
            </p>
          </div>
        ) : null}
        <ConfigurationTab
          config={trainingConfig}
          updateConfig={updateConfig}
          authenticated={authenticated}
          flavors={flavors}
          hardwareLoading={hardwareLoading}
        />
        {needsUpload ? (
          <div className="max-w-3xl mx-auto mt-6">
            <LocalDatasetCloudNotice
              repoId={datasetRepoId}
              sizeBytes={datasetSizeBytes}
              offline={offline}
              uploading={uploading}
              errorMessage={uploadError}
            />
          </div>
        ) : null}
        <div className="max-w-3xl mx-auto mt-6 flex flex-col items-end gap-2">
          {resumeStepError ? (
            <p className="text-sm text-red-400">{resumeStepError}</p>
          ) : null}
          {(() => {
            const startButton = (
              <Button
                onClick={handleStart}
                disabled={startDisabled}
                size="lg"
                className="bg-green-500 hover:bg-green-600 text-white font-semibold px-6"
              >
                {uploading ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" /> Uploading…
                  </>
                ) : isStarting ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" /> Starting…
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5 mr-2" />{" "}
                    {resumeSource
                      ? "Continue Training"
                      : finetuneSource
                        ? "Start Fine-tuning"
                        : needsUpload
                          ? "Upload & start training"
                          : "Start Training"}
                  </>
                )}
              </Button>
            );
            // Native `title` doesn't fire reliably on disabled buttons across
            // browsers — and since Radix's tooltip relies on pointer events
            // that a disabled button swallows, wrap in a span so the trigger
            // still receives hover/focus.
            if (!startTooltip) return startButton;
            return (
              <Tooltip>
                <TooltipTrigger asChild>
                  <span tabIndex={0}>{startButton}</span>
                </TooltipTrigger>
                <TooltipContent>{startTooltip}</TooltipContent>
              </Tooltip>
            );
          })()}
        </div>
      </div>

      {policyExtra && (
        <PolicyExtraDialog
          open={!!policyExtra}
          onOpenChange={(o) => !o && setPolicyExtra(null)}
          policyType={policyExtra.policyType}
          packageName={policyExtra.packageName}
          installTarget={policyExtra.installTarget}
          installHint={policyExtra.installHint}
        />
      )}
    </div>
  );
};

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
      <div className="min-h-screen bg-slate-900 text-white p-4">
        <div className="max-w-7xl mx-auto space-y-4">
          <Button
            variant="ghost"
            onClick={() => navigate("/")}
            className="text-slate-400"
          >
            <ArrowLeft className="w-4 h-4 mr-2" /> Back to Jobs
          </Button>
          <p className="text-red-300">
            Couldn't load job {jobId}: {error}
          </p>
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
            <Button
              variant="ghost"
              onClick={() => navigate("/")}
              className="text-slate-400 hover:text-white"
            >
              <ArrowLeft className="w-4 h-4 mr-2" /> Jobs
            </Button>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-semibold text-white">
                  {jobDisplayName(job)}
                </h1>
                {job.runner === "hf_cloud" ? (
                  <span className="text-xs px-2 py-0.5 rounded bg-amber-900/40 text-amber-200 border border-amber-700">
                    HF · {job.hf_flavor ?? "cloud"}
                  </span>
                ) : (
                  <span className="text-xs px-2 py-0.5 rounded bg-slate-700 text-slate-200 border border-slate-600">
                    Local
                  </span>
                )}
                {job.runner === "hf_cloud" &&
                  job.hf_repo_id &&
                  job.state === "done" && (
                    <a
                      href={`https://huggingface.co/${job.hf_repo_id}`}
                      target="_blank"
                      rel="noreferrer"
                      className="text-xs text-amber-300 hover:text-amber-200 underline"
                    >
                      View on Hub ↗
                    </a>
                  )}
                {job.wandb_run_url && (
                  <a
                    href={job.wandb_run_url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-yellow-300 hover:text-yellow-200 underline"
                  >
                    View on W&B ↗
                  </a>
                )}
              </div>
              {/* When aliased, keep the immutable run id visible as subtext. */}
              {job.display_name ? (
                <p className="text-[11px] text-slate-500">{job.id}</p>
              ) : null}
              <p className="text-xs text-slate-400">
                {job.state}
                {job.error_message ? ` — ${job.error_message}` : ""}
              </p>
            </div>
          </div>
          {isRunning ? (
            <Button
              onClick={handleStop}
              className="bg-red-500 hover:bg-red-600 text-white"
            >
              <Square className="w-4 h-4 mr-2" /> Stop
            </Button>
          ) : (
            <Button
              onClick={handleDelete}
              variant="ghost"
              className="text-slate-400 hover:text-white"
            >
              <Trash2 className="w-4 h-4 mr-2" /> Delete
            </Button>
          )}
        </div>

        <MonitoringStats
          jobId={jobId}
          trainingStatus={jobToStatus(job, false)}
          getProgressPercentage={getProgressPercentage}
          formatTime={formatTime}
        />
        <div className="bg-slate-800/40 border border-slate-700 rounded-lg p-4 flex items-center gap-3">
          <span className="text-sm font-semibold text-slate-300">
            Run inference
          </span>
          {checkpoints.length === 0 ? (
            <span className="text-xs text-slate-500">
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
                className="bg-green-500 hover:bg-green-600 text-white"
              >
                <Play className="w-4 h-4 mr-2" />
                Run on robot
              </Button>
            </>
          )}
        </div>
        <InferenceModal
          open={inferenceModalOpen}
          onOpenChange={setInferenceModalOpen}
          robot={selectedRecord}
          jobId={jobId}
          initialStep={selectedStep}
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
