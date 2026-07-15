import React, { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useHfAuth } from "@/contexts/HfAuthContext";

import { TrainingConfig } from "@/components/training/types";
import ConfigurationTab from "@/components/training/ConfigurationTab";
import TrainingExtraGate from "@/components/training/TrainingExtraGate";
import PolicyExtraDialog from "@/components/training/PolicyExtraDialog";
import HfAuthBanner from "@/components/landing/HfAuthBanner";

import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";
import { Loader2, Play } from "lucide-react";

import { useSelectedDataset } from "@/hooks/useSelectedDataset";
import {
  TrainingRequest,
  listJobs,
  startTrainingJob,
  listRunnerHardware,
  RunnerFlavor,
} from "@/lib/jobsApi";
import { useDatasets } from "@/hooks/useDatasets";
import { validateDatasetRepoId } from "@/lib/datasetName";
import { useDatasetUpload } from "@/hooks/useDatasetUpload";
import { getDatasetInfo } from "@/lib/replayApi";
import LocalDatasetCloudNotice from "@/components/training/config/LocalDatasetCloudNotice";

// The policy type is chosen before landing here (home-page model buttons, or
// inherited by the Continue/Fine-tune flows) and arrives via router state.
// Router state is lost on a hard refresh / direct visit, so mirror the last
// resolved type in sessionStorage and fall back to it — the frozen Policy
// display then survives a refresh instead of silently reverting to "act".
const POLICY_TYPE_STORAGE_KEY = "makerlab.training.policyType";

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

/**
 * The "New training run" config panel — extracted verbatim from Training.tsx's
 * former ConfigurationMode (the no-jobId branch). Toggled open by TrainDeploy;
 * it still reads resume/finetune/policyType off router state exactly as before,
 * so the frozen `/training` state contracts are preserved. Rendered as an
 * inline card inside the StageLayout instead of its own AppShell.
 */
const TrainingConfigPanel: React.FC = () => {
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
      toast({ title: "Training started", description: job.name });
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

    // Pre-flight: smolvla/pi0/diffusion need an optional package. Catch it here
    // with a one-click installer instead of a buried ImportError after the job
    // has already started.
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
      // Check failed (offline / older backend) — fall through and let the job
      // report any problem itself.
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
      const err = await startUpload([], true /* private */);
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
      <Card className="mb-5">
        <CardContent className="flex items-center justify-center py-16 text-muted-foreground">
          <Loader2 className="mr-3 h-6 w-6 animate-spin" />
          Checking training environment…
        </CardContent>
      </Card>
    );
  }

  if (trainingExtraAvailable === false) {
    return (
      <Card className="mb-5">
        <CardContent className="pt-6">
          <TrainingExtraGate installHint={trainingExtraInstallHint} />
        </CardContent>
      </Card>
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
  // The dataset field is free text now (any Hub dataset) — block Start on a
  // malformed id instead of launching a job doomed to fail.
  const datasetRepoIdError = datasetRepoId
    ? validateDatasetRepoId(datasetRepoId)
    : null;
  const startDisabled =
    isStarting ||
    uploading ||
    !datasetRepoId ||
    datasetRepoIdError != null ||
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
          : datasetRepoIdError
            ? datasetRepoIdError
            : undefined;

  return (
    <Card className="mb-5">
      <CardContent className="space-y-6 pt-6">
        <HfAuthBanner />
        {resumeSource ? (
          <div className="mx-auto max-w-3xl rounded-md border border-info/40 bg-info/10 p-4 text-sm text-foreground">
            <div className="font-display font-semibold">
              Continuing “{resumeSource.name}”
              {resumeSource.step != null
                ? ` from step ${resumeSource.step.toLocaleString()}`
                : " from its latest checkpoint"}
            </div>
            <p className="mt-1 text-muted-foreground">
              The dataset, policy, batch size, and optimizer are inherited from
              the checkpoint — only{" "}
              <span className="font-medium text-foreground">Steps</span> applies
              here. Set it above the resumed step to train further (prefilled to{" "}
              {trainingConfig.steps.toLocaleString()}).
            </p>
          </div>
        ) : null}
        {finetuneSource ? (
          <div className="mx-auto max-w-3xl rounded-md border border-info/40 bg-info/10 p-4 text-sm text-foreground">
            <div className="font-display font-semibold">
              Fine-tuning from “{finetuneSource.name}”
              {finetuneSource.step != null
                ? ` (step ${finetuneSource.step.toLocaleString()})`
                : " (latest checkpoint)"}
            </div>
            <p className="mt-1 text-muted-foreground">
              This starts a{" "}
              <span className="font-medium text-foreground">fresh run</span> (new
              optimizer, from step 0) with the policy weights initialized from
              that model. Pick a{" "}
              <span className="font-medium text-foreground">dataset</span> to
              train on and set your training parameters as usual.
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
          <div className="mx-auto max-w-3xl">
            <LocalDatasetCloudNotice
              repoId={datasetRepoId}
              sizeBytes={datasetSizeBytes}
              offline={offline}
              uploading={uploading}
              errorMessage={uploadError}
            />
          </div>
        ) : null}
        <div className="mx-auto flex max-w-3xl flex-col items-end gap-2">
          {resumeStepError ? (
            <p className="text-sm text-destructive">{resumeStepError}</p>
          ) : null}
          {(() => {
            const startButton = (
              <Button
                onClick={handleStart}
                disabled={startDisabled}
                size="lg"
                variant="brand"
                className="px-6"
              >
                {uploading ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" /> Uploading…
                  </>
                ) : isStarting ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" /> Starting…
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-5 w-5" />{" "}
                    {resumeSource
                      ? "Continue training"
                      : finetuneSource
                        ? "Start fine-tuning"
                        : needsUpload
                          ? "Upload & start training"
                          : "Start training"}
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
      </CardContent>
    </Card>
  );
};

export default TrainingConfigPanel;
