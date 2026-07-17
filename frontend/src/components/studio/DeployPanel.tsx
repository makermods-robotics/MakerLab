import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle,
  ChevronDown,
  Loader2,
  Play,
  Square,
  VideoOff,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useStudio } from "@/contexts/StudioContext";
import { useInferenceSession } from "@/contexts/InferenceSessionContext";
import { useRobots, robotSetupGap } from "@/hooks/useRobots";
import { useInferenceLaunch } from "@/hooks/useInferenceLaunch";
import {
  JobCheckpoint,
  PolicyConfigSummary,
  getCheckpointPolicyConfig,
  listJobCheckpoints,
} from "@/lib/checkpointsApi";
import {
  InferenceStatus,
  getInferenceStatus,
  startInference,
  stopInference,
} from "@/lib/inferenceApi";
import { JobRecord, getJob, jobDisplayName, listJobs } from "@/lib/jobsApi";
import { ModelItem, getModels } from "@/lib/modelsApi";
import { findJobForModel, importSourceForModel } from "@/lib/inferenceLaunch";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import ModelsLibrary from "@/components/jobs/ModelsLibrary";
import {
  LibrarySection,
  PanelHeader,
  SLIDE,
} from "@/components/studio/panel/primitives";
import {
  AvailableCamera,
  useAvailableCameras,
} from "@/hooks/useAvailableCameras";
import { useCameraStream } from "@/hooks/useCameraStream";

/**
 * Studio panel 3 · Deploy — run a skill (local trained checkpoint or an
 * imported Hub model) on the corner robot. Every "Run on robot" action lands
 * here via `useStudio().deployPrefill`.
 *
 * This is a PARALLEL surface to the legacy `InferenceModal` (still used by
 * JobsSection + the Landing Models panel through `useInferenceLaunch`). To keep
 * those consumers untouched and avoid drift, the checkpoint/policy-config
 * fetch, the bimanual `left_` camera-prefix round-trip, the state_dim 6-vs-12
 * arm-count guard, the camera thumbnails and the start flow are ported VERBATIM
 * from `components/landing/InferenceModal.tsx` (only the palette becomes token
 * classes). The Hub lazy-import reuses `useInferenceLaunch().importSource` so
 * the husk-repo messaging is identical, not re-implemented.
 */

const DEFAULT_FPS = 30;
const JOB_SCAN_LIMIT = 200;

const cameraKey = (cam: AvailableCamera) => String(cam.index);

/** Small getUserMedia preview for verifying which physical camera a role binds
 * to. `paused` drops the browser stream so the rollout subprocess can open the
 * same device via OpenCV without contending. (Ported from InferenceModal.) */
const CameraThumbnail: React.FC<{ deviceId: string; paused: boolean }> = ({
  deviceId,
  paused,
}) => {
  const { videoRef, hasError } = useCameraStream(deviceId, paused);
  if (paused || hasError || !deviceId) {
    return (
      <div className="flex h-24 w-32 flex-col items-center justify-center rounded border border-border bg-muted">
        <VideoOff className="mb-1 h-5 w-5 text-muted-foreground" />
        <span className="text-[10px] text-muted-foreground">
          {paused ? "Released" : "No preview"}
        </span>
      </div>
    );
  }
  return (
    <video
      ref={videoRef}
      autoPlay
      muted
      playsInline
      className="h-24 w-32 rounded border border-border bg-muted object-cover"
    />
  );
};

/**
 * One camera as the panel sees it. The BiSO prefix round-trip lives here so the
 * future per-arm routing work has a single obvious place to extend.
 *
 * (Verbatim port of InferenceModal's CameraMapping / cameraMappings — see that
 * file's doc comment for the full BiSO `left_` prefix rationale. Kept here so
 * the legacy modal stays untouched for its existing consumers.)
 */
interface CameraMapping {
  /** Checkpoint feature key — the key into `policyConfig.image_features`. */
  feature: string;
  /** Name shown in the UI. Stripped bare name (bimanual) or `feature`. */
  display: string;
  /** Key used in the start-inference camera dict. Equals `display`. */
  requestKey: string;
}

const ARM_PREFIX_RE = /^(left|right)_/;

/**
 * Build the display/request mapping for the checkpoint's camera features.
 *
 * Single-arm: identity — bare names pass through untouched.
 *
 * Bimanual: strip the `left_`/`right_` prefix that BiSO added at record time,
 * so the user sees the name they chose and the rollout re-prefixes it correctly.
 * Guard against collisions: if two features would strip to the same bare name,
 * fall back to the FULL feature name for every colliding entry — correctness
 * over cosmetics.
 */
function cameraMappings(
  features: string[],
  isBimanual: boolean,
): CameraMapping[] {
  if (!isBimanual) {
    return features.map((f) => ({ feature: f, display: f, requestKey: f }));
  }
  const strippedCounts = new Map<string, number>();
  for (const f of features) {
    const bare = f.replace(ARM_PREFIX_RE, "");
    strippedCounts.set(bare, (strippedCounts.get(bare) ?? 0) + 1);
  }
  return features.map((f) => {
    const bare = f.replace(ARM_PREFIX_RE, "");
    const name = strippedCounts.get(bare) === 1 ? bare : f;
    return { feature: f, display: name, requestKey: name };
  });
}

const DeployPanel: React.FC = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { open, deployPrefill, clearDeployPrefill } = useStudio();
  const { openInferenceSession } = useInferenceSession();
  const { selectedRecord: robot } = useRobots();
  // Reuse the shared lazy-import (husk-repo messaging + idempotent registration)
  // so a Hub skill resolves to a pseudo-job exactly as the Jobs cards do.
  const { importSource } = useInferenceLaunch();

  // --- Skill picker state ------------------------------------------------
  const [models, setModels] = useState<ModelItem[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<JobRecord | null>(null);
  const [resolving, setResolving] = useState(false);

  // --- Inference config state (ported from InferenceModal) ---------------
  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [task, setTask] = useState("");
  const [durationS, setDurationS] = useState(60);
  const [submitting, setSubmitting] = useState(false);

  const [policyConfig, setPolicyConfig] = useState<PolicyConfigSummary | null>(
    null,
  );
  const [policyConfigLoading, setPolicyConfigLoading] = useState(false);
  const [policyConfigError, setPolicyConfigError] = useState<string | null>(
    null,
  );

  // Per camera DISPLAY name → user-selected physical camera key (raw cv2 index
  // string). Keyed by the stripped display name (== requestKey).
  const [cameraBindings, setCameraBindings] = useState<
    Record<string, string | null>
  >({});
  const { cameras: availableCameras } = useAvailableCameras({ enabled: open });

  // Light status poll while the panel is visible so ⏹ Stop enables only when a
  // rollout is actually active.
  const [status, setStatus] = useState<InferenceStatus | null>(null);
  const [stopping, setStopping] = useState(false);

  // The settings block (robot, checkpoint, run parameters, cameras) collapses
  // as one so a configured deploy can be folded down to picker + actions.
  const [settingsOpen, setSettingsOpen] = useState(true);

  const jobId = selectedJob?.id ?? null;
  const isBimanual = robot?.mode === "bimanual";

  const cameraMap = useMemo(
    () =>
      cameraMappings(Object.keys(policyConfig?.image_features ?? {}), isBimanual),
    [policyConfig, isBimanual],
  );

  const liveCameraByKey = useCallback(
    (key: string | null | undefined) =>
      key == null
        ? undefined
        : availableCameras.find((cam) => cameraKey(cam) === key),
    [availableCameras],
  );

  // Load the skill listing (local runs + Hub models) when the studio opens.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setModelsLoading(true);
    getModels(baseUrl, fetchWithHeaders)
      .then((m) => {
        if (!cancelled) setModels(m);
      })
      .catch(() => {
        if (!cancelled) setModels([]);
      })
      .finally(() => {
        if (!cancelled) setModelsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, baseUrl, fetchWithHeaders]);

  // Apply a "Run on robot" prefill: source "job" selects that job (+ optional
  // step); source "hub" lazy-imports the repo, then selects the pseudo-job.
  // Cleared only by the run that actually finished resolving THIS prefill — a
  // cancelled (superseded) run must not clear a newer prefill out from under
  // the run that is handling it.
  useEffect(() => {
    if (!deployPrefill) return;
    let cancelled = false;
    (async () => {
      setResolving(true);
      // "Run on robot" means the user is heading for Start — surface the
      // settings block (robot, checkpoint, cameras) even if they collapsed it.
      setSettingsOpen(true);
      try {
        if (deployPrefill.source === "job") {
          const job = await getJob(baseUrl, fetchWithHeaders, deployPrefill.id);
          if (cancelled) return;
          setSelectedStep(deployPrefill.step ?? null);
          setSelectedJob(job);
          setSelectedModelId(job.id);
        } else {
          const imported = await importSource(deployPrefill.id);
          if (cancelled || !imported) return;
          setSelectedStep(deployPrefill.step ?? null);
          setSelectedJob(imported);
          setSelectedModelId(imported.id);
        }
      } catch (e) {
        if (!cancelled) {
          toast({
            title: "Couldn't load the skill",
            description: e instanceof Error ? e.message : String(e),
            variant: "destructive",
          });
        }
      } finally {
        if (!cancelled) {
          setResolving(false);
          clearDeployPrefill();
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [
    deployPrefill,
    baseUrl,
    fetchWithHeaders,
    importSource,
    clearDeployPrefill,
    toast,
  ]);

  // Manual skill pick: resolve the chosen model to a launchable job (its own
  // registry id, an already-imported repo, or a fresh lazy import).
  const handlePickSkill = useCallback(
    async (modelId: string) => {
      setSelectedModelId(modelId);
      const model = models.find((m) => m.id === modelId);
      if (!model) return;
      // New skill → drop the prior step so the load effect picks the new job's
      // latest checkpoint.
      setSelectedStep(null);
      setResolving(true);
      try {
        const jobs = await listJobs(
          baseUrl,
          fetchWithHeaders,
          JOB_SCAN_LIMIT,
        );
        const hit = findJobForModel(model, jobs);
        if (hit) {
          setSelectedJob(hit);
          return;
        }
        const imported = await importSource(importSourceForModel(model));
        if (imported) setSelectedJob(imported);
      } catch {
        // Resolution failed → leave the prior selection; a toast already fired
        // for the import path.
      } finally {
        setResolving(false);
      }
    },
    [models, baseUrl, fetchWithHeaders, importSource],
  );

  // Load checkpoints when the selected job changes.
  useEffect(() => {
    if (!open || !jobId) {
      setCheckpoints([]);
      return;
    }
    let cancelled = false;
    listJobCheckpoints(baseUrl, fetchWithHeaders, jobId)
      .then((cks) => {
        if (cancelled) return;
        setCheckpoints(cks);
        if (cks.length > 0) {
          const latest = cks[cks.length - 1].step;
          setSelectedStep((prev) => (prev != null ? prev : latest));
        }
      })
      .catch(() => {
        if (cancelled) return;
        setCheckpoints([]);
      });
    return () => {
      cancelled = true;
    };
  }, [open, baseUrl, fetchWithHeaders, jobId]);

  // Load policy config when step changes.
  useEffect(() => {
    if (!open || !jobId || selectedStep == null) {
      setPolicyConfig(null);
      setPolicyConfigError(null);
      return;
    }
    let cancelled = false;
    setPolicyConfigLoading(true);
    setPolicyConfigError(null);
    getCheckpointPolicyConfig(baseUrl, fetchWithHeaders, jobId, selectedStep)
      .then((cfg) => {
        if (cancelled) return;
        setPolicyConfig(cfg);
        const mappings = cameraMappings(
          Object.keys(cfg.image_features),
          isBimanual,
        );
        setCameraBindings((prev) => {
          const next: Record<string, string | null> = {};
          for (const m of mappings) {
            next[m.requestKey] = prev[m.requestKey] ?? null;
          }
          return next;
        });
      })
      .catch((e) => {
        if (cancelled) return;
        setPolicyConfig(null);
        setPolicyConfigError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setPolicyConfigLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, baseUrl, fetchWithHeaders, jobId, selectedStep, isBimanual]);

  // Auto-bind robot cameras whose names match a policy-expected camera. Match
  // against the DISPLAY name (bare name), preferring stored device_id then
  // camera_index. (Ported verbatim.)
  useEffect(() => {
    if (!policyConfig) return;
    const robotCams = robot?.cameras ?? [];
    if (robotCams.length === 0 || availableCameras.length === 0) return;
    setCameraBindings((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const m of cameraMap) {
        if (next[m.requestKey] != null) continue;
        const robotCam = robotCams.find(
          (c) => c.name.toLowerCase() === m.display.toLowerCase(),
        );
        if (!robotCam) continue;
        const live = robotCam.device_id
          ? availableCameras.find((c) => c.deviceId === robotCam.device_id)
          : availableCameras.find((c) => c.index === robotCam.camera_index);
        if (live) {
          next[m.requestKey] = cameraKey(live);
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [policyConfig, robot, availableCameras, cameraMap]);

  // Drop a binding whose physical camera has vanished. (Ported verbatim.)
  useEffect(() => {
    if (!policyConfig || availableCameras.length === 0) return;
    setCameraBindings((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const [name, key] of Object.entries(prev)) {
        if (key != null && !liveCameraByKey(key)) {
          next[name] = null;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [policyConfig, availableCameras, liveCameraByKey]);

  // Poll inference status while visible so ⏹ Stop reflects a live rollout.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const s = await getInferenceStatus(baseUrl, fetchWithHeaders);
        if (!cancelled) setStatus(s);
      } catch {
        // Transient; the next tick retries.
      }
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [open, baseUrl, fetchWithHeaders]);

  const selectedRef =
    selectedStep != null
      ? checkpoints.find((c) => c.step === selectedStep)?.ref ?? null
      : null;

  // Arm-count mismatch between CHECKPOINT and ROBOT — client mirror of the
  // server's `_arm_count_mismatch` 409 guard. (Ported verbatim.)
  const SO101_DOF = 6;
  const checkpointDim =
    policyConfig?.state_dim ?? policyConfig?.action_dim ?? null;
  const checkpointArms =
    checkpointDim != null && checkpointDim % SO101_DOF === 0
      ? checkpointDim / SO101_DOF
      : null;
  const checkpointIsBimanual = checkpointArms != null && checkpointArms >= 2;
  const robotCheckpointArmMismatch =
    !!robot &&
    !!policyConfig &&
    checkpointArms != null &&
    checkpointIsBimanual !== isBimanual;

  const allCamerasBound = cameraMap.every(
    (m) => liveCameraByKey(cameraBindings[m.requestKey]) != null,
  );

  const inferenceActive = status?.inference_active === true;

  const canStart =
    !!robot &&
    robot.is_clean &&
    !robotCheckpointArmMismatch &&
    selectedRef != null &&
    !!policyConfig &&
    allCamerasBound &&
    !submitting &&
    !inferenceActive;

  const handleStart = async () => {
    if (
      !robot ||
      robotCheckpointArmMismatch ||
      selectedRef == null ||
      !policyConfig
    )
      return;
    // Drops every CameraThumbnail's browser stream so the rollout subprocess can
    // open the same camera index via OpenCV without colliding on the device.
    setSubmitting(true);
    await new Promise((r) => setTimeout(r, 300));
    // Emit camera dict keys under the DISPLAY/request name (bare in bimanual
    // mode). The rollout hands these to the BiSO left_arm_config, which
    // re-prefixes with `left_` — reconstructing the checkpoint's `left_<name>`
    // feature. Resolution comes from the checkpoint feature's dims.
    const cameraDict: Record<
      string,
      {
        type: string;
        camera_index?: number;
        width: number;
        height: number;
        fps?: number;
        fourcc?: string;
      }
    > = {};
    for (const m of cameraMap) {
      const key = cameraBindings[m.requestKey];
      if (key == null) continue;
      const live = liveCameraByKey(key);
      if (!live) continue;
      const dims = policyConfig.image_features[m.feature];
      cameraDict[m.requestKey] = {
        type: "opencv",
        camera_index: live.index,
        width: dims.width,
        height: dims.height,
        fps: DEFAULT_FPS,
      };
    }
    try {
      await startInference(baseUrl, fetchWithHeaders, {
        follower_port: robot.follower_port,
        follower_config: robot.follower_config,
        policy_ref: selectedRef,
        task,
        cameras: cameraDict,
        duration_s: durationS,
        mode: robot.mode,
        right_follower_port: robot.right_follower_port,
        right_follower_config: robot.right_follower_config,
        robot_name: robot.name,
        checkpoint_state_dim: policyConfig.state_dim ?? undefined,
      });
      // The run surfaces as the InferenceSessionDialog over this panel —
      // closing it lands back here (the studio stays open underneath).
      openInferenceSession();
      // The POST claims the inference slot synchronously, so a status fetch
      // issued now reflects THIS run — hand the released-previews / disabled-
      // Start duty from `submitting` to `inferenceActive` (kept fresh by the
      // poll). Unlike the modal this panel never unmounts, so `submitting`
      // must be cleared here or Start stays stuck on "Starting…" forever.
      try {
        setStatus(await getInferenceStatus(baseUrl, fetchWithHeaders));
      } catch {
        // The 2s poll catches up on its next tick.
      }
      setSubmitting(false);
    } catch (e) {
      toast({
        title: "Couldn't start inference",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
      // Failure: bring the previews back so the user can adjust.
      setSubmitting(false);
    }
  };

  const handleStop = async () => {
    setStopping(true);
    try {
      await stopInference(baseUrl, fetchWithHeaders);
      toast({ title: "Stopping inference", description: "The rollout is winding down." });
    } catch (e) {
      toast({
        title: "Stop failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setStopping(false);
    }
  };

  const onCameraBindingChange = (name: string, value: string) => {
    setCameraBindings((prev) => ({ ...prev, [name]: value }));
  };

  const selectedSkillLabel = selectedJob ? jobDisplayName(selectedJob) : null;

  return (
    <div className="flex flex-1 flex-col gap-5 p-5">
      <PanelHeader step="3" title="Deploy policy">
        {resolving ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
        ) : null}
      </PanelHeader>

      {/* Skill picker — the panel's entry control. Unlabeled so it sits at
          the same level as Collect's and Train's entry controls; the
          placeholder carries the meaning. ----------------------------------- */}
      <div className="space-y-2">
        <Select
          value={selectedModelId ?? undefined}
          onValueChange={handlePickSkill}
          disabled={resolving}
        >
          <SelectTrigger className="w-full">
            {selectedSkillLabel ? (
              <span className="truncate">{selectedSkillLabel}</span>
            ) : (
              <SelectValue placeholder="Pick a policy" />
            )}
          </SelectTrigger>
          <SelectContent>
            {modelsLoading ? (
              <div className="px-2 py-1.5 text-xs text-muted-foreground">
                Loading skills…
              </div>
            ) : models.length === 0 ? (
              <div className="px-2 py-1.5 text-xs text-muted-foreground">
                No trained or imported skills yet
              </div>
            ) : (
              models.map((m) => (
                <SelectItem key={m.id} value={m.id}>
                  <span className="truncate">{m.name}</span>
                  <span className="ml-2 text-[10px] uppercase tracking-wide text-muted-foreground">
                    {m.source === "hub"
                      ? "hub"
                      : m.source === "both"
                        ? "local · hub"
                        : "local"}
                  </span>
                </SelectItem>
              ))
            )}
          </SelectContent>
        </Select>
        {!selectedJob ? (
          <p className="text-xs text-muted-foreground">
            Pick a trained checkpoint or an imported Hub model to run on your
            robot.
          </p>
        ) : null}
      </div>

      {/* Settings & configuration — robot, checkpoint, run parameters and
          cameras collapse as one block. ------------------------------------ */}
      <Collapsible
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        className="group space-y-5"
      >
        <CollapsibleTrigger className="flex w-full items-center justify-between border-b border-border pb-2 text-sm font-semibold text-foreground">
          <span>Settings &amp; configuration</span>
          <ChevronDown className="h-4 w-4 transition-transform group-data-[state=open]:rotate-180" />
        </CollapsibleTrigger>
        <CollapsibleContent className={SLIDE}>
          <div className="space-y-5">
          {/* Runs-on row ------------------------------------------------------ */}
          <div className="space-y-2">
            <h3 className="eyebrow">Runs on</h3>
            {!robot ? (
              <Alert className="border-warn/40 text-warn [&>svg]:text-warn">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  Select a robot in the corner to deploy.
                </AlertDescription>
              </Alert>
            ) : !robot.is_clean ? (
              <Alert className="border-warn/40 text-warn [&>svg]:text-warn">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  <strong>{robot.name}</strong> {robotSetupGap(robot)}. Open
                  Robot settings before running inference.
                </AlertDescription>
              </Alert>
            ) : (
              <div className="flex items-center gap-2 text-sm">
                <CheckCircle className="h-4 w-4 text-ok" />
                <span className="text-foreground">{robot.name}</span>
                <span className="rounded border border-border px-1.5 py-0.5 text-[11px] text-muted-foreground">
                  {isBimanual ? "bimanual — both followers" : "single arm"}
                </span>
              </div>
            )}
          </div>

          {/* Checkpoint ------------------------------------------------------- */}
          {selectedJob ? (
            <div className="space-y-2">
              <h3 className="eyebrow">Checkpoint</h3>
              {checkpoints.length === 0 ? (
                <Alert className="border-warn/40 text-warn [&>svg]:text-warn">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    No checkpoints available for this skill yet.
                  </AlertDescription>
                </Alert>
              ) : (
                <CheckpointDropdown
                  checkpoints={checkpoints}
                  selectedStep={selectedStep}
                  onChange={setSelectedStep}
                />
              )}
              {robotCheckpointArmMismatch ? (
                <Alert className="border-warn/40 text-warn [&>svg]:text-warn">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    {checkpointIsBimanual ? (
                      <>
                        This checkpoint was trained on a{" "}
                        <strong>bimanual robot</strong> ({checkpointDim}-dim state,{" "}
                        {checkpointArms} arms), but <strong>{robot?.name}</strong> is
                        a single-arm robot. Pick a single-arm checkpoint, or select a
                        bimanual robot in the corner.
                      </>
                    ) : (
                      <>
                        This checkpoint was trained on a{" "}
                        <strong>single-arm robot</strong> ({checkpointDim}-dim
                        state), but <strong>{robot?.name}</strong> is a bimanual
                        robot. Pick a bimanual checkpoint, or select a single-arm
                        robot in the corner.
                      </>
                    )}
                  </AlertDescription>
                </Alert>
              ) : null}
            </div>
          ) : null}

          {/* Run parameters --------------------------------------------------- */}
          {selectedJob && policyConfig ? (
            <div className="space-y-3">
              <h3 className="eyebrow">Run parameters</h3>
              {policyConfig.requires_task ? (
                <div className="space-y-1.5">
                  <Label htmlFor="deploy-task" className="text-sm font-medium">
                    Task description
                  </Label>
                  <Input
                    id="deploy-task"
                    value={task}
                    onChange={(e) => setTask(e.target.value)}
                    placeholder="e.g., pick up the red block"
                  />
                  <p className="text-xs text-muted-foreground">
                    This policy is language-conditioned ({policyConfig.policy_type}).
                  </p>
                </div>
              ) : null}
              <div className="space-y-1.5">
                <Label htmlFor="deploy-duration" className="text-sm font-medium">
                  Max duration (seconds)
                </Label>
                <NumberInput
                  id="deploy-duration"
                  min={1}
                  value={durationS}
                  onChange={(v) => {
                    if (v !== undefined) setDurationS(v);
                  }}
                />
              </div>
            </div>
          ) : null}

          {/* Cameras ---------------------------------------------------------- */}
          {selectedJob ? (
            <div className="space-y-2">
              <h3 className="eyebrow">Cameras</h3>
              {policyConfigLoading ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Reading policy config…
                </div>
              ) : policyConfigError ? (
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    Couldn't load policy config: {policyConfigError}
                  </AlertDescription>
                </Alert>
              ) : !policyConfig ? null : cameraMap.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  This policy doesn't use cameras.
                </p>
              ) : (
                <div className="space-y-3">
                  <p className="text-xs text-muted-foreground">
                    Bind a physical camera to each name the policy was trained with.
                    Resolution comes from the checkpoint.
                  </p>
                  {cameraMap.map((m) => {
                    const dims = policyConfig.image_features[m.feature];
                    const value = cameraBindings[m.requestKey];
                    const selectedCamera = liveCameraByKey(value);
                    return (
                      <div key={m.requestKey} className="flex items-center gap-3">
                        <div className="flex-1">
                          <Label className="text-sm font-medium">{m.display}</Label>
                          <p className="text-xs text-muted-foreground">
                            {dims.width}×{dims.height}
                          </p>
                        </div>
                        <Select
                          value={value ?? undefined}
                          onValueChange={(v) => onCameraBindingChange(m.requestKey, v)}
                        >
                          <SelectTrigger className="w-52">
                            <SelectValue placeholder="Select a camera" />
                          </SelectTrigger>
                          <SelectContent>
                            {availableCameras.length === 0 ? (
                              <div className="px-2 py-1.5 text-xs text-muted-foreground">
                                No cameras detected
                              </div>
                            ) : (
                              availableCameras.map((cam) => (
                                <SelectItem
                                  key={cameraKey(cam)}
                                  value={cameraKey(cam)}
                                >
                                  #{cam.index} — {cam.name}
                                </SelectItem>
                              ))
                            )}
                          </SelectContent>
                        </Select>
                        <CameraThumbnail
                          deviceId={selectedCamera?.deviceId ?? ""}
                          paused={submitting || inferenceActive}
                        />
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          ) : null}
          </div>
        </CollapsibleContent>
      </Collapsible>

      {/* Actions — pinned directly above the model library. Side by side so
          the row sits level with Collect's and Train's single Start. -------- */}
      <div className="mt-auto flex gap-2 pt-2">
        <Button
          onClick={handleStart}
          disabled={!canStart}
          className="flex-1 gap-2"
        >
          <Play className="h-4 w-4" />
          {submitting ? "Starting…" : "Start inference"}
        </Button>
        <Button
          onClick={handleStop}
          variant="outline"
          disabled={!inferenceActive || stopping}
          className="flex-1 gap-2"
        >
          <Square className="h-4 w-4" />
          {stopping ? "Stopping…" : "Stop inference"}
        </Button>
      </div>

      {/* Model / policy library — imported models + uploaded Hub repos.
          Picking a card selects it as the skill above (step null → the
          checkpoint loader falls back to the latest). mt-0 keeps it glued to
          the actions block above, which carries the panel's mt-auto. */}
      <LibrarySection className="mt-0">
        <ModelsLibrary
          onPick={(job, step) => {
            setSelectedStep(step);
            setSelectedJob(job);
            setSelectedModelId(job.id);
          }}
        />
      </LibrarySection>
    </div>
  );
};

export default DeployPanel;
