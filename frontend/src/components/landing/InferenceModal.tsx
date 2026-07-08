import React, { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AlertTriangle, CheckCircle, Loader2, Play } from "lucide-react";
import { RobotRecord } from "@/hooks/useRobots";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useNavigate } from "react-router-dom";
import {
  JobCheckpoint,
  PolicyConfigSummary,
  getCheckpointPolicyConfig,
  listJobCheckpoints,
} from "@/lib/checkpointsApi";
import { startInference } from "@/lib/inferenceApi";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import { useAvailableCameras } from "@/hooks/useAvailableCameras";
import CameraTile from "@/components/CameraTile";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  robot: RobotRecord | null;
  jobId: string;
  initialStep: number | null;
}

const DEFAULT_FPS = 30;

/**
 * One camera as the modal sees it. The BiSO prefix round-trip lives here so the
 * future per-arm routing work has a single obvious place to extend.
 *
 * `feature` is the checkpoint's camera key exactly as it comes back from
 * `get_policy_config_summary` (the suffix after `observation.images.`). For a
 * bimanual checkpoint recorded through lelab, lerobot's BiSOFollower parks every
 * camera on the LEFT arm and auto-prefixes each feature with `left_` when it
 * writes the dataset — so a camera the user named `front` at record time becomes
 * the feature `left_front`.
 *
 * Inference mirrors that: the rollout hands the request's camera dict to
 * `--robot.left_arm_config.cameras`, and BiSO re-prefixes with `left_` at
 * runtime. So the modal must bind + send under the BARE name (`front`), which
 * the rollout re-prefixes back to `left_front` — matching the checkpoint. If we
 * bound/sent under the literal `left_front` we'd emit `left_left_front` (double
 * prefix → policy mismatch).
 *
 * `display` / `requestKey` are therefore the stripped bare name in bimanual
 * mode, and identical to `feature` in single-arm mode (single-arm checkpoints
 * already carry bare names — a camera legitimately named e.g. `left_side` must
 * never be mangled). We only strip when it's unambiguous: see `cameraMappings`.
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
 * Guard against collisions: if two features would strip to the same bare name
 * (a checkpoint carrying BOTH `left_x` and `right_x`, or a bare `x` alongside
 * `left_x`), fall back to the FULL feature name for every colliding entry —
 * correctness over cosmetics. lelab can't produce `right_*` checkpoints today,
 * so this is a defensive branch for externally-recorded bimanual checkpoints;
 * it must not silently mis-bind them.
 */
function cameraMappings(
  features: string[],
  isBimanual: boolean,
): CameraMapping[] {
  if (!isBimanual) {
    return features.map((f) => ({ feature: f, display: f, requestKey: f }));
  }
  // Count how many features want each stripped bare name so we can detect
  // collisions before committing to the shortened form.
  const strippedCounts = new Map<string, number>();
  for (const f of features) {
    const bare = f.replace(ARM_PREFIX_RE, "");
    strippedCounts.set(bare, (strippedCounts.get(bare) ?? 0) + 1);
  }
  return features.map((f) => {
    const bare = f.replace(ARM_PREFIX_RE, "");
    // Strip only when the bare name is unique across all features; otherwise
    // keep the full feature name so the two colliding cameras stay distinct.
    const name = strippedCounts.get(bare) === 1 ? bare : f;
    return { feature: f, display: name, requestKey: name };
  });
}

const InferenceModal: React.FC<Props> = ({
  open,
  onOpenChange,
  robot,
  jobId,
  initialStep,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const navigate = useNavigate();

  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(initialStep);
  const [task, setTask] = useState("");
  const [durationS, setDurationS] = useState(60);
  const [submitting, setSubmitting] = useState(false);

  const [policyConfig, setPolicyConfig] = useState<PolicyConfigSummary | null>(null);
  const [policyConfigLoading, setPolicyConfigLoading] = useState(false);
  const [policyConfigError, setPolicyConfigError] = useState<string | null>(null);

  // Per camera DISPLAY name → user-selected physical camera index (or null).
  // Keyed by the stripped display name (== requestKey), not the checkpoint
  // feature key — see `cameraMappings` / the CameraMapping doc for the round-trip.
  const [cameraBindings, setCameraBindings] = useState<Record<string, number | null>>({});
  const { cameras: availableCameras } = useAvailableCameras({ enabled: open });

  // `lerobot-rollout` drives any Robot generically, including `bi_so_follower`,
  // so a bimanual record now runs inference on BOTH followers — the server
  // stages the two follower calibrations and builds a `bi_so_follower` command.
  // We no longer block bimanual robots here.
  const isBimanual = robot?.mode === "bimanual";

  // Checkpoint feature ↔ display/request-key mapping. Bimanual checkpoints carry
  // BiSO's `left_`-prefixed camera features; the modal shows + binds + sends the
  // bare name so the rollout re-prefixes it back to match the checkpoint.
  const cameraMap = React.useMemo(
    () => cameraMappings(Object.keys(policyConfig?.image_features ?? {}), isBimanual),
    [policyConfig, isBimanual],
  );

  // Load checkpoints when modal opens.
  useEffect(() => {
    if (!open) return;
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
    if (!open || selectedStep == null) {
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
        // Reset camera bindings to one entry per DISPLAY name (bare name in
        // bimanual mode). Preserve any prior selection that's still relevant.
        const mappings = cameraMappings(Object.keys(cfg.image_features), isBimanual);
        setCameraBindings((prev) => {
          const next: Record<string, number | null> = {};
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

  // If the selected robot has cameras whose names match a policy-expected
  // camera, auto-bind them. Match against the DISPLAY name (the bare name the
  // user chose at record time — that's what the robot record stores), not the
  // `left_`-prefixed checkpoint feature. Prefer matching by browser device_id
  // (stable across cv2 index drift); fall back to the saved camera_index.
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
        const live =
          (robotCam.device_id &&
            availableCameras.find((c) => c.deviceId === robotCam.device_id)) ||
          availableCameras.find((c) => c.index === robotCam.camera_index);
        if (live) {
          next[m.requestKey] = live.index;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [policyConfig, robot, availableCameras, cameraMap]);

  const selectedRef =
    selectedStep != null
      ? checkpoints.find((c) => c.step === selectedStep)?.ref ?? null
      : null;

  // Arm-count mismatch between the CHECKPOINT and the selected ROBOT. A
  // bimanual-trained SO-101 checkpoint carries a 12-dim state/action (two 6-DOF
  // arms) and left_/right_-prefixed camera names; a single-arm checkpoint is
  // 6-dim. Running a policy on the wrong arm count crashes on a shape mismatch
  // deep in the rollout subprocess. Detect it here from the checkpoint's state
  // dim (fall back to action dim) and explain it before Start. This is the
  // client mirror of the server's `_arm_count_mismatch` 409 guard — we forward
  // `checkpoint_state_dim` so the server enforces the same rule authoritatively.
  const SO101_DOF = 6;
  const checkpointDim = policyConfig?.state_dim ?? policyConfig?.action_dim ?? null;
  const checkpointArms =
    checkpointDim != null && checkpointDim % SO101_DOF === 0
      ? checkpointDim / SO101_DOF
      : null;
  const checkpointIsBimanual = checkpointArms != null && checkpointArms >= 2;
  // Flag both directions: a bimanual checkpoint on a single-arm robot, AND a
  // single-arm checkpoint on a bimanual robot. Only assert a mismatch when the
  // checkpoint exposes a recognisable arm count (checkpointArms != null) — a
  // vision-only checkpoint with no state dim can't be judged here, so we let
  // the server's post-mortem shape check speak instead of guessing.
  const robotCheckpointArmMismatch =
    !!robot &&
    !!policyConfig &&
    checkpointArms != null &&
    checkpointIsBimanual !== isBimanual;

  const allCamerasBound = cameraMap.every(
    (m) => cameraBindings[m.requestKey] != null,
  );

  const canStart =
    !!robot &&
    robot.is_clean &&
    !robotCheckpointArmMismatch &&
    selectedRef != null &&
    !!policyConfig &&
    allCamerasBound &&
    !submitting;

  const handleStart = async () => {
    if (
      !robot ||
      robotCheckpointArmMismatch ||
      selectedRef == null ||
      !policyConfig
    )
      return;
    // Setting submitting=true makes every CameraPreview drop its
    // browser stream — required so the rollout subprocess can open the
    // same camera index via OpenCV without colliding on the device.
    setSubmitting(true);
    await new Promise((r) => setTimeout(r, 300));
    // Emit camera dict keys under the DISPLAY/request name (bare in bimanual
    // mode). The rollout hands these to the BiSO left_arm_config, which
    // re-prefixes with `left_` — reconstructing the checkpoint's `left_<name>`
    // feature. Resolution still comes from the checkpoint feature's dims.
    const cameraDict: Record<string, {
      type: string; camera_index?: number; width: number; height: number; fps?: number;
    }> = {};
    for (const m of cameraMap) {
      const idx = cameraBindings[m.requestKey];
      if (idx == null) continue;
      const dims = policyConfig.image_features[m.feature];
      cameraDict[m.requestKey] = {
        type: "opencv",
        camera_index: idx,
        width: dims.width,
        height: dims.height,
        fps: DEFAULT_FPS,
      };
    }
    try {
      // The POST now returns immediately (it only validates cheaply, then the
      // server downloads the model + preflights the arm in the background), so
      // this navigates to the inference page right away — the download and its
      // progress, any warn-but-allow arm finding, and any failure all surface
      // there via /inference-status polling.
      await startInference(baseUrl, fetchWithHeaders, {
        follower_port: robot.follower_port,
        follower_config: robot.follower_config,
        policy_ref: selectedRef,
        task,
        cameras: cameraDict,
        duration_s: durationS,
        // Follower torque limit for the session (10-100% of full power).
        motor_power: robot.motor_power ?? 100,
        // Bimanual: forward the mode + right-arm follower so the server builds a
        // `bi_so_follower` command staging both follower calibrations. In single
        // mode the right_* fields are inert (mode defaults to "single"
        // server-side). robot_name is the BiSO staging base id.
        mode: robot.mode,
        right_follower_port: robot.right_follower_port,
        right_follower_config: robot.right_follower_config,
        robot_name: robot.name,
        // Forward the checkpoint's flat state width so the server enforces the
        // same arm-count guard authoritatively (null when the checkpoint omits
        // observation.state — the server then defers to its shape check).
        checkpoint_state_dim: policyConfig.state_dim ?? undefined,
      });
      onOpenChange(false);
      navigate("/inference");
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

  const onCameraBindingChange = (name: string, value: string) => {
    const idx = Number(value);
    setCameraBindings((prev) => ({ ...prev, [name]: idx }));
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-gray-900 border-gray-800 text-white sm:max-w-[600px] p-8 max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex justify-center items-center mb-4">
            <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
              <Play className="w-4 h-4 text-white" />
            </div>
          </div>
          <DialogTitle className="text-white text-center text-2xl font-bold">
            Configure Inference
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6 py-4">
          <DialogDescription className="text-gray-400 text-base leading-relaxed text-center">
            Pick a checkpoint and confirm hardware. The selected policy will
            drive the follower autonomously for the configured duration.
          </DialogDescription>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">
              Robot Configuration
            </h3>
            {!robot ? (
              <Alert className="bg-amber-900/40 border-amber-700 text-amber-100">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  Select and configure a robot on the Landing page first.
                </AlertDescription>
              </Alert>
            ) : !robot.is_clean ? (
              <Alert className="bg-amber-900/40 border-amber-700 text-amber-100">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  <strong>{robot.name}</strong> is missing a calibration.
                  Configure it before running inference.
                </AlertDescription>
              </Alert>
            ) : (
              <div className="flex items-center gap-2 text-sm">
                <CheckCircle className="w-4 h-4 text-green-400" />
                <span className="text-slate-200">
                  Running on <strong>{robot.name}</strong>
                  {isBimanual ? " (bimanual — both followers)" : ""}
                </span>
              </div>
            )}
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">
              Checkpoint
            </h3>
            {checkpoints.length === 0 ? (
              <Alert className="bg-amber-900/40 border-amber-700 text-amber-100">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  No checkpoints available for this job yet.
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
              <Alert className="bg-amber-900/40 border-amber-700 text-amber-100">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  {checkpointIsBimanual ? (
                    <>
                      This checkpoint was trained on a{" "}
                      <strong>bimanual robot</strong> ({checkpointDim}-dim state,{" "}
                      {checkpointArms} arms), but <strong>{robot?.name}</strong>{" "}
                      is a single-arm robot. Pick a single-arm checkpoint, or
                      select a bimanual robot on the Landing page.
                    </>
                  ) : (
                    <>
                      This checkpoint was trained on a{" "}
                      <strong>single-arm robot</strong> ({checkpointDim}-dim
                      state), but <strong>{robot?.name}</strong> is a bimanual
                      robot. Pick a bimanual checkpoint, or select a single-arm
                      robot on the Landing page.
                    </>
                  )}
                </AlertDescription>
              </Alert>
            ) : null}
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">
              Run parameters
            </h3>
            {policyConfig?.requires_task ? (
              <div className="space-y-2">
                <Label htmlFor="task" className="text-sm font-medium text-gray-300">
                  Task description
                </Label>
                <Input
                  id="task"
                  value={task}
                  onChange={(e) => setTask(e.target.value)}
                  placeholder="e.g., pick up the red block"
                  className="bg-gray-800 border-gray-700 text-white"
                />
                <p className="text-xs text-gray-500">
                  This policy is language-conditioned ({policyConfig.policy_type}).
                </p>
              </div>
            ) : null}
            <div className="space-y-2">
              <Label htmlFor="durationS" className="text-sm font-medium text-gray-300">
                Max duration (seconds)
              </Label>
              <NumberInput
                id="durationS"
                min={1}
                value={durationS}
                onChange={(v) => {
                  if (v !== undefined) setDurationS(v);
                }}
                className="bg-gray-800 border-gray-700 text-white"
              />
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">
              Cameras
            </h3>
            {policyConfigLoading ? (
              <div className="flex items-center gap-2 text-sm text-slate-400">
                <Loader2 className="w-4 h-4 animate-spin" />
                Reading policy config…
              </div>
            ) : policyConfigError ? (
              <Alert className="bg-red-900/40 border-red-700 text-red-100">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  Couldn't load policy config: {policyConfigError}
                </AlertDescription>
              </Alert>
            ) : !policyConfig ? null : cameraMap.length === 0 ? (
              <p className="text-xs text-gray-500">
                This policy doesn't use cameras.
              </p>
            ) : (
              <div className="space-y-3">
                <p className="text-xs text-gray-500">
                  Bind a physical camera to each name the policy was trained
                  with. Resolution comes from the checkpoint.
                </p>
                {cameraMap.map((m) => {
                  const dims = policyConfig.image_features[m.feature];
                  const value = cameraBindings[m.requestKey];
                  return (
                    <div key={m.requestKey} className="flex items-center gap-3">
                      <div className="flex-1">
                        <Label className="text-sm font-medium text-gray-200">
                          {m.display}
                        </Label>
                        <p className="text-xs text-gray-500">
                          {dims.width}×{dims.height}
                        </p>
                      </div>
                      <Select
                        value={value != null ? String(value) : undefined}
                        onValueChange={(v) => onCameraBindingChange(m.requestKey, v)}
                      >
                        <SelectTrigger className="bg-gray-800 border-gray-700 text-white w-56">
                          <SelectValue placeholder="Select a camera" />
                        </SelectTrigger>
                        <SelectContent className="bg-gray-900 border-gray-700 text-white">
                          {availableCameras.length === 0 ? (
                            <div className="px-2 py-1.5 text-xs text-gray-500">
                              No cameras detected
                            </div>
                          ) : (
                            availableCameras.map((cam) => (
                              <SelectItem
                                key={cam.index}
                                value={String(cam.index)}
                              >
                                #{cam.index} — {cam.name}
                              </SelectItem>
                            ))
                          )}
                        </SelectContent>
                      </Select>
                      <CameraTile
                        size="sm"
                        cameraIndex={value ?? undefined}
                        paused={submitting}
                        emptyLabel="No preview"
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <Button
              onClick={handleStart}
              disabled={!canStart}
              className="w-full sm:w-auto bg-green-500 hover:bg-green-600 text-white px-10 py-6 text-lg disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Play className="w-5 h-5 mr-2" />
              {submitting ? "Starting…" : "Start Inference"}
            </Button>
            <Button
              onClick={() => onOpenChange(false)}
              variant="outline"
              className="w-full sm:w-auto border-gray-500 hover:border-gray-200 px-10 py-6 text-lg text-zinc-500 bg-zinc-900 hover:bg-zinc-800"
            >
              Cancel
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default InferenceModal;
