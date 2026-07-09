import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  ArrowLeft,
  Settings,
  Activity,
  CheckCircle,
  XCircle,
  AlertCircle,
  AlertTriangle,
  Loader2,
  Play,
  Square,
  Circle,
  Camera,
  ShieldQuestion,
  Hand,
  RefreshCw,
  Wand2,
  Trash2,
  FolderOpen,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import Logo from "@/components/Logo";
import { useApi } from "@/contexts/ApiContext";
import { useSessionExitGuard } from "@/hooks/useSessionExitGuard";
import { isMotorRangeComplete } from "@/lib/calibrationTargets";
import CameraConfiguration, {
  CameraConfig,
} from "@/components/recording/CameraConfiguration";
import CalibrationLibrary from "@/components/calibration/CalibrationLibrary";
import { RobotRecord } from "@/hooks/useRobots";

const DISCONTINUITY_ERROR_PREFIX = "Motor discontinuity detected";

interface CalibrationStatus {
  calibration_active: boolean;
  status: string; // "idle", "connecting", "recording", "completed", "error", "stopping"
  device_type: string | null;
  error: string | null;
  message: string;
  step: number;
  total_steps: number;
  current_positions: Record<string, number> | null;
  recorded_ranges: Record<
    string,
    { min: number; max: number; current: number }
  > | null;
}

interface CalibrationRequest {
  device_type: string; // "robot" or "teleop"
  port: string;
  config_file: string;
  robot_name: string | null;
  overwrite?: boolean; // must be true to replace an existing config of the same name
  arm?: "left" | "right"; // which arm of a bimanual robot ("left" = the single pair)
}

// One selectable arm slot in the multi-arm auto-calibration picker. `key`
// uniquely identifies the (device_type, arm) slot; cfgField/portField map it to
// the robot record's fields so the slot can prefill its name + port.
interface ArmSlot {
  key: string;
  label: string;
  device: "teleop" | "robot";
  arm: "left" | "right";
  cfgField: keyof RobotRecord;
  portField: keyof RobotRecord;
}

// Per-arm terminal/running state in a concurrent batch (from the backend).
interface BatchArmStatus {
  name: string;
  port: string;
  device_type: string;
  arm: string;
  status: string; // running | completed | failed | stopped | stopping | idle
  error: string | null;
  logs: string[];
}

interface BatchAutoCalStatus {
  active: boolean;
  arms: BatchArmStatus[];
  total: number;
  completed: number;
  failed: number;
  logs: string[];
}

const Calibration = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const robotName =
    (location.state as { robot_name?: string } | null)?.robot_name ?? null;
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();

  const consoleRef = useRef<HTMLDivElement>(null);
  const demoVideoRef = useRef<HTMLDivElement>(null);

  const [deviceType, setDeviceType] = useState<string>("teleop");
  const [arm, setArm] = useState<"left" | "right">("left");
  const [port, setPort] = useState<string>("");
  // `robot` is the last-fetched SERVER baseline. Client-initiated config edits on
  // this page (ports, cameras, motor power) NEVER write straight to the record;
  // they accumulate in local draft state and are persisted only when the user
  // presses Save (a single batched POST). Dirtiness is the draft-vs-baseline diff.
  const [robot, setRobot] = useState<RobotRecord | null>(null);
  // Draft overlay for the four port slots. A field present here (including "" for
  // a cleared slot) overrides the baseline until Save. `draftPort` reads through
  // the overlay so every port-derived value (dropdown, conflict checks, batch
  // slots) reflects unsaved edits.
  const [portDraft, setPortDraft] = useState<
    Partial<Record<keyof RobotRecord, string>>
  >({});
  const draftPort = useCallback(
    (field: keyof RobotRecord): string =>
      portDraft[field] ?? ((robot?.[field] as string) || ""),
    [portDraft, robot],
  );
  const [saving, setSaving] = useState(false);
  const [quitPromptOpen, setQuitPromptOpen] = useState(false);

  const isBimanual = robot?.mode === "bimanual";
  // In single (or left) mode the primary leader/follower fields are used; in
  // bimanual mode the right arm uses the right_* fields. Maps the current
  // device_type + arm to the record's port and config field names.
  const isRight = arm === "right";
  const portField = (
    deviceType === "teleop"
      ? isRight
        ? "right_leader_port"
        : "leader_port"
      : isRight
        ? "right_follower_port"
        : "follower_port"
  ) as keyof RobotRecord;
  const configField = (
    deviceType === "teleop"
      ? isRight
        ? "right_leader_config"
        : "leader_config"
      : isRight
        ? "right_follower_config"
        : "follower_config"
  ) as keyof RobotRecord;

  const assignedConfig = robot ? (robot[configField] as string) : "";
  // Calibration names are arbitrary in every mode — bimanual no longer forces
  // "<robot>_<arm>" (lerobot's "<base>_left/right" convention is satisfied by a
  // per-session staging copy on the backend, not by the on-disk name). Default
  // to the in-use config for this slot, else a per-arm suggestion so a fresh
  // bimanual robot doesn't propose the same name for all four slots.
  const defaultConfigName = assignedConfig?.trim()
    ? assignedConfig
    : ((isBimanual ? `${robotName}_${arm}` : robotName) ?? "");

  // No name is chosen in the UI. Calibration always saves to the robot's own
  // default config name for this slot and silently replaces it (see overwrite
  // below). To keep an old calibration under a different name, the user renames
  // it afterward via the existing per-side rename feature.
  const calibrationConfigName = defaultConfigName;

  // Bumped when a calibration completes so the per-side CalibrationLibrary
  // dropdowns re-fetch and surface any newly-named file.
  const [calibReloadToken, setCalibReloadToken] = useState(0);

  // Ports already assigned to the OTHER arms of this robot — each physical arm
  // needs its own serial port, so these are greyed out in the dropdown. The
  // right-arm ports only count in bimanual mode (mirrors the backend guard), so
  // a single-arm robot's stale right_* ports don't get shown as taken.
  const portFields =
    robot?.mode === "bimanual"
      ? ([
          "leader_port",
          "follower_port",
          "right_leader_port",
          "right_follower_port",
        ] as const)
      : (["leader_port", "follower_port"] as const);
  const otherArmPorts = robot
    ? portFields
        .filter((f) => f !== portField)
        .map((f) => draftPort(f))
        .filter(Boolean)
    : [];

  // Human-readable name for a port slot, matching the labels the "Robot
  // calibration" checklist renders. Bimanual distinguishes left/right; single
  // mode has just Leader/Follower. Used by Detect's reassign toast to name the
  // slot whose port it just took over.
  const portFieldLabel = (field: keyof RobotRecord): string => {
    switch (field) {
      case "leader_port":
        return isBimanual ? "Left Leader" : "Leader";
      case "follower_port":
        return isBimanual ? "Left Follower" : "Follower";
      case "right_leader_port":
        return "Right Leader";
      case "right_follower_port":
        return "Right Follower";
      default:
        return String(field);
    }
  };
  const [wiggling, setWiggling] = useState(false);
  // Touch-to-identify: watching every port for a hand-moved shoulder-pan swing.
  const [detecting, setDetecting] = useState(false);
  // Picking a port that's in use by another arm (via the dropdown OR Detect)
  // stages the assignment here and opens a confirmation dialog instead of
  // applying immediately. Two shapes, distinguished by `source`:
  //  - When the OTHER slot holds this port and THIS slot already had a port,
  //    confirming SWAPS: the other slot receives this slot's old port, so no
  //    slot ends up empty. `swapPort` carries the old port for the message and
  //    the patch.
  //  - When this slot had no port, the swap degenerates to a take-with-warning:
  //    the other slot is left empty. `swapPort` is null in that case.
  // `releasedField`/`releasedLabel` are null when the port isn't in use at all
  // (plain Detect assign) — then confirming is just a straight assignment.
  const [portAssignPrompt, setPortAssignPrompt] = useState<{
    source: "detect" | "manual";
    port: string;
    message: string;
    targetLabel: string;
    releasedField: keyof RobotRecord | null;
    releasedLabel: string | null;
    swapPort: string | null;
  } | null>(null);
  // --- Concurrent multi-arm auto-calibration ---
  // Auto-calibration IS the multi-arm flow: the main "Auto-calibrate" button
  // opens this picker, where the user ticks 1-4 arm slots and every one's
  // hands-off auto-cal subprocess runs at the SAME TIME, each on its own port.
  // The manual step-by-step flow is untouched and stays available separately.
  const [batchAutoCalOpen, setBatchAutoCalOpen] = useState(false);
  const [batchAutoCalPromptOpen, setBatchAutoCalPromptOpen] = useState(false);
  // Which arm slots are ticked. Each slot's port comes straight from its
  // assignment on the robot record; each slot's save name is the robot's own
  // default config for that slot (no per-arm name input).
  const [batchSelected, setBatchSelected] = useState<Record<string, boolean>>(
    {},
  );
  const [batchAutoCal, setBatchAutoCal] = useState<BatchAutoCalStatus>({
    active: false,
    arms: [],
    total: 0,
    completed: 0,
    failed: 0,
    logs: [],
  });
  const [availablePorts, setAvailablePorts] = useState<string[]>([]);
  const [portsLoading, setPortsLoading] = useState(false);
  const [cameras, setCameras] = useState<CameraConfig[]>([]);
  const releaseStreamsRef = useRef<(() => void) | null>(null);
  // Off by default so merely opening the calibration page never grabs a camera.
  // The user explicitly starts a scan, which is when cameras are turned on,
  // enumerated, and the browser permission prompt is requested.
  const [camerasActive, setCamerasActive] = useState(false);

  const handleCamerasActiveChange = (active: boolean) => {
    if (!active) {
      releaseStreamsRef.current?.();
    }
    setCamerasActive(active);
  };

  useEffect(() => {
    return () => {
      releaseStreamsRef.current?.();
    };
  }, []);

  // Arm slots the multi-arm auto-cal picker can offer. Bimanual exposes all
  // four (left/right × leader/follower); single-arm exposes the leader +
  // follower pair. Each maps to the record's config/port fields for prefill.
  const armSlots: ArmSlot[] = useMemo(
    () =>
      isBimanual
        ? [
            {
              key: "teleop:left",
              label: "Left Leader",
              device: "teleop",
              arm: "left",
              cfgField: "leader_config",
              portField: "leader_port",
            },
            {
              key: "robot:left",
              label: "Left Follower",
              device: "robot",
              arm: "left",
              cfgField: "follower_config",
              portField: "follower_port",
            },
            {
              key: "teleop:right",
              label: "Right Leader",
              device: "teleop",
              arm: "right",
              cfgField: "right_leader_config",
              portField: "right_leader_port",
            },
            {
              key: "robot:right",
              label: "Right Follower",
              device: "robot",
              arm: "right",
              cfgField: "right_follower_config",
              portField: "right_follower_port",
            },
          ]
        : [
            {
              key: "teleop:left",
              label: "Leader",
              device: "teleop",
              arm: "left",
              cfgField: "leader_config",
              portField: "leader_port",
            },
            {
              key: "robot:left",
              label: "Follower",
              device: "robot",
              arm: "left",
              cfgField: "follower_config",
              portField: "follower_port",
            },
          ],
    [isBimanual],
  );

  const fetchRobot = useCallback(async (): Promise<RobotRecord | null> => {
    if (!robotName) return null;
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/robots/${encodeURIComponent(robotName)}`,
      );
      if (!res.ok) return null;
      const data = await res.json();
      const r = (data.robot as RobotRecord | null) ?? null;
      setRobot(r);
      return r;
    } catch (e) {
      console.error("Failed to load robot record:", e);
      return null;
    }
  }, [robotName, baseUrl, fetchWithHeaders]);

  // Open the side's calibration folder in the OS file browser (Finder/Explorer/
  // xdg-open). A local, non-network action handled server-side; the dir is
  // created there if missing so a fresh install still opens an empty folder.
  const openCalibrationFolder = useCallback(
    async (device: "teleop" | "robot") => {
      try {
        const res = await fetchWithHeaders(`${baseUrl}/open-calibration-folder`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ device_type: device }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok || !data.opened) {
          toast({
            title: "Couldn't open folder",
            description: data.message,
            variant: "destructive",
          });
        }
      } catch (e) {
        toast({
          title: "Couldn't open folder",
          description: String(e),
          variant: "destructive",
        });
      }
    },
    [baseUrl, fetchWithHeaders, toast],
  );

  // List the USB-serial ports for the dropdown (filtered to arm-like devices by
  // the backend). Refreshable so plugging in an arm and rescanning works.
  const fetchPorts = useCallback(async () => {
    setPortsLoading(true);
    try {
      const res = await fetchWithHeaders(`${baseUrl}/available-ports`);
      const data = await res.json();
      setAvailablePorts(Array.isArray(data.ports) ? data.ports : []);
    } catch (e) {
      console.error("Failed to list ports:", e);
    } finally {
      setPortsLoading(false);
    }
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    fetchPorts();
  }, [fetchPorts]);

  // Initial fetch + form prefill on arrival.
  useEffect(() => {
    if (!robotName) return;
    let cancelled = false;
    (async () => {
      const r = await fetchRobot();
      if (!r || cancelled) return;
      // Default to the first incomplete side in the checklist (leader, then follower).
      const defaultDevice = !r.leader_config
        ? "teleop"
        : !r.follower_config
          ? "robot"
          : "teleop";
      setDeviceType(defaultDevice);
      setPort(
        defaultDevice === "teleop"
          ? r.leader_port || ""
          : r.follower_port || "",
      );
      setCameras(r.cameras ?? []);
    })();
    return () => {
      cancelled = true;
    };
  }, [robotName, fetchRobot]);

  // Camera edits (adds/removes/edits AND CameraConfiguration's automatic
  // resync corrections) update the local draft only. Nothing is written to the
  // robot record until Save. `cameras` is the draft; `robot.cameras` the baseline.
  const handleCamerasChange = (next: CameraConfig[]) => {
    setCameras(next);
  };

  const [calibrationStatus, setCalibrationStatus] = useState<CalibrationStatus>(
    {
      calibration_active: false,
      status: "idle",
      device_type: null,
      error: null,
      message: "",
      step: 0,
      total_steps: 1,
      current_positions: null,
      recorded_ranges: null,
    },
  );
  const [isPolling, setIsPolling] = useState(false);

  // Manual (step-by-step) calibration liveness, in state so the shared exit
  // guard below can react to it. Set optimistically at start (so a leave in the
  // sub-second before the first status poll still aborts) and cleared when the
  // session reaches a terminal status.
  //
  // Scope note: this guards the MANUAL flow ONLY. The batch auto-calibration
  // subprocess is deliberately designed to SURVIVE navigation — it resumes on
  // remount (see the batch-status resume/poll effects) — so it is intentionally
  // NOT aborted on leave. Aborting it here would break that resume feature; a
  // running batch is stopped only via its explicit "Stop all" button.
  const [manualCalibLive, setManualCalibLive] = useState(false);
  useEffect(() => {
    if (calibrationStatus.calibration_active) {
      setManualCalibLive(true);
    } else if (
      ["idle", "completed", "error"].includes(calibrationStatus.status)
    ) {
      setManualCalibLive(false);
    }
  }, [calibrationStatus.calibration_active, calibrationStatus.status]);

  // Shared page-leave safety net (same hook as Recording & Inference). Manual
  // calibration holds the serial port in a singleton that would otherwise block
  // the next start ("Calibration already active"); an unintentional exit aborts
  // it. The arm is LIMP during manual range recording (torque is disabled), so
  // this is a clean teardown, not a mid-motion stop. The abort reuses the
  // module's existing /stop-calibration teardown.
  const { markHandled: markCalibHandled } = useSessionExitGuard({
    active: manualCalibLive,
    confirmMessage:
      "Leaving aborts this calibration — nothing will be saved and the arm is released. Continue?",
    beaconUrl: `${baseUrl}/stop-calibration`,
    onLeave: () => {
      fetchWithHeaders(`${baseUrl}/stop-calibration`, { method: "POST" }).catch(
        (e) => console.error("Failed to stop calibration on leave:", e),
      );
    },
    beaconFlagKey: "lelab:calibration-stopped",
  });

  const pollStatus = async () => {
    try {
      const response = await fetchWithHeaders(`${baseUrl}/calibration-status`);
      if (response.ok) {
        const status = await response.json();
        setCalibrationStatus(status);

        if (
          !status.calibration_active &&
          (status.status === "completed" ||
            status.status === "error" ||
            status.status === "idle")
        ) {
          setIsPolling(false);
        }
      }
    } catch (error) {
      console.error("Error polling status:", error);
    }
  };

  const handleWiggle = async () => {
    if (!port) {
      toast({
        title: "Missing port",
        description:
          "Enter or detect the port first, then wiggle to confirm the arm.",
        variant: "destructive",
      });
      return;
    }
    setWiggling(true);
    try {
      const res = await fetchWithHeaders(`${baseUrl}/wiggle`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ port }),
      });
      const data = await res.json();
      if (data.success) {
        toast({ title: "Wiggling gripper", description: data.message });
      } else {
        toast({
          title: "Wiggle failed",
          description: data.message,
          variant: "destructive",
        });
      }
    } catch (e) {
      toast({
        title: "Wiggle failed",
        description: String(e),
        variant: "destructive",
      });
    } finally {
      setWiggling(false);
    }
  };

  // The inverse of Wiggle: instead of driving a motor, the backend watches
  // every detected port (read-only) while the user swings the arm's base by
  // hand, then reports which port saw the motion. On success the detected
  // port is STAGED for confirmation (see handleConfirmDetectedPort) — nothing
  // is selected or persisted until the user confirms in the dialog.
  //
  // Detect is physical ground truth — the user just swung THIS arm on THIS
  // port — so if the record currently assigns the detected port to a DIFFERENT
  // slot, that slot's entry is stale (typical after a cable swap). We surface
  // that in the confirmation dialog and, on confirm, SWAP: the other slot
  // receives this slot's previous port (if any) while this slot takes the
  // detected port, in a single upsert (the backend's port-conflict guard
  // evaluates the prospective merged record, so a two-slot swap passes). If
  // this slot had no port the swap degenerates to a take-with-warning that
  // leaves the other slot empty. Confirm/messaging happen in
  // handleConfirmPortAssign.
  const handleDetect = async () => {
    setDetecting(true);
    try {
      const res = await fetchWithHeaders(`${baseUrl}/identify-arm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}), // empty = watch all detected ports
      });
      const data = await res.json();
      if (data.success && data.port) {
        // Which OTHER slot (if any) currently holds the detected port? Reuses
        // the same portFields set the dropdown uses (right_* only in bimanual),
        // so a single-arm robot's stale right_* ports don't trigger a release.
        const conflictingField = robot
          ? portFields.find(
              (f) => f !== portField && draftPort(f) === data.port,
            )
          : undefined;
        // The port THIS slot currently holds — handed to the other slot on a
        // swap. Null/empty means the swap degenerates to a take-with-warning.
        const currentPort = draftPort(portField);

        // Stage the result and open the confirmation dialog. No assignment or
        // persist happens here — that's deferred to handleConfirmPortAssign.
        setPortAssignPrompt({
          source: "detect",
          port: data.port,
          message: data.message,
          targetLabel: portFieldLabel(portField),
          releasedField: conflictingField ?? null,
          releasedLabel: conflictingField
            ? portFieldLabel(conflictingField)
            : null,
          swapPort: conflictingField && currentPort ? currentPort : null,
        });
      } else {
        toast({
          title: "No arm detected",
          description: data.message,
          variant: "destructive",
        });
      }
    } catch (e) {
      toast({
        title: "Detect failed",
        description: String(e),
        variant: "destructive",
      });
    } finally {
      setDetecting(false);
    }
  };

  // Apply a staged port assignment (from Detect or the manual dropdown) once
  // the user confirms. Cancel simply closes the dialog (setPortAssignPrompt(null))
  // and leaves everything as-is. Three cases:
  //  - releasedField + swapPort: SWAP — this slot takes the port, the other slot
  //    takes this slot's old port. One upsert; the backend's port-conflict guard
  //    evaluates the merged record, so a two-slot swap of distinct ports passes.
  //  - releasedField, no swapPort: take-with-warning — this slot had no port, so
  //    the other slot is left empty.
  //  - neither: straight assign (port wasn't in use anywhere).
  const handleConfirmPortAssign = async () => {
    const prompt = portAssignPrompt;
    if (!prompt) return;
    setPortAssignPrompt(null);

    setPort(prompt.port);
    const detected = prompt.source === "detect";

    if (prompt.releasedField) {
      const nextRobot = await persistPorts({
        [prompt.releasedField]: prompt.swapPort ?? "",
        [portField]: prompt.port,
      });
      if (nextRobot) {
        if (prompt.swapPort) {
          toast({
            title: detected ? "Arm identified — ports swapped" : "Ports swapped",
            description: `${detected ? `${prompt.message} ` : ""}${prompt.port} is now this arm's; the ${prompt.releasedLabel} took ${prompt.swapPort}.`,
          });
        } else {
          toast({
            title: detected ? "Arm identified — port moved" : "Port moved",
            description: `${detected ? `${prompt.message} ` : ""}${prompt.port} was assigned to the ${prompt.releasedLabel}; moved it here. The ${prompt.releasedLabel} now needs a port.`,
          });
        }
      }
      // persistPorts surfaces its own error toast on failure.
    } else {
      persistPort(prompt.port);
      toast({
        title: detected ? "Arm identified" : "Port assigned",
        description: detected
          ? `${prompt.message} Port assigned to this arm.`
          : `${prompt.port} assigned to this arm.`,
      });
    }
  };

  // Manual dropdown pick. In-use ports are now selectable (no longer greyed
  // out): picking one that another slot holds stages a swap/take confirmation
  // (same dialog as Detect). Picking a free port assigns immediately.
  const handleSelectPort = (nextPort: string) => {
    const conflictingField = robot
      ? portFields.find(
          (f) => f !== portField && draftPort(f) === nextPort,
        )
      : undefined;
    if (conflictingField) {
      const currentPort = draftPort(portField);
      setPortAssignPrompt({
        source: "manual",
        port: nextPort,
        message: "",
        targetLabel: portFieldLabel(portField),
        releasedField: conflictingField,
        releasedLabel: portFieldLabel(conflictingField),
        swapPort: currentPort || null,
      });
      return;
    }
    setPort(nextPort);
    persistPort(nextPort);
  };

  // --- Concurrent multi-arm auto-calibration ---

  // Each arm's port as designated on the robot record (assigned in the per-arm
  // flow above). Raw value — may name a port that isn't currently plugged in.
  const slotSavedPort = useCallback(
    (slot: ArmSlot) => draftPort(slot.portField).trim(),
    [draftPort],
  );

  // The port the batch will actually use: the saved port ONLY if it's currently
  // detected. A saved-but-undetected port (arm unplugged, moved, or renamed by
  // the OS) is treated as no port at all — you can't calibrate against an absent
  // bus, and the subprocess would just fail to open it. Single source of truth
  // for batch ports; never re-entered by the user.
  const slotPort = useCallback(
    (slot: ArmSlot) => {
      const saved = slotSavedPort(slot);
      return saved && availablePorts.includes(saved) ? saved : "";
    },
    [slotSavedPort, availablePorts],
  );

  // Single-arm picker: the selected port only counts if it's actually detected.
  // A saved-but-unplugged port is treated as no port — same rule as the batch
  // flow — so calibration can't start against an absent bus. `port` stays set to
  // the saved value so it re-selects automatically once the arm is plugged back
  // in and ports are rescanned.
  const portDetected = !!port && availablePorts.includes(port);

  // The slots the user ticked, in canonical order, with their inputs.
  const selectedBatchSlots = armSlots.filter((s) => batchSelected[s.key]);

  // Resume the batch panel if a run is in progress (e.g. page reload).
  useEffect(() => {
    (async () => {
      try {
        const res = await fetchWithHeaders(
          `${baseUrl}/auto-calibration-batch-status`,
        );
        const data = await res.json();
        setBatchAutoCal(data);
        if (data.active) setBatchAutoCalOpen(true);
      } catch {
        // ignore
      }
    })();
  }, [baseUrl, fetchWithHeaders]);

  // Poll batch status + logs while a run is active.
  useEffect(() => {
    if (!batchAutoCal.active) return;
    const id = setInterval(async () => {
      try {
        const res = await fetchWithHeaders(
          `${baseUrl}/auto-calibration-batch-status`,
        );
        const data: BatchAutoCalStatus = await res.json();
        setBatchAutoCal(data);
        if (!data.active) {
          setCalibReloadToken((t) => t + 1);
          fetchRobot();
          if (data.failed === 0) {
            toast({
              title: `Auto-calibrated ${data.completed} arm(s)`,
            });
          } else {
            toast({
              title: "Batch auto-calibration finished with issues",
              description: `${data.completed} completed, ${data.failed} failed/stopped.`,
              variant: data.completed > 0 ? "default" : "destructive",
            });
          }
        }
      } catch {
        // transient; keep polling
      }
    }, 700);
    return () => clearInterval(id);
  }, [batchAutoCal.active, baseUrl, fetchWithHeaders, fetchRobot, toast]);

  const startBatchAutoCalibration = async () => {
    setBatchAutoCalPromptOpen(false);
    if (!robotName) return;
    const slots = selectedBatchSlots;
    if (slots.length === 0) {
      toast({
        title: "No arms selected",
        description: "Tick at least one arm to auto-calibrate.",
        variant: "destructive",
      });
      return;
    }
    // Ports come from each arm's assignment on the robot record — the batch
    // never re-collects them. Guards mirror the backend; the missing-port case
    // is normally prevented by gating selection on an assigned port.
    const missingPort = slots.find((s) => !slotPort(s));
    if (missingPort) {
      toast({
        title: "Arm has no detected port",
        description: `${missingPort.label} has no port that's currently plugged in — assign/reconnect it above before starting.`,
        variant: "destructive",
      });
      return;
    }
    const ports = slots.map((s) => slotPort(s));
    if (new Set(ports).size !== ports.length) {
      toast({
        title: "Duplicate port",
        description: "Each arm needs its own serial port.",
        variant: "destructive",
      });
      return;
    }

    // Each arm saves to its own default name: the in-use config for that slot,
    // else a per-arm "<robot>_<arm>" (bimanual) / "<robot>" suggestion.
    const arms = slots.map((s) => ({
      device_type: s.device,
      port: slotPort(s),
      config_file:
        ((robot?.[s.cfgField] as string) || "").trim() ||
        (isBimanual ? `${robotName}_${s.arm}` : robotName || ""),
      arm: s.arm,
    }));

    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/start-auto-calibration-batch`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          // Each arm saves to its own default name, so replacing that arm's
          // existing calibration is the expected outcome — overwrite is always
          // on and the old name-taken confirmation is gone.
          body: JSON.stringify({ robot_name: robotName, overwrite: true, arms }),
        },
      );
      const data = await res.json();
      if (data.success) {
        setBatchAutoCal({
          active: true,
          arms: [],
          total: data.total ?? arms.length,
          completed: 0,
          failed: 0,
          logs: [],
        });
        toast({
          title: `Auto-calibration started on ${data.launched ?? arms.length} arm(s)`,
          description: "The arms are moving — keep the workspace clear.",
        });
      } else {
        toast({
          title: "Couldn't start auto-calibration",
          description: data.message,
          variant: "destructive",
        });
      }
    } catch (e) {
      toast({
        title: "Couldn't start auto-calibration",
        description: String(e),
        variant: "destructive",
      });
    }
  };

  const stopBatchAutoCalibration = async () => {
    try {
      await fetchWithHeaders(`${baseUrl}/stop-auto-calibration-batch`, {
        method: "POST",
      });
    } catch (e) {
      console.error("Failed to stop batch auto-calibration:", e);
    }
  };

  const handleStartCalibration = async () => {
    if (!robotName) {
      toast({
        title: "No robot selected",
        description:
          "Open Calibration from a robot's gear icon on the Landing page.",
        variant: "destructive",
      });
      return;
    }
    if (!port) {
      toast({
        title: "Missing port",
        description: "Set the device's serial port before starting.",
        variant: "destructive",
      });
      return;
    }

    const request: CalibrationRequest = {
      device_type: deviceType,
      port: port,
      config_file: calibrationConfigName,
      robot_name: robotName,
      // The name is always the robot's own default for this slot, so replacing
      // its existing calibration is the expected outcome — overwrite is always
      // on and the old name-taken confirmation prompt is gone. To keep the old
      // calibration, rename it afterward via the per-side rename feature.
      overwrite: true,
      arm,
    };

    // Optimistically mark as active so the leave guard will fire even if the
    // user navigates away before the backend reports calibration_active=true.
    // Reverted below if the start request fails.
    setManualCalibLive(true);

    try {
      const response = await fetchWithHeaders(`${baseUrl}/start-calibration`, {
        method: "POST",
        body: JSON.stringify(request),
      });

      const result = await response.json();

      if (result.success) {
        toast({
          title: "Calibration Started",
          description: `Calibration started for ${deviceType}`,
        });
        setIsPolling(true);
      } else {
        setManualCalibLive(false);
        toast({
          title: "Calibration Failed",
          description: result.message || "Failed to start calibration",
          variant: "destructive",
        });
      }
    } catch (error) {
      setManualCalibLive(false);
      console.error("Error starting calibration:", error);
      toast({
        title: "Error",
        description: "Failed to start calibration",
        variant: "destructive",
      });
    }
  };

  const handleStopCalibration = async () => {
    // Explicit Cancel — mark handled so the leave guard doesn't also fire while
    // the session winds down (the hook re-arms for any later calibration).
    markCalibHandled();
    try {
      const response = await fetchWithHeaders(`${baseUrl}/stop-calibration`, {
        method: "POST",
      });

      const result = await response.json();

      if (result.success) {
        // The 200ms polling interval will pick up the stopped state.
        toast({
          title: "Calibration Stopped",
          description: "Calibration has been stopped",
        });
      } else {
        toast({
          title: "Error",
          description: result.message || "Failed to stop calibration",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error("Error stopping calibration:", error);
      toast({
        title: "Error",
        description: "Failed to stop calibration",
        variant: "destructive",
      });
    }
  };

  const handleCompleteStep = async () => {
    if (!calibrationStatus.calibration_active) return;

    try {
      const response = await fetchWithHeaders(
        `${baseUrl}/complete-calibration-step`,
        { method: "POST" },
      );

      const data = await response.json();

      if (data.success) {
        toast({
          title: "Step Completed",
          description: data.message,
        });
      } else {
        toast({
          title: "Step Failed",
          description: data.message || "Could not complete step",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error("Error completing step:", error);
      toast({
        title: "Error",
        description: "Could not complete calibration step",
        variant: "destructive",
      });
    }
  };

  useEffect(() => {
    if (
      calibrationStatus.status === "error" &&
      calibrationStatus.error?.startsWith(DISCONTINUITY_ERROR_PREFIX)
    ) {
      demoVideoRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }, [calibrationStatus.status, calibrationStatus.error]);

  useEffect(() => {
    if (!isPolling) return;
    // Single stable interval. Reads calibration_active from the ref each tick so
    // the interval doesn't tear down/recreate on every status change.
    pollStatus();
    const interval = setInterval(() => {
      pollStatus();
    }, 200);
    return () => clearInterval(interval);
    // pollStatus is stable enough — it only reads via fetchWithHeaders + setState.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPolling]);

  // Load default port when device type changes (skip when arriving from a tile —
  // the robot-record prefill above wins)
  useEffect(() => {
    const loadDefaultPort = async () => {
      if (!deviceType) return;
      if (robotName) return;

      try {
        const robotType = deviceType === "robot" ? "follower" : "leader";
        const response = await fetchWithHeaders(
          `${baseUrl}/robot-port/${robotType}`,
        );
        const data = await response.json();
        if (data.status === "success") {
          const portToUse = data.saved_port || data.default_port;
          if (portToUse) {
            setPort(portToUse);
          }
        }
      } catch (error) {
        console.error("Error loading default port:", error);
      }
    };

    loadDefaultPort();
  }, [deviceType, robotName, baseUrl, fetchWithHeaders]);

  const handleDeviceTypeChange = (next: string) => {
    setDeviceType(next);
    // Port is re-synced from the record by the device/arm effect below.
  };

  // Keep the port field in sync with the selected device_type + arm's saved
  // port whenever either changes (single uses leader/follower; bimanual right
  // uses the right_* fields). Port is a dropdown, so overwriting it is safe.
  useEffect(() => {
    if (!robot) return;
    setPort(draftPort(portField) || "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deviceType, arm, robot, portDraft]);

  // Refresh the robot record when a calibration completes so the checklist
  // flips to ✓ for the side that was just saved, and advance Device Type to
  // the next still-incomplete side (or stay on the current side if both done).
  useEffect(() => {
    if (calibrationStatus.status !== "completed") return;
    // A completed calibration may have written a new named file — nudge the
    // per-side libraries to re-fetch their config lists so it shows up.
    setCalibReloadToken((t) => t + 1);
    (async () => {
      const r = await fetchRobot();
      if (!r) return;
      const nextDevice = !r.leader_config
        ? "teleop"
        : !r.follower_config
          ? "robot"
          : "teleop";
      setDeviceType(nextDevice);
      // Port re-syncs via the device/arm effect.
    })();
  }, [calibrationStatus.status, fetchRobot]);

  // Stage the current side's port into the local draft (no network write). A
  // re-detected USB port (which shuffles on reboot/reconnect) is recorded here
  // and only committed on Save. An empty string is a valid value: it CLEARS the
  // assignment (arm disconnected). The batched Save sends every dirty port slot
  // together so the backend's duplicate-port guard sees the merged record.
  const persistPort = useCallback(
    (nextPort: string) => {
      if (!robotName) return;
      setPortDraft((prev) => ({ ...prev, [portField]: nextPort }));
    },
    [robotName, portField],
  );

  // Stage several port slots at once into the draft (used by Detect's reassign
  // path: clear the stale slot AND set the current one in one edit). Both land
  // in the same batched Save request, so the backend's duplicate-port guard —
  // which evaluates the prospective merged record — passes for a legitimate
  // swap. Returns the applied patch (truthy) so callers can gate their toast.
  const persistPorts = useCallback(
    (patch: Partial<Record<keyof RobotRecord, string>>) => {
      if (!robotName) return null;
      setPortDraft((prev) => ({ ...prev, ...patch }));
      return patch;
    },
    [robotName],
  );

  // --- Motor power (per-robot, persisted) -------------------------------
  // The backend stores/applies motor_power as a PERCENT of full torque
  // (10-100; see lelab/utils/config.py clamp_motor_power). The servo actually
  // scales torque via its RAM Torque_Limit register on a 0-1000 scale, and
  // apply_motor_power writes `percent * _TORQUE_LIMIT_PER_PERCENT` to it (see
  // lelab/motor_power.py, _TORQUE_LIMIT_PER_PERCENT = 10). The UI below is
  // expressed in those RAW register units — same scale as autocal's
  // DEFAULT_TORQUE_LIMIT = 380 (lelab/vendor/feetech_autocal/
  // calibration_defaults.py) — so operators can reason in one vocabulary.
  // This is a DISPLAY/units change only: we convert raw<->percent at the edges
  // and still persist a percent, so clamp_motor_power / torque_limit_from_percent
  // / the apply path are untouched.
  const TORQUE_LIMIT_PER_PERCENT = 10; // must match lelab/motor_power.py
  const MOTOR_POWER_MIN_PERCENT = 10; // must match lelab/utils/config.py
  const MOTOR_POWER_MAX_PERCENT = 100; // must match lelab/utils/config.py
  const TORQUE_LIMIT_MIN = MOTOR_POWER_MIN_PERCENT * TORQUE_LIMIT_PER_PERCENT; // 100
  const TORQUE_LIMIT_MAX = MOTOR_POWER_MAX_PERCENT * TORQUE_LIMIT_PER_PERCENT; // 1000
  // Autocal's operating torque, shown as a reference marker only. NOT applied
  // as the default here — changing the persisted default would be a torque
  // (safety) behavior change; see config.py DEFAULT_MOTOR_POWER.
  const DEFAULT_TORQUE_LIMIT_REF = 380; // lelab/vendor/.../calibration_defaults.py

  // Local slider position (in PERCENT). Held as a draft and committed to the
  // robot record only on Save (previously it POSTed on slider release). Applied
  // to the follower's motors at the start of each teleop/record/inference
  // session. Fallback matches backend DEFAULT_MOTOR_POWER (38% = Torque_Limit
  // 380); re-syncs from the baseline whenever the saved value changes.
  const [powerDraft, setPowerDraft] = useState(38);
  useEffect(() => {
    setPowerDraft(robot?.motor_power ?? 38);
  }, [robot?.motor_power]);

  // Slider is in raw Torque_Limit units; convert to the percent the draft holds.
  const torqueLimitDraft = Math.round(powerDraft) * TORQUE_LIMIT_PER_PERCENT;
  // The integer percent the draft would persist, clamped to the backend's 10-100.
  const motorPercent = Math.min(100, Math.max(10, Math.round(powerDraft)));

  // --- Draft dirtiness + batched Save ------------------------------------
  // A field is dirty when its draft differs from the last-fetched baseline.
  // Save is the ONLY path that writes the record; it POSTs every dirty field in
  // one request (batching matters for ports: the backend's duplicate-port guard
  // evaluates the merged record, so clearing one slot and assigning another must
  // arrive together). Server-side writes elsewhere (calibration completion,
  // config assignment via the library) are out of scope and untouched.
  const camerasDirty = useMemo(
    () =>
      !!robot &&
      JSON.stringify(cameras ?? []) !== JSON.stringify(robot.cameras ?? []),
    [cameras, robot],
  );
  const portsDirty = useMemo(
    () =>
      !!robot &&
      Object.entries(portDraft).some(
        ([f, v]) => (v ?? "") !== ((robot[f as keyof RobotRecord] as string) || ""),
      ),
    [portDraft, robot],
  );
  const motorDirty = !!robot && motorPercent !== robot.motor_power;
  const isDirty = camerasDirty || portsDirty || motorDirty;

  const handleSave = useCallback(async () => {
    if (!robotName || !robot) return;
    const patch: Record<string, unknown> = {};
    if (camerasDirty) patch.cameras = cameras;
    if (motorDirty) patch.motor_power = motorPercent;
    if (portsDirty) {
      for (const [f, v] of Object.entries(portDraft)) {
        if ((v ?? "") !== ((robot[f as keyof RobotRecord] as string) || "")) {
          patch[f] = v ?? "";
        }
      }
    }
    if (Object.keys(patch).length === 0) return;
    setSaving(true);
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/robots/${encodeURIComponent(robotName)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(patch),
        },
      );
      const data = await res.json();
      if (res.ok && data.robot) {
        // Adopt the server record as the new baseline and clear the drafts.
        // powerDraft re-syncs via its effect when motor_power changes.
        setRobot(data.robot);
        setPortDraft({});
        setCameras((data.robot as RobotRecord).cameras ?? []);
        toast({ title: "Changes saved" });
      } else {
        // Surface the backend guard (e.g. duplicate-port 409) and stay put.
        toast({
          title: "Couldn't save changes",
          description: data.message || "Failed to save the configuration.",
          variant: "destructive",
        });
      }
    } catch (e) {
      toast({
        title: "Couldn't save changes",
        description: String(e),
        variant: "destructive",
      });
    } finally {
      setSaving(false);
    }
  }, [
    robotName,
    robot,
    camerasDirty,
    motorDirty,
    portsDirty,
    cameras,
    motorPercent,
    portDraft,
    baseUrl,
    fetchWithHeaders,
    toast,
  ]);

  // Quit returns to wherever the page's back-navigation goes (same as the
  // header's back arrow: navigate(-1)). Unsaved drafts prompt for confirmation
  // first so an accidental exit never silently discards edits.
  const handleQuit = useCallback(() => {
    if (isDirty) {
      setQuitPromptOpen(true);
      return;
    }
    navigate(-1);
  }, [isDirty, navigate]);

  const confirmQuit = useCallback(() => {
    setQuitPromptOpen(false);
    navigate(-1);
  }, [navigate]);

  const getStatusDisplay = () => {
    switch (calibrationStatus.status) {
      case "idle":
        return {
          color: "bg-slate-500",
          icon: <Settings className="w-4 h-4" />,
          text: "Idle",
        };
      case "connecting":
        return {
          color: "bg-yellow-500",
          icon: <Loader2 className="w-4 h-4 animate-spin" />,
          text: "Connecting",
        };
      case "recording":
        return {
          color: "bg-purple-500",
          icon: <Activity className="w-4 h-4" />,
          text: "Recording Ranges",
        };
      case "completed":
        return {
          color: "bg-green-500",
          icon: <CheckCircle className="w-4 h-4" />,
          text: "Completed",
        };
      case "error":
        return {
          color: "bg-red-500",
          icon: <XCircle className="w-4 h-4" />,
          text: "Error",
        };
      case "stopping":
        return {
          color: "bg-orange-500",
          icon: <Square className="w-4 h-4" />,
          text: "Stopping",
        };
      default:
        return {
          color: "bg-slate-500",
          icon: <Settings className="w-4 h-4" />,
          text: "Unknown",
        };
    }
  };

  const statusDisplay = getStatusDisplay();

  return (
    <div className="min-h-screen bg-slate-900 text-white p-4">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center gap-4 mb-6">
          <Button
            variant="ghost"
            size="icon"
            onClick={handleQuit}
            className="text-slate-400 hover:text-white hover:bg-slate-800"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <div className="flex items-center gap-3">
            <Logo iconOnly />
            <h1 className="text-3xl font-bold">
              {robotName ? `Calibrate "${robotName}"` : "Device Calibration"}
            </h1>
          </div>
        </div>

        {!robotName && (
          <Alert className="mb-6 bg-amber-900/40 border-amber-700 text-amber-100">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Open Calibration from a robot's gear icon on the Landing page.
              Each robot has its own calibration; running this page directly is
              not supported.
            </AlertDescription>
          </Alert>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="bg-slate-800/60 border-slate-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-slate-200">
                <Settings className="w-5 h-5 text-blue-400" />
                Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label
                  htmlFor="deviceType"
                  className="text-sm font-medium text-slate-300"
                >
                  Device Type *
                </Label>
                <Select
                  value={deviceType}
                  onValueChange={handleDeviceTypeChange}
                >
                  <SelectTrigger className="bg-slate-700 border-slate-600 text-white rounded-md">
                    <SelectValue placeholder="Select device type" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-slate-700 text-white">
                    <SelectItem value="teleop" className="hover:bg-slate-700">
                      Teleoperator (Leader)
                    </SelectItem>
                    <SelectItem value="robot" className="hover:bg-slate-700">
                      Robot (Follower)
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {isBimanual && (
                <div className="space-y-2">
                  <Label
                    htmlFor="arm"
                    className="text-sm font-medium text-slate-300"
                  >
                    Arm *
                  </Label>
                  <Select
                    value={arm}
                    onValueChange={(v) => setArm(v as "left" | "right")}
                  >
                    <SelectTrigger className="bg-slate-700 border-slate-600 text-white rounded-md">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-700 text-white">
                      <SelectItem value="left" className="hover:bg-slate-700">
                        Left arm
                      </SelectItem>
                      <SelectItem value="right" className="hover:bg-slate-700">
                        Right arm
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}

              <div className="space-y-2">
                <Label
                  htmlFor="port"
                  className="text-sm font-medium text-slate-300"
                >
                  Port *
                </Label>
                <div className="flex flex-wrap gap-2">
                  <Select value={port} onValueChange={handleSelectPort}>
                    <SelectTrigger
                      id="port"
                      className="bg-slate-700 border-slate-600 text-white rounded-md flex-1 min-w-[200px]"
                    >
                      <SelectValue
                        placeholder={
                          availablePorts.length
                            ? "Select a port"
                            : "No arms detected — plug in & refresh"
                        }
                      />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-700 text-white">
                      {availablePorts.map((p) => {
                        // In-use ports stay selectable: picking one prompts a
                        // swap (this slot's current port goes to the other arm)
                        // or, if this slot is empty, a take-with-warning.
                        const usedByOtherArm = otherArmPorts.includes(p);
                        return (
                          <SelectItem key={p} value={p} className="text-white">
                            <span className="flex items-center gap-2">
                              {p}
                              {usedByOtherArm && (
                                <span className="text-[10px] uppercase tracking-wide text-amber-400 border border-amber-500/40 rounded px-1">
                                  other arm
                                </span>
                              )}
                            </span>
                          </SelectItem>
                        );
                      })}
                      {/* A saved-but-undetected port is intentionally NOT offered
                          here: an unplugged bus can't be calibrated against, so
                          it's treated as no port. The trigger falls back to the
                          placeholder, and the port re-selects on its own once the
                          arm is plugged back in and ports are rescanned. */}
                    </SelectContent>
                  </Select>
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={() => {
                      setPort("");
                      persistPort("");
                    }}
                    // Also gated during calibration: clearing wouldn't stop the
                    // running session (the subprocess holds the serial port),
                    // it would just desync the UI from the arm being measured.
                    disabled={
                      !port ||
                      calibrationStatus.calibration_active ||
                      batchAutoCal.active
                    }
                    title="Clear port — release it without assigning another"
                    aria-label="Clear port"
                    className="border-slate-600 hover:border-red-500 text-slate-400 hover:text-red-400 bg-slate-700 hover:bg-slate-600 shrink-0"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={fetchPorts}
                    disabled={portsLoading}
                    title="Rescan ports"
                    className="border-slate-600 hover:border-blue-500 text-slate-400 hover:text-blue-400 bg-slate-700 hover:bg-slate-600 shrink-0"
                  >
                    <RefreshCw
                      className={`w-4 h-4 ${portsLoading ? "animate-spin" : ""}`}
                    />
                  </Button>
                </div>
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handleDetect}
                    disabled={
                      detecting ||
                      wiggling ||
                      calibrationStatus.calibration_active ||
                      batchAutoCal.active
                    }
                    title="Identify by hand: swing the arm's base left and right"
                    className="w-32 shrink-0 border-slate-600 hover:border-emerald-500 text-slate-400 hover:text-emerald-400 bg-slate-700 hover:bg-slate-600"
                  >
                    {detecting ? (
                      <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                    ) : (
                      <Hand className="w-4 h-4 mr-1" />
                    )}
                    {detecting ? "Watching…" : "Detect"}
                  </Button>
                  <p className="flex-1 min-w-[200px] text-xs text-slate-400">
                    Identify by hand — swing the arm's base left and right; the
                    port that moves is assigned.
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handleWiggle}
                    disabled={
                      !port ||
                      wiggling ||
                      detecting ||
                      calibrationStatus.calibration_active ||
                      batchAutoCal.active
                    }
                    title="Move the gripper on this port to see which arm it is"
                    className="w-32 shrink-0 border-slate-600 hover:border-yellow-500 text-slate-400 hover:text-yellow-400 bg-slate-700 hover:bg-slate-600"
                  >
                    <Hand className="w-4 h-4 mr-1" />
                    {wiggling ? "Wiggling…" : "Wiggle"}
                  </Button>
                  <p className="flex-1 min-w-[200px] text-xs text-slate-400">
                    Confirms an arm is on this port — briefly drives its gripper
                    so you can see which arm responds.
                  </p>
                </div>
                {detecting && (
                  <p className="text-xs text-emerald-400">
                    Swing the base of the arm left and right — the port that
                    sees the motion will be assigned to this arm.
                  </p>
                )}
              </div>

              <Separator className="bg-slate-700" />

              <div className="flex flex-col gap-3">
                {calibrationStatus.calibration_active ? (
                  <Button
                    onClick={handleStopCalibration}
                    variant="destructive"
                    className="w-full rounded-full py-6 text-lg"
                  >
                    <Square className="w-5 h-5 mr-2" />
                    Cancel Calibration
                  </Button>
                ) : batchAutoCal.active ? (
                  <Button
                    onClick={stopBatchAutoCalibration}
                    variant="destructive"
                    className="w-full rounded-full py-6 text-lg"
                  >
                    <Square className="w-5 h-5 mr-2" />
                    Stop all auto-calibration
                  </Button>
                ) : (
                  // Auto-calibrate is the default calibration mode: it's the
                  // prominent primary action and opens the multi-arm picker
                  // below (single-arm robots see just their leader+follower
                  // slots there). Manual step-by-step calibration stays fully
                  // available as the secondary button — a user who wants it just
                  // clicks it, but landing here nudges toward the hands-off auto
                  // flow.
                  <>
                    <Button
                      onClick={() => setBatchAutoCalOpen(true)}
                      className="w-full bg-purple-600 hover:bg-purple-700 text-white rounded-full py-6 text-lg"
                      disabled={!robotName}
                    >
                      <Wand2 className="w-5 h-5 mr-2" />
                      Auto-calibrate
                    </Button>
                    <Button
                      onClick={() => handleStartCalibration()}
                      variant="outline"
                      disabled={!robotName || !deviceType || !portDetected}
                      className="w-full border-blue-500/50 text-blue-700 hover:bg-blue-900/20 hover:text-blue-800 dark:text-blue-300 dark:hover:text-blue-200 rounded-full py-5"
                    >
                      <Play className="w-5 h-5 mr-2" />
                      Calibrate manually
                    </Button>
                  </>
                )}

                {(batchAutoCalOpen || batchAutoCal.active) && (
                  <div className="rounded-lg border border-purple-800/50 bg-purple-950/20 p-3 space-y-3">
                    <div className="flex items-center gap-2 text-sm font-medium text-purple-200">
                      <Wand2 className="w-4 h-4" />
                      Multi-arm auto-calibration
                    </div>
                    {!batchAutoCal.active ? (
                      <>
                        <p className="text-xs text-slate-400">
                          Pick the arms to calibrate. Each runs its own hands-off
                          calibration <strong>at the same time</strong> on its
                          assigned port — one arm failing doesn't stop the
                          others. Ports come from each arm's assignment above;
                          an arm with no port yet can't be picked. Each arm
                          replaces its own existing calibration; rename any of
                          them afterward from the calibration list below.
                        </p>
                        <div className="space-y-2">
                          {armSlots.map((slot) => {
                            const selected = !!batchSelected[slot.key];
                            const assignedPort = slotPort(slot);
                            const hasPort = !!assignedPort;
                            // Distinguish "never assigned" from "assigned but
                            // not currently detected" so the hint is actionable.
                            const savedButUndetected =
                              !hasPort && !!slotSavedPort(slot);
                            return (
                              <div
                                key={slot.key}
                                className={`rounded-md border p-2 ${
                                  selected
                                    ? "border-purple-600 bg-slate-800/60"
                                    : "border-slate-700 bg-slate-800/20"
                                } ${hasPort ? "" : "opacity-60"}`}
                              >
                                <label
                                  className={`flex items-center gap-2 ${
                                    hasPort ? "cursor-pointer" : "cursor-not-allowed"
                                  }`}
                                >
                                  <input
                                    type="checkbox"
                                    checked={selected}
                                    disabled={!hasPort}
                                    onChange={(e) =>
                                      setBatchSelected((prev) => ({
                                        ...prev,
                                        [slot.key]: e.target.checked,
                                      }))
                                    }
                                    className="accent-purple-500"
                                  />
                                  <span className="text-sm text-slate-200">
                                    {slot.label}
                                  </span>
                                  <span
                                    className={`ml-auto text-xs font-mono ${
                                      hasPort ? "text-slate-400" : "text-amber-400/80"
                                    }`}
                                  >
                                    {hasPort
                                      ? assignedPort
                                      : savedButUndetected
                                        ? "port not detected"
                                        : "no port — assign above"}
                                  </span>
                                </label>
                              </div>
                            );
                          })}
                        </div>
                        <div className="flex gap-2">
                          <Button
                            onClick={() => setBatchAutoCalPromptOpen(true)}
                            disabled={selectedBatchSlots.length === 0}
                            className="flex-1 bg-purple-600 hover:bg-purple-700 text-white rounded-full py-4"
                          >
                            <Wand2 className="w-4 h-4 mr-2" />
                            Auto-calibrate {selectedBatchSlots.length || 0} arm
                            {selectedBatchSlots.length === 1 ? "" : "s"}
                          </Button>
                          <Button
                            onClick={() => setBatchAutoCalOpen(false)}
                            variant="outline"
                            className="shrink-0 border-slate-600 text-slate-300 hover:text-slate-100 rounded-full py-4"
                          >
                            Cancel
                          </Button>
                        </div>
                      </>
                    ) : (
                      <p className="text-xs text-slate-400">
                        {batchAutoCal.completed + batchAutoCal.failed} of{" "}
                        {batchAutoCal.total} done — the arms are moving. Keep the
                        workspace clear.
                      </p>
                    )}

                    {/* Per-arm status rows (running + terminal), shown live. */}
                    {batchAutoCal.arms.length > 0 && (
                      <div className="space-y-1">
                        {batchAutoCal.arms.map((a) => (
                          <div
                            key={`${a.device_type}:${a.port}`}
                            className="flex items-center justify-between gap-2 text-xs rounded bg-slate-900/60 px-2 py-1"
                          >
                            <span className="font-mono text-slate-300 truncate">
                              {a.name || a.port}
                            </span>
                            <span
                              className={
                                a.status === "completed"
                                  ? "text-green-400"
                                  : a.status === "failed"
                                    ? "text-red-400"
                                    : a.status === "stopped"
                                      ? "text-amber-400"
                                      : "text-purple-300"
                              }
                              title={a.error ?? undefined}
                            >
                              {a.status === "completed"
                                ? "✓ done"
                                : a.status === "failed"
                                  ? "✗ failed"
                                  : a.status === "stopped"
                                    ? "stopped"
                                    : "running…"}
                            </span>
                          </div>
                        ))}
                        {!batchAutoCal.active && batchAutoCal.total > 0 && (
                          <p className="text-xs text-slate-400 pt-1">
                            {batchAutoCal.completed} completed,{" "}
                            {batchAutoCal.failed} failed/stopped.
                          </p>
                        )}
                      </div>
                    )}

                    {batchAutoCal.logs.length > 0 && (
                      <div className="bg-slate-900 rounded border border-slate-700 p-2 max-h-40 overflow-auto text-xs font-mono text-slate-300 whitespace-pre-wrap">
                        {batchAutoCal.logs.slice(-120).map((line, i) => (
                          <div key={i}>{line}</div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {robot && (
                  <div className="space-y-1 pt-1">
                    <div className="flex items-center gap-3">
                      <Label
                        htmlFor="motorPower"
                        className="text-sm font-medium text-slate-300 shrink-0"
                      >
                        Motor torque limit
                      </Label>
                      <input
                        id="motorPower"
                        type="range"
                        min={TORQUE_LIMIT_MIN}
                        max={TORQUE_LIMIT_MAX}
                        step={TORQUE_LIMIT_PER_PERCENT}
                        value={torqueLimitDraft}
                        onChange={(e) => {
                          // Slider is in raw Torque_Limit units; store as percent.
                          setPowerDraft(
                            Number(e.target.value) / TORQUE_LIMIT_PER_PERCENT,
                          );
                        }}
                        list="motorTorqueTicks"
                        className="flex-1 h-1.5 accent-blue-500 cursor-pointer"
                        aria-label="Motor torque limit (Torque_Limit register, 0-1000 scale)"
                      />
                      <datalist id="motorTorqueTicks">
                        {/* Autocal's operating torque, as a reference tick. */}
                        <option value={DEFAULT_TORQUE_LIMIT_REF} />
                      </datalist>
                      <span className="text-sm font-mono text-slate-200 w-12 text-right shrink-0">
                        {torqueLimitDraft}
                      </span>
                    </div>
                    <div className="flex items-start justify-between gap-2 text-xs text-slate-500">
                      <span>
                        Raw servo{" "}
                        <code className="text-slate-400">Torque_Limit</code>{" "}
                        (0–1000 scale). Lower = gentler movements and weaker grip;
                        below {TORQUE_LIMIT_MIN} the arm can't hold its own weight.
                        Autocal drives at {DEFAULT_TORQUE_LIMIT_REF}. Resets to the
                        saved value each session.
                      </span>
                    </div>
                  </div>
                )}

                {/* Manual calibration only: torque is off the whole session,
                    which surprises novices (the arm is deliberately floppy).
                    Auto-cal needs no standing warning — it ends gracefully
                    (fold on completion, freeze + return-to-start on Stop) and
                    the multi-arm pre-start confirmation dialog carries the
                    safety guidance. */}
                {calibrationStatus.calibration_active && (
                  <Alert className="bg-amber-900/40 border-amber-700 text-amber-100">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      Motor torque is off — the arm won't hold its pose during
                      calibration, and stays limp after you cancel or finish.
                      Keep it low and supported so it can't drop onto the table
                      edge.
                    </AlertDescription>
                  </Alert>
                )}

              </div>

              <Dialog
                open={batchAutoCalPromptOpen}
                onOpenChange={setBatchAutoCalPromptOpen}
              >
                <DialogContent className="bg-slate-900 border-slate-800 text-white">
                  <DialogHeader>
                    <DialogTitle>
                      Auto-calibrate multiple arms — they will move
                    </DialogTitle>
                    <DialogDescription className="text-slate-400">
                      {selectedBatchSlots.length} arm
                      {selectedBatchSlots.length === 1 ? "" : "s"} will{" "}
                      <strong>move on their own under power</strong> at the same
                      time to find each joint's range. Clear the workspace and
                      keep hands away from every arm. Each arm replaces its own
                      existing calibration.
                    </DialogDescription>
                  </DialogHeader>
                  <DialogFooter className="flex gap-2 justify-end">
                    <Button
                      variant="outline"
                      className="border-slate-600 text-slate-700 dark:text-slate-300"
                      onClick={() => setBatchAutoCalPromptOpen(false)}
                    >
                      Cancel
                    </Button>
                    <Button
                      className="bg-purple-600 hover:bg-purple-700 text-white"
                      onClick={() => startBatchAutoCalibration()}
                    >
                      Start auto-calibration
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>

              <AlertDialog
                open={portAssignPrompt !== null}
                onOpenChange={(open) => {
                  if (!open) setPortAssignPrompt(null);
                }}
              >
                <AlertDialogContent className="bg-slate-900 border-slate-800 text-white">
                  <AlertDialogHeader>
                    <AlertDialogTitle>
                      {portAssignPrompt?.swapPort
                        ? "Swap ports?"
                        : portAssignPrompt?.source === "detect"
                          ? "Assign detected port?"
                          : "Assign port?"}
                    </AlertDialogTitle>
                    <AlertDialogDescription className="text-slate-400">
                      {portAssignPrompt?.source === "detect"
                        ? "Detected "
                        : "Assign "}
                      <span className="font-mono text-slate-200">
                        {portAssignPrompt?.port}
                      </span>{" "}
                      {portAssignPrompt?.source === "detect"
                        ? "— assign it to the "
                        : "to the "}
                      <strong>{portAssignPrompt?.targetLabel}</strong>?
                      {portAssignPrompt?.releasedLabel &&
                        (portAssignPrompt.swapPort ? (
                          <>
                            {" "}
                            It's currently assigned to the{" "}
                            <strong>{portAssignPrompt.releasedLabel}</strong>;
                            confirming swaps them — the{" "}
                            <strong>{portAssignPrompt.releasedLabel}</strong>{" "}
                            takes this arm's current port{" "}
                            <span className="font-mono text-slate-200">
                              {portAssignPrompt.swapPort}
                            </span>{" "}
                            in exchange, so neither arm is left without a port.
                          </>
                        ) : (
                          <>
                            {" "}
                            It's currently assigned to the{" "}
                            <strong>{portAssignPrompt.releasedLabel}</strong>;
                            this arm has no port to swap back, so confirming
                            moves it here and leaves the{" "}
                            <strong>{portAssignPrompt.releasedLabel}</strong>{" "}
                            without a port.
                          </>
                        ))}
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter className="flex gap-2 justify-end">
                    <AlertDialogCancel className="border-slate-600 text-slate-700 dark:text-slate-300">
                      Cancel
                    </AlertDialogCancel>
                    <AlertDialogAction
                      className="bg-emerald-600 hover:bg-emerald-700 text-white"
                      onClick={handleConfirmPortAssign}
                    >
                      {portAssignPrompt?.swapPort
                        ? "Swap ports"
                        : portAssignPrompt?.releasedLabel
                          ? "Move & assign"
                          : "Assign port"}
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>

              {robot && (
                <div className="space-y-2 pt-2">
                  <div className="flex items-center gap-2 text-sm font-medium text-slate-300">
                    <span>Robot calibration</span>
                    {/* One folder per device type — both same-side slots share a
                        single directory (so101_leader / so101_follower), so a
                        single leader + follower pair covers single AND bimanual
                        modes (no per-slot duplication). */}
                    <Button
                      size="sm"
                      variant="ghost"
                      className="ml-auto h-6 gap-1.5 px-2 text-xs text-slate-400 hover:text-white"
                      onClick={() => openCalibrationFolder("teleop")}
                      aria-label="Open leader calibrations folder"
                      title="Open leader calibrations folder"
                    >
                      <FolderOpen className="w-4 h-4" />
                      Leader
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 gap-1.5 px-2 text-xs text-slate-400 hover:text-white"
                      onClick={() => openCalibrationFolder("robot")}
                      aria-label="Open follower calibrations folder"
                      title="Open follower calibrations folder"
                    >
                      <FolderOpen className="w-4 h-4" />
                      Follower
                    </Button>
                  </div>
                  {(isBimanual
                    ? // Bimanual: each of the four slots gets the same free-naming
                      // picker as single mode — names are arbitrary now, and the
                      // SLOT (not the name) decides which arm the file drives.
                      ([
                        {
                          label: "Left Leader (Teleoperator)",
                          device: "teleop",
                          cfgField: "leader_config",
                        },
                        {
                          label: "Left Follower (Robot)",
                          device: "robot",
                          cfgField: "follower_config",
                        },
                        {
                          label: "Right Leader (Teleoperator)",
                          device: "teleop",
                          cfgField: "right_leader_config",
                        },
                        {
                          label: "Right Follower (Robot)",
                          device: "robot",
                          cfgField: "right_follower_config",
                        },
                      ] as const)
                    : ([
                        {
                          label: "Leader (Teleoperator)",
                          device: "teleop",
                          cfgField: "leader_config",
                        },
                        {
                          label: "Follower (Robot)",
                          device: "robot",
                          cfgField: "follower_config",
                        },
                      ] as const)
                  ).map((row) => {
                    const cfg = (robot[row.cfgField] as string) || "";
                    // The same config may drive both same-side slots only by
                    // mistake (one physical arm on two arms), so exclude the
                    // counterpart slot's config from this picker in bimanual mode.
                    const counterpartField =
                      row.cfgField === "leader_config"
                        ? "right_leader_config"
                        : row.cfgField === "right_leader_config"
                          ? "leader_config"
                          : row.cfgField === "follower_config"
                            ? "right_follower_config"
                            : "follower_config";
                    const excludeConfig = isBimanual
                      ? (robot[counterpartField] as string) || undefined
                      : undefined;
                    // The counterpart slot's config field, so the library can
                    // SWAP assignments when the user picks its in-use config
                    // (this slot takes it; the counterpart takes this slot's).
                    const excludeConfigField = isBimanual
                      ? counterpartField
                      : undefined;
                    return (
                      <div key={row.label}>
                        <div className="flex items-center gap-2 text-sm">
                          {cfg ? (
                            <CheckCircle className="w-4 h-4 text-green-400" />
                          ) : (
                            <Circle className="w-4 h-4 text-slate-500" />
                          )}
                          <span
                            className={cfg ? "text-slate-200" : "text-slate-400"}
                          >
                            {row.label}
                          </span>
                        </div>
                        <CalibrationLibrary
                          device={row.device}
                          assignedConfig={cfg}
                          configField={row.cfgField}
                          excludeConfig={excludeConfig}
                          excludeConfigField={excludeConfigField}
                          robotName={robotName}
                          onAssigned={fetchRobot}
                          reloadToken={calibReloadToken}
                        />
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-slate-800/60 border-slate-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-slate-200">
                <Activity className="w-5 h-5 text-teal-400" />
                Status
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-slate-900/50 rounded-md">
                <span className="text-slate-300">Status:</span>
                <Badge
                  className={`${statusDisplay.color} text-white rounded-md`}
                >
                  {statusDisplay.icon}
                  <span className="ml-2">{statusDisplay.text}</span>
                </Badge>
              </div>

              {calibrationStatus.status === "recording" &&
                calibrationStatus.recorded_ranges && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Activity className="w-4 h-4 text-purple-400" />
                      <span className="text-sm font-medium text-slate-300">
                        Live Position Data
                      </span>
                    </div>
                    <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                      <div className="space-y-3">
                        {Object.entries(calibrationStatus.recorded_ranges).map(
                          ([motor, range]) => {
                            const totalRange = range.max - range.min;
                            const currentOffset = range.current - range.min;
                            const progressPercent =
                              totalRange > 0
                                ? (currentOffset / totalRange) * 100
                                : 50;
                            const rangeComplete = isMotorRangeComplete(
                              calibrationStatus.device_type,
                              motor,
                              totalRange,
                            );

                            return (
                              <div key={motor} className="space-y-2">
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-2">
                                    <span className="text-white font-semibold text-sm">
                                      {motor}
                                    </span>
                                    {rangeComplete && (
                                      <CheckCircle
                                        className="w-4 h-4 text-green-400"
                                        aria-label="Range complete"
                                      />
                                    )}
                                  </div>
                                  <span className="text-slate-300 text-xs font-mono">
                                    {range.current}
                                  </span>
                                </div>
                                <div className="relative">
                                  <div className="w-full bg-slate-700 rounded-full h-3">
                                    <div
                                      className="bg-slate-600 h-3 rounded-full relative"
                                      style={{ width: "100%" }}
                                    >
                                      <div
                                        className={`absolute top-0 w-1 h-3 rounded-full transition-all duration-100 ${
                                          rangeComplete
                                            ? "bg-green-400"
                                            : "bg-yellow-400"
                                        }`}
                                        style={{
                                          left: `${Math.max(
                                            0,
                                            Math.min(100, progressPercent),
                                          )}%`,
                                          transform: "translateX(-50%)",
                                        }}
                                      />
                                    </div>
                                  </div>
                                  <div className="flex justify-between text-xs text-slate-400 mt-1">
                                    <span>{range.min}</span>
                                    <span>{range.max}</span>
                                  </div>
                                </div>
                              </div>
                            );
                          },
                        )}
                      </div>
                    </div>
                  </div>
                )}

              {calibrationStatus.status === "connecting" && (
                <Alert className="bg-yellow-900/50 border-yellow-700 text-yellow-200">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    Connecting to the device. Please ensure it's connected.
                  </AlertDescription>
                </Alert>
              )}

              {calibrationStatus.status === "recording" &&
                (() => {
                  const ranges = calibrationStatus.recorded_ranges ?? {};
                  const motors = Object.entries(ranges);
                  const allComplete =
                    motors.length > 0 &&
                    motors.every(([motor, range]) =>
                      isMotorRangeComplete(
                        calibrationStatus.device_type,
                        motor,
                        range.max - range.min,
                      ),
                    );
                  return (
                    <div className="space-y-3">
                      <div className="flex justify-center">
                        <Button
                          onClick={handleCompleteStep}
                          disabled={!calibrationStatus.calibration_active}
                          className={`px-8 py-3 rounded-full transition-colors ${
                            allComplete
                              ? "bg-green-600 hover:bg-green-700"
                              : "bg-orange-500 hover:bg-orange-600"
                          }`}
                        >
                          {allComplete ? (
                            <CheckCircle className="w-4 h-4 mr-2" />
                          ) : (
                            <AlertCircle className="w-4 h-4 mr-2" />
                          )}
                          Save Calibration
                        </Button>
                      </div>
                      <Alert className="bg-purple-900/50 border-purple-700 text-purple-200">
                        <Activity className="h-4 w-4" />
                        <AlertDescription>
                          <strong>Important:</strong> Move each joint through
                          its full range —{" "}
                          <strong>except the wrist roll</strong>: leave it near
                          the middle. It rotates continuously and its range is
                          set automatically. A check appears next to each joint
                          once its range is wide enough.
                        </AlertDescription>
                      </Alert>
                    </div>
                  );
                })()}

              {calibrationStatus.status === "completed" && (
                <Alert className="bg-green-900/50 border-green-700 text-green-200">
                  <CheckCircle className="h-4 w-4" />
                  <AlertDescription>
                    Calibration completed successfully!
                  </AlertDescription>
                </Alert>
              )}

              {calibrationStatus.status === "error" &&
                calibrationStatus.error &&
                (calibrationStatus.error.startsWith(
                  DISCONTINUITY_ERROR_PREFIX,
                ) ? (
                  <Alert className="bg-red-900/50 border-red-700 text-red-200">
                    <XCircle className="h-4 w-4" />
                    <AlertDescription>
                      <div className="font-semibold text-base mb-1">
                        Motor discontinuity detected
                      </div>
                      <div>
                        Make sure to start the calibration with the robot in a
                        middle position — all joints in the middle of their
                        ranges. See the calibration demo below for the correct
                        starting pose.
                      </div>
                    </AlertDescription>
                  </Alert>
                ) : (
                  <Alert className="bg-red-900/50 border-red-700 text-red-200">
                    <XCircle className="h-4 w-4" />
                    <AlertDescription>
                      <strong>Error:</strong> {calibrationStatus.error}
                    </AlertDescription>
                  </Alert>
                ))}

              <div
                ref={demoVideoRef}
                className="bg-slate-900/50 p-4 rounded-lg border border-slate-700"
              >
                <h4 className="font-semibold mb-3 text-slate-200">
                  Calibration Demo:
                </h4>
                <div className="relative rounded-lg overflow-hidden bg-slate-800">
                  <video
                    className="w-full h-auto rounded-md"
                    controls
                    preload="auto"
                    muted
                  >
                    <source
                      src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/calibrate_so101_2.mp4"
                      type="video/mp4"
                    />
                    <p className="text-slate-400 text-sm text-center py-4">
                      Your browser does not support the video tag.
                      <br />
                      <a
                        href="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/calibrate_so101_2.mp4"
                        className="text-blue-400 hover:text-blue-300 underline"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Click here to view the calibration video
                      </a>
                    </p>
                  </video>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {robotName && (
          <Card className="bg-slate-800/60 border-slate-700 backdrop-blur-sm mt-6">
            <CardHeader className="flex-row items-center justify-between space-y-0">
              <CardTitle className="flex items-center gap-2 text-slate-200">
                <Settings className="w-5 h-5 text-blue-400" />
                Attached cameras
              </CardTitle>
              <div className="flex items-center gap-2">
                <Label
                  htmlFor="cameras-toggle"
                  className="text-sm text-slate-400 cursor-pointer"
                >
                  {camerasActive ? "On" : "Off"}
                </Label>
                <Switch
                  id="cameras-toggle"
                  checked={camerasActive}
                  onCheckedChange={handleCamerasActiveChange}
                  className="data-[state=checked]:bg-green-500"
                  aria-label="Turn cameras on or off"
                />
              </div>
            </CardHeader>
            <CardContent>
              {camerasActive ? (
                <CameraConfiguration
                  cameras={cameras}
                  onCamerasChange={handleCamerasChange}
                  releaseStreamsRef={releaseStreamsRef}
                />
              ) : (
                <div className="rounded-lg border border-slate-700 bg-slate-900/40 p-6 text-center space-y-3">
                  <Camera className="w-10 h-10 mx-auto text-slate-500" />
                  <div className="space-y-1">
                    <p className="text-slate-200 font-medium">
                      Cameras are off
                    </p>
                    <p className="text-sm text-slate-400 max-w-md mx-auto">
                      Turn cameras on to scan for connected devices and preview
                      them. The browser may briefly open a camera to read device
                      labels, and configured cameras stay active while previews
                      are visible; your browser will ask for camera permission.
                      Nothing is recorded.
                    </p>
                    {cameras.length > 0 && (
                      <p className="text-xs text-slate-500 pt-1">
                        {cameras.length} camera
                        {cameras.length === 1 ? "" : "s"} saved to this robot.
                      </p>
                    )}
                  </div>
                  <p className="flex items-center justify-center gap-1.5 text-xs text-slate-500">
                    <ShieldQuestion className="w-3.5 h-3.5" />
                    You'll be asked to grant camera access.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Sticky Save / Quit bar. Save is the ONLY path that writes the robot
            record — every port, camera, and motor-power edit stays a local draft
            until pressed. Quit returns to the previous page, confirming first if
            there are unsaved drafts. */}
        {robotName && (
          <div className="sticky bottom-0 z-10 mt-6 -mx-4 flex items-center justify-between gap-3 border-t border-slate-700 bg-slate-900/95 px-4 py-3 backdrop-blur">
            <span
              className={`text-sm ${
                isDirty ? "text-amber-300" : "text-slate-500"
              }`}
            >
              {isDirty ? "Unsaved changes" : "All changes saved"}
            </span>
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={handleQuit}
                className="border-slate-600 text-slate-300 hover:text-white hover:bg-slate-800 rounded-full px-6"
              >
                Quit
              </Button>
              <Button
                onClick={handleSave}
                disabled={!isDirty || saving}
                className="bg-blue-600 hover:bg-blue-700 text-white rounded-full px-8"
              >
                {saving ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <CheckCircle className="w-4 h-4 mr-2" />
                )}
                {saving ? "Saving…" : "Save"}
              </Button>
            </div>
          </div>
        )}

        <AlertDialog open={quitPromptOpen} onOpenChange={setQuitPromptOpen}>
          <AlertDialogContent className="bg-slate-900 border-slate-800 text-white">
            <AlertDialogHeader>
              <AlertDialogTitle>Discard unsaved changes?</AlertDialogTitle>
              <AlertDialogDescription className="text-slate-400">
                You have unsaved configuration changes (ports, cameras, or motor
                torque). Leaving now discards them — nothing was written to the
                robot. Save first to keep them.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter className="flex gap-2 justify-end">
              <AlertDialogCancel className="border-slate-600 text-slate-700 dark:text-slate-300">
                Keep editing
              </AlertDialogCancel>
              <AlertDialogAction
                className="bg-red-600 hover:bg-red-700 text-white"
                onClick={confirmQuit}
              >
                Discard &amp; quit
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </div>
  );
};

export default Calibration;
