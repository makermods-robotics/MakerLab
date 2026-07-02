import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
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
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import Logo from "@/components/Logo";
import { useApi } from "@/contexts/ApiContext";
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
  const [robot, setRobot] = useState<RobotRecord | null>(null);

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
  // Bimanual MUST follow lerobot's "<base>_left"/"<base>_right" convention, so
  // the name is forced to "<robot>_<arm>" regardless of any assigned config.
  // Single-arm recalibration defaults to the in-use config (or the robot name).
  const defaultConfigName = isBimanual
    ? `${robotName}_${arm}`
    : ((assignedConfig?.trim() ? assignedConfig : robotName) ?? "");

  // Editable "save as" name (single-arm only) so one robot can own multiple
  // named calibrations instead of overwriting. Blank falls back to the default;
  // bimanual stays locked to the lerobot naming convention. The field re-syncs
  // to the default whenever the target side changes (device/arm switch, robot
  // load, or a just-saved calibration reassigning the robot).
  const [configNameInput, setConfigNameInput] = useState("");
  useEffect(() => {
    setConfigNameInput(defaultConfigName);
  }, [defaultConfigName]);
  const calibrationConfigName = isBimanual
    ? defaultConfigName
    : configNameInput.trim() || defaultConfigName;

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
        .map((f) => (robot[f] as string) || "")
        .filter(Boolean)
    : [];
  const [overwritePromptOpen, setOverwritePromptOpen] = useState(false);
  const [wiggling, setWiggling] = useState(false);
  // Touch-to-identify: watching every port for a hand-moved shoulder-pan swing.
  const [detecting, setDetecting] = useState(false);
  const [autoCalPromptOpen, setAutoCalPromptOpen] = useState(false);
  const [autoCal, setAutoCal] = useState<{
    active: boolean;
    status: string;
    message: string;
    error: string | null;
    logs: string[];
  }>({ active: false, status: "idle", message: "", error: null, logs: [] });
  const [availablePorts, setAvailablePorts] = useState<string[]>([]);
  const [portsLoading, setPortsLoading] = useState(false);
  const [cameras, setCameras] = useState<CameraConfig[]>([]);
  // Off by default so merely opening the calibration page never grabs a camera.
  // The user explicitly starts a scan, which is when cameras are turned on,
  // enumerated, and the browser permission prompt is requested.
  const [camerasActive, setCamerasActive] = useState(false);
  const cameraSaveTimerRef = useRef<NodeJS.Timeout | null>(null);

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

  // Persist camera changes back to the robot record (debounced).
  const handleCamerasChange = (next: CameraConfig[]) => {
    setCameras(next);
    if (!robotName) return;
    if (cameraSaveTimerRef.current) {
      clearTimeout(cameraSaveTimerRef.current);
    }
    cameraSaveTimerRef.current = setTimeout(async () => {
      try {
        await fetchWithHeaders(
          `${baseUrl}/robots/${encodeURIComponent(robotName)}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ cameras: next }),
          },
        );
      } catch (e) {
        console.error("Failed to save cameras to robot record:", e);
      }
    }, 500);
  };

  useEffect(() => {
    return () => {
      if (cameraSaveTimerRef.current) {
        clearTimeout(cameraSaveTimerRef.current);
      }
    };
  }, []);

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

  // Mirror calibration_active into a ref so the unmount cleanup below can read
  // the latest value without re-firing on every status change.
  const calibrationActiveRef = useRef(false);
  useEffect(() => {
    calibrationActiveRef.current = calibrationStatus.calibration_active;
  }, [calibrationStatus.calibration_active]);

  // If the user leaves this page (back arrow, browser back, programmatic nav)
  // while calibration is running, the backend singleton stays active and the
  // next Start request fails with "Calibration already active". Stop it on
  // unmount as a catch-all.
  useEffect(() => {
    return () => {
      if (calibrationActiveRef.current) {
        fetchWithHeaders(`${baseUrl}/stop-calibration`, {
          method: "POST",
        }).catch((e) =>
          console.error("Failed to stop calibration on unmount:", e),
        );
      }
    };
  }, [baseUrl, fetchWithHeaders]);

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
  // port is assigned to the CURRENT device/arm slot.
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
        setPort(data.port);
        persistPort(data.port);
        toast({
          title: "Arm identified",
          description: `${data.message} Port assigned to this arm.`,
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

  // Resume the auto-cal panel if a run is in progress (e.g. page reload).
  useEffect(() => {
    (async () => {
      try {
        const res = await fetchWithHeaders(
          `${baseUrl}/auto-calibration-status`,
        );
        const data = await res.json();
        setAutoCal(data);
      } catch {
        // ignore
      }
    })();
  }, [baseUrl, fetchWithHeaders]);

  // Poll auto-cal status + logs while a run is active.
  useEffect(() => {
    if (!autoCal.active) return;
    const id = setInterval(async () => {
      try {
        const res = await fetchWithHeaders(
          `${baseUrl}/auto-calibration-status`,
        );
        const data = await res.json();
        setAutoCal(data);
        if (!data.active) {
          if (data.status === "completed") {
            toast({ title: "Auto-calibration complete" });
            setCalibReloadToken((t) => t + 1);
            fetchRobot();
          } else if (data.status === "failed") {
            toast({
              title: "Auto-calibration failed",
              description: data.error || "See the log.",
              variant: "destructive",
            });
          }
        }
      } catch {
        // transient; keep polling
      }
    }, 600);
    return () => clearInterval(id);
  }, [autoCal.active, baseUrl, fetchWithHeaders, fetchRobot, toast]);

  const startAutoCalibration = async () => {
    setAutoCalPromptOpen(false);
    if (!robotName || !port) return;
    try {
      const res = await fetchWithHeaders(`${baseUrl}/start-auto-calibration`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          device_type: deviceType,
          port,
          config_file: calibrationConfigName,
          robot_name: robotName,
          arm,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setAutoCal({
          active: true,
          status: "running",
          message: "",
          error: null,
          logs: [],
        });
        toast({
          title: "Auto-calibration started",
          description: "The arm is moving — keep the workspace clear.",
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

  const stopAutoCalibration = async () => {
    try {
      await fetchWithHeaders(`${baseUrl}/stop-auto-calibration`, {
        method: "POST",
      });
    } catch (e) {
      console.error("Failed to stop auto-calibration:", e);
    }
  };

  const handleStartCalibration = async (overwrite = false) => {
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
      overwrite,
      arm,
    };

    // Optimistically mark as active so the unmount cleanup will fire even if
    // the user navigates away before the backend reports calibration_active=true.
    // Reverted below if the start request fails.
    calibrationActiveRef.current = true;

    try {
      const response = await fetchWithHeaders(`${baseUrl}/start-calibration`, {
        method: "POST",
        body: JSON.stringify(request),
      });

      const result = await response.json();

      if (result.success) {
        setOverwritePromptOpen(false);
        toast({
          title: "Calibration Started",
          description: `Calibration started for ${deviceType}`,
        });
        setIsPolling(true);
      } else if (result.code === "name_taken") {
        // Existing config of the same name — confirm before overwriting.
        calibrationActiveRef.current = false;
        setOverwritePromptOpen(true);
      } else {
        calibrationActiveRef.current = false;
        toast({
          title: "Calibration Failed",
          description: result.message || "Failed to start calibration",
          variant: "destructive",
        });
      }
    } catch (error) {
      calibrationActiveRef.current = false;
      console.error("Error starting calibration:", error);
      toast({
        title: "Error",
        description: "Failed to start calibration",
        variant: "destructive",
      });
    }
  };

  const handleStopCalibration = async () => {
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
    setPort((robot[portField] as string) || "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deviceType, arm, robot]);

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

  // Write the port for the current side straight into the robot record, so a
  // re-detected USB port (which shuffles on reboot/reconnect) sticks without
  // needing a full re-calibration. Mirrors the camera write-back above.
  // An empty string is a valid value: it CLEARS the assignment (arm
  // disconnected), which the backend merge accepts and never treats as a
  // port conflict.
  const persistPort = useCallback(
    async (nextPort: string) => {
      if (!robotName) return;
      // Skip redundant writes when the value already matches the record.
      if (robot && robot[portField] === nextPort) return;
      try {
        const res = await fetchWithHeaders(
          `${baseUrl}/robots/${encodeURIComponent(robotName)}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ [portField]: nextPort }),
          },
        );
        const data = await res.json();
        if (res.ok && data.robot) {
          setRobot(data.robot);
        } else if (!res.ok) {
          // Backstop for the same-port-on-two-arms guard (409).
          toast({
            title: "Couldn't assign port",
            description: data.message || "Failed to save the port.",
            variant: "destructive",
          });
        }
      } catch (e) {
        console.error("Failed to save port to robot record:", e);
      }
    },
    [robotName, portField, robot, baseUrl, fetchWithHeaders, toast],
  );

  // --- Motor power (per-robot, persisted) -------------------------------
  // Local slider position while dragging; persisted to the robot record on
  // release so we don't fire a POST per pixel. Applied to the follower's
  // motors at the start of each teleop/record/inference session.
  const [powerDraft, setPowerDraft] = useState(100);
  useEffect(() => {
    setPowerDraft(robot?.motor_power ?? 100);
  }, [robot?.motor_power]);

  const commitMotorPower = useCallback(async () => {
    if (!robotName || !robot) return;
    // powerDraft may be fractional when the slider is geared in decivolts;
    // persist the nearest integer percent, clamped to the backend's 10-100.
    const percentInt = Math.min(100, Math.max(10, Math.round(powerDraft)));
    if (percentInt === robot.motor_power) return;
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/robots/${encodeURIComponent(robotName)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ motor_power: percentInt }),
        },
      );
      const data = await res.json();
      if (res.ok && data.robot) {
        setRobot(data.robot);
      }
    } catch (e) {
      console.error("Failed to save motor power:", e);
    }
  }, [robotName, robot, powerDraft, baseUrl, fetchWithHeaders]);

  // One-shot supply-voltage reading (a REAL measured voltage from the servos'
  // Present_Voltage register — distinct from the motor-power torque fraction).
  // Read once per port selection, never polled: the backend connects, reads,
  // and releases the port immediately so calibration/teleop can grab it.
  const [voltage, setVoltage] = useState<number | null>(null);
  useEffect(() => {
    setVoltage(null);
    if (!port) return;
    // Don't touch the serial port while a calibration session may hold it.
    if (calibrationStatus.calibration_active || autoCal.active) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetchWithHeaders(
          `${baseUrl}/supply-voltage?port=${encodeURIComponent(port)}`,
        );
        const data = await res.json();
        if (!cancelled && data.success && typeof data.voltage === "number") {
          setVoltage(data.voltage);
        }
      } catch {
        // Informational only — leave the reading blank on failure.
      }
    })();
    return () => {
      cancelled = true;
    };
    // Re-read only when the port changes; the active flags are read at fire
    // time but must not re-trigger a read when a session ends.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [port, baseUrl, fetchWithHeaders]);

  // Slider gearing: with a supply reading, the slider steps in 0.1 V of the
  // computed drive-voltage ceiling (decivolt units: min = 10% floor, max =
  // full supply, step = 1 dv). Without a reading, plain 1% steps.
  const supplyDv =
    voltage != null && voltage > 0 ? Math.round(voltage * 10) : null;
  const powerDv =
    supplyDv != null ? Math.round((supplyDv * powerDraft) / 100) : null;

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
            onClick={() => navigate(-1)}
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
                  <Select
                    value={port}
                    onValueChange={(v) => {
                      setPort(v);
                      persistPort(v);
                    }}
                  >
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
                        const usedByOtherArm = otherArmPorts.includes(p);
                        return (
                          <SelectItem
                            key={p}
                            value={p}
                            disabled={usedByOtherArm}
                            className="text-white"
                          >
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
                      {/* Keep a persisted port selectable even if it's unplugged. */}
                      {port && !availablePorts.includes(port) && (
                        <SelectItem value={port} className="text-white">
                          {port} (saved, not detected)
                        </SelectItem>
                      )}
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
                    disabled={!port}
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
                      autoCal.active
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
                    disabled={!port || wiggling}
                    title="Move the gripper on this port to see which arm it is"
                    className="w-32 shrink-0 border-slate-600 hover:border-yellow-500 text-slate-400 hover:text-yellow-400 bg-slate-700 hover:bg-slate-600"
                  >
                    <Hand className="w-4 h-4 mr-1" />
                    {wiggling ? "Wiggling…" : "Wiggle"}
                  </Button>
                  <p className="flex-1 min-w-[200px] text-xs text-slate-400">
                    Legacy: drives the gripper ±200 ticks to confirm a port —
                    prefer Detect when possible.
                  </p>
                </div>
                {detecting && (
                  <p className="text-xs text-emerald-400">
                    Swing the base of the arm left and right — the port that
                    sees the motion will be assigned to this arm.
                  </p>
                )}
              </div>

              {!isBimanual && (
                <div className="space-y-2">
                  <Label
                    htmlFor="configName"
                    className="text-sm font-medium text-slate-300"
                  >
                    Calibration name
                  </Label>
                  <Input
                    id="configName"
                    value={configNameInput}
                    onChange={(e) => setConfigNameInput(e.target.value)}
                    placeholder={defaultConfigName}
                    disabled={
                      calibrationStatus.calibration_active || autoCal.active
                    }
                    className="bg-slate-700 border-slate-600 text-white rounded-md"
                  />
                  <p className="text-xs text-slate-500">
                    Saves as{" "}
                    <span className="font-mono text-slate-400">
                      {calibrationConfigName || "…"}
                    </span>
                    . Change it to keep the current calibration and save a new
                    one instead of overwriting.
                  </p>
                </div>
              )}

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
                ) : autoCal.active ? (
                  <Button
                    onClick={stopAutoCalibration}
                    variant="destructive"
                    className="w-full rounded-full py-6 text-lg"
                  >
                    <Square className="w-5 h-5 mr-2" />
                    Stop auto-calibration
                  </Button>
                ) : (
                  <>
                    <Button
                      onClick={() => handleStartCalibration()}
                      className="w-full bg-blue-600 hover:bg-blue-700 text-white rounded-full py-6 text-lg"
                      disabled={!robotName || !deviceType || !port}
                    >
                      <Play className="w-5 h-5 mr-2" />
                      Start Calibration
                    </Button>
                    <Button
                      onClick={() => setAutoCalPromptOpen(true)}
                      variant="outline"
                      disabled={!robotName || !deviceType || !port}
                      className="w-full border-purple-500/50 text-purple-700 hover:bg-purple-900/20 hover:text-purple-800 dark:text-purple-300 dark:hover:text-purple-200 rounded-full py-5"
                    >
                      <Wand2 className="w-5 h-5 mr-2" />
                      Auto-calibrate
                    </Button>
                  </>
                )}

                {robot && (
                  <div className="space-y-1 pt-1">
                    <div className="flex items-center gap-3">
                      <Label
                        htmlFor="motorPower"
                        className="text-sm font-medium text-slate-300 shrink-0"
                      >
                        Motor power
                      </Label>
                      <input
                        id="motorPower"
                        type="range"
                        min={supplyDv != null ? Math.round(supplyDv * 0.1) : 10}
                        max={supplyDv ?? 100}
                        step={1}
                        value={powerDv ?? powerDraft}
                        onChange={(e) => {
                          const v = Number(e.target.value);
                          setPowerDraft(
                            supplyDv != null ? (v / supplyDv) * 100 : v,
                          );
                        }}
                        onPointerUp={commitMotorPower}
                        onKeyUp={commitMotorPower}
                        onBlur={commitMotorPower}
                        className="flex-1 h-1.5 accent-blue-500 cursor-pointer"
                        aria-label="Motor power"
                      />
                      {powerDv != null ? (
                        <span
                          className="font-mono text-right leading-tight shrink-0"
                          title="Approximate maximum average drive voltage: supply × power setting."
                        >
                          <span className="block text-sm text-slate-200">
                            ≈ {(powerDv / 10).toFixed(1)} V
                          </span>
                          <span className="block text-[11px] text-slate-500">
                            {Math.round(powerDraft)}%
                          </span>
                        </span>
                      ) : (
                        <span className="text-sm font-mono text-slate-200 w-12 text-right shrink-0">
                          {Math.round(powerDraft)}%
                        </span>
                      )}
                    </div>
                    <div className="flex items-start justify-between gap-2 text-xs text-slate-500">
                      <span>
                        Lower = gentler movements and weaker grip; below 10% the
                        arm can't hold its own weight. Resets to the saved value
                        each session.
                      </span>
                      {voltage != null && (
                        <span
                          className="font-mono text-slate-400 shrink-0"
                          title="Measured servo bus supply voltage on the selected port"
                        >
                          Supply: {voltage.toFixed(1)}V
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {/* Persistent while a session is active: ending calibration
                    (cancel, finish, or error) leaves motor torque released, so
                    the arm won't hold itself up. */}
                {(calibrationStatus.calibration_active || autoCal.active) && (
                  <Alert className="bg-amber-900/40 border-amber-700 text-amber-100">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      {autoCal.active
                        ? "The arm goes limp when auto-calibration ends — motor torque is released on completion or failure and the arm falls under gravity. Keep the space beneath it clear and be ready to support it."
                        : "Motor torque is off — the arm won't hold its pose during calibration, and stays limp after you cancel or finish. Keep it low and supported so it can't drop onto the table edge."}
                    </AlertDescription>
                  </Alert>
                )}

                {autoCal.logs.length > 0 && autoCal.status !== "idle" && (
                  <div className="bg-slate-900 rounded border border-slate-700 p-2 max-h-40 overflow-auto text-xs font-mono text-slate-300 whitespace-pre-wrap">
                    {autoCal.status === "completed" && (
                      <div className="text-green-400 mb-1">
                        ✓ Auto-calibration complete
                      </div>
                    )}
                    {(autoCal.status === "failed" ||
                      autoCal.status === "stopped") && (
                      <div className="text-red-400 mb-1">
                        {autoCal.status === "stopped"
                          ? "Stopped"
                          : `Failed: ${autoCal.error ?? ""}`}
                      </div>
                    )}
                    {autoCal.logs.slice(-120).map((line, i) => (
                      <div key={i}>{line}</div>
                    ))}
                  </div>
                )}
              </div>

              <Dialog
                open={autoCalPromptOpen}
                onOpenChange={setAutoCalPromptOpen}
              >
                <DialogContent className="bg-slate-900 border-slate-800 text-white">
                  <DialogHeader>
                    <DialogTitle>
                      Auto-calibrate — the arm will move
                    </DialogTitle>
                    <DialogDescription className="text-slate-400">
                      The arm will <strong>move on its own under power</strong>{" "}
                      to find each joint's range. Clear the workspace and keep
                      hands away. This will save/replace the calibration{" "}
                      <strong>"{calibrationConfigName}"</strong>.
                    </DialogDescription>
                  </DialogHeader>
                  <DialogFooter className="flex gap-2 justify-end">
                    <Button
                      variant="outline"
                      className="border-slate-600 text-slate-700 dark:text-slate-300"
                      onClick={() => setAutoCalPromptOpen(false)}
                    >
                      Cancel
                    </Button>
                    <Button
                      className="bg-purple-600 hover:bg-purple-700 text-white"
                      onClick={startAutoCalibration}
                    >
                      Start auto-calibration
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>

              <Dialog
                open={overwritePromptOpen}
                onOpenChange={setOverwritePromptOpen}
              >
                <DialogContent className="bg-slate-900 border-slate-800 text-white">
                  <DialogHeader>
                    <DialogTitle>Overwrite existing calibration?</DialogTitle>
                    <DialogDescription className="text-slate-400">
                      A calibration named "{calibrationConfigName}" already
                      exists for this side. Continuing will replace its data
                      when calibration completes. To keep it, cancel and
                      download or rename it first.
                    </DialogDescription>
                  </DialogHeader>
                  <DialogFooter className="flex gap-2 justify-end">
                    <Button
                      variant="outline"
                      className="border-slate-600 text-slate-700 dark:text-slate-300"
                      onClick={() => setOverwritePromptOpen(false)}
                    >
                      Cancel
                    </Button>
                    <Button
                      className="bg-red-500 hover:bg-red-600 text-white"
                      onClick={() => handleStartCalibration(true)}
                    >
                      Overwrite & calibrate
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>

              {robot && (
                <div className="space-y-2 pt-2">
                  <div className="text-sm font-medium text-slate-300">
                    Robot calibration
                  </div>
                  {isBimanual
                    ? // Bimanual configs are fixed by lerobot's convention
                      // ("<robot>_left"/"<robot>_right"), so there's no config
                      // picker — each arm just shows its convention name + whether
                      // it's been calibrated to it.
                      (
                        [
                          {
                            label: "Left Leader (Teleoperator)",
                            cfgField: "leader_config",
                            side: "left",
                          },
                          {
                            label: "Left Follower (Robot)",
                            cfgField: "follower_config",
                            side: "left",
                          },
                          {
                            label: "Right Leader (Teleoperator)",
                            cfgField: "right_leader_config",
                            side: "right",
                          },
                          {
                            label: "Right Follower (Robot)",
                            cfgField: "right_follower_config",
                            side: "right",
                          },
                        ] as const
                      ).map((row) => {
                        const expected = `${robotName}_${row.side}`;
                        const current = (robot[row.cfgField] as string) || "";
                        const compliant = current === expected;
                        return (
                          <div key={row.label}>
                            <div className="flex items-center gap-2 text-sm">
                              {compliant ? (
                                <CheckCircle className="w-4 h-4 text-green-400" />
                              ) : (
                                <Circle className="w-4 h-4 text-slate-500" />
                              )}
                              <span
                                className={
                                  compliant
                                    ? "text-slate-200"
                                    : "text-slate-400"
                                }
                              >
                                {row.label}
                              </span>
                              <span className="ml-auto font-mono text-xs text-slate-500">
                                {expected}
                              </span>
                            </div>
                            {!compliant && (
                              <div className="ml-6 text-xs text-amber-400">
                                {current
                                  ? `Currently "${current}" — recalibrate this arm to use ${expected}.`
                                  : "Not calibrated yet — calibrate this arm."}
                              </div>
                            )}
                          </div>
                        );
                      })
                    : (
                        [
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
                        ] as const
                      ).map((row) => {
                        const cfg = (robot[row.cfgField] as string) || "";
                        return (
                          <div key={row.label}>
                            <div className="flex items-center gap-2 text-sm">
                              {cfg ? (
                                <CheckCircle className="w-4 h-4 text-green-400" />
                              ) : (
                                <Circle className="w-4 h-4 text-slate-500" />
                              )}
                              <span
                                className={
                                  cfg ? "text-slate-200" : "text-slate-400"
                                }
                              >
                                {row.label}
                              </span>
                            </div>
                            <CalibrationLibrary
                              device={row.device}
                              assignedConfig={cfg}
                              configField={row.cfgField}
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
                  onCheckedChange={setCamerasActive}
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
      </div>
    </div>
  );
};

export default Calibration;
