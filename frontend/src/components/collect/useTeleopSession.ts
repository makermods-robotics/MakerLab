import { useCallback, useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { RobotRecord } from "@/hooks/useRobots";

interface TeleopStartResponse {
  success?: boolean;
  message?: string;
  warning?: string;
}

interface TeleopStopResponse extends TeleopStartResponse {
  releasing?: boolean;
}

interface TeleopStatusResponse {
  teleoperation_active?: boolean;
  releasing?: boolean;
  last_cleanup_error?: string | null;
  message?: string;
}

const POLL_MS = 2000;

export const useTeleopSession = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [active, setActive] = useState(false);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [status, setStatus] = useState<TeleopStatusResponse | null>(null);
  const activeRef = useRef(false);

  const sessionRunning = active || status?.releasing === true;

  useEffect(() => {
    activeRef.current = sessionRunning;
  }, [sessionRunning]);

  const pollStatus = useCallback(async () => {
    const res = await fetchWithHeaders(`${baseUrl}/teleoperation-status`);
    const data = (await res.json()) as TeleopStatusResponse;
    setStatus(data);
    setActive(data.teleoperation_active === true);
    if (data.last_cleanup_error) {
      toast({
        title: "Check the arm",
        description: data.last_cleanup_error,
        variant: "destructive",
      });
    }
    return data;
  }, [baseUrl, fetchWithHeaders, toast]);

  useEffect(() => {
    if (!sessionRunning) return;
    const id = window.setInterval(() => {
      pollStatus().catch(() => {
        setStatus((prev) => ({
          ...prev,
          message: "Could not read teleoperation status.",
        }));
      });
    }, POLL_MS);
    return () => window.clearInterval(id);
  }, [pollStatus, sessionRunning]);

  const stopTeleoperation = useCallback(async () => {
    setStopping(true);
    try {
      const res = await fetchWithHeaders(`${baseUrl}/stop-teleoperation`, {
        method: "POST",
      });
      const data = (await res.json()) as TeleopStopResponse;
      setStatus(data);
      if (data.warning) {
        toast({
          title: "Teleoperation stopped - check the arm",
          description: data.warning,
          variant: "destructive",
        });
      } else if (data.releasing) {
        toast({
          title: "Teleoperation stopped",
          description:
            data.message ??
            "The arm returns to its starting position, then goes limp.",
        });
      } else if (data.success) {
        toast({
          title: "Teleoperation stopped",
          description: data.message ?? "The arm was disconnected cleanly.",
        });
      }
      setActive(data.releasing === true);
      await pollStatus().catch(() => undefined);
    } catch {
      toast({
        title: "Stop failed",
        description: "Could not connect to the backend server.",
        variant: "destructive",
      });
    } finally {
      setStopping(false);
    }
  }, [baseUrl, fetchWithHeaders, pollStatus, toast]);

  const startTeleoperation = useCallback(
    async (robot: RobotRecord) => {
      setStarting(true);
      try {
        const res = await fetchWithHeaders(`${baseUrl}/move-arm`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            leader_port: robot.leader_port,
            follower_port: robot.follower_port,
            leader_config: robot.leader_config,
            follower_config: robot.follower_config,
            // Bimanual: include the mode + right arm so the backend builds a BiSO pair.
            mode: robot.mode,
            right_leader_port: robot.right_leader_port,
            right_follower_port: robot.right_follower_port,
            right_leader_config: robot.right_leader_config,
            right_follower_config: robot.right_follower_config,
            // Robot name -> BiSO staging base id (bimanual). Names the per-session
            // staging dir; does not affect which calibration drives which arm.
            robot_name: robot.name,
            // Raw follower torque limit for the session (0-1000, default 380).
            max_torque_limit: robot.max_torque_limit ?? 380,
          }),
        });
        const data = (await res.json()) as TeleopStartResponse;
        if (res.ok && data.success) {
          if (data.warning) {
            toast({
              title: "Started with a warning",
              description: data.warning,
              duration: 10000,
            });
          } else {
            toast({
              title: "Teleoperation Started",
              description:
                data.message || `Started teleoperation for ${robot.name}.`,
            });
          }
          setActive(true);
          await pollStatus().catch(() => undefined);
        } else {
          toast({
            title: "Error Starting Teleoperation",
            description: data.message || "Failed to start.",
            variant: "destructive",
          });
        }
      } catch {
        toast({
          title: "Connection Error",
          description: "Could not connect to the backend server.",
          variant: "destructive",
        });
      } finally {
        setStarting(false);
      }
    },
    [baseUrl, fetchWithHeaders, pollStatus, toast],
  );

  useEffect(() => {
    const handlePageHide = () => {
      if (!activeRef.current) return;
      try {
        sessionStorage.setItem("makerlab:teleop-stopped", "1");
      } catch {
        /* sessionStorage may be unavailable; the stop below still runs */
      }
      fetch(`${baseUrl}/stop-teleoperation`, {
        method: "POST",
        keepalive: true,
      }).catch(() => {});
    };
    window.addEventListener("pagehide", handlePageHide);

    return () => {
      window.removeEventListener("pagehide", handlePageHide);
      if (activeRef.current) {
        fetchWithHeaders(`${baseUrl}/stop-teleoperation`, {
          method: "POST",
        }).catch(() => {});
      }
    };
  }, [baseUrl, fetchWithHeaders]);

  return {
    active: sessionRunning,
    starting,
    stopping,
    status,
    startTeleoperation,
    stopTeleoperation,
  };
};
