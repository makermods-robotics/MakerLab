import React, { useCallback, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import VisualizerPanel from "@/components/control/VisualizerPanel";
import TeleopCameraPanel from "@/components/control/TeleopCameraPanel";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useRobots } from "@/hooks/useRobots";

const TeleoperationPage = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();
  // The teleop session is for the currently-selected robot; show two arms when
  // it's bimanual.
  const { selectedRecord } = useRobots();
  const bimanual = selectedRecord?.mode === "bimanual";

  // Stop teleoperation exactly once, however the user leaves, so the back
  // button, an in-app link, and the unmount safety net can't double-stop or
  // double-toast.
  const stoppedRef = useRef(false);
  const stopTeleoperation = useCallback(async () => {
    if (stoppedRef.current) return;
    stoppedRef.current = true;
    try {
      const res = await fetchWithHeaders(`${baseUrl}/stop-teleoperation`, {
        method: "POST",
      });
      const data = await res.json();
      if (data?.warning) {
        // Cleanup could not release an arm — torque may still be enabled and
        // the arm can stay rigid. Make this loud instead of claiming success.
        toast({
          title: "Teleoperation stopped — check the arm",
          description: data.warning,
          variant: "destructive",
        });
      } else if (data?.releasing) {
        // The backend drives the follower straight back to its session-start
        // pose (no timed hold), then releases torque.
        toast({
          title: "Teleoperation stopped",
          description:
            data.message ?? "The arm returns to its starting position, then goes limp.",
        });
        // The release happens after this response returns, so check once
        // after the return (progress-based, 10 s ceiling) whether it actually
        // succeeded (the toast store is global, so this fires even after
        // navigating away).
        setTimeout(async () => {
          try {
            const status = await fetchWithHeaders(`${baseUrl}/teleoperation-status`).then((r) =>
              r.json()
            );
            if (status?.last_cleanup_error) {
              toast({
                title: "Check the arm",
                description: status.last_cleanup_error,
                variant: "destructive",
              });
            }
          } catch {
            /* best-effort */
          }
        }, 13000);
      } else if (data?.success) {
        toast({
          title: "Teleoperation stopped",
          description: "The arm was disconnected cleanly.",
        });
      }
    } catch {
      /* best-effort */
    }
  }, [baseUrl, fetchWithHeaders, toast]);

  // Cover every exit path so a session can't keep running and block the next
  // start with "already active":
  //   - the back button awaits stopTeleoperation() then navigates (below);
  //   - any other in-app navigation unmounts this component → stop via cleanup;
  //   - a browser-level leave (URL change, reload, tab close) never runs React
  //     cleanup, so `pagehide` fires a keepalive stop that survives the unload
  //     and stashes a flag the next page reads to confirm the clean disconnect.
  //     It uses a bare fetch (no JSON Content-Type) so the request stays a CORS
  //     "simple request" and isn't dropped to a preflight mid-unload.
  useEffect(() => {
    const handlePageHide = () => {
      try {
        sessionStorage.setItem("lelab:teleop-stopped", "1");
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
      stopTeleoperation();
    };
  }, [baseUrl, stopTeleoperation]);

  const handleGoBack = async () => {
    await stopTeleoperation();
    navigate("/");
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-2 sm:p-4">
      <div className="w-full h-[95vh] flex">
        <VisualizerPanel
          onGoBack={handleGoBack}
          className="lg:w-full"
          bimanual={bimanual}
          rightSlot={<TeleopCameraPanel />}
        />
      </div>
    </div>
  );
};

export default TeleoperationPage;
