import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import VisualizerPanel from "@/components/control/VisualizerPanel";
import TeleopCameraPanel from "@/components/control/TeleopCameraPanel";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useRobots } from "@/hooks/useRobots";

export interface TeleopDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/**
 * Live teleoperation in a dialog (not a fullscreen page): the URDF
 * visualizer(s) + camera panel over the launchpad. The session state machine
 * is ported verbatim from pages/Teleoperation.tsx — every exit path (Done,
 * ESC/overlay close, unmount, browser-level leave) stops the session exactly
 * once, and a mid-loop death surfaces as an inline banner.
 */
const TeleopDialog: React.FC<TeleopDialogProps> = ({ open, onOpenChange }) => {
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();
  // The teleop session is for the currently-selected robot; show two arms when
  // it's bimanual.
  const { selectedRecord } = useRobots();
  const bimanual = selectedRecord?.mode === "bimanual";

  // Stop teleoperation exactly once, however the user leaves, so Done, the
  // dialog close, and the unmount safety net can't double-stop or
  // double-toast. Reset when a new session opens the dialog.
  const stoppedRef = useRef(false);

  // Terminal outcome of a session that ended UNDER us (the status poll below
  // caught the worker dying mid-loop) — rendered as an inline banner: failed
  // red, ran_with_warning amber. Null while the session is live.
  const [finished, setFinished] = useState<{
    outcome: "ran_with_warning" | "failed";
    error: string | null;
    hint: string | null;
  } | null>(null);

  // Fresh session per open.
  useEffect(() => {
    if (open) {
      stoppedRef.current = false;
      setFinished(null);
    }
  }, [open]);

  // Poll the session status so a mid-loop death (unplugged bus, camera crash)
  // surfaces here instead of failing silently — the backend clears the
  // outcome fields on start, so a previous session's result can't trigger
  // this on mount. Stops polling once we've caught an outcome or the user
  // initiated a stop (the stop flow owns its own toasts).
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const tick = async () => {
      if (cancelled || stoppedRef.current) return;
      try {
        const res = await fetchWithHeaders(`${baseUrl}/teleoperation-status`);
        if (!res.ok) return;
        const status = await res.json();
        if (cancelled || stoppedRef.current) return;
        if (
          !status.teleoperation_active &&
          !status.releasing &&
          (status.outcome === "failed" || status.outcome === "ran_with_warning")
        ) {
          // The session is already gone — mark the leave safety net handled so
          // close doesn't POST a spurious stop against a dead session.
          stoppedRef.current = true;
          setFinished({
            outcome: status.outcome,
            error: status.error ?? null,
            hint: status.hint ?? null,
          });
        }
      } catch {
        /* best-effort; the next tick retries */
      }
    };
    const id = setInterval(tick, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [open, baseUrl, fetchWithHeaders]);

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
            data.message ??
            "The arm returns to its starting position, then goes limp.",
        });
        // The release happens after this response returns, so check once
        // after the return (progress-based, 10 s ceiling) whether it actually
        // succeeded (the toast store is global, so this fires even after the
        // dialog closes).
        setTimeout(async () => {
          try {
            const status = await fetchWithHeaders(
              `${baseUrl}/teleoperation-status`,
            ).then((r) => r.json());
            if (status?.last_cleanup_error) {
              toast({
                title: "Check the arm",
                // Lead with the plain-language hint when the backend mapped
                // one (e.g. gripper overload) — the raw text follows.
                description: status.hint
                  ? `${status.hint} (${status.last_cleanup_error})`
                  : status.last_cleanup_error,
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

  // Cover every exit path while a session is live so it can't keep running and
  // block the next start with "already active":
  //   - Done and dialog-close await/fire stopTeleoperation;
  //   - unmount while open stops via cleanup;
  //   - a browser-level leave (URL change, reload, tab close) never runs React
  //     cleanup, so `pagehide` fires a keepalive stop that survives the unload
  //     and stashes a flag the next page reads to confirm the clean disconnect.
  //     It uses a bare fetch (no JSON Content-Type) so the request stays a CORS
  //     "simple request" and isn't dropped to a preflight mid-unload.
  useEffect(() => {
    if (!open) return;
    const handlePageHide = () => {
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
      stopTeleoperation();
    };
  }, [open, baseUrl, stopTeleoperation]);

  const handleDone = async () => {
    await stopTeleoperation();
    onOpenChange(false);
  };

  const finishedWarn = finished?.outcome === "ran_with_warning";

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        // ESC / overlay / ✕ — the open-scoped effect's cleanup fires the stop.
        if (!o) stopTeleoperation();
        onOpenChange(o);
      }}
    >
      <DialogContent className="flex h-[85vh] max-w-5xl flex-col gap-3 p-4 sm:p-5">
        {/* Visually the VisualizerPanel's own header titles the dialog; keep an
            sr-only title for the dialog's accessible name. */}
        <DialogHeader className="sr-only">
          <DialogTitle>
            Teleoperation{selectedRecord ? ` — ${selectedRecord.name}` : ""}
          </DialogTitle>
        </DialogHeader>

        {finished && (
          <div
            className={`shrink-0 rounded-lg border p-4 ${
              finishedWarn
                ? "border-warn/40 bg-warn/10"
                : "border-destructive/40 bg-destructive/10"
            }`}
          >
            <div
              className={`flex items-center gap-2 text-sm font-semibold ${
                finishedWarn ? "text-warn" : "text-destructive"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  finishedWarn ? "bg-warn" : "bg-destructive"
                }`}
              />
              {finishedWarn
                ? "Teleoperation ended with a cleanup warning"
                : "Teleoperation failed"}
            </div>
            {finished.hint && (
              <p
                className={`mt-2 text-sm leading-relaxed ${
                  finishedWarn ? "text-warn/90" : "text-destructive/90"
                }`}
              >
                {finished.hint}
              </p>
            )}
            {finished.error && (
              <pre className="mt-3 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-muted p-2 text-xs text-muted-foreground">
                {finished.error}
              </pre>
            )}
          </div>
        )}

        <div className="flex min-h-0 flex-1">
          <VisualizerPanel
            onGoBack={handleDone}
            className="w-full"
            bimanual={bimanual}
            rightSlot={<TeleopCameraPanel />}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default TeleopDialog;
