import React, { useEffect, useRef, useState } from "react";
import { Loader2, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
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
  InferenceStatus,
  InferencePhase,
  getInferenceStatus,
  getInferenceLog,
  stopInference,
} from "@/lib/inferenceApi";
import LogPanel from "@/components/LogPanel";
import { formatBytes } from "@/lib/formatBytes";
import { useSessionExitGuard } from "@/hooks/useSessionExitGuard";

const POLL_MS = 1000;

// Human-readable label + tone for each startup sub-phase. Drives the status
// line above the log panel so a slow startup names its substep ("Downloading
// model…", "Connecting to arm…") instead of an opaque spinner. `pulse` marks
// the still-working phases; terminal phases render steady.
const PHASE_META: Record<
  InferencePhase,
  { label: string; tone: "amber" | "green" | "red"; pulse: boolean }
> = {
  downloading_model: { label: "Downloading model…", tone: "amber", pulse: true },
  starting: { label: "Starting up…", tone: "amber", pulse: true },
  loading_policy: { label: "Loading policy…", tone: "amber", pulse: true },
  connecting: { label: "Connecting to arm…", tone: "amber", pulse: true },
  running: { label: "Running", tone: "green", pulse: true },
  stopping: { label: "Stopping…", tone: "amber", pulse: true },
  stopped: { label: "Stopped", tone: "green", pulse: false },
  error: { label: "Error — see log", tone: "red", pulse: false },
};

const PHASE_DOT: Record<"amber" | "green" | "red", string> = {
  amber: "bg-warn",
  green: "bg-ok",
  red: "bg-destructive",
};

const PHASE_TEXT: Record<"amber" | "green" | "red", string> = {
  amber: "text-warn",
  green: "text-ok",
  red: "text-destructive",
};

// Pill (status chip) background + text per tone. Mirrors the dot/text maps so
// the finished-failed/warning states reuse the same palette as the phases.
const PILL_BG: Record<"amber" | "green" | "red", string> = {
  amber: "bg-warn/15 text-warn",
  green: "bg-ok/15 text-ok",
  red: "bg-destructive/15 text-destructive",
};

function formatTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

/**
 * The live inference run as a modal dialog over whatever launched it —
 * replaces the old /inference page (the polling/safety logic is ported
 * verbatim; every navigate-home became `onExit`). While the run is live
 * (including the connecting/setup window before the first status lands) the
 * dialog can't be dismissed by ESC / outside click / X: leaving stops the arm,
 * so the only exits are the explicit Stop flow, the clean-finish auto-close,
 * or the shared exit guard's confirmed leave. Once the run has ended,
 * dismissal is free.
 */
const InferenceSessionDialog: React.FC<{
  /** Called for every exit — closes the dialog, landing back where the run
   * was launched from. */
  onExit: () => void;
}> = ({ onExit }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [status, setStatus] = useState<InferenceStatus | null>(null);
  const [logs, setLogs] = useState("");
  const [showStopConfirm, setShowStopConfirm] = useState(false);
  const exitedRef = useRef(false);
  // Independent flag: we may request a stop (safety net) before the run
  // is actually inactive. We must not flip exitedRef yet — that
  // would block the natural completion path on the next tick.
  const stopRequestedRef = useRef(false);
  // Set once we've captured a finished (exited) payload we want to stay on —
  // a failure/warning we're surfacing inline. Freezes further polling so the
  // next idle status (which lacks outcome/error/hint, since the subprocess is
  // already reaped) can't clobber the error display.
  const doneRef = useRef(false);
  // The warn-but-allow arm-identity finding now arrives on the status payload
  // (the preflight runs server-side in the background), not the start response.
  // Toast it once when first seen so it isn't repeated on every poll.
  const warnedRef = useRef(false);

  // Safety net: a policy must never keep driving the arm with nobody watching.
  // While a session is active (any phase, INCLUDING downloading_model), an
  // unintentional exit stops the run — in-app back gets a blocking confirm, a
  // browser unload fires a best-effort stop beacon. There's no artifact and no
  // Done/Quit split here: the only semantic is STOP. After the run ends
  // (inference_active false) the guard disarms and navigation is free.
  const { markHandled } = useSessionExitGuard({
    active: status?.inference_active === true,
    confirmMessage: "Leaving stops the running inference. Continue?",
    beaconUrl: `${baseUrl}/stop-inference`,
    onLeave: () => {
      stopInference(baseUrl, fetchWithHeaders).catch(() => {});
    },
    beaconFlagKey: "makerlab:inference-stopped",
  });

  useEffect(() => {
    let cancelled = false;
    const stopIfHung = async () => {
      try {
        await stopInference(baseUrl, fetchWithHeaders);
      } catch {
        // The next status poll will surface the failure if it persists.
      }
    };
    const tick = async () => {
      // Once we've frozen on a finished-with-error payload, stop polling: a
      // later idle status would drop the outcome/error/hint we're showing.
      if (doneRef.current) return;
      try {
        const next = await getInferenceStatus(baseUrl, fetchWithHeaders);
        if (cancelled) return;
        setStatus(next);
        // Surface the server's warn-but-allow arm-identity finding once.
        if (next.warning && !warnedRef.current) {
          warnedRef.current = true;
          toast({
            title: "Started with a warning",
            description: next.warning,
            duration: 10000,
          });
        }
        // Pull the rollout log tail on the same tick so the panel stays live.
        // Best-effort: a log fetch failure must not disturb status handling.
        try {
          const log = await getInferenceLog(baseUrl, fetchWithHeaders);
          if (!cancelled) setLogs(log.logs);
        } catch {
          // Ignore; the next tick retries.
        }
        // Handle a finished run.
        if (!next.inference_active && !exitedRef.current) {
          // A real failure or a cleanup-warning: keep the user here so the
          // hint + error snippet (rendered near the log panel) are readable
          // instead of flashing a toast and bouncing away. Freeze polling on
          // this payload.
          if (next.exited && next.outcome && next.outcome !== "ok") {
            doneRef.current = true;
            return;
          }
          // A clean finish (completed / user stop): toast + auto-close.
          // Mark the exit handled so the leave guard doesn't fire a spurious
          // stop on the imminent unmount.
          markHandled();
          exitedRef.current = true;
          doneRef.current = true;
          if (next.exited) {
            toast({
              title: "Inference finished",
              description: "Run completed.",
            });
          }
          onExit();
          return;
        }
        // Safety net: only fire after the rollout *main loop* has actually
        // started (lerobot honours --duration there). Setup time — policy
        // load, snapshot_download, bus connect, camera connect — can take
        // 10–30s and must NOT count against the user's configured duration.
        if (
          next.inference_active &&
          next.rollout_started_at != null &&
          next.duration_s != null &&
          next.duration_s > 0 &&
          next.rollout_elapsed_s > next.duration_s + 10 &&
          !stopRequestedRef.current
        ) {
          stopRequestedRef.current = true;
          toast({
            title: "Inference seems hung",
            description: `Rollout past duration by ${Math.round(
              next.rollout_elapsed_s - next.duration_s,
            )}s. Stopping.`,
            variant: "destructive",
          });
          stopIfHung();
        }
      } catch (e) {
        if (!cancelled) {
          toast({
            title: "Lost connection to backend",
            description: e instanceof Error ? e.message : String(e),
            variant: "destructive",
          });
        }
      }
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [baseUrl, fetchWithHeaders, onExit, toast, markHandled]);

  const handleStop = async () => {
    setShowStopConfirm(false);
    // Explicit Stop — mark handled so the leave guard doesn't double-fire while
    // the run winds down.
    markHandled();
    try {
      await stopInference(baseUrl, fetchWithHeaders);
      // Status poll will catch the inactive state and close the dialog.
    } catch (e) {
      toast({
        title: "Stop failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  // Dismissal is blocked while the run is (or may still be) live — before the
  // first status lands we treat the session as live, since the launcher just
  // started it.
  const live = status == null || status.inference_active === true;

  const setupElapsed = status?.elapsed_s ?? 0;
  const rolloutElapsed = status?.rollout_elapsed_s ?? 0;
  const duration = status?.duration_s ?? 0;
  const isSettingUp =
    status != null && status.inference_active && status.rollout_started_at == null;
  const isRunning =
    status != null && status.inference_active && status.rollout_started_at != null;

  // A finished run we're staying on to surface (see the tick): a real failure
  // (red) or a cleanup-only warning (amber). `ran_with_warning` must NOT read
  // as the red failed state — the run actually worked, only teardown was noisy.
  const isFinished = status?.exited === true && !status?.inference_active;
  const outcome = status?.outcome ?? null;
  const finishedWarn = isFinished && outcome === "ran_with_warning";
  const finishedFailed = isFinished && outcome === "failed";
  const showOutcome = finishedWarn || finishedFailed;

  // When setting up: progress is uncertain — show a soft pulsing bar.
  // When rolling out: progress is rolloutElapsed / duration.
  const pct =
    isRunning && duration > 0
      ? Math.min(100, (rolloutElapsed / duration) * 100)
      : 0;
  const pillTone: "amber" | "green" | "red" = finishedFailed
    ? "red"
    : finishedWarn
    ? "amber"
    : isSettingUp
    ? "amber"
    : "green";
  const pillLabel = finishedFailed
    ? "FAILED"
    : finishedWarn
    ? "RAN WITH WARNING"
    : isSettingUp
    ? "SETTING UP"
    : isRunning
    ? "RUNNING"
    : "FINISHED";
  const timerSeconds = isRunning ? rolloutElapsed : setupElapsed;

  // Granular startup phase (from the same status poll). Suppressed once we're
  // showing the terminal outcome banner, which carries its own tone + label.
  // Null before any session has seeded a phase, or for an unrecognised value —
  // then we show nothing and let the timer/pill carry the state.
  const phaseMeta =
    !showOutcome && status?.phase ? PHASE_META[status.phase] ?? null : null;

  // Hub model download: show a real byte-progress bar during the
  // downloading_model phase. Indeterminate (pulsing) until the total is known —
  // the total can grow as file sizes are discovered, so the bar may legitimately
  // step backwards. Mirrors the sibling branch's DownloadProgressBar shape.
  const isDownloading = !showOutcome && status?.phase === "downloading_model";
  const dlDone = status?.download_bytes_done ?? null;
  const dlTotal = status?.download_bytes_total ?? null;
  const dlPercent = status?.download_percent ?? null;
  const dlDeterminate = dlPercent != null && dlTotal != null;

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !live) onExit();
      }}
    >
      <DialogContent
        hideClose
        onEscapeKeyDown={(e) => {
          if (live) e.preventDefault();
        }}
        onPointerDownOutside={(e) => {
          if (live) e.preventDefault();
        }}
        onInteractOutside={(e) => {
          if (live) e.preventDefault();
        }}
        className="max-h-[92vh] max-w-xl gap-0 overflow-y-auto p-6"
        aria-describedby={undefined}
      >
        <DialogTitle className="sr-only">Inference session</DialogTitle>

        {!status ? (
          <div className="flex items-center justify-center py-20 text-muted-foreground">
            <Loader2 className="w-6 h-6 animate-spin mr-3" /> Connecting to
            inference…
          </div>
        ) : (
          <div>
            <div className="text-center mb-6">
              <div
                className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold tracking-widest ${PILL_BG[pillTone]}`}
              >
                <span
                  className={`w-2 h-2 rounded-full ${PHASE_DOT[pillTone]} ${
                    isFinished ? "" : "animate-pulse"
                  }`}
                />
                {pillLabel}
              </div>
            </div>

            {!isFinished && (
              <>
                <div className="text-center mb-4">
                  <div
                    className={`text-7xl font-mono font-bold leading-none ${
                      isSettingUp ? "text-warn" : "text-ok"
                    }`}
                  >
                    {formatTime(timerSeconds)}
                  </div>
                  <div className="text-sm text-muted-foreground mt-2">
                    {isSettingUp
                      ? "Loading policy & connecting hardware…"
                      : `/ ${formatTime(duration)}`}
                  </div>
                </div>

                <div className="w-full bg-muted rounded-full h-1.5 mb-8">
                  <div
                    className={`h-1.5 rounded-full transition-all duration-500 ${
                      isSettingUp ? "bg-warn/40 animate-pulse w-full" : "bg-ok"
                    }`}
                    style={isSettingUp ? undefined : { width: `${pct}%` }}
                  />
                </div>
              </>
            )}

            <div className="text-xs text-muted-foreground break-all mb-6">
              policy: {status.policy_ref ?? "(unknown)"}
            </div>

            {showOutcome && (
              <div
                className={`mb-6 rounded-lg border p-4 ${
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
                    className={`w-2 h-2 rounded-full ${
                      finishedWarn ? "bg-warn" : "bg-destructive"
                    }`}
                  />
                  {finishedWarn ? "Ran with a cleanup warning" : "Run failed"}
                </div>
                {status.hint && (
                  <p
                    className={`mt-2 text-sm leading-relaxed ${
                      finishedWarn ? "text-warn/90" : "text-destructive/90"
                    }`}
                  >
                    {status.hint}
                  </p>
                )}
                {status.error && (
                  <pre className="mt-3 max-h-40 overflow-auto rounded bg-muted p-2 text-xs text-muted-foreground whitespace-pre-wrap break-words">
                    {status.error}
                  </pre>
                )}
              </div>
            )}

            {isFinished ? (
              <Button
                onClick={onExit}
                variant="secondary"
                className="w-full font-semibold py-6 text-lg"
              >
                Close
              </Button>
            ) : (
              <Button
                onClick={() => setShowStopConfirm(true)}
                disabled={!status.inference_active}
                variant="destructive"
                className="w-full font-semibold py-6 text-lg disabled:opacity-50"
              >
                <Square className="w-5 h-5 mr-2" />
                Stop
              </Button>
            )}

            {phaseMeta && (
              <div className="mt-6 flex items-center gap-2 text-sm">
                <span
                  className={`w-2 h-2 rounded-full ${PHASE_DOT[phaseMeta.tone]} ${
                    phaseMeta.pulse ? "animate-pulse" : ""
                  }`}
                />
                <span className={`font-medium ${PHASE_TEXT[phaseMeta.tone]}`}>
                  {phaseMeta.label}
                </span>
              </div>
            )}

            {isDownloading && (
              <div className="mt-3 space-y-1">
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                  {dlDeterminate ? (
                    <div
                      className="h-full rounded-full bg-warn transition-[width] duration-500"
                      style={{ width: `${dlPercent}%` }}
                    />
                  ) : (
                    <div className="h-full w-full animate-pulse rounded-full bg-warn/40" />
                  )}
                </div>
                <div className="text-[11px] tabular-nums text-muted-foreground">
                  {dlDeterminate
                    ? `${formatBytes(dlDone ?? 0)} / ${formatBytes(dlTotal)}`
                    : dlDone != null
                      ? `${formatBytes(dlDone)} so far`
                      : "Starting download…"}
                </div>
              </div>
            )}

            <div className="mt-4">
              <LogPanel logs={logs} title="Inference log" defaultCollapsed />
            </div>
          </div>
        )}

        <AlertDialog open={showStopConfirm} onOpenChange={setShowStopConfirm}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Stop inference?</AlertDialogTitle>
              <AlertDialogDescription>
                The follower eases back to the pose it started the run in, then
                releases torque and goes limp. You can launch another run from
                the job tile.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Keep running</AlertDialogCancel>
              <AlertDialogAction
                onClick={handleStop}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                Stop
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </DialogContent>
    </Dialog>
  );
};

export default InferenceSessionDialog;
