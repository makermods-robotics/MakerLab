import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  MoreHorizontal,
  Play,
  RotateCcw,
  SkipForward,
  Square,
  Volume2,
  VolumeX,
} from "lucide-react";

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
import { Button, buttonVariants } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import {
  getMuted,
  setMuted as persistMuted,
  playRecordingStartCue,
  playResetStartCue,
  playAutoAdvanceWarning,
} from "@/lib/recordingAudio";
import { cn } from "@/lib/utils";

/** The /start-recording request body. Only the fields the dialog reads are
 * typed; Collect builds the full payload (ports, cameras, torque, …) and it is
 * forwarded verbatim. */
export interface RecordingConfig {
  dataset_repo_id: string;
  single_task: string;
  num_episodes: number;
  episode_time_s: number;
  reset_time_s: number;
  push_to_hub: boolean;
  resume: boolean;
  [key: string]: unknown;
}

/** What a finished session hands back to Collect for the completed banner. */
export interface RecordingResult {
  dataset_repo_id: string;
  single_task: string;
  num_episodes: number;
  saved_episodes: number;
  session_elapsed_seconds: number;
  // The backend discards a session that saved zero episodes; the banner shows
  // a "nothing was saved" variant when this is set.
  discarded_empty: boolean;
}

interface RecordingDialogProps {
  /** The session's request body. The dialog POSTs /start-recording on mount —
   * mount it only when a session should begin. */
  config: RecordingConfig;
  /** Called exactly once when the session is over: with a result when it ran
   * (normally or stopped), with null when it never started. */
  onClose: (result: RecordingResult | null) => void;
}

type Phase = "preparing" | "recording" | "resetting" | "completed";

interface BackendStatus {
  recording_active: boolean;
  current_phase: string;
  current_episode?: number;
  total_episodes?: number;
  saved_episodes?: number;
  phase_elapsed_seconds?: number;
  phase_time_limit_s?: number;
  session_elapsed_seconds?: number;
  session_ended?: boolean;
  dataset_repo_id?: string;
  discarded_empty?: boolean;
  warning?: string;
  available_controls: {
    stop_recording: boolean;
    exit_early: boolean;
    rerecord_episode: boolean;
  };
}

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
};

/**
 * Active-recording session UI, as a modal over Collect (upstream leLab's
 * compact card layout in our design system): meta row, status pill, one giant
 * phase timer, a thin progress bar, and a single phase-colored action button.
 * The reset phase reuses the same card — only colors and the button change.
 */
const RecordingDialog: React.FC<RecordingDialogProps> = ({
  config,
  onClose,
}) => {
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();

  // Backend status is the single source of truth.
  const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(
    null,
  );
  const [recordingSessionStarted, setRecordingSessionStarted] = useState(false);

  const [optimisticPhase, setOptimisticPhase] = useState<Phase | null>(null);
  const [showStopConfirm, setShowStopConfirm] = useState(false);
  const [muted, setMutedState] = useState<boolean>(() => getMuted());
  const prevRealPhaseRef = useRef<Phase | null>(null);
  // Bumps on each re-record so the auto-advance warning re-fires for the same episode number.
  const [rerecordTick, setRerecordTick] = useState(0);
  const warningFiredForPhaseRef = useRef<{
    phase: Phase | null;
    episode: number | null;
    tick: number;
  }>({ phase: null, episode: null, tick: 0 });
  // Guards against React StrictMode double-invocation of the start effect.
  const startInitiatedRef = useRef(false);
  // Stop the session exactly once, however the user leaves. Set true when a
  // session ends normally, fails to start, or is stopped via the Stop flow —
  // so the leave safety net below (unmount / pagehide) never fires a spurious
  // second stop on a completed or never-started session.
  const stoppedRef = useRef(false);
  // The arm-identity guard runs inside the backend's recording worker (after
  // the start response), so its warn-but-allow findings arrive via the status
  // poll. Show them once, not on every 1s tick.
  const identityWarningShownRef = useRef(false);
  // Close the dialog exactly once, whatever ends the session.
  const closedRef = useRef(false);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  const closeOnce = useCallback((result: RecordingResult | null) => {
    if (closedRef.current) return;
    closedRef.current = true;
    onCloseRef.current(result);
  }, []);

  const toggleMute = useCallback(() => {
    setMutedState((prev) => {
      const next = !prev;
      persistMuted(next);
      return next;
    });
  }, []);

  // Start the recording session on mount. The ref guard prevents React
  // StrictMode (and any future re-renders) from firing /start-recording
  // twice — the second call returns 409 and kills the session.
  useEffect(() => {
    if (startInitiatedRef.current) return;
    startInitiatedRef.current = true;

    const start = async () => {
      try {
        const response = await fetchWithHeaders(`${baseUrl}/start-recording`, {
          method: "POST",
          body: JSON.stringify(config),
        });
        const data = await response.json();
        // The endpoint answers HTTP 200 even for refusals ({success: false} —
        // mutex held, existing directory, invalid name), so response.ok alone
        // would leave this non-dismissible dialog polling a session that never
        // started.
        if (response.ok && data?.success !== false) {
          setRecordingSessionStarted(true);
          toast({
            title: "Recording started",
            description: `Recording ${config.num_episodes} episodes to ${config.dataset_repo_id}`,
          });
        } else {
          // The backend rejected the start (409 already-active, config error)
          // — no session is ours to stop, so keep the safety net from firing
          // a stop that would kill an unrelated in-flight session.
          stoppedRef.current = true;
          toast({
            title: "Error starting recording",
            description: data.message || "Failed to start recording session.",
            variant: "destructive",
          });
          closeOnce(null);
        }
      } catch {
        // Never reached the backend — nothing started, nothing to stop.
        stoppedRef.current = true;
        toast({
          title: "Connection error",
          description: "Could not connect to the backend server.",
          variant: "destructive",
        });
        closeOnce(null);
      }
    };
    void start();
    // config is captured for the lifetime of the mount by design: the parent
    // mounts a fresh dialog per session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Refs so the poll interval below stays stable and reads the latest values
  // without tearing itself down on every state change.
  const optimisticPhaseRef = useRef(optimisticPhase);
  optimisticPhaseRef.current = optimisticPhase;
  const rerecordTickRef = useRef(rerecordTick);
  rerecordTickRef.current = rerecordTick;

  // Poll backend status continuously to stay in sync.
  useEffect(() => {
    if (!recordingSessionStarted) return;

    const pollStatus = async () => {
      try {
        const response = await fetchWithHeaders(`${baseUrl}/recording-status`);
        if (!response.ok) return;
        const status: BackendStatus = await response.json();
        setBackendStatus(status);

        if (status.warning && !identityWarningShownRef.current) {
          identityWarningShownRef.current = true;
          toast({
            title: "Recording started with a warning",
            description: status.warning,
            duration: 10000,
          });
        }

        const currentOptimistic = optimisticPhaseRef.current;
        if (currentOptimistic && status.current_phase === currentOptimistic) {
          setOptimisticPhase(null);
        }

        const real = status.current_phase as Phase;
        const prev = prevRealPhaseRef.current;
        if (prev !== real) {
          if (real === "recording" && prev !== null) {
            playRecordingStartCue();
          } else if (real === "resetting") {
            playResetStartCue();
          }
          prevRealPhaseRef.current = real;
          warningFiredForPhaseRef.current = {
            phase: null,
            episode: null,
            tick: 0,
          };
        }

        const elapsed = status.phase_elapsed_seconds || 0;
        const limit = status.phase_time_limit_s || 0;
        const inFinalThreeSeconds = limit > 3 && elapsed >= limit - 3;
        const ep = status.current_episode ?? null;
        const tick = rerecordTickRef.current;
        const warned = warningFiredForPhaseRef.current;
        if (
          inFinalThreeSeconds &&
          currentOptimistic === null &&
          (warned.phase !== real ||
            warned.episode !== ep ||
            warned.tick !== tick)
        ) {
          playAutoAdvanceWarning();
          warningFiredForPhaseRef.current = { phase: real, episode: ep, tick };
        }

        if (!status.recording_active && status.session_ended) {
          // The session finished on its own (or a stop we issued completed and
          // the backend returned to rest). This is a normal exit — mark the
          // safety net as handled so the imminent unmount doesn't POST a
          // spurious /stop-recording against a session that's already gone.
          stoppedRef.current = true;
          closeOnce({
            dataset_repo_id:
              status.dataset_repo_id || config.dataset_repo_id,
            single_task: config.single_task,
            num_episodes: config.num_episodes,
            saved_episodes: status.saved_episodes || 0,
            session_elapsed_seconds: status.session_elapsed_seconds || 0,
            discarded_empty: status.discarded_empty || false,
          });
        }
      } catch (error) {
        console.error("Error polling recording status:", error);
      }
    };

    void pollStatus();
    const statusInterval = setInterval(pollStatus, 1000);
    return () => clearInterval(statusInterval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recordingSessionStarted, baseUrl, fetchWithHeaders]);

  // Idempotent stop for leave paths (unmount while a session is live).
  // Distinct from the Stop button, which keeps its confirm dialog — unmounting
  // IS the decision. Runs at most once (stoppedRef), so a completed or
  // never-started session, or an already-issued stop, won't POST a spurious
  // second stop.
  const stopRecordingForLeave = useCallback(async () => {
    if (stoppedRef.current) return;
    stoppedRef.current = true;
    try {
      const res = await fetchWithHeaders(`${baseUrl}/stop-recording`, {
        method: "POST",
      });
      const data = await res.json().catch(() => null);
      if (data?.success) {
        toast({
          title: "Recording stopped",
          description:
            data.message ??
            "The in-progress episode was discarded; saved episodes are kept.",
        });
      }
    } catch {
      /* best-effort — the dialog is going away regardless */
    }
  }, [baseUrl, fetchWithHeaders, toast]);

  // Cover every leave path so a headless recording session can't keep the arm
  // torqued after the user is gone:
  //   - unmount (in-app navigation away from Collect) → stop via cleanup;
  //   - a browser-level leave (URL change, reload, tab close) never runs React
  //     cleanup, so `pagehide` fires a keepalive stop that survives the unload.
  //     It uses a bare fetch (no JSON Content-Type) so the request stays a
  //     CORS "simple request" and isn't dropped to a preflight mid-unload.
  // Normal completion and start-failure paths set stoppedRef first, so those
  // exits fall through here without issuing a spurious stop.
  useEffect(() => {
    const handlePageHide = () => {
      if (stoppedRef.current) return;
      stoppedRef.current = true;
      try {
        sessionStorage.setItem("makerlab:recording-stopped", "1");
      } catch {
        /* sessionStorage may be unavailable; the stop below still runs */
      }
      fetch(`${baseUrl}/stop-recording`, {
        method: "POST",
        keepalive: true,
      }).catch(() => {});
    };
    window.addEventListener("pagehide", handlePageHide);

    return () => {
      window.removeEventListener("pagehide", handlePageHide);
      void stopRecordingForLeave();
    };
  }, [baseUrl, stopRecordingForLeave]);

  const handleExitEarly = useCallback(async () => {
    if (!backendStatus?.available_controls.exit_early) return;
    if (optimisticPhase !== null) return;

    const realPhase = backendStatus.current_phase as Phase;
    const next: Phase | null =
      realPhase === "recording"
        ? "resetting"
        : realPhase === "resetting"
          ? "recording"
          : null;
    if (!next) return;

    setOptimisticPhase(next);

    try {
      const response = await fetchWithHeaders(
        `${baseUrl}/recording-exit-early`,
        { method: "POST" },
      );
      if (!response.ok) {
        const data = await response.json();
        setOptimisticPhase(null);
        toast({
          title: "Error",
          description: data.message,
          variant: "destructive",
        });
      }
    } catch {
      setOptimisticPhase(null);
      toast({
        title: "Connection error",
        description: "Could not connect to the backend server.",
        variant: "destructive",
      });
    }
  }, [backendStatus, optimisticPhase, baseUrl, fetchWithHeaders, toast]);

  const handleRerecordEpisode = useCallback(async () => {
    if (!backendStatus?.available_controls.rerecord_episode) return;

    try {
      const response = await fetchWithHeaders(
        `${baseUrl}/recording-rerecord-episode`,
        { method: "POST" },
      );
      const data = await response.json();
      if (response.ok) {
        setRerecordTick((t) => t + 1);
        toast({
          title: "Re-recording episode",
          description: `Episode ${backendStatus.current_episode} will be re-recorded.`,
        });
      } else {
        toast({
          title: "Error",
          description: data.message,
          variant: "destructive",
        });
      }
    } catch {
      toast({
        title: "Connection error",
        description: "Could not connect to the backend server.",
        variant: "destructive",
      });
    }
  }, [backendStatus, baseUrl, fetchWithHeaders, toast]);

  const handleStopRecording = useCallback(async () => {
    if (!backendStatus?.available_controls.stop_recording) return;
    try {
      await fetchWithHeaders(`${baseUrl}/stop-recording`, { method: "POST" });
      toast({
        title: "Stopping recording",
        description: "Finalizing dataset…",
      });
    } catch {
      toast({
        title: "Error",
        description: "Failed to stop recording.",
        variant: "destructive",
      });
    }
  }, [backendStatus, baseUrl, fetchWithHeaders, toast]);

  const requestStopRecording = useCallback(() => {
    if (!backendStatus?.available_controls.stop_recording) return;
    setShowStopConfirm(true);
  }, [backendStatus]);

  const confirmStopRecording = useCallback(async () => {
    setShowStopConfirm(false);
    await handleStopRecording();
  }, [handleStopRecording]);

  const handlersRef = useRef({
    handleExitEarly,
    handleRerecordEpisode,
    requestStopRecording,
    showStopConfirm,
  });
  useEffect(() => {
    handlersRef.current = {
      handleExitEarly,
      handleRerecordEpisode,
      requestStopRecording,
      showStopConfirm,
    };
  });

  const sessionReady = recordingSessionStarted && backendStatus !== null;

  // Keyboard shortcuts (upstream leLab's): Space/→ advance, ← re-record,
  // Esc stop (with confirm). Ignored while typing in inputs.
  useEffect(() => {
    if (!sessionReady) return;

    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)
      ) {
        return;
      }
      if (e.key === " " || e.code === "Space" || e.key === "ArrowRight") {
        e.preventDefault();
        void handlersRef.current.handleExitEarly();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        void handlersRef.current.handleRerecordEpisode();
      } else if (e.key === "Escape") {
        if (handlersRef.current.showStopConfirm) return;
        handlersRef.current.requestStopRecording();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [sessionReady]);

  const realPhase = (backendStatus?.current_phase ?? "preparing") as Phase;
  const currentPhase: Phase = optimisticPhase ?? realPhase;
  const currentEpisode = backendStatus?.current_episode ?? 1;
  const totalEpisodes = backendStatus?.total_episodes ?? config.num_episodes;

  const phaseElapsedTime = optimisticPhase
    ? 0
    : backendStatus?.phase_elapsed_seconds || 0;
  const phaseTimeLimit =
    currentPhase === "recording"
      ? config.episode_time_s
      : currentPhase === "resetting"
        ? config.reset_time_s
        : backendStatus?.phase_time_limit_s || 0;
  const sessionElapsedTime = backendStatus?.session_elapsed_seconds || 0;

  const statusText =
    currentPhase === "recording"
      ? `Recording episode ${currentEpisode}`
      : currentPhase === "resetting"
        ? "Reset — get ready"
        : currentPhase === "preparing"
          ? "Preparing session"
          : "Session complete";

  // Phase palette: red dot while the arm records, orange for the reset
  // window; the timer/bar/action track the phase you're IN.
  const dotClass =
    currentPhase === "recording"
      ? "bg-destructive"
      : currentPhase === "resetting"
        ? "bg-warn"
        : currentPhase === "completed"
          ? "bg-ok"
          : "bg-muted-foreground";
  const timerClass =
    currentPhase === "recording"
      ? "text-ok"
      : currentPhase === "resetting"
        ? "text-warn"
        : "text-foreground";
  const barClass =
    currentPhase === "recording"
      ? "bg-ok"
      : currentPhase === "resetting"
        ? "bg-warn"
        : "bg-primary";
  const actionClass =
    currentPhase === "recording"
      ? "bg-ok text-background hover:bg-ok/90"
      : currentPhase === "resetting"
        ? "bg-warn text-background hover:bg-warn/90"
        : "";

  return (
    <Dialog open>
      <DialogContent
        hideClose
        className="sm:max-w-xl"
        aria-describedby={undefined}
        // The session owns the modal: outside clicks and Esc must never
        // silently dismiss it. Esc is handled by the keydown listener above
        // (opens the stop confirm).
        onEscapeKeyDown={(e) => e.preventDefault()}
        onPointerDownOutside={(e) => e.preventDefault()}
        onInteractOutside={(e) => e.preventDefault()}
      >
        <DialogTitle className="sr-only">Recording session</DialogTitle>

        {!sessionReady ? (
          <div className="flex flex-col items-center gap-4 py-16">
            <div className="h-10 w-10 animate-spin rounded-full border-b-2 border-primary" />
            <p className="text-sm text-muted-foreground">
              Connecting to recording session…
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-5">
            {/* Meta row */}
            <div className="flex items-center justify-end gap-4 text-sm text-muted-foreground">
              <span>
                Episode{" "}
                <span className="font-semibold text-foreground">
                  {currentEpisode}
                </span>{" "}
                / {totalEpisodes}
              </span>
              <span className="font-mono">{formatTime(sessionElapsedTime)}</span>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={toggleMute}
                aria-label={muted ? "Unmute" : "Mute"}
              >
                {muted ? (
                  <VolumeX className="h-4 w-4" />
                ) : (
                  <Volume2 className="h-4 w-4" />
                )}
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    aria-label="More options"
                  >
                    <MoreHorizontal className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem
                    disabled={!backendStatus?.available_controls.rerecord_episode}
                    onClick={() => void handleRerecordEpisode()}
                  >
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Re-record episode
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    disabled={!backendStatus?.available_controls.stop_recording}
                    onClick={requestStopRecording}
                    className="text-destructive focus:text-destructive"
                  >
                    <Square className="mr-2 h-4 w-4" />
                    Stop recording
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>

            {/* Status pill */}
            <div className="flex justify-center" role="status" aria-live="polite">
              <span className="inline-flex items-center gap-2 rounded-full border border-border bg-secondary px-3 py-1 font-mono text-xs uppercase tracking-[0.1em] text-foreground">
                <span
                  className={cn(
                    "h-2 w-2 rounded-full",
                    dotClass,
                    currentPhase !== "completed" && "animate-pulse",
                  )}
                />
                {statusText}
              </span>
            </div>

            {/* Phase timer */}
            <div className="text-center">
              <div
                className={cn(
                  "font-mono text-7xl font-bold leading-none tracking-tight",
                  timerClass,
                )}
              >
                {formatTime(phaseElapsedTime)}
              </div>
              {phaseTimeLimit > 0 && (
                <div className="mt-2 font-mono text-sm text-muted-foreground">
                  / {formatTime(phaseTimeLimit)}
                </div>
              )}
            </div>

            {/* Phase progress */}
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
              <div
                className={cn("h-full transition-all duration-500", barClass)}
                style={{
                  width: `${
                    phaseTimeLimit > 0
                      ? Math.min((phaseElapsedTime / phaseTimeLimit) * 100, 100)
                      : 0
                  }%`,
                }}
              />
            </div>

            {/* Primary action */}
            <Button
              onClick={() => void handleExitEarly()}
              disabled={
                !backendStatus?.available_controls.exit_early ||
                optimisticPhase !== null ||
                currentPhase === "completed" ||
                currentPhase === "preparing"
              }
              className={cn("w-full py-6 text-lg", actionClass)}
            >
              {currentPhase === "resetting" ? (
                <Play className="mr-2 h-5 w-5" />
              ) : (
                <SkipForward className="mr-2 h-5 w-5" />
              )}
              {currentPhase === "recording"
                ? "End episode"
                : currentPhase === "resetting"
                  ? "Start next episode"
                  : currentPhase === "completed"
                    ? "Session complete"
                    : "Preparing…"}
              <kbd className="ml-3 rounded border border-border/40 bg-background/20 px-1.5 py-0.5 font-mono text-[10px] font-normal">
                SPACE / →
              </kbd>
            </Button>

            {currentPhase === "completed" ? (
              <p className="text-center text-sm text-muted-foreground">
                Recording complete — finalizing dataset…
              </p>
            ) : (
              <p className="text-center font-mono text-xs text-muted-foreground">
                episodes save to {config.dataset_repo_id}
              </p>
            )}
          </div>
        )}

        <AlertDialog open={showStopConfirm} onOpenChange={setShowStopConfirm}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Stop recording?</AlertDialogTitle>
              <AlertDialogDescription>
                Saved episodes are kept. The arm returns to its starting
                position, then goes limp, and you'll be back on Collect.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Keep recording</AlertDialogCancel>
              <AlertDialogAction
                onClick={() => void confirmStopRecording()}
                className={buttonVariants({ variant: "destructive" })}
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

export default RecordingDialog;
