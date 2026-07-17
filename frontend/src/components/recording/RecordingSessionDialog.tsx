import React, { useState, useEffect, useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import {
  RotateCcw,
  SkipForward,
  Play,
  Volume2,
  VolumeX,
} from "lucide-react";
import {
  getMuted,
  setMuted as persistMuted,
  playRecordingStartCue,
  playResetStartCue,
  playAutoAdvanceWarning,
} from "@/lib/recordingAudio";
import { useApi } from "@/contexts/ApiContext";
import { useSessionExitGuard } from "@/hooks/useSessionExitGuard";
import {
  doneConfirmCopy,
  quitConfirmCopy,
  leaveDiscardMessage,
} from "@/lib/recordingExit";
import LogPanel from "@/components/LogPanel";
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

export interface RecordingConfig {
  leader_port: string;
  follower_port: string;
  leader_config: string;
  follower_config: string;
  // Follower torque limit for the session (10-100% of full power).
  motor_power?: number;
  dataset_repo_id: string;
  single_task: string;
  num_episodes: number;
  episode_time_s: number;
  reset_time_s: number;
  fps: number;
  video: boolean;
  push_to_hub: boolean;
  resume: boolean;
  streaming_encoding: boolean;
}

/** Handoff payload for a session that saved (or discarded) a dataset —
 * consumed by CollectHandoff via router state (see CollectPanel's onExit). */
export interface RecordedInfo {
  repo_id: string;
  saved_episodes: number;
  discarded_empty: boolean;
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
  // True when the session saved zero episodes and its dataset directory was
  // discarded (see record.py). The exit handoff shows a "nothing was saved"
  // variant when this is set.
  discarded_empty?: boolean;
  // Terminal error taxonomy for an ended session (see record.py): `outcome`
  // classifies it, `error` is the caught exception's "Type: message" text,
  // `hint` a plain-language headline. ran_with_warning = the episodes are
  // safe and only teardown tripped — styled amber, not as a failure.
  outcome?: "ok" | "ran_with_warning" | "failed" | null;
  error?: string | null;
  hint?: string | null;
  available_controls: {
    stop_recording: boolean;
    exit_early: boolean;
    rerecord_episode: boolean;
  };
}

/**
 * The live recording session as a modal dialog over the Launchpad/studio —
 * replaces the old /recording page (the session logic is ported verbatim;
 * every navigate-home became `onExit`). The dialog can't be dismissed by
 * ESC / outside click / X: leaving a live session is only ever an explicit
 * Done or Quit (or the shared exit guard's confirmed leave).
 */
const RecordingSessionDialog: React.FC<{
  config: RecordingConfig;
  /** Called for every exit. `recorded` present ⇒ the session ended with a
   * dataset outcome to hand off (train/upload banner); absent ⇒ plain leave
   * (quit, discard, start failure). */
  onExit: (recorded?: RecordedInfo) => void;
}> = ({ config: recordingConfig, onExit }) => {
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();

  // Backend status state - this is the single source of truth
  const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(
    null
  );
  const [recordingSessionStarted, setRecordingSessionStarted] = useState(false);
  const [logs, setLogs] = useState("");

  const [optimisticPhase, setOptimisticPhase] = useState<Phase | null>(null);
  // The two explicit exits while a session is live: Done (finish + keep) and
  // Quit (end without saving / discard). Each gets its own confirm dialog.
  const [showDoneConfirm, setShowDoneConfirm] = useState(false);
  const [showQuitConfirm, setShowQuitConfirm] = useState(false);
  const [muted, setMutedState] = useState<boolean>(() => getMuted());
  const prevRealPhaseRef = useRef<Phase | null>(null);
  // Bumps on each re-record so the auto-advance warning re-fires for the same episode number.
  const [rerecordTick, setRerecordTick] = useState(0);
  const warningFiredForPhaseRef = useRef<{ phase: Phase | null; episode: number | null; tick: number }>({ phase: null, episode: null, tick: 0 });
  // Guards against React StrictMode double-invocation of the start effect.
  const startInitiatedRef = useRef(false);
  // The arm-identity guard runs inside the backend's recording worker (after
  // the start response), so its warn-but-allow findings arrive via the status
  // poll. Show them once, not on every 1s tick.
  const identityWarningShownRef = useRef(false);
  // Set once we've captured an ended session with a failure/warning outcome
  // we're surfacing inline (banner near the log panel, mirroring
  // Inference.tsx). Freezes further polling so a later tick can't clobber the
  // outcome/error/hint we're showing.
  const doneRef = useRef(false);

  // Latest onExit without threading it through every effect dependency (the
  // caller's closure identity may change per render).
  const onExitRef = useRef(onExit);
  onExitRef.current = onExit;

  const toggleMute = useCallback(() => {
    setMutedState((prev) => {
      const next = !prev;
      persistMuted(next);
      return next;
    });
  }, []);

  // The session has ended (any outcome) once the backend reports it inactive
  // AND session_ended. Live chrome (Advance/Stop, HUD, keyboard) unmounts at
  // this point; the guard below disarms so post-end navigation is free.
  const sessionEnded =
    backendStatus !== null &&
    !backendStatus.recording_active &&
    backendStatus.session_ended === true;
  const resume = recordingConfig?.resume ?? false;

  // Unintentional leave (back button, tab close, incidental route change) is
  // treated as QUIT: end WITHOUT saving. Fresh → the backend deletes the whole
  // dataset; resume → already-saved episodes stay, only the in-flight take is
  // dropped. Best-effort discard POST; the guard's own latch runs it once.
  const stopRecordingForLeave = useCallback(async () => {
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/stop-recording?discard=true`,
        { method: "POST" }
      );
      const data = await res.json().catch(() => null);
      if (data?.success) {
        toast({
          title: "Recording discarded",
          description: data.message ?? leaveDiscardMessage(resume),
        });
      }
    } catch {
      /* best-effort — the session UI is going away regardless */
    }
  }, [baseUrl, fetchWithHeaders, toast, resume]);

  // One shared page-leave safety net (also used by Inference & Calibration):
  //  - browser unload → native confirm + keepalive discard beacon;
  //  - in-app back → blocking confirm;
  //  - other in-app nav (this dialog unmounting) → discard on unmount.
  // Armed only while the session is live; disarmed the moment it ends. The
  // deliberate exits (Done/Quit buttons, natural end, start failure) call
  // markHandled() so the guard doesn't fire a spurious second discard.
  const guardActive = recordingSessionStarted && !sessionEnded;
  const { markHandled } = useSessionExitGuard({
    active: guardActive,
    confirmMessage: leaveDiscardMessage(resume),
    beaconUrl: `${baseUrl}/stop-recording?discard=true`,
    onLeave: stopRecordingForLeave,
    beaconFlagKey: "makerlab:recording-stopped",
  });

  // Start recording session when the dialog mounts. The ref guard prevents
  // React StrictMode (and any future re-renders) from firing /start-recording
  // twice — the second call returns 409 and bounces the user out.
  useEffect(() => {
    if (!startInitiatedRef.current) {
      startInitiatedRef.current = true;
      startRecordingSession();
    }
    // startRecordingSession is intentionally omitted: re-running this effect
    // on its identity change would re-fire /start-recording.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Refs so the poll interval below stays stable and reads the latest values
  // without tearing itself down on every state change.
  const optimisticPhaseRef = useRef(optimisticPhase);
  optimisticPhaseRef.current = optimisticPhase;
  const rerecordTickRef = useRef(rerecordTick);
  rerecordTickRef.current = rerecordTick;

  // Poll backend status continuously to stay in sync
  useEffect(() => {
    if (!recordingSessionStarted) return;

    const pollStatus = async () => {
      // Once we've frozen on an ended-with-issue payload, stop polling: a
      // later status would drop the outcome/error/hint we're showing.
      if (doneRef.current) return;
      try {
        const response = await fetchWithHeaders(
          `${baseUrl}/recording-status`
        );
        if (!response.ok) return;
        const status = await response.json();
        setBackendStatus(status);

        // Pull the recording log tail on the same tick so the panel stays live.
        // Best-effort: a log fetch failure must not disturb status handling.
        try {
          const logRes = await fetchWithHeaders(`${baseUrl}/recording-log`);
          if (logRes.ok) {
            const logData = await logRes.json();
            setLogs(logData.logs ?? "");
          }
        } catch {
          // Ignore; the next tick retries.
        }

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
          warningFiredForPhaseRef.current = { phase: null, episode: null, tick: 0 };
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
          // spurious discard against a session that's already gone.
          markHandled();
          // A real failure or a cleanup-only warning: keep the user here so
          // the hint + error banner (near the log panel) is readable instead
          // of bouncing straight to upload. Freeze polling on this payload;
          // the banner's Continue button exits on the user's own click.
          if (
            status.outcome === "failed" ||
            status.outcome === "ran_with_warning"
          ) {
            doneRef.current = true;
            return;
          }
          onExitRef.current({
            repo_id: status.dataset_repo_id || recordingConfig.dataset_repo_id,
            saved_episodes: status.saved_episodes || 0,
            // The backend discards a session that saved zero episodes; when it
            // did, the launchpad handoff shows a "nothing was saved" variant.
            discarded_empty: status.discarded_empty || false,
          });
        }
      } catch (error) {
        console.error("Error polling recording status:", error);
      }
    };

    pollStatus();
    const statusInterval = setInterval(pollStatus, 1000);
    return () => clearInterval(statusInterval);
  }, [recordingSessionStarted, recordingConfig, baseUrl, fetchWithHeaders, toast, markHandled]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}`;
  };

  const startRecordingSession = async () => {
    try {
      const response = await fetchWithHeaders(`${baseUrl}/start-recording`, {
        method: "POST",
        body: JSON.stringify(recordingConfig),
      });

      const data = await response.json();

      if (response.ok) {
        setRecordingSessionStarted(true);
        toast({
          title: "Recording Started",
          description: `Started recording ${recordingConfig.num_episodes} episodes`,
        });
      } else {
        // The backend rejected the start (e.g. 409 already-active, or a config
        // error) — no session is ours to stop, so keep the safety net from
        // firing a stop that would kill an unrelated in-flight session.
        markHandled();
        toast({
          title: "Error Starting Recording",
          description: data.message || "Failed to start recording session.",
          variant: "destructive",
        });
        onExitRef.current();
      }
    } catch (error) {
      // Never reached the backend — nothing started, nothing to stop.
      markHandled();
      toast({
        title: "Connection Error",
        description: "Could not connect to the backend server.",
        variant: "destructive",
      });
      onExitRef.current();
    }
  };

  const handleExitEarly = useCallback(async () => {
    if (!backendStatus?.available_controls.exit_early) return;
    if (optimisticPhase !== null) return;

    const realPhase = backendStatus.current_phase as Phase;
    const next: Phase | null =
      realPhase === "recording" ? "resetting" :
      realPhase === "resetting" ? "recording" : null;

    if (!next) return;

    setOptimisticPhase(next);

    try {
      const response = await fetchWithHeaders(
        `${baseUrl}/recording-exit-early`,
        { method: "POST" }
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
    } catch (error) {
      setOptimisticPhase(null);
      toast({
        title: "Connection Error",
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
        {
          method: "POST",
        }
      );
      const data = await response.json();

      if (response.ok) {
        setRerecordTick((t) => t + 1);
        toast({
          title: "Re-recording Episode",
          description: `Episode ${backendStatus.current_episode} will be re-recorded.`,
        });
      } else {
        toast({
          title: "Error",
          description: data.message,
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Connection Error",
        description: "Could not connect to the backend server.",
        variant: "destructive",
      });
    }
  }, [backendStatus, baseUrl, fetchWithHeaders, toast]);

  // POST the session-end request. discard=false is DONE (keep saved episodes,
  // the poll then exits with the handoff); discard=true is QUIT (end without
  // saving). markHandled() first so the page-leave guard doesn't also fire.
  const doStopRecording = useCallback(
    async (discard: boolean) => {
      if (!backendStatus?.available_controls.stop_recording) return;
      markHandled();
      const url = discard
        ? `${baseUrl}/stop-recording?discard=true`
        : `${baseUrl}/stop-recording`;
      try {
        await fetchWithHeaders(url, { method: "POST" });
        toast(
          discard
            ? { title: "Quitting", description: "Discarding the recording…" }
            : { title: "Finishing", description: "Finalizing dataset…" }
        );
      } catch (error) {
        toast({
          title: "Error",
          description: "Failed to end the recording session.",
          variant: "destructive",
        });
      }
    },
    [backendStatus, baseUrl, fetchWithHeaders, toast, markHandled]
  );

  const requestDone = useCallback(() => {
    if (!backendStatus?.available_controls.stop_recording) return;
    setShowDoneConfirm(true);
  }, [backendStatus]);

  const requestQuit = useCallback(() => {
    if (!backendStatus?.available_controls.stop_recording) return;
    setShowQuitConfirm(true);
  }, [backendStatus]);

  const confirmDone = useCallback(async () => {
    setShowDoneConfirm(false);
    await doStopRecording(false);
    // The poll detects session_ended and exits with the handoff payload.
  }, [doStopRecording]);

  const confirmQuit = useCallback(async () => {
    setShowQuitConfirm(false);
    await doStopRecording(true);
    // Quit means leave now — don't wait for the backend's rest-return/cleanup
    // (it finishes on its own in the worker). Close the dialog.
    onExitRef.current();
  }, [doStopRecording]);

  // Leave the frozen outcome banner for the launchpad handoff — same `recorded`
  // payload the auto-exit builds, on the user's own click. The handoff banner
  // handles the zero-saved case (discarded_empty → "nothing was saved").
  const continueToUpload = useCallback(() => {
    if (!backendStatus) return;
    onExitRef.current({
      repo_id:
        backendStatus.dataset_repo_id || recordingConfig.dataset_repo_id,
      saved_episodes: backendStatus.saved_episodes || 0,
      discarded_empty: backendStatus.discarded_empty || false,
    });
  }, [backendStatus, recordingConfig]);

  // "Discard & exit" from an ENDED failed/warning session that kept episodes on
  // disk: the session is already over (no active-session stop to issue), so
  // delete the dataset directory outright, then close.
  const discardAndExit = useCallback(async () => {
    const repoId =
      backendStatus?.dataset_repo_id || recordingConfig?.dataset_repo_id;
    if (repoId) {
      try {
        await fetchWithHeaders(`${baseUrl}/delete-dataset`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dataset_repo_id: repoId }),
        });
      } catch {
        /* best-effort — we're leaving regardless */
      }
    }
    onExitRef.current();
  }, [backendStatus, recordingConfig, baseUrl, fetchWithHeaders]);

  // Re-record is keyboard-driven again, on BACKSPACE (explicit user request;
  // the old ArrowLeft binding was removed because a back-gesture keystroke
  // discarding the in-progress episode felt accidental — Backspace is a
  // deliberate "delete" gesture, and the handler still preventDefaults so it
  // can never double as browser back-navigation).
  const anyExitDialogOpen = showDoneConfirm || showQuitConfirm;
  const handlersRef = useRef({
    handleExitEarly,
    handleRerecordEpisode,
    requestDone,
    anyExitDialogOpen,
  });
  useEffect(() => {
    handlersRef.current = {
      handleExitEarly,
      handleRerecordEpisode,
      requestDone,
      anyExitDialogOpen,
    };
  });

  // Keyboard shortcuts are LIVE-only: once the session has ended, SPACE/→
  // (advance) and Escape (finish) deactivate so a keystroke can't act on a
  // session that no longer exists.
  const keyboardActive =
    recordingSessionStarted && backendStatus !== null && !sessionEnded;

  useEffect(() => {
    if (!keyboardActive) return;

    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
        return;
      }
      if (e.key === " " || e.code === "Space" || e.key === "ArrowRight") {
        e.preventDefault();
        handlersRef.current.handleExitEarly();
      } else if (e.key === "Backspace") {
        // Deliberate delete gesture: discard and re-record the current episode.
        // preventDefault also guarantees it never acts as browser back-nav.
        e.preventDefault();
        handlersRef.current.handleRerecordEpisode();
      } else if (e.key === "Escape") {
        if (handlersRef.current.anyExitDialogOpen) return;
        // Escape = finish & keep (the least destructive exit). Quit is an
        // explicit button click only.
        handlersRef.current.requestDone();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [keyboardActive]);

  const realPhase = backendStatus?.current_phase as Phase;
  const currentPhase: Phase = optimisticPhase ?? realPhase;
  const currentEpisode = backendStatus?.current_episode ?? 1;
  const totalEpisodes =
    backendStatus?.total_episodes ?? recordingConfig.num_episodes;

  const phaseElapsedTime = optimisticPhase
    ? 0
    : backendStatus?.phase_elapsed_seconds || 0;
  const phaseTimeLimit =
    currentPhase === "recording"
      ? recordingConfig.episode_time_s
      : currentPhase === "resetting"
      ? recordingConfig.reset_time_s
      : backendStatus?.phase_time_limit_s || 0;

  const sessionElapsedTime = backendStatus?.session_elapsed_seconds || 0;

  // An ended session we froze on to surface its outcome (see the poll): a real
  // failure (red) or a cleanup-only warning (amber). `ran_with_warning` must
  // NOT read as the red failed state — the episodes are safe, only teardown
  // was noisy (mirrors Inference.tsx).
  const outcome = backendStatus?.outcome ?? null;
  const endedWithIssue =
    !!backendStatus &&
    !backendStatus.recording_active &&
    backendStatus.session_ended === true &&
    (outcome === "failed" || outcome === "ran_with_warning");
  const endedWarn = endedWithIssue && outcome === "ran_with_warning";

  const getStatusText = () => {
    if (currentPhase === "recording") return `RECORDING EPISODE ${currentEpisode}`;
    if (currentPhase === "resetting") return "RESET — GET READY";
    // Finer "preparing" substeps (and terminal states) the backend now reports
    // (record.py). These aren't in the Phase-narrowed set that drives the
    // recording/resetting colors, so read the raw backend string: they render
    // with the neutral preparing styling below, but the label names which
    // substep the startup is actually in.
    const raw = backendStatus?.current_phase;
    if (raw === "connecting_robot") return "CONNECTING ARM & CAMERAS…";
    if (raw === "connecting_teleop") return "CONNECTING LEADER ARM…";
    if (raw === "stopping") return "STOPPING…";
    if (raw === "error") return "SESSION ERROR — SEE LOG";
    if (currentPhase === "preparing") return "PREPARING SESSION";
    return "SESSION COMPLETE";
  };

  const phaseColor =
    currentPhase === "recording"
      ? { dot: "bg-red-500", pill: "bg-red-500/15 text-red-600 dark:text-red-300", timer: "text-green-600 dark:text-green-400", bar: "bg-green-500", button: "bg-green-500 hover:bg-green-600" }
      : currentPhase === "resetting"
      ? { dot: "bg-orange-500", pill: "bg-orange-500/15 text-orange-600 dark:text-orange-300", timer: "text-orange-600 dark:text-orange-400", bar: "bg-orange-500", button: "bg-orange-500 hover:bg-orange-600" }
      : { dot: "bg-muted-foreground", pill: "bg-muted text-muted-foreground", timer: "text-muted-foreground", bar: "bg-muted-foreground", button: "bg-muted-foreground" };

  const primaryLabel =
    currentPhase === "recording"
      ? "End Episode"
      : currentPhase === "resetting"
      ? "Start Next Episode"
      : "Advance";

  const PrimaryIcon = currentPhase === "recording" ? SkipForward : Play;

  return (
    <Dialog open>
      {/* A live-hardware session owns its exits: no X, and ESC / outside
          clicks never dismiss (ESC instead requests Done via the keyboard
          handler above). */}
      <DialogContent
        hideClose
        onEscapeKeyDown={(e) => e.preventDefault()}
        onPointerDownOutside={(e) => e.preventDefault()}
        onInteractOutside={(e) => e.preventDefault()}
        className="max-h-[92vh] max-w-2xl gap-0 overflow-y-auto p-6"
        aria-describedby={undefined}
      >
        <DialogTitle className="sr-only">Recording session</DialogTitle>

        {/* Loading state while waiting for the first backend status */}
        {!backendStatus ? (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <div className="mb-4 h-12 w-12 animate-spin rounded-full border-b-2 border-red-500" />
            <p className="text-lg">Connecting to recording session...</p>
          </div>
        ) : (
          <>
            {/* Two explicit exits, LIVE-only. Once the session has ended these
                unmount — no control may imply the session is still alive. */}
            {!sessionEnded && (
              <div className="mb-6 flex justify-end gap-3">
                <Button
                  onClick={requestDone}
                  disabled={!backendStatus.available_controls.stop_recording}
                  className="bg-green-600 hover:bg-green-700 text-white flex-shrink-0"
                >
                  Done
                </Button>
                <Button
                  onClick={requestQuit}
                  disabled={!backendStatus.available_controls.stop_recording}
                  variant="outline"
                  className="border-red-500/50 text-red-600 dark:text-red-300 hover:bg-red-500/10 flex-shrink-0"
                >
                  Quit
                </Button>
              </div>
            )}

            <div className="bg-card rounded-lg border border-border p-8">
              {/* LIVE session chrome: episode HUD, phase pill, timers, Advance.
                  Replaced wholesale by the end-state UI once the session ends. */}
              {!sessionEnded && (
                <>
                  <div className="flex justify-end items-center gap-4 mb-6 text-sm text-muted-foreground">
                    <span aria-label={`Episode ${currentEpisode} of ${totalEpisodes}`}>
                      Episode <span className="text-foreground font-semibold">{currentEpisode}</span> / {totalEpisodes}
                    </span>
                    <span className="font-mono" aria-label={`Total session time ${formatTime(sessionElapsedTime)}`}>
                      {formatTime(sessionElapsedTime)}
                    </span>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={toggleMute}
                      aria-label={muted ? "Unmute" : "Mute"}
                      className="h-8 w-8 text-muted-foreground hover:text-foreground hover:bg-muted"
                    >
                      {muted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
                    </Button>
                  </div>

                  <div className="text-center mb-6">
                    <div
                      role="status"
                      aria-live="polite"
                      className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold tracking-widest ${phaseColor.pill}`}
                    >
                      <span className={`w-2 h-2 rounded-full ${phaseColor.dot} ${currentPhase !== "completed" ? "animate-pulse" : ""}`} />
                      {getStatusText()}
                    </div>
                  </div>

                  <div className="text-center mb-4">
                    <div className={`text-7xl font-mono font-bold leading-none ${phaseColor.timer}`}>
                      {formatTime(phaseElapsedTime)}
                    </div>
                    <div className="text-sm text-muted-foreground mt-2">
                      / {formatTime(phaseTimeLimit)}
                    </div>
                  </div>

                  <div className="w-full bg-muted rounded-full h-1.5 mb-8">
                    <div
                      className={`h-1.5 rounded-full transition-all duration-500 ${phaseColor.bar}`}
                      style={{
                        width: `${Math.min((phaseElapsedTime / phaseTimeLimit) * 100, 100)}%`,
                      }}
                    />
                  </div>

                  {/* Primary advance + re-record side by side: retaking a bad take
                      is a first-class action during collection, not a hidden menu
                      item. Backspace mirrors the button (a deliberate delete
                      gesture — see the keydown handler note). */}
                  <div className="flex gap-3">
                    <Button
                      onClick={handleExitEarly}
                      disabled={
                        !backendStatus.available_controls.exit_early ||
                        optimisticPhase !== null ||
                        currentPhase === "completed"
                      }
                      className={`flex-1 text-white font-semibold py-6 text-lg disabled:opacity-50 ${phaseColor.button}`}
                    >
                      <PrimaryIcon className="w-5 h-5 mr-2" />
                      {primaryLabel}
                      {currentPhase !== "completed" && (
                        <span className="ml-3 px-2 py-0.5 rounded text-xs font-mono bg-white/20 text-white/90">SPACE / →</span>
                      )}
                    </Button>
                    <Button
                      onClick={handleRerecordEpisode}
                      disabled={!backendStatus.available_controls.rerecord_episode}
                      variant="outline"
                      className="py-6 text-lg font-semibold border-border bg-transparent text-foreground hover:bg-muted disabled:opacity-50"
                    >
                      <RotateCcw className="w-5 h-5 mr-2" />
                      Re-record
                      <span className="ml-3 px-2 py-0.5 rounded text-xs font-mono bg-muted text-muted-foreground">⌫</span>
                    </Button>
                  </div>
                </>
              )}

              {/* END-STATE UI. A clean finish auto-exits with the handoff (see
                  the poll); this text covers the brief window before that fires. */}
              {sessionEnded && !endedWithIssue && (
                <p className="text-center text-sm text-muted-foreground">
                  Recording complete — returning home…
                </p>
              )}

              {endedWithIssue && (() => {
                const savedEpisodes = backendStatus.saved_episodes ?? 0;
                const keptSomething = savedEpisodes > 0 && !backendStatus.discarded_empty;
                return (
                  <div
                    className={`rounded-lg border p-4 ${
                      endedWarn
                        ? "border-amber-500/40 bg-amber-500/10"
                        : "border-red-500/40 bg-red-500/10"
                    }`}
                  >
                    <div
                      className={`flex items-center gap-2 text-sm font-semibold ${
                        endedWarn
                          ? "text-amber-600 dark:text-amber-300"
                          : "text-red-600 dark:text-red-300"
                      }`}
                    >
                      <span
                        className={`w-2 h-2 rounded-full ${
                          endedWarn ? "bg-amber-500" : "bg-red-500"
                        }`}
                      />
                      {endedWarn
                        ? "Session finished with a cleanup warning — your episodes are safe"
                        : "Recording session failed"}
                    </div>
                    {backendStatus.hint && (
                      <p
                        className={`mt-2 text-sm leading-relaxed ${
                          endedWarn
                            ? "text-amber-700 dark:text-amber-100/90"
                            : "text-red-700 dark:text-red-100/90"
                        }`}
                      >
                        {backendStatus.hint}
                      </p>
                    )}
                    {backendStatus.error && (
                      <pre className="mt-3 max-h-40 overflow-auto rounded bg-muted p-2 text-xs text-muted-foreground whitespace-pre-wrap break-words">
                        {backendStatus.error}
                      </pre>
                    )}
                    <div className="mt-4 flex flex-col gap-2">
                      {/* Keep the saved episodes — hidden when nothing was saved
                          (discarded_empty) since there's no dataset to upload. */}
                      {keptSomething && (
                        <Button
                          onClick={continueToUpload}
                          className="w-full font-semibold"
                        >
                          Keep episodes &amp; continue
                        </Button>
                      )}
                      {/* Discard the kept episodes and leave (quit path from an
                          already-ended session). Only when there's something to
                          discard. */}
                      {keptSomething && (
                        <Button
                          onClick={discardAndExit}
                          variant="outline"
                          className="w-full border-red-500/50 text-red-600 dark:text-red-300 hover:bg-red-500/10"
                        >
                          Discard &amp; exit
                        </Button>
                      )}
                      <Button
                        onClick={() => onExitRef.current()}
                        variant="ghost"
                        className="w-full text-muted-foreground hover:text-foreground hover:bg-muted"
                      >
                        Back to home
                      </Button>
                    </div>
                  </div>
                );
              })()}
            </div>

            <div className="mt-6">
              <LogPanel logs={logs} title="Recording log" defaultCollapsed />
            </div>
          </>
        )}

        <AlertDialog open={showDoneConfirm} onOpenChange={setShowDoneConfirm}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>{doneConfirmCopy().title}</AlertDialogTitle>
              <AlertDialogDescription>
                {doneConfirmCopy().description}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>
                Keep recording
              </AlertDialogCancel>
              <AlertDialogAction
                onClick={confirmDone}
                className="bg-green-600 hover:bg-green-700 text-white"
              >
                Done
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        <AlertDialog open={showQuitConfirm} onOpenChange={setShowQuitConfirm}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>{quitConfirmCopy(resume).title}</AlertDialogTitle>
              <AlertDialogDescription>
                {quitConfirmCopy(resume).description}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>
                Keep recording
              </AlertDialogCancel>
              <AlertDialogAction
                onClick={confirmQuit}
                className="bg-red-500 hover:bg-red-600 text-white"
              >
                Quit without saving
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </DialogContent>
    </Dialog>
  );
};

export default RecordingSessionDialog;
