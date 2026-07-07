import React, { useState, useEffect, useCallback, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useToast } from "@/hooks/use-toast";
import {
  MoreHorizontal,
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
import { AppShell } from "@/components/shell/AppShell";
import { Card } from "@/components/ui/card";
import { Eyebrow } from "@/components/ui/eyebrow";
import { StatusPill, type SessionPhase } from "@/components/ui/status-pill";
import { buttonVariants } from "@/components/ui/button";

interface RecordingConfig {
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
  // discarded (see record.py). The post-recording page shows a "nothing was
  // saved" variant when this is set.
  discarded_empty?: boolean;
  available_controls: {
    stop_recording: boolean;
    exit_early: boolean;
    rerecord_episode: boolean;
  };
}

const Recording = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { toast } = useToast();
  const { baseUrl, wsBaseUrl, fetchWithHeaders } = useApi();

  // Get recording config from navigation state
  const recordingConfig = location.state?.recordingConfig as RecordingConfig;

  // Backend status state - this is the single source of truth
  const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(
    null
  );
  const [recordingSessionStarted, setRecordingSessionStarted] = useState(false);

  const [optimisticPhase, setOptimisticPhase] = useState<Phase | null>(null);
  const [showStopConfirm, setShowStopConfirm] = useState(false);
  const [muted, setMutedState] = useState<boolean>(() => getMuted());
  const prevRealPhaseRef = useRef<Phase | null>(null);
  // Bumps on each re-record so the auto-advance warning re-fires for the same episode number.
  const [rerecordTick, setRerecordTick] = useState(0);
  const warningFiredForPhaseRef = useRef<{ phase: Phase | null; episode: number | null; tick: number }>({ phase: null, episode: null, tick: 0 });
  // Guards against React StrictMode double-invocation of the start effect.
  const startInitiatedRef = useRef(false);
  // Stop the session exactly once, however the user leaves. Set true when a
  // session ends normally (poll navigates to /upload), fails to start, or is
  // stopped via the on-page Stop button — so the page-leave safety net below
  // (back button / unmount / pagehide) never fires a spurious second stop on a
  // completed or never-started session.
  const stoppedRef = useRef(false);
  // The arm-identity guard runs inside the backend's recording worker (after
  // the start response), so its warn-but-allow findings arrive via the status
  // poll. Show them once, not on every 1s tick.
  const identityWarningShownRef = useRef(false);

  const toggleMute = useCallback(() => {
    setMutedState((prev) => {
      const next = !prev;
      persistMuted(next);
      return next;
    });
  }, []);

  // Redirect if no config provided
  useEffect(() => {
    if (!recordingConfig) {
      // No session was ever started here — nothing to stop, so mark the safety
      // net as already-handled before bouncing home.
      stoppedRef.current = true;
      toast({
        title: "No Configuration",
        description: "Please start recording from the main page.",
        variant: "destructive",
      });
      navigate("/");
    }
  }, [recordingConfig, navigate, toast]);

  // Start recording session when component loads. The ref guard prevents
  // React StrictMode (and any future re-renders) from firing /start-recording
  // twice — the second call returns 409 and bounces the user home.
  useEffect(() => {
    if (recordingConfig && !startInitiatedRef.current) {
      startInitiatedRef.current = true;
      startRecordingSession();
    }
    // startRecordingSession is intentionally omitted: re-running this effect
    // on its identity change would re-fire /start-recording.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recordingConfig]);

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
      try {
        const response = await fetchWithHeaders(
          `${baseUrl}/recording-status`
        );
        if (!response.ok) return;
        const status = await response.json();
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
          // spurious /stop-recording against a session that's already gone.
          stoppedRef.current = true;
          const datasetInfo = {
            dataset_repo_id:
              status.dataset_repo_id || recordingConfig.dataset_repo_id,
            single_task: recordingConfig.single_task,
            num_episodes: recordingConfig.num_episodes,
            saved_episodes: status.saved_episodes || 0,
            session_elapsed_seconds: status.session_elapsed_seconds || 0,
            // The backend discards a session that saved zero episodes; when it
            // did, the post-recording page shows a "nothing was saved" variant.
            discarded_empty: status.discarded_empty || false,
          };
          navigate("/upload", { state: { datasetInfo } });
        }
      } catch (error) {
        console.error("Error polling recording status:", error);
      }
    };

    pollStatus();
    const statusInterval = setInterval(pollStatus, 1000);
    return () => clearInterval(statusInterval);
  }, [recordingSessionStarted, recordingConfig, navigate, baseUrl, fetchWithHeaders, toast]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}`;
  };

  // Idempotent stop for page-leave paths (back button, in-app navigation via
  // unmount). Distinct from the on-page Stop button, which keeps its confirm
  // dialog — leaving the page IS the decision, so the safety net skips the
  // dialog. Runs at most once (stoppedRef), so a completed/never-started
  // session, or an already-issued stop, won't POST a spurious second stop.
  const stopRecordingForLeave = useCallback(async () => {
    if (stoppedRef.current) return;
    stoppedRef.current = true;
    try {
      const res = await fetchWithHeaders(`${baseUrl}/stop-recording`, {
        method: "POST",
      });
      const data = await res.json().catch(() => null);
      if (data?.success) {
        // The stop response carries only a session_ending flag; the incomplete
        // in-progress episode (if any) is discarded, previously saved episodes
        // stay saved, and the arm returns to rest before going limp.
        toast({
          title: "Recording stopped",
          description:
            data.message ??
            "The in-progress episode was discarded; saved episodes are kept.",
        });
      }
    } catch {
      /* best-effort — the page is going away regardless */
    }
  }, [baseUrl, fetchWithHeaders, toast]);

  // Cover every page-leave path so a headless recording session can't keep the
  // arm torqued after the user is gone. (Unlike teleop, the active recording
  // screen has no dedicated back button — the only in-page exits are the Stop
  // confirm flow and the browser's own back/close — so the back-button layer
  // collapses into the unmount cleanup here.)
  //   - any in-app navigation unmounts this component → stop via cleanup;
  //   - a browser-level leave (URL change, reload, tab close) never runs React
  //     cleanup, so `pagehide` fires a keepalive stop that survives the unload
  //     and stashes a flag the next page can read. It uses a bare fetch (no JSON
  //     Content-Type) so the request stays a CORS "simple request" and isn't
  //     dropped to a preflight mid-unload.
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
      stopRecordingForLeave();
    };
  }, [baseUrl, stopRecordingForLeave]);

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
        stoppedRef.current = true;
        toast({
          title: "Error Starting Recording",
          description: data.message || "Failed to start recording session.",
          variant: "destructive",
        });
        navigate("/");
      }
    } catch (error) {
      // Never reached the backend — nothing started, nothing to stop.
      stoppedRef.current = true;
      toast({
        title: "Connection Error",
        description: "Could not connect to the backend server.",
        variant: "destructive",
      });
      navigate("/");
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

  const handleStopRecording = useCallback(async () => {
    if (!backendStatus?.available_controls.stop_recording) return;
    try {
      await fetchWithHeaders(`${baseUrl}/stop-recording`, {
        method: "POST",
      });

      toast({
        title: "Stopping recording",
        description: "Finalizing dataset…",
      });
    } catch (error) {
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

  // Re-record is no longer keyboard-driven (the ArrowLeft binding was removed —
  // a back-gesture keystroke shouldn't discard the in-progress episode), so it's
  // reached only via the on-screen dropdown item and doesn't need to sit in the
  // stable keydown-handler ref.
  const handlersRef = useRef({
    handleExitEarly,
    requestStopRecording,
    showStopConfirm,
  });
  useEffect(() => {
    handlersRef.current = {
      handleExitEarly,
      requestStopRecording,
      showStopConfirm,
    };
  });

  const sessionReady = recordingSessionStarted && backendStatus !== null;

  useEffect(() => {
    if (!sessionReady) return;

    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
        return;
      }
      if (e.key === " " || e.code === "Space" || e.key === "ArrowRight") {
        e.preventDefault();
        handlersRef.current.handleExitEarly();
      } else if (e.key === "Escape") {
        if (handlersRef.current.showStopConfirm) return;
        handlersRef.current.requestStopRecording();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [sessionReady]);

  if (!recordingConfig) {
    return (
      <AppShell fullBleed>
        <div className="grid-bg flex min-h-[calc(100vh-52px)] items-center justify-center px-4 py-8">
          <div className="text-center">
            <p className="text-lg text-foreground">
              No recording configuration found.
            </p>
            <Button onClick={() => navigate("/")} variant="brand" className="mt-4">
              Return to home
            </Button>
          </div>
        </div>
      </AppShell>
    );
  }

  // Show loading state while waiting for backend status
  if (!backendStatus) {
    return (
      <AppShell fullBleed>
        <div className="grid-bg flex min-h-[calc(100vh-52px)] items-center justify-center px-4 py-8">
          <div className="text-center">
            <div className="mx-auto mb-4 h-12 w-12 animate-spin rounded-full border-b-2 border-primary"></div>
            <p className="text-lg text-foreground">
              Connecting to recording session…
            </p>
          </div>
        </div>
      </AppShell>
    );
  }

  const realPhase = backendStatus.current_phase as Phase;
  const currentPhase: Phase = optimisticPhase ?? realPhase;
  const currentEpisode = backendStatus.current_episode ?? 1;
  const totalEpisodes =
    backendStatus.total_episodes ?? recordingConfig.num_episodes;

  const phaseElapsedTime = optimisticPhase
    ? 0
    : backendStatus.phase_elapsed_seconds || 0;
  const phaseTimeLimit =
    currentPhase === "recording"
      ? recordingConfig.episode_time_s
      : currentPhase === "resetting"
      ? recordingConfig.reset_time_s
      : backendStatus.phase_time_limit_s || 0;

  const sessionElapsedTime = backendStatus.session_elapsed_seconds || 0;

  const pillPhase: SessionPhase =
    currentPhase === "recording"
      ? "recording"
      : currentPhase === "resetting"
      ? "resetting"
      : currentPhase === "preparing"
      ? "setup"
      : "idle";

  const statusLabel =
    currentPhase === "recording"
      ? `recording · ep ${currentEpisode}/${totalEpisodes}`
      : currentPhase === "resetting"
      ? "reset · get ready"
      : currentPhase === "preparing"
      ? "preparing session"
      : "session complete";

  const primaryLabel =
    currentPhase === "recording"
      ? "End episode"
      : currentPhase === "resetting"
      ? "Start next episode"
      : "Advance";

  const PrimaryIcon = currentPhase === "recording" ? SkipForward : Play;

  return (
    <AppShell
      fullBleed
      logoLink={false}
      status={
        <div role="status" aria-live="polite">
          <StatusPill
            phase={pillPhase}
            label={statusLabel}
            pulse={currentPhase !== "completed"}
          />
        </div>
      }
      actions={
        <Button
          onClick={requestStopRecording}
          disabled={!backendStatus.available_controls.stop_recording}
          variant="destructive"
          size="sm"
        >
          Stop
        </Button>
      }
    >
      <div className="grid-bg flex min-h-[calc(100vh-52px)] items-center justify-center px-4 py-8">
        <Card variant="notch" className="w-full max-w-md p-8">
          <div className="flex items-center justify-between gap-4">
            <Eyebrow>
              [ episode {String(currentEpisode).padStart(2, "0")} ·{" "}
              {recordingConfig.dataset_repo_id} ]
            </Eyebrow>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleMute}
                aria-label={muted ? "Unmute" : "Mute"}
                className="h-8 w-8"
              >
                {muted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    aria-label="More actions"
                  >
                    <MoreHorizontal className="w-5 h-5" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent
                  align="end"
                  onCloseAutoFocus={(e) => e.preventDefault()}
                >
                  <DropdownMenuItem
                    onClick={handleRerecordEpisode}
                    disabled={!backendStatus.available_controls.rerecord_episode}
                  >
                    <RotateCcw className="w-4 h-4 mr-2" />
                    Re-record episode
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>

          <div className="mt-8 text-center">
            <div className="font-mono text-6xl font-bold leading-none text-foreground">
              {formatTime(phaseElapsedTime)}
            </div>
            <div className="mt-2 font-mono text-sm text-muted-foreground">
              / {formatTime(phaseTimeLimit)}
            </div>
          </div>

          <div className="mt-6 h-1 w-full bg-secondary">
            <div
              className="h-1 bg-primary transition-all duration-500"
              style={{
                width: `${Math.min((phaseElapsedTime / phaseTimeLimit) * 100, 100)}%`,
              }}
            />
          </div>

          <Button
            onClick={handleExitEarly}
            disabled={
              !backendStatus.available_controls.exit_early ||
              optimisticPhase !== null ||
              currentPhase === "completed"
            }
            variant="brand"
            className="mt-8 w-full py-6 text-base"
          >
            <PrimaryIcon className="w-5 h-5 mr-2" />
            {primaryLabel}
          </Button>

          {currentPhase !== "completed" && (
            <div className="mt-3 text-center font-mono text-xs text-muted-foreground">
              [ space / → ] {primaryLabel.toLowerCase()}
            </div>
          )}

          <div className="mt-6 flex items-center justify-center gap-4 font-mono text-xs text-muted-foreground">
            <span aria-label={`Episode ${currentEpisode} of ${totalEpisodes}`}>
              ep <span className="text-foreground">{currentEpisode}</span> /{" "}
              {totalEpisodes}
            </span>
            <span
              aria-label={`Total session time ${formatTime(sessionElapsedTime)}`}
            >
              {formatTime(sessionElapsedTime)}
            </span>
          </div>

          {currentPhase === "completed" && (
            <p className="mt-6 text-center text-sm text-muted-foreground">
              Recording complete — redirecting to upload…
            </p>
          )}
        </Card>
      </div>

      <AlertDialog open={showStopConfirm} onOpenChange={setShowStopConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Stop recording?</AlertDialogTitle>
            <AlertDialogDescription>
              Saved episodes are kept. The arm returns to its starting position, then goes limp, and
              you'll be taken to the upload page.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep recording</AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmStopRecording}
              className={buttonVariants({ variant: "destructive" })}
            >
              Stop
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </AppShell>
  );
};

export default Recording;
