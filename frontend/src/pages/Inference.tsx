import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Loader2, Square } from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
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
import { cn } from "@/lib/utils";
import {
  InferenceStatus,
  getInferenceStatus,
  stopInference,
} from "@/lib/inferenceApi";

const POLL_MS = 1000;

function formatTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

const Inference: React.FC = () => {
  const navigate = useNavigate();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [status, setStatus] = useState<InferenceStatus | null>(null);
  const [showStopConfirm, setShowStopConfirm] = useState(false);
  const navigatedAwayRef = useRef(false);
  // Independent flag: we may request a stop (safety net) before the run
  // is actually inactive. We must not flip navigatedAwayRef yet — that
  // would block the natural completion path on the next tick.
  const stopRequestedRef = useRef(false);

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
      try {
        const next = await getInferenceStatus(baseUrl, fetchWithHeaders);
        if (cancelled) return;
        setStatus(next);
        // Auto-bounce home once the run is done.
        if (!next.inference_active && !navigatedAwayRef.current) {
          navigatedAwayRef.current = true;
          if (next.exited) {
            toast({
              title: "Inference finished",
              description:
                next.exit_code === 0
                  ? "Run completed."
                  : `Exit code ${next.exit_code}. See ${next.log_path}.`,
              variant: next.exit_code === 0 ? "default" : "destructive",
            });
          }
          navigate("/");
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
  }, [baseUrl, fetchWithHeaders, navigate, toast]);

  const handleStop = async () => {
    setShowStopConfirm(false);
    try {
      await stopInference(baseUrl, fetchWithHeaders);
      // Status poll will catch the inactive state and navigate home.
    } catch (e) {
      toast({
        title: "Stop failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  if (!status) {
    return (
      <AppShell fullBleed back={{ to: "/" }}>
        <div className="grid-bg flex min-h-[calc(100vh-52px)] items-center justify-center px-4 py-8 text-foreground">
          <Loader2 className="mr-3 h-6 w-6 animate-spin" /> Connecting to inference…
        </div>
      </AppShell>
    );
  }

  const setupElapsed = status.elapsed_s ?? 0;
  const rolloutElapsed = status.rollout_elapsed_s ?? 0;
  const duration = status.duration_s ?? 0;
  const isSettingUp = status.inference_active && status.rollout_started_at == null;
  const isRunning = status.inference_active && status.rollout_started_at != null;
  // When setting up: progress is uncertain — show a soft pulsing bar.
  // When rolling out: progress is rolloutElapsed / duration.
  const pct =
    isRunning && duration > 0
      ? Math.min(100, (rolloutElapsed / duration) * 100)
      : 0;
  const pillPhase: SessionPhase = isSettingUp
    ? "setup"
    : isRunning
    ? "running"
    : "idle";
  const pillLabel = isSettingUp ? "setting up" : isRunning ? "running" : "finished";
  const timerSeconds = isRunning ? rolloutElapsed : setupElapsed;

  return (
    <AppShell
      fullBleed
      back={{ to: "/" }}
      status={<StatusPill phase={pillPhase} label={pillLabel} />}
      actions={
        <Button
          onClick={() => setShowStopConfirm(true)}
          disabled={!status.inference_active}
          variant="destructive"
          size="sm"
        >
          <Square className="w-4 h-4 mr-2" />
          Stop
        </Button>
      }
    >
      <div className="grid-bg flex min-h-[calc(100vh-52px)] items-center justify-center px-4 py-8">
        <Card variant="notch" className="w-full max-w-md p-8">
          <Eyebrow>[ inference run ]</Eyebrow>

          <div className="mt-8 text-center">
            <div className="font-mono text-6xl font-bold leading-none text-foreground">
              {formatTime(timerSeconds)}
            </div>
            <div className="mt-2 font-mono text-sm text-muted-foreground">
              {isSettingUp
                ? "Loading policy & connecting hardware…"
                : `/ ${formatTime(duration)}`}
            </div>
          </div>

          <div className="mt-6 h-1 w-full bg-secondary">
            <div
              className={cn(
                "h-1 transition-all duration-500",
                isSettingUp ? "w-full animate-pulse bg-primary/40" : "bg-primary"
              )}
              style={isSettingUp ? undefined : { width: `${pct}%` }}
            />
          </div>

          <div className="mt-6 break-all font-mono text-xs text-muted-foreground">
            policy: {status.policy_ref ?? "(unknown)"}
          </div>
        </Card>
      </div>

      <AlertDialog open={showStopConfirm} onOpenChange={setShowStopConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Stop inference?</AlertDialogTitle>
            <AlertDialogDescription>
              The follower eases back to the pose it started the run in, then
              releases torque and goes limp. You can launch another run from the
              job tile.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep running</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleStop}
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

export default Inference;
