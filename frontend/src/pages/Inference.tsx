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
import { StatusPill, type SessionPhase } from "@/components/ui/status-pill";
import { Badge } from "@/components/ui/badge";
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
          navigate("/training");
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
      logoLink={false}
      status={<StatusPill phase={pillPhase} label={pillLabel} />}
    >
      <div className="grid-bg min-h-[calc(100vh-52px)] px-4 pb-28 pt-6">
        <div className="mx-auto w-full max-w-[1280px]">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowStopConfirm(true)}
            disabled={!status.inference_active}
            className="-ml-3 mb-4 text-muted-foreground hover:text-foreground"
          >
            ← Train &amp; Deploy (stop run)
          </Button>

          <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <h1 className="font-display text-3xl font-bold leading-tight tracking-normal text-foreground md:text-5xl">
                Running — {status.policy_ref ?? "unknown policy"}
              </h1>
              <p className="font-mono text-sm text-muted-foreground">
                {status.policy_ref ?? "checkpoint unknown"} · robot active · elapsed{" "}
                {formatTime(timerSeconds)}
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="secondary">
                {isSettingUp ? "setup" : isRunning ? "rollout" : "finished"}
              </Badge>
              <Badge variant="outline">{formatTime(duration)}</Badge>
            </div>
          </div>

          <div className="mb-6 h-1 w-full bg-secondary" aria-label={`progress ${Math.round(pct)}%`}>
            <div
              className={cn(
                "h-1 transition-all duration-500",
                isSettingUp ? "w-full animate-pulse bg-primary/40" : "bg-primary"
              )}
              style={isSettingUp ? undefined : { width: `${pct}%` }}
            />
          </div>

          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
            <Card className="p-6">
              <div className="mb-8 flex items-center justify-between gap-4">
                <div>
                  <h2 className="font-display text-xl font-semibold text-foreground">
                    Run
                  </h2>
                  <p className="font-mono text-xs text-muted-foreground">
                    {isSettingUp
                      ? "loading policy and connecting hardware"
                      : "policy rollout in progress"}
                  </p>
                </div>
                <StatusPill phase={pillPhase} label={pillLabel} />
              </div>
              <div className="font-mono text-6xl font-medium leading-none tracking-normal text-foreground">
                {formatTime(timerSeconds)}
              </div>
              <div className="mt-3 font-mono text-sm text-muted-foreground">
                {isSettingUp ? "setup elapsed" : `/ ${formatTime(duration)}`}
              </div>
              <div className="mt-8 grid gap-3 font-mono text-xs text-muted-foreground sm:grid-cols-3">
                <p>progress {Math.round(pct)}%</p>
                <p>{isSettingUp ? "rollout pending" : "rollout active"}</p>
                <p>{status.log_path ? "log attached" : "log pending"}</p>
              </div>
            </Card>

            <Card className="p-6">
              <div className="mb-6">
                <h2 className="font-display text-xl font-semibold text-foreground">
                  Session
                </h2>
              </div>
              <div className="grid gap-4 text-sm">
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-[0.1em] text-muted-foreground">
                    Model
                  </p>
                  <p className="break-all font-medium text-foreground">
                    {status.policy_ref ?? "(unknown)"}
                  </p>
                </div>
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-[0.1em] text-muted-foreground">
                    Checkpoint
                  </p>
                  <p className="break-all font-medium text-foreground">
                    {status.policy_ref ?? "(unknown)"}
                  </p>
                </div>
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-[0.1em] text-muted-foreground">
                    Robot
                  </p>
                  <p className="font-medium text-foreground">active robot</p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>

      <div className="fixed bottom-0 left-0 right-0 z-30 border-t border-border bg-background/90 px-4 py-4 backdrop-blur-[8px]">
        <div className="mx-auto flex max-w-[1280px] justify-end">
          <Button
            onClick={() => setShowStopConfirm(true)}
            disabled={!status.inference_active}
            variant="destructive"
          >
            <Square />
            Stop run
          </Button>
        </div>
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
