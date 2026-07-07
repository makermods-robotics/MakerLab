import React, { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatNumber } from "@/components/ui/stat-number";
import { TrainingStatus } from "../types";
import { CheckCircle, Activity } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { getJobMetricsHistory } from "@/lib/jobsApi";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface MonitoringStatsProps {
  jobId: string;
  trainingStatus: TrainingStatus;
  getProgressPercentage: () => number;
  formatTime: (seconds: number) => string;
}

interface LossPoint {
  step: number;
  loss: number;
}

interface LrPoint {
  step: number;
  lr: number;
}

const HISTORY_CAP = 2000;

const MonitoringStats: React.FC<MonitoringStatsProps> = ({
  jobId,
  trainingStatus,
  getProgressPercentage,
  formatTime,
}) => {
  const [lossHistory, setLossHistory] = useState<LossPoint[]>([]);
  const [lrHistory, setLrHistory] = useState<LrPoint[]>([]);
  const lastStepRef = useRef(0);
  // The last loss value we actually charted. The backend refreshes
  // current_loss/current_lr only on a real log line (every log_freq steps) and
  // forward-fills the stale values on the tqdm ticks in between. Keying appends
  // off a loss-value change lets us plot one point per real log emission
  // instead of a flat run of duplicated points across the intervening steps.
  const lastLossValRef = useRef<number | null>(null);
  const { baseUrl, fetchWithHeaders } = useApi();

  // Seed the curves from the persisted log on mount (and when the active job
  // changes). Without this, the chart starts empty on every page reload,
  // after navigating away and back, or after a makerlab restart re-attaches to
  // a still-running job. Live-append continues from the last seeded step.
  useEffect(() => {
    let cancelled = false;
    getJobMetricsHistory(baseUrl, fetchWithHeaders, jobId)
      .then((points) => {
        if (cancelled || points.length === 0) return;
        const lossSeed: LossPoint[] = points
          .filter((p) => p.loss != null)
          .map((p) => ({ step: p.step, loss: p.loss as number }))
          .slice(-HISTORY_CAP);
        const lrSeed: LrPoint[] = points
          .filter((p) => p.lr != null)
          .map((p) => ({ step: p.step, lr: p.lr as number }))
          .slice(-HISTORY_CAP);
        setLossHistory(lossSeed);
        setLrHistory(lrSeed);
        // Prime the loss ref so the first live tick (which forward-fills the
        // last logged value) doesn't re-append a point we already seeded.
        lastLossValRef.current = lossSeed[lossSeed.length - 1]?.loss ?? null;
        // Pin lastStepRef to the last seeded step so the first live tick
        // (whose step is >= the seed's last step) doesn't trigger the
        // step-regressed reset in the live-append effect below.
        const lastSeededStep = points[points.length - 1]?.step ?? 0;
        lastStepRef.current = lastSeededStep;
      })
      .catch(() => {
        // 404 or transient — fall through; live ticks will populate from empty.
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders, jobId]);

  // Append new metric points as they arrive; reset when a new run starts
  // (current_step resets back to 0).
  useEffect(() => {
    const step = trainingStatus.current_step;
    if (step < lastStepRef.current) {
      setLossHistory([]);
      setLrHistory([]);
      lastLossValRef.current = null;
    }
    lastStepRef.current = step;

    // A new log line refreshes current_loss (and current_lr alongside it).
    // Between log lines the backend forward-fills the stale values on every
    // tqdm tick, so we key off a loss-value change to detect a genuine new
    // emission and append both series once per real log point — not once per
    // step. Loss effectively never repeats to 4+ decimals, so this won't drop
    // real points, and lr gets a point at each log step even on a constant
    // schedule.
    if (
      step > 0 &&
      trainingStatus.current_loss != null &&
      trainingStatus.current_loss !== lastLossValRef.current
    ) {
      const loss = trainingStatus.current_loss;
      const lr = trainingStatus.current_lr;
      lastLossValRef.current = loss;
      setLossHistory((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.step === step) return prev;
        return [...prev, { step, loss }].slice(-HISTORY_CAP);
      });
      if (lr != null) {
        setLrHistory((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.step === step) return prev;
          return [...prev, { step, lr }].slice(-HISTORY_CAP);
        });
      }
    }
  }, [
    trainingStatus.current_step,
    trainingStatus.current_loss,
    trainingStatus.current_lr,
  ]);

  const progress = getProgressPercentage();
  // Until tqdm fires its first progress line, total_steps is 0 — show
  // "Training starting…" instead of a misleading 0/0 0% reading.
  const isStarting =
    trainingStatus.training_active && trainingStatus.total_steps === 0;
  const etaLabel =
    trainingStatus.eta_seconds != null
      ? formatTime(trainingStatus.eta_seconds)
      : "—";

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="p-6">
          <div className="flex flex-wrap items-end justify-between gap-6">
            <StatNumber
              label="step"
              value={
                isStarting
                  ? "starting…"
                  : trainingStatus.current_step.toLocaleString()
              }
              sublabel={
                isStarting
                  ? "training starting"
                  : `of ${trainingStatus.total_steps.toLocaleString()} steps`
              }
              accent
            />
            <StatNumber label="eta" value={etaLabel} />
          </div>
          <div className="mt-4 h-1 w-full overflow-hidden rounded-sm bg-secondary">
            <div
              className="h-full bg-primary transition-[width] duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="mt-2 font-mono text-[10px] tabular-nums text-muted-foreground">
            {isStarting ? "warming up…" : `${progress.toFixed(1)}%`}
          </p>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-3 text-base text-foreground">
              <div className="flex h-8 w-8 items-center justify-center rounded-md bg-secondary text-muted-foreground">
                <CheckCircle className="h-4 w-4" />
              </div>
              <span>
                Loss{" "}
                <span className="font-mono text-sm font-normal text-muted-foreground">
                  ({trainingStatus.current_loss?.toFixed(4) ?? "—"})
                </span>
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="h-48">
              {lossHistory.length === 0 ? (
                <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                  Waiting for first metric tick…
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={lossHistory}
                    margin={{ top: 8, right: 12, left: 0, bottom: 0 }}
                  >
                    <XAxis
                      dataKey="step"
                      type="number"
                      scale="linear"
                      domain={["dataMin", "dataMax"]}
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                      stroke="#475569"
                    />
                    <YAxis
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                      stroke="#475569"
                      width={48}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#1e293b",
                        border: "1px solid #475569",
                        borderRadius: 8,
                      }}
                      labelStyle={{ color: "#cbd5e1" }}
                      itemStyle={{ color: "#34d399" }}
                      formatter={(v: number) => v.toFixed(4)}
                    />
                    <Line
                      type="monotone"
                      dataKey="loss"
                      stroke="#34d399"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-3 text-base text-foreground">
              <div className="flex h-8 w-8 items-center justify-center rounded-md bg-secondary text-muted-foreground">
                <Activity className="h-4 w-4" />
              </div>
              <span>
                Learning rate{" "}
                <span className="font-mono text-sm font-normal text-muted-foreground">
                  ({trainingStatus.current_lr?.toExponential(2) ?? "—"})
                </span>
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="h-48">
              {lrHistory.length === 0 ? (
                <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                  Waiting for first metric tick…
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={lrHistory}
                    margin={{ top: 8, right: 12, left: 0, bottom: 0 }}
                  >
                    <XAxis
                      dataKey="step"
                      type="number"
                      scale="linear"
                      domain={["dataMin", "dataMax"]}
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                      stroke="#475569"
                    />
                    <YAxis
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                      stroke="#475569"
                      width={48}
                      tickFormatter={(v: number) => v.toExponential(0)}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#1e293b",
                        border: "1px solid #475569",
                        borderRadius: 8,
                      }}
                      labelStyle={{ color: "#cbd5e1" }}
                      itemStyle={{ color: "#fb923c" }}
                      formatter={(v: number) => v.toExponential(2)}
                    />
                    <Line
                      type="monotone"
                      dataKey="lr"
                      stroke="#fb923c"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default MonitoringStats;
