import React, { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { TrainingStatus } from '../types';
import { CheckCircle, Activity, Clock } from 'lucide-react';
import { useApi } from '@/contexts/ApiContext';
import { getJobMetricsHistory } from '@/lib/jobsApi';
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

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
  // after navigating away and back, or after a lelab restart re-attaches to
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
  const isStarting = trainingStatus.training_active && trainingStatus.total_steps === 0;
  const stepLabel = isStarting
    ? 'Training starting…'
    : `${trainingStatus.current_step.toLocaleString()} / ${trainingStatus.total_steps.toLocaleString()}`;
  const etaLabel =
    trainingStatus.eta_seconds != null ? formatTime(trainingStatus.eta_seconds) : '—';

  return (
    <div className="space-y-6">
      <Card className="bg-slate-800/50 border-slate-700 rounded-xl">
        <CardContent className="p-6">
          <div className="flex items-baseline justify-between mb-3">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-500/20 text-blue-400">
                <Activity className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-sm text-slate-400">Progress</h3>
                <div className="text-base font-semibold text-white">{stepLabel}</div>
              </div>
            </div>
            <div className="flex items-center gap-2 text-slate-300">
              <Clock className="w-4 h-4 text-purple-400" />
              <span className="text-sm">
                ETA <span className="font-semibold text-white">{etaLabel}</span>
              </span>
            </div>
          </div>
          <div className="relative h-8 w-full overflow-hidden rounded-md bg-slate-900 border border-slate-700">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-sky-400 transition-[width] duration-500"
              style={{ width: `${progress}%` }}
            />
            <div className="absolute inset-0 flex items-center justify-center font-semibold text-white text-sm tabular-nums drop-shadow">
              {isStarting ? 'warming up…' : `${progress.toFixed(1)}%`}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-slate-800/50 border-slate-700 rounded-xl">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-3 text-white text-base">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-green-500/20 text-green-400">
                <CheckCircle className="w-4 h-4" />
              </div>
              <span>
                Loss{' '}
                <span className="text-slate-400 text-sm font-normal">
                  ({trainingStatus.current_loss?.toFixed(4) ?? '—'})
                </span>
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="h-48">
              {lossHistory.length === 0 ? (
                <div className="flex h-full items-center justify-center text-slate-500 text-sm">
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
                      tick={{ fill: '#94a3b8', fontSize: 11 }}
                      stroke="#475569"
                    />
                    <YAxis
                      tick={{ fill: '#94a3b8', fontSize: 11 }}
                      stroke="#475569"
                      width={48}
                    />
                    <Tooltip
                      contentStyle={{
                        background: '#1e293b',
                        border: '1px solid #475569',
                        borderRadius: 8,
                      }}
                      labelStyle={{ color: '#cbd5e1' }}
                      itemStyle={{ color: '#34d399' }}
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

        <Card className="bg-slate-800/50 border-slate-700 rounded-xl">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-3 text-white text-base">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-orange-500/20 text-orange-400">
                <Activity className="w-4 h-4" />
              </div>
              <span>
                Learning Rate{' '}
                <span className="text-slate-400 text-sm font-normal">
                  ({trainingStatus.current_lr?.toExponential(2) ?? '—'})
                </span>
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="h-48">
              {lrHistory.length === 0 ? (
                <div className="flex h-full items-center justify-center text-slate-500 text-sm">
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
                      tick={{ fill: '#94a3b8', fontSize: 11 }}
                      stroke="#475569"
                    />
                    <YAxis
                      tick={{ fill: '#94a3b8', fontSize: 11 }}
                      stroke="#475569"
                      width={48}
                      tickFormatter={(v: number) => v.toExponential(0)}
                    />
                    <Tooltip
                      contentStyle={{
                        background: '#1e293b',
                        border: '1px solid #475569',
                        borderRadius: 8,
                      }}
                      labelStyle={{ color: '#cbd5e1' }}
                      itemStyle={{ color: '#fb923c' }}
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
