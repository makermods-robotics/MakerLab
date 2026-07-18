
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { FileText } from 'lucide-react';
import { LogEntry } from '../types';

interface TrainingLogsProps {
  logs: LogEntry[];
  logContainerRef: React.RefObject<HTMLDivElement>;
}

// A tqdm progress-bar render line, e.g.
//   "Training:  83%|████▎ | 8304/10000 [41:48<08:28,  3.34step/s]"
// lerobot emits one every few steps, so they dominate the log and drown the
// real metric lines. We hide them by default (client-side only — the API keeps
// returning raw lines). Anchored at "Training:" + a percentage + a bar pipe so
// it can't match genuine lines that merely contain the word "training"
// (e.g. "Start offline training on a fixed dataset ...").
const TQDM_BAR_RE = /^\s*Training:\s*\d+%[^|]*\|/;

const TrainingLogs: React.FC<TrainingLogsProps> = ({ logs, logContainerRef }) => {
  const [showRaw, setShowRaw] = React.useState(false);

  const visibleLogs = showRaw
    ? logs
    : logs.filter((log) => !TQDM_BAR_RE.test(log.message));
  const hiddenCount = logs.length - visibleLogs.length;

  return (
    <Card className="bg-slate-800/50 border-slate-700 rounded-xl">
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-3 text-white">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-700">
              <FileText className="w-5 h-5 text-sky-400" />
            </div>
            Training Logs
          </div>
          <button
            type="button"
            onClick={() => setShowRaw((v) => !v)}
            className="text-xs font-medium text-slate-300 hover:text-white bg-slate-700/70 hover:bg-slate-700 border border-slate-600 rounded-md px-2.5 py-1 transition-colors"
            title={
              showRaw
                ? 'Hide tqdm progress-bar lines'
                : 'Show every raw log line, including tqdm progress bars'
            }
          >
            {showRaw
              ? 'Hide progress bars'
              : `Show raw${hiddenCount > 0 ? ` (+${hiddenCount})` : ''}`}
          </button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div
          ref={logContainerRef}
          className="bg-slate-900 rounded-lg p-4 h-96 overflow-y-auto font-mono text-sm border border-slate-700"
        >
          {visibleLogs.length === 0 ? (
            <div className="text-slate-500 py-8">
              No training logs yet. Start training to see output.
            </div>
          ) : (
            visibleLogs.map((log, index) => (
              <div
                key={index}
                className="text-slate-300 break-words whitespace-pre-wrap"
              >
                <span className="text-slate-500 mr-2 select-none">
                  {new Date(log.timestamp * 1000).toLocaleTimeString()}
                </span>
                {log.message}
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default TrainingLogs;
