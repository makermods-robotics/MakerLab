import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { FileText } from 'lucide-react';
import { LogEntry } from '../types';

interface TrainingLogsProps {
  logs: LogEntry[];
  logContainerRef: React.RefObject<HTMLDivElement>;
}

const TrainingLogs: React.FC<TrainingLogsProps> = ({ logs, logContainerRef }) => {
  return (
    <Card variant="inverted">
      <CardHeader>
        <CardTitle className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary-foreground/10">
            <FileText className="h-5 w-5" />
          </div>
          Training logs
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div
          ref={logContainerRef}
          className="h-96 overflow-y-auto rounded-md border border-primary-foreground/15 p-4 font-mono text-xs leading-relaxed"
        >
          {logs.length === 0 ? (
            <div className="py-8 text-primary-foreground/50">
              No training logs yet. Start training to see output.
            </div>
          ) : (
            logs.map((log, index) => (
              <div
                key={index}
                className="whitespace-pre-wrap break-words text-primary-foreground/90"
              >
                <span className="mr-2 select-none text-primary-foreground/40">
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
