import React from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { FileText } from "lucide-react";
import { LogEntry } from "../types";

interface TrainingLogsProps {
  logs: LogEntry[];
  logContainerRef: React.RefObject<HTMLDivElement>;
}

const TrainingLogs: React.FC<TrainingLogsProps> = ({
  logs,
  logContainerRef,
}) => {
  return (
    <Card className="bg-card border-border rounded-md">
      <CardHeader className="pb-2">
        <h3 className="eyebrow flex items-center gap-1.5">
          <FileText className="h-3.5 w-3.5" /> Training logs
        </h3>
      </CardHeader>
      <CardContent>
        <div
          ref={logContainerRef}
          className="h-96 overflow-y-auto rounded-md border border-border bg-muted p-4 font-mono text-xs"
        >
          {logs.length === 0 ? (
            <div className="py-8 text-muted-foreground">
              No training logs yet. Start training to see output.
            </div>
          ) : (
            logs.map((log, index) => (
              <div
                key={index}
                className="text-foreground break-words whitespace-pre-wrap"
              >
                <span className="text-muted-foreground mr-2 select-none">
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
