import React from "react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import {
  AlertTriangle,
  CheckCircle2,
  Copy,
  Loader2,
  XCircle,
} from "lucide-react";
import type { InstallState, LogEntry } from "@/hooks/useInstallExtra";

interface InstallProgressProps {
  state: InstallState;
  error: string | null;
  logs: LogEntry[];
  logBoxRef: React.RefObject<HTMLDivElement>;
  onInstall: () => void;
  onRetry: () => void;

  installHint: string;
  packageName: string;
  idleTitle: string;
  idleDescription: React.ReactNode;
  doneDescription: React.ReactNode;
}

export function installTitle(state: InstallState, idleTitle: string): string {
  switch (state) {
    case "done":
      return "Install complete";
    case "error":
      return "Install failed";
    case "installing":
      return "Installing…";
    default:
      return idleTitle;
  }
}

export function InstallTitleIcon({ state }: { state: InstallState }) {
  if (state === "done") return <CheckCircle2 className="h-6 w-6 text-ok" />;
  if (state === "error") return <XCircle className="h-6 w-6 text-destructive" />;
  if (state === "installing")
    return <Loader2 className="h-6 w-6 animate-spin text-info" />;
  return <AlertTriangle className="h-6 w-6 text-warn" />;
}

export const InstallProgress: React.FC<InstallProgressProps> = ({
  state,
  error,
  logs,
  logBoxRef,
  onInstall,
  onRetry,
  installHint,
  packageName,
  idleDescription,
  doneDescription,
}) => {
  const { toast } = useToast();

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(installHint);
      toast({ title: "Copied", description: installHint });
    } catch {
      toast({
        title: "Copy failed",
        description: "Select the command and copy manually.",
        variant: "destructive",
      });
    }
  };

  return (
    <>
      {state === "idle" && (
        <>
          <p className="text-muted-foreground">{idleDescription}</p>
          <div className="flex items-center gap-2">
            <code className="flex-1 rounded-sm border border-input bg-secondary px-3 py-2 font-mono text-sm text-foreground">
              {installHint}
            </code>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleCopy}
              aria-label="Copy install command"
            >
              <Copy className="w-4 h-4" />
            </Button>
          </div>
          <Button onClick={onInstall}>Install now</Button>
        </>
      )}

      {state === "installing" && (
        <p className="text-muted-foreground">
          Installing{" "}
          <code className="rounded-sm bg-secondary px-1 py-0.5 font-mono text-info">
            {packageName}
          </code>
          . This usually takes about 10 seconds.
        </p>
      )}

      {state === "done" && (
        <div className="space-y-3 text-muted-foreground">{doneDescription}</div>
      )}

      {state === "error" && (
        <>
          <p className="text-destructive">{error || "Install failed."}</p>
          <Button onClick={onRetry} variant="secondary">
            Try again
          </Button>
        </>
      )}

      {state === "error" && logs.length > 0 && (
        <div
          ref={logBoxRef}
          className="h-48 overflow-y-auto whitespace-pre-wrap break-words rounded-sm border border-border bg-secondary p-3 font-mono text-xs text-muted-foreground"
        >
          {logs.map((log, idx) => (
            <div key={idx}>{log.message}</div>
          ))}
        </div>
      )}
    </>
  );
};

// The installed package is only consumed by training subprocesses (fresh
// processes that see the new install immediately), and the backend probes
// availability live per request — so no server restart is needed. Reload the
// page to pick it up.
export const ReadyInstructions: React.FC<{ purpose: string }> = ({
  purpose,
}) => (
  <p>
    Install complete — {purpose} is available immediately, no restart needed.
    Reload the page if it doesn't unlock on its own.
  </p>
);
