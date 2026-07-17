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
      return "Install Complete";
    case "error":
      return "Install Failed";
    case "installing":
      return "Installing…";
    default:
      return idleTitle;
  }
}

export function InstallTitleIcon({ state }: { state: InstallState }) {
  if (state === "done") return <CheckCircle2 className="w-6 h-6 text-ok" />;
  if (state === "error") return <XCircle className="w-6 h-6 text-destructive" />;
  if (state === "installing")
    return <Loader2 className="w-6 h-6 text-info animate-spin" />;
  return <AlertTriangle className="w-6 h-6 text-warn" />;
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
            <code className="flex-1 bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground font-mono">
              {installHint}
            </code>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleCopy}
              className="text-muted-foreground hover:text-foreground"
              aria-label="Copy install command"
            >
              <Copy className="w-4 h-4" />
            </Button>
          </div>
          <Button
            onClick={onInstall}
            className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold"
          >
            Install Now
          </Button>
        </>
      )}

      {state === "installing" && (
        <p className="text-muted-foreground">
          Installing{" "}
          <code className="px-1 py-0.5 rounded bg-muted text-info">
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
          <Button
            onClick={onRetry}
            className="bg-secondary hover:bg-secondary/80 text-secondary-foreground"
          >
            Try again
          </Button>
        </>
      )}

      {state === "error" && logs.length > 0 && (
        <div
          ref={logBoxRef}
          className="bg-muted rounded-lg p-3 h-48 overflow-y-auto font-mono text-xs border border-border text-muted-foreground whitespace-pre-wrap break-words"
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
