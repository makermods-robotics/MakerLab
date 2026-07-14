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
  if (state === "done") return <CheckCircle2 className="w-6 h-6 text-green-400" />;
  if (state === "error") return <XCircle className="w-6 h-6 text-red-400" />;
  if (state === "installing")
    return <Loader2 className="w-6 h-6 text-sky-400 animate-spin" />;
  return <AlertTriangle className="w-6 h-6 text-amber-400" />;
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
          <p className="text-slate-300">{idleDescription}</p>
          <div className="flex items-center gap-2">
            <code className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 font-mono">
              {installHint}
            </code>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleCopy}
              className="text-slate-400 hover:text-white"
              aria-label="Copy install command"
            >
              <Copy className="w-4 h-4" />
            </Button>
          </div>
          <Button
            onClick={onInstall}
            className="bg-green-500 hover:bg-green-600 text-white font-semibold"
          >
            Install Now
          </Button>
        </>
      )}

      {state === "installing" && (
        <p className="text-slate-300">
          Installing{" "}
          <code className="px-1 py-0.5 rounded bg-slate-900 text-sky-300">
            {packageName}
          </code>
          . This usually takes about 10 seconds.
        </p>
      )}

      {state === "done" && (
        <div className="space-y-3 text-slate-300">{doneDescription}</div>
      )}

      {state === "error" && (
        <>
          <p className="text-red-300">{error || "Install failed."}</p>
          <Button
            onClick={onRetry}
            className="bg-slate-700 hover:bg-slate-600 text-white"
          >
            Try again
          </Button>
        </>
      )}

      {state === "error" && logs.length > 0 && (
        <div
          ref={logBoxRef}
          className="bg-slate-900 rounded-lg p-3 h-48 overflow-y-auto font-mono text-xs border border-slate-700 text-slate-300 whitespace-pre-wrap break-words"
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
