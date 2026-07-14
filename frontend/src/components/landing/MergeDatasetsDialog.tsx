import React, { useEffect, useRef, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Loader2, CheckCircle2, XCircle, GitMerge } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { validateDatasetRepoId } from "@/lib/datasetName";
import {
  DatasetItem,
  MergeStatus,
  getDatasetMergeStatus,
  startDatasetMerge,
} from "@/lib/replayApi";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasets: DatasetItem[];
  onMerged: () => void;
}

const POLL_MS = 1500;

const MergeDatasetsDialog: React.FC<Props> = ({
  open,
  onOpenChange,
  datasets,
  onMerged,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [output, setOutput] = useState("");
  const [status, setStatus] = useState<MergeStatus | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const logBoxRef = useRef<HTMLDivElement>(null);
  const notifiedDone = useRef(false);

  // Reset on open, and re-attach to an already-running merge (survives closing
  // the dialog / reloading the page) by seeding from the backend.
  useEffect(() => {
    if (!open) return;
    setSelected(new Set());
    setOutput("");
    setStartError(null);
    notifiedDone.current = false;
    getDatasetMergeStatus(baseUrl, fetchWithHeaders)
      .then((s) => setStatus(s.state === "running" ? s : null))
      .catch(() => setStatus(null));
  }, [open, baseUrl, fetchWithHeaders]);

  // Poll while a merge runs; accumulate the drained log lines.
  useEffect(() => {
    if (!open || status?.state !== "running") return;
    const id = setInterval(async () => {
      try {
        const s = await getDatasetMergeStatus(baseUrl, fetchWithHeaders);
        setStatus((prev) =>
          prev ? { ...s, logs: [...prev.logs, ...s.logs] } : s,
        );
        if (s.state === "done" && !notifiedDone.current) {
          notifiedDone.current = true;
          onMerged();
        }
      } catch {
        // transient — retry next tick
      }
    }, POLL_MS);
    return () => clearInterval(id);
  }, [open, status?.state, baseUrl, fetchWithHeaders, onMerged]);

  useEffect(() => {
    if (logBoxRef.current)
      logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
  }, [status?.logs]);

  const toggle = (repoId: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(repoId)) next.delete(repoId);
      else next.add(repoId);
      return next;
    });

  // A bare output name (no "/") inherits the sources' namespace when they all
  // share one. Without this, typing "merged" created a namespace-less dataset
  // at the cache root — inconsistent with every other dataset, and rename can
  // never fix it (rename only touches the final path segment). Mixed-namespace
  // sources make no single answer right, so a bare name then stays bare and
  // the user can type the full id explicitly.
  const sourceNamespaces = [...selected].map((id) =>
    id.includes("/") ? id.split("/")[0] : null,
  );
  const commonNamespace =
    sourceNamespaces.length > 0 &&
    sourceNamespaces[0] !== null &&
    sourceNamespaces.every((ns) => ns === sourceNamespaces[0])
      ? sourceNamespaces[0]
      : null;
  const trimmedOutput = output.trim();
  const effectiveOutput =
    trimmedOutput && !trimmedOutput.includes("/") && commonNamespace
      ? `${commonNamespace}/${trimmedOutput}`
      : trimmedOutput;
  const outputError = effectiveOutput ? validateDatasetRepoId(effectiveOutput) : null;
  const canMerge =
    selected.size >= 2 &&
    effectiveOutput.length > 0 &&
    outputError === null &&
    !selected.has(effectiveOutput) &&
    status?.state !== "running";

  const handleMerge = async () => {
    setStarting(true);
    setStartError(null);
    try {
      const res = await startDatasetMerge(
        baseUrl,
        fetchWithHeaders,
        [...selected],
        effectiveOutput,
      );
      if (!res.started) {
        setStartError(res.message);
        return;
      }
      // Seed a running status so the poll effect attaches immediately.
      setStatus({
        state: "running",
        error: null,
        output_repo_id: effectiveOutput,
        logs: [],
      });
    } catch (e) {
      setStartError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  };

  const state = status?.state ?? "idle";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-slate-800 border-slate-700 text-white max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-white">
            <GitMerge className="w-5 h-5" /> Merge datasets
          </DialogTitle>
          <DialogDescription className="text-slate-400">
            Combine episodes from two or more datasets into a new one. Sources
            must share the same robot, fps, and cameras.
          </DialogDescription>
        </DialogHeader>

        {state === "idle" ? (
          <div className="space-y-4">
            <div>
              <Label className="text-slate-300">
                Sources ({selected.size} selected)
              </Label>
              <div className="mt-1 max-h-56 overflow-auto rounded-md border border-slate-700 divide-y divide-slate-700/60">
                {datasets.length === 0 ? (
                  <p className="p-3 text-sm text-slate-500">
                    No datasets found.
                  </p>
                ) : (
                  datasets.map((d) => (
                    <label
                      key={d.repo_id}
                      className="flex items-start gap-2 p-2 hover:bg-slate-700/40 cursor-pointer text-sm"
                    >
                      <Checkbox
                        className="shrink-0 mt-0.5"
                        checked={selected.has(d.repo_id)}
                        onCheckedChange={() => toggle(d.repo_id)}
                      />
                      <span className="min-w-0 break-all">{d.repo_id}</span>
                    </label>
                  ))
                )}
              </div>
            </div>
            <div>
              <Label htmlFor="merge-output" className="text-slate-300">
                Output dataset name
              </Label>
              <Input
                id="merge-output"
                value={output}
                onChange={(e) => setOutput(e.target.value)}
                placeholder="user/merged_dataset"
                aria-invalid={outputError !== null}
                className="mt-1 bg-slate-900 border-slate-600 text-white aria-[invalid=true]:border-red-500/70"
              />
              {outputError && (
                <p className="mt-1 text-xs text-red-400">{outputError}</p>
              )}
              {!outputError && effectiveOutput !== trimmedOutput && (
                <p className="mt-1 text-xs text-slate-400">
                  Will be created as{" "}
                  <code className="text-sky-300">{effectiveOutput}</code>
                </p>
              )}
            </div>
            {startError ? (
              <p className="text-sm text-red-300">{startError}</p>
            ) : null}
            <div className="flex justify-end">
              <Button
                onClick={handleMerge}
                disabled={!canMerge || starting}
                className="bg-green-500 hover:bg-green-600 text-white"
              >
                {starting ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Starting…
                  </>
                ) : (
                  <>
                    <GitMerge className="w-4 h-4 mr-2" /> Merge {selected.size}{" "}
                    datasets
                  </>
                )}
              </Button>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm text-slate-200">
              {state === "running" ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin text-sky-400" />
                  Merging into{" "}
                  <code className="text-sky-300">{status?.output_repo_id}</code>
                  …
                </>
              ) : state === "done" ? (
                <>
                  <CheckCircle2 className="w-4 h-4 text-green-400" />
                  Created{" "}
                  <code className="text-green-300">
                    {status?.output_repo_id}
                  </code>
                </>
              ) : (
                <>
                  <XCircle className="w-4 h-4 text-red-400" /> Merge failed
                </>
              )}
            </div>
            <div
              ref={logBoxRef}
              className="max-h-56 overflow-auto rounded-md border border-slate-700 bg-slate-900 p-2 font-mono text-xs text-slate-300 whitespace-pre-wrap"
            >
              {(status?.logs ?? []).map((l, i) => (
                <div key={i}>{l.message}</div>
              ))}
            </div>
            {status?.error ? (
              <p className="text-sm text-red-300">{status.error}</p>
            ) : null}
            <div className="flex justify-end">
              <Button
                variant="outline"
                className="text-slate-900 dark:text-slate-100"
                onClick={() => onOpenChange(false)}
              >
                {state === "done" ? "Done" : "Close"}
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default MergeDatasetsDialog;
