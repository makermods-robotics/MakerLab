import React, { useEffect, useState } from "react";
import { Loader2, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useApi } from "@/contexts/ApiContext";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import { JobCheckpoint, listJobCheckpoints } from "@/lib/checkpointsApi";
import { JobRecord, listJobs } from "@/lib/jobsApi";
import { ModelItem } from "@/lib/modelsApi";
import { findJobForModel, importSourceForModel } from "@/lib/inferenceLaunch";

/** How many registry records to scan when mapping the selected model to an
 * existing job (run id or already-imported repo). Matches the backend's own
 * listing bound order of magnitude; the browse registry never needs more. */
const JOB_SCAN_LIMIT = 200;

interface ModelLaunchFooterProps {
  /** The currently selected model, or null when nothing is selected. */
  model: ModelItem | null;
  /** useInferenceLaunch.play — opens the shared InferenceModal. */
  onPlay: (job: JobRecord, step: number | null) => void;
  /** useInferenceLaunch.importSource — the Jobs cards' lazy auto-import for a
   * model no job tracks yet; returns null on failure (already toasted). */
  onImportSource: (source: string) => Promise<JobRecord | null>;
}

/**
 * "Deploy the selected model" footer for the Landing Models panel: a checkpoint
 * dropdown + a Run inference button, mirroring the dataset panel's footer row.
 *
 * The checkpoint list is fetched LAZILY when a model is selected (never for the
 * whole listing), from the same endpoint the Jobs cards use
 * (/jobs/{id}/checkpoints) once the selection maps to a job:
 *
 *   * local-run / "both" models — their id IS a registry job id;
 *   * repos already covered by an imported/cloud record — matched on
 *     hf_repo_id (findJobForModel, the JobsSection trackedRepoIds rule);
 *   * anything else has NO job yet, so there is nothing to list without
 *     registering it — the dropdown stays on "Latest checkpoint" and the
 *     button routes through the same lazy auto-import the Jobs cards use,
 *     after which the InferenceModal loads the checkpoints and auto-selects
 *     the latest (initialStep null). Import happens on PLAY, never on mere
 *     selection, so browsing the picker doesn't mutate the job registry.
 *
 * The latest/final checkpoint is pre-selected when the list loads, so the
 * one-click case stays one click.
 */
const ModelLaunchFooter: React.FC<ModelLaunchFooterProps> = ({
  model,
  onPlay,
  onImportSource,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();

  // The job covering the selected model (null = untracked → lazy import on
  // play), its checkpoints, and the user's step choice.
  const [job, setJob] = useState<JobRecord | null>(null);
  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [launching, setLaunching] = useState(false);

  useEffect(() => {
    setJob(null);
    setCheckpoints([]);
    setSelectedStep(null);
    if (!model) return;

    const controller = new AbortController();
    setLoading(true);
    (async () => {
      try {
        const jobs = await listJobs(
          baseUrl,
          fetchWithHeaders,
          JOB_SCAN_LIMIT,
          controller.signal,
        );
        const hit = findJobForModel(model, jobs);
        if (controller.signal.aborted) return;
        setJob(hit);
        if (hit) {
          const cks = await listJobCheckpoints(
            baseUrl,
            fetchWithHeaders,
            hit.id,
            controller.signal,
          );
          if (controller.signal.aborted) return;
          setCheckpoints(cks);
          // Default to the latest/final checkpoint so one click deploys it.
          if (cks.length > 0) setSelectedStep(cks[cks.length - 1].step);
        }
      } catch {
        // Mapping/list failure degrades to the untracked path: the button
        // stays usable (lazy import → the modal loads checkpoints itself).
        if (!controller.signal.aborted) {
          setJob(null);
          setCheckpoints([]);
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    })();
    return () => controller.abort();
  }, [baseUrl, fetchWithHeaders, model]);

  const handleRun = async () => {
    if (!model || launching) return;
    if (job) {
      // Tracked job: play the chosen checkpoint (or let the modal pick latest).
      onPlay(job, selectedStep);
      return;
    }
    // Untracked: the Jobs cards' lazy auto-import, then play latest.
    setLaunching(true);
    try {
      const record = await onImportSource(importSourceForModel(model));
      if (record) onPlay(record, null);
    } finally {
      setLaunching(false);
    }
  };

  // Disabled reasons, most specific first. A tracked job whose checkpoint list
  // definitively loaded empty (husk / absent repo) can't run; an untracked
  // model stays runnable — the lazy import gives the real answer on click.
  const noSelection = !model;
  const emptyTracked = !!model && !loading && job != null && checkpoints.length === 0;
  const disabled = noSelection || loading || launching || emptyTracked;
  const disabledReason = noSelection
    ? "Select a model first"
    : loading
      ? "Loading checkpoints…"
      : emptyTracked
        ? "No checkpoints available for this model"
        : undefined;

  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-xs text-gray-400">Deploy</span>
      <div className="flex items-center gap-2">
        <CheckpointDropdown
          checkpoints={checkpoints}
          selectedStep={selectedStep}
          onChange={setSelectedStep}
          // Loading, no selection, or an untracked model (no listable
          // checkpoints without registering it) → disabled; the placeholder
          // says what will run.
          disabled={disabled || checkpoints.length === 0}
          placeholder="Latest checkpoint"
        />
        <Button
          size="sm"
          variant="outline"
          onClick={handleRun}
          disabled={disabled}
          aria-label="Run inference with the selected model"
          title={disabledReason ?? "Run inference with the selected model"}
          className="h-8 gap-1.5 border-green-500/50 px-3 text-xs text-green-700 dark:text-green-300 hover:bg-green-500/10"
        >
          {launching ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Play className="h-3.5 w-3.5" />
          )}
          Run inference
        </Button>
      </div>
    </div>
  );
};

export default ModelLaunchFooter;
