import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { JobRecord, jobDisplayName } from "@/lib/jobsApi";
import {
  Globe,
  HardDrive,
  Play,
  FastForward,
  Download,
  Sparkles,
  Upload,
  Box,
  History,
} from "lucide-react";
import MetaRows from "@/components/library/MetaRows";
import { useApi } from "@/contexts/ApiContext";
import { useStudio } from "@/contexts/StudioContext";
import { useToast } from "@/hooks/use-toast";
import { JobCheckpoint, listJobCheckpoints } from "@/lib/checkpointsApi";

interface Props {
  /** The model, represented by its backing job record (an import, or — when
   * the parent chooses to surface them — a finished local/cloud training). */
  model: JobRecord;
  onStop: (id: string) => void;
  onPlay: (job: JobRecord, step: number) => void;
  /** Runs this model was resumed/fine-tuned from, nearest-parent first. Their
   * checkpoints join this model's own in the history dropdown so a resumed
   * lineage reads as one model with a run history behind it. */
  ancestors?: JobRecord[];
}

function relativeTime(epochSec: number): string {
  const diff = Math.max(0, Date.now() / 1000 - epochSec);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

/**
 * Model-centric card for the Train panel's model library. The redesign moves
 * the rich affordances — Run, Fine-tune, Download, Resume, and the step
 * counter — onto the *model* (the primary unit), and folds the training runs
 * that produced it into a lightweight dropdown of past runs / checkpoints
 * behind it.
 *
 * Every affordance is gated by what the backing record actually supports, so
 * this one component renders correctly whether the model is a weights-only
 * import (Run + Fine-tune only) or a finished local/cloud training (adds the
 * step counter, Download, and Resume). Today the parent (ModelsLibrary) feeds
 * it only imports; the card is built to light the rest up the moment finished
 * local/cloud runs are surfaced as models — a data-source decision left to the
 * coordinator, not made here.
 */
const ModelCard: React.FC<Props> = ({
  model,
  onStop,
  onPlay,
  ancestors = [],
}) => {
  const navigate = useNavigate();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { openStudio } = useStudio();

  const isRunning = model.state === "running";
  const isImported = model.runner === "imported";
  const isHubImport = isImported && !!model.hf_repo_id;
  const displayName = jobDisplayName(model);
  const importedSource = model.hf_repo_id || model.output_dir;

  // Where the model came from — the header chip mirrors the dataset/job cards'
  // source badges (Imported / from Hub / Local / Cloud).
  const originLabel = isImported
    ? "Imported"
    : model.runner === "hf_cloud"
      ? "Cloud"
      : "Local";
  const OriginIcon = isImported
    ? Box
    : model.runner === "hf_cloud"
      ? Globe
      : HardDrive;

  const subtitle = isImported
    ? importedSource
    : model.ended_at != null
      ? `trained ${relativeTime(model.ended_at)}`
      : `created ${relativeTime(model.started_at)}`;

  // Checkpoints across the resume lineage (this model + the runs it descends
  // from), each tagged with its owning run so inference/continue/download route
  // to the right record. Newest-step-first so the latest sits on top.
  const [lineageCheckpoints, setLineageCheckpoints] = useState<
    { job: JobRecord; ckpt: JobCheckpoint }[]
  >([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);

  // Key ancestors by id+count so frequent list refreshes (new array refs)
  // don't refetch unless the lineage actually changed.
  const ancestorKey = ancestors
    .map((a) => `${a.id}:${a.checkpoint_count}`)
    .join("|");

  useEffect(() => {
    const lineage = [model, ...ancestors].filter((j) => j.checkpoint_count > 0);
    if (lineage.length === 0) {
      setLineageCheckpoints([]);
      setSelectedStep(null);
      return;
    }
    let cancelled = false;
    Promise.all(
      lineage.map((j) =>
        listJobCheckpoints(baseUrl, fetchWithHeaders, j.id)
          .then((cks) => cks.map((ckpt) => ({ job: j, ckpt })))
          .catch(() => [] as { job: JobRecord; ckpt: JobCheckpoint }[]),
      ),
    ).then((results) => {
      if (cancelled) return;
      const combined = results.flat().sort((a, b) => b.ckpt.step - a.ckpt.step);
      setLineageCheckpoints(combined);
      setSelectedStep((prev) =>
        prev != null && combined.some((c) => c.ckpt.step === prev)
          ? prev
          : (combined[0]?.ckpt.step ?? null),
      );
    });
    return () => {
      cancelled = true;
    };
    // model/ancestors captured via id+count keys above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseUrl, fetchWithHeaders, model.id, model.checkpoint_count, ancestorKey]);

  // The selected checkpoint may belong to this model or an inherited source
  // run; route actions to whichever run owns it.
  const selected =
    lineageCheckpoints.find((c) => c.ckpt.step === selectedStep) ?? null;
  const selectedJob = selected?.job ?? model;

  const handlePlay = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedStep == null) return;
    onPlay(selectedJob, selectedStep);
  };

  // Resume (local): needs a saved checkpoint carrying optimizer/step state on
  // this machine — a finished local training. Weights-only imports have no
  // training_state, so this is correctly absent on them.
  const canContinue =
    selectedJob.runner === "local" &&
    !isRunning &&
    lineageCheckpoints.length > 0 &&
    selectedStep != null;

  // Resume (cloud): an HF Job is immutable once ended, so this launches a NEW
  // cloud job continuing from the parent's Hub checkpoint. Offered only when a
  // cloud run ended BEFORE its step target (failed/interrupted with a saved
  // checkpoint) — a `done` run reached its target, nothing to resume.
  const endedBeforeTarget =
    (selectedJob.state === "failed" || selectedJob.state === "interrupted") &&
    (selectedJob.config.steps === 0 ||
      selectedStep == null ||
      selectedStep < selectedJob.config.steps);
  const canResumeCloud =
    selectedJob.runner === "hf_cloud" &&
    !isRunning &&
    lineageCheckpoints.length > 0 &&
    selectedStep != null &&
    endedBeforeTarget;

  const goToResume = (runner: "local" | "hf_cloud") => {
    if (selectedStep == null) return;
    navigate("/training", {
      state: {
        resume: {
          jobId: selectedJob.id,
          step: selectedStep,
          name: jobDisplayName(selectedJob),
          datasetRepoId: selectedJob.config.dataset_repo_id,
          policyType: selectedJob.config.policy_type,
          sourceSteps: selectedJob.config.steps,
          logFreq: selectedJob.config.log_freq,
          saveFreq: selectedJob.config.save_freq,
          runner,
          flavor:
            runner === "hf_cloud"
              ? (selectedJob.hf_flavor ?? undefined)
              : undefined,
        },
      },
    });
  };

  // Fine-tune: start a FRESH run whose weights init from this model's
  // checkpoint. Unlike Resume (which needs optimizer/step state), fine-tuning
  // works from weights-only imports — the common model here.
  const canFinetune =
    selectedJob.runner === "imported" &&
    !isRunning &&
    lineageCheckpoints.length > 0 &&
    selectedStep != null;

  // Unified Resume: routes to the local or cloud resume flow based on which run
  // owns the selected checkpoint. Only reachable while enabled (canContinue ||
  // canResumeCloud), so the runner is always local or hf_cloud here.
  const handleResume = (e: React.MouseEvent) => {
    e.stopPropagation();
    goToResume(selectedJob.runner === "hf_cloud" ? "hf_cloud" : "local");
  };
  const handleFinetune = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedStep == null) return;
    openStudio("train", {
      train: {
        baseJobId: selectedJob.id,
        baseStep: selectedStep,
        baseName: jobDisplayName(selectedJob),
      },
    });
  };

  // Download: exports a local checkpoint as a zip. Local-only (a cloud/imported
  // model's weights live on the Hub, not on disk here), so absent otherwise.
  const canDownload =
    selectedJob.runner === "local" &&
    lineageCheckpoints.length > 0 &&
    selectedStep != null;

  const handleDownload = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedStep == null) return;
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/jobs/${selectedJob.id}/checkpoints/${selectedStep}/download`,
      );
      if (!res.ok) {
        toast({ title: "Download failed", variant: "destructive" });
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${jobDisplayName(selectedJob)}_step_${selectedStep}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      toast({
        title: "Download failed",
        description: String(err),
        variant: "destructive",
      });
    }
  };

  const hasCheckpoints =
    lineageCheckpoints.length > 0 && selectedStep != null;

  // Presentation model for the affordance row: every control always renders,
  // but a control whose backing predicate is unmet renders DISABLED with a
  // specific, human-readable reason (surfaced on hover). The enabled decisions
  // reuse the predicates above unchanged — only the presentation flips from
  // hidden→disabled. Each reason maps to *why* that particular predicate failed.

  // Run: needs a checkpoint to load.
  const runEnabled = hasCheckpoints;
  const runReason = runEnabled
    ? undefined
    : "No checkpoint available to run yet.";

  // A hub-imported base whose weights live on the Hub, not on disk here. The
  // "keep training" affordances (Resume) need a LOCAL copy, so for this case
  // their disabled reason points at the missing local download rather than at
  // "no training state" — you'd have to download the model first to continue
  // training it. (Gating is unchanged; only the reason wording differs.)
  const selectedIsHubImport =
    selectedJob.runner === "imported" && !!selectedJob.hf_repo_id;

  // Resume: unifies the local (canContinue) and cloud (canResumeCloud) paths
  // into one control. Enabled when either applies.
  const resumeEnabled = canContinue || canResumeCloud;
  let resumeReason: string | undefined;
  if (!resumeEnabled) {
    if (isRunning)
      resumeReason = "Can't resume while this run is still in progress.";
    else if (!hasCheckpoints)
      resumeReason = "No checkpoint to resume from yet.";
    else if (selectedJob.runner === "imported")
      resumeReason = selectedIsHubImport
        ? "Download this model locally to resume training."
        : "Imported models have no training state to resume from.";
    else if (selectedJob.runner === "hf_cloud")
      resumeReason =
        "This cloud run already reached its training target — nothing left to resume.";
    else resumeReason = "This run has no resumable training state.";
  }

  // Fine-tune: currently gated to imported base models only (see report note —
  // enabling it for local/cloud trained runs is a separate functional change).
  const finetuneEnabled = canFinetune;
  let finetuneReason: string | undefined;
  if (!finetuneEnabled) {
    if (isRunning)
      finetuneReason = "Can't finetune while this run is still in progress.";
    else if (!hasCheckpoints)
      finetuneReason = "No checkpoint to finetune from yet.";
    else finetuneReason = "Finetune is available for imported base models.";
  }

  // Download: exports a local on-disk checkpoint as a zip. Gated to
  // runner === "local" on BOTH sides — server.py's download_checkpoint rejects
  // anything else — because a local run's output_dir is server-generated under
  // the jobs root while an imported one is a path the user typed, so serving it
  // would make this "zip any directory on the server", which --lan mode exposes
  // to the network. The gate is deliberate; the wording below explains why
  // rather than implying the checkpoint doesn't exist (for a disk import it
  // plainly does — it's just not ours to re-serve).
  const downloadEnabled = canDownload;
  let downloadReason: string | undefined;
  if (!downloadEnabled) {
    if (!hasCheckpoints) {
      downloadReason = "No checkpoint available to download yet.";
    } else if (selectedIsHubImport) {
      downloadReason =
        "This model's weights live on the Hub, not on this machine.";
    } else if (selectedJob.runner === "imported") {
      downloadReason = selectedJob.output_dir
        ? `Imported from disk — the checkpoint is already at ${selectedJob.output_dir}`
        : "Imported models aren't re-exported — open the original folder.";
    } else if (selectedJob.runner === "hf_cloud") {
      downloadReason =
        "Cloud runs keep their checkpoints on the Hub, not on this machine.";
    } else {
      downloadReason =
        "Only local training runs have a downloadable checkpoint.";
    }
  }

  // Run-history picker: only meaningful when there's more than one checkpoint
  // across the lineage to choose between.
  const selectorEnabled = hasCheckpoints && lineageCheckpoints.length > 1;
  let selectorReason: string | undefined;
  if (!selectorEnabled) {
    if (!hasCheckpoints)
      selectorReason = "No checkpoints to choose from yet.";
    else selectorReason = "Only one checkpoint — nothing to choose.";
  }

  // Wraps a control so its disabled reason is reachable on hover. A natively
  // `disabled` button emits no pointer events, so the tooltip trigger is a
  // `span` wrapper (the pattern used in RobotCorner). When there's no reason
  // (the control is enabled) the child renders bare, with its own handlers.
  const withHint = (
    reason: string | undefined,
    node: React.ReactNode,
    wrapperClassName?: string,
  ) =>
    reason ? (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className={cn("inline-flex", wrapperClassName)}>{node}</span>
        </TooltipTrigger>
        <TooltipContent side="top">{reason}</TooltipContent>
      </Tooltip>
    ) : (
      <>{node}</>
    );

  // Run history: one option per checkpoint across the lineage. The label
  // carries both the step and — when the model spans more than one run — which
  // run produced it, so the dropdown reads as a history of past runs, not just
  // a bare step list. step 0 is the imported-single-model sentinel → "latest".
  const spansMultipleRuns = useMemo(
    () => new Set(lineageCheckpoints.map((c) => c.job.id)).size > 1,
    [lineageCheckpoints],
  );
  const stepLabel = (step: number) =>
    step === 0 ? "latest" : `step ${step.toLocaleString()}`;
  const historyLabel = (entry: { job: JobRecord; ckpt: JobCheckpoint }) =>
    spansMultipleRuns
      ? `${jobDisplayName(entry.job)} · ${stepLabel(entry.ckpt.step)}`
      : stepLabel(entry.ckpt.step);

  // Metadata block (same format as the dataset/job cards). Steps pairs the
  // selected checkpoint with the run's step TARGET — the dropdown alone says
  // where the model stopped but never what it was aiming at, so a card can't
  // otherwise tell "20,000 of 20,000" from "20,000 of 200,000". Step and target
  // are read off the run that OWNS the selected checkpoint (which may be an
  // ancestor), so the two halves of the ratio always belong together.
  // Imported models carry a step-0 sentinel and no target, so they get no row
  // rather than a dishonest "0 / 0".
  const metaRows: Array<[string, string]> = [];
  if (model.config?.policy_type)
    metaRows.push(["Policy", model.config.policy_type]);
  if (
    model.config?.dataset_repo_id &&
    model.config.dataset_repo_id !== "(imported)"
  )
    metaRows.push(["Dataset", model.config.dataset_repo_id]);
  const stepTarget = selectedJob.config?.steps ?? 0;
  if (stepTarget > 0)
    metaRows.push([
      "Steps",
      selectedStep != null && selectedStep > 0
        ? `${selectedStep.toLocaleString()} / ${stepTarget.toLocaleString()}`
        : stepTarget.toLocaleString(),
    ]);

  return (
    <Card className="@container bg-card border-border rounded-md transition-colors h-full">
      <CardContent className="flex h-full flex-col gap-2.5 p-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-foreground">
              <OriginIcon className="w-3.5 h-3.5" />
              {originLabel}
            </div>
            {isHubImport ? (
              <div
                className="flex items-center gap-1 text-[11px] font-medium text-info"
                title="Imported from a Hugging Face Hub repo"
              >
                <Upload className="w-3 h-3" />
                from Hub
              </div>
            ) : null}
          </div>
        </div>

        <div>
          <div
            className="text-foreground font-semibold truncate"
            title={displayName}
          >
            {displayName}
          </div>
          {/* When aliased, keep the true identity visible: the run id for
              trained models (imports already show their repo id / path in the
              subtitle below). */}
          {!isImported && model.display_name ? (
            <div
              className="text-[11px] text-muted-foreground truncate"
              title={model.id}
            >
              {model.id}
            </div>
          ) : null}
          {/* Imported subtitles are file paths — truncate the *start* (rtl
              flips the ellipsis) so the useful tail stays visible; the leading
              LRM keeps the first "/" from being bidi-reordered. */}
          <div
            className="text-xs text-muted-foreground truncate"
            title={subtitle}
            style={
              isImported ? { direction: "rtl", textAlign: "left" } : undefined
            }
          >
            {isImported ? "\u200e" + subtitle : subtitle}
          </div>
        </div>

        <MetaRows rows={metaRows} />

        {/* Run history — the single step indicator + the training runs behind
            this model, collapsed into a lightweight dropdown. The dropdown is
            deliberately narrow so it doesn't span the card, leaving room on its
            right for the Download icon-button. The selector is disabled when
            there's only a single checkpoint (the common import case) with
            nothing to choose. Download exports the selected local checkpoint as
            a zip; it always renders — greyed with its reason on hover when the
            backing record has nothing downloadable. */}
        <div className="flex items-center gap-1.5">
          <History className="w-3.5 h-3.5 shrink-0 text-muted-foreground" />
          <Select
            value={selectedStep != null ? String(selectedStep) : undefined}
            onValueChange={(v) => setSelectedStep(Number(v))}
            disabled={!selectorEnabled}
          >
            {withHint(
              selectorReason,
              <SelectTrigger
                className="bg-background border-input h-8 w-36 min-w-0 px-2 text-xs"
                onClick={(e) => e.stopPropagation()}
                aria-label="Select a past run / checkpoint"
              >
                <SelectValue placeholder="Run history" />
              </SelectTrigger>,
              "shrink-0",
            )}
            <SelectContent className="bg-popover border-border">
              {lineageCheckpoints.map((entry) => (
                <SelectItem
                  key={`${entry.job.id}:${entry.ckpt.step}`}
                  value={String(entry.ckpt.step)}
                  onClick={(e) => e.stopPropagation()}
                >
                  {historyLabel(entry)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <div className="min-w-0 flex-1" />
          {withHint(
            downloadReason,
            <Button
              size="icon"
              variant="outline"
              onClick={handleDownload}
              disabled={!downloadEnabled}
              className="h-6 w-6 shrink-0 p-0 border-border text-muted-foreground hover:bg-muted"
              aria-label="Download this checkpoint"
            >
              <Download className="w-3.5 h-3.5" />
            </Button>,
            "shrink-0",
          )}
        </div>

        {/* Action footer. Run is the model's primary affordance; Resume /
            Finetune fold onto a second row (Download sits on the right of the
            run-history row above). Every control always renders — those the
            backing record doesn't support render disabled with a specific reason
            on hover (see the *Reason / *Enabled derivation above), not hidden. */}
        <div className="mt-auto flex flex-col gap-1.5 pt-1">
          {withHint(
            runReason,
            <Button
              size="sm"
              onClick={handlePlay}
              disabled={!runEnabled}
              className="h-8 w-full gap-1 bg-primary hover:bg-primary/90 text-primary-foreground"
              aria-label="Run inference with this model"
            >
              <Play className="w-3.5 h-3.5" /> Run
            </Button>,
            "w-full",
          )}
          <div className="flex items-center gap-1.5">
            {withHint(
              resumeReason,
              <Button
                size="sm"
                variant="outline"
                onClick={handleResume}
                disabled={!resumeEnabled}
                className="h-8 shrink-0 gap-1 border-info/50 text-info hover:bg-info/10"
                aria-label="Resume training from this checkpoint"
              >
                <FastForward className="w-3.5 h-3.5" />
                <span className="hidden @[13rem]:inline">Resume</span>
              </Button>,
              "shrink-0",
            )}
            {withHint(
              finetuneReason,
              <Button
                size="sm"
                variant="outline"
                onClick={handleFinetune}
                disabled={!finetuneEnabled}
                className="h-8 shrink-0 gap-1 border-primary/40 text-primary hover:bg-primary/10"
                aria-label="Finetune a new run from this model's weights"
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span className="hidden @[13rem]:inline">Finetune</span>
              </Button>,
              "shrink-0",
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ModelCard;
