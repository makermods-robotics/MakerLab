import React, { useCallback, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useRobots } from "@/hooks/useRobots";
import { ApiError } from "@/lib/apiClient";
import { JobRecord, importModel } from "@/lib/jobsApi";
import InferenceModal from "@/components/landing/InferenceModal";

/**
 * The shared "run inference on a checkpoint" launch sequence, lifted out of
 * JobsSection so the Landing Models panel's footer and the Jobs cards drive
 * the SAME machinery instead of re-implementing it:
 *
 *   * `play(job, step)` — open the InferenceModal on a job's checkpoint
 *     (step null ⇒ the modal loads the job's checkpoints and auto-selects the
 *     latest). The modal owns robot/camera binding, policy-config checks, and
 *     the actual POST /inference/start.
 *   * `importSource(source)` — the Jobs cards' LAZY AUTO-IMPORT for a model
 *     that isn't a tracked job yet: registers the repo id / local path as an
 *     imported pseudo-job (idempotent — a re-import returns the existing
 *     record) with the same husk-repo messaging (a cloud run that died before
 *     its first checkpoint save 400s and gets the plain "no checkpoints"
 *     answer instead of a broken modal). Returns null on failure (already
 *     toasted).
 *   * `modal` — the rendered InferenceModal element; the consumer places it
 *     once in its tree. Robot selection comes from the shared useRobots store,
 *     exactly as JobsSection always passed it.
 */
export const useInferenceLaunch = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { selectedRecord } = useRobots();

  const [open, setOpen] = useState(false);
  const [job, setJob] = useState<JobRecord | null>(null);
  const [step, setStep] = useState<number | null>(null);

  const play = useCallback((j: JobRecord, s: number | null) => {
    setJob(j);
    setStep(s);
    setOpen(true);
  }, []);

  const importSource = useCallback(
    async (source: string): Promise<JobRecord | null> => {
      try {
        return await importModel(baseUrl, fetchWithHeaders, source);
      } catch (e) {
        const isHusk =
          e instanceof ApiError && (e.status === 400 || e.status === 404);
        toast({
          title: isHusk ? "No checkpoints in this repo" : "Import failed",
          description: isHusk
            ? "The run likely died before its first checkpoint save."
            : e instanceof Error
              ? e.message
              : String(e),
          variant: "destructive",
        });
        return null;
      }
    },
    [baseUrl, fetchWithHeaders, toast],
  );

  const modal = job ? (
    <InferenceModal
      open={open}
      onOpenChange={setOpen}
      robot={selectedRecord}
      jobId={job.id}
      initialStep={step}
    />
  ) : null;

  return { play, importSource, modal };
};

export default useInferenceLaunch;
