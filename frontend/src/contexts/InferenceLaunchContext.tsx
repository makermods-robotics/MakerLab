import React, {
  createContext,
  useCallback,
  useContext,
  useMemo,
} from "react";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useInferenceLaunch } from "@/hooks/useInferenceLaunch";
import {
  JOB_SCAN_LIMIT,
  findJobForModel,
  importSourceForModel,
} from "@/lib/inferenceLaunch";
import { JobRecord, listJobs } from "@/lib/jobsApi";
import { ModelItem } from "@/lib/modelsApi";

/**
 * Hosts the InferenceModal (the "Configure inference" step) above the router,
 * so every "run this model" entry point can launch and then CLOSE ITSELF.
 *
 * That hosting is the whole point. `useInferenceLaunch` returns a `modal`
 * element its consumer must render, so a consumer that unmounts as part of Run
 * — the skill detail dialog closing, the training monitor calling `onExit` —
 * would tear down the modal it just opened and the button would silently do
 * nothing. Mounting it once here, as a sibling of InferenceSessionProvider's
 * session dialog, makes the modal outlive whatever surface launched it.
 *
 * The launch logic itself still lives in `useInferenceLaunch` (checkpoint
 * hand-off, lazy auto-import, husk-repo toasts) — this provider only owns
 * where the element lives and exposes imperative entry points.
 */
interface InferenceLaunchContextValue {
  /** Run a known job's checkpoint (step null ⇒ the modal picks the latest). */
  launchJob: (job: JobRecord, step: number | null) => void;
  /** Run a library/marketplace model: resolve it to a job, then launch. */
  launchModel: (model: ModelItem) => Promise<void>;
  /** The shared lazy auto-import, for callers that need the pseudo-job record
   * itself (the Models library registers an untracked Hub repo, then refreshes
   * its listing, before handing the record on). */
  importSource: (source: string) => Promise<JobRecord | null>;
}

const InferenceLaunchContext =
  createContext<InferenceLaunchContextValue | null>(null);

export const InferenceLaunchProvider: React.FC<{
  children: React.ReactNode;
}> = ({ children }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { play, importSource, modal } = useInferenceLaunch();

  const launchJob = useCallback(
    (job: JobRecord, step: number | null) => play(job, step),
    [play],
  );

  const launchModel = useCallback(
    async (model: ModelItem) => {
      // Only a Hub-ONLY model may take the repo-id (lazy-import) path. A model
      // with a local copy (`local`/`both`) already has a job registry entry —
      // its run id IS the job id — and must run through it: importing a second
      // Hub pseudo-job would duplicate the record and break offline runs. So
      // the import fallback is gated on hub-only rather than firing whenever
      // the registry scan misses.
      const hubOnly = model.source === "hub";
      try {
        const jobs = await listJobs(baseUrl, fetchWithHeaders, JOB_SCAN_LIMIT);
        // Matches a local run by registry id first, an already-imported repo
        // second. step null ⇒ the modal loads checkpoints and picks the latest.
        const hit = findJobForModel(model, jobs);
        if (hit) {
          play(hit, null);
          return;
        }
        if (!hubOnly) {
          toast({
            title: "Couldn't find this skill's run",
            description:
              "It has a local copy but no training-registry entry. Re-import it from the models library.",
            variant: "destructive",
          });
          return;
        }
        const imported = await importSource(importSourceForModel(model));
        if (imported) play(imported, null); // else: importSource already toasted.
      } catch (e) {
        toast({
          title: "Couldn't load the skill",
          description: e instanceof Error ? e.message : String(e),
          variant: "destructive",
        });
      }
    },
    [baseUrl, fetchWithHeaders, importSource, play, toast],
  );

  const value = useMemo(
    () => ({ launchJob, launchModel, importSource }),
    [launchJob, launchModel, importSource],
  );

  return (
    <InferenceLaunchContext.Provider value={value}>
      {children}
      {modal}
    </InferenceLaunchContext.Provider>
  );
};

export const useLaunchInference = (): InferenceLaunchContextValue => {
  const ctx = useContext(InferenceLaunchContext);
  if (!ctx) {
    throw new Error(
      "useLaunchInference must be used within InferenceLaunchProvider",
    );
  }
  return ctx;
};
