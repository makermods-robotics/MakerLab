import React, { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { Plus, X } from "lucide-react";
import TrainingConfigPanel from "@/components/train/TrainingConfigPanel";
import JobsSection from "@/components/jobs/JobsSection";
import { Button } from "@/components/ui/button";

/**
 * Train & Deploy — the stage page at /training: start new training runs and
 * manage jobs + models in one place. Monitoring a specific run lives at
 * /training/:jobId (Training.tsx). TrainingConfigPanel reads router state
 * (policyType / resume / finetune) itself; we only auto-expand when a flow
 * deep-links here with that state.
 */
const TrainDeploy: React.FC = () => {
  const location = useLocation();
  const navState = location.state as
    | { policyType?: string; resume?: unknown; finetune?: unknown }
    | null;
  const deepLinked = Boolean(
    navState?.policyType || navState?.resume || navState?.finetune
  );
  const [configOpen, setConfigOpen] = useState(deepLinked);

  // A later deep-link (e.g. Fine-tune from a model card while already on the
  // page) must also open the panel.
  useEffect(() => {
    if (deepLinked) setConfigOpen(true);
  }, [deepLinked, location.key]);

  return (
    <div>
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-[22px] font-bold tracking-tight">
            Train &amp; Deploy
          </h1>
          <p className="mt-1 text-[13.5px] text-muted-foreground">
            Turn demos into skills — train on your datasets, test checkpoints on
            the active robot.
          </p>
        </div>
        <Button onClick={() => setConfigOpen((v) => !v)}>
          {configOpen ? (
            <>
              <X className="h-4 w-4" /> Close config
            </>
          ) : (
            <>
              <Plus className="h-4 w-4" /> New training run
            </>
          )}
        </Button>
      </div>

      {configOpen && (
        <div className="mt-5">
          <TrainingConfigPanel />
        </div>
      )}

      <div className="mt-6">
        <JobsSection />
      </div>
    </div>
  );
};

export default TrainDeploy;
