import React from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useInstallExtra } from "@/hooks/useInstallExtra";
import {
  InstallProgress,
  InstallTitleIcon,
  ReadyInstructions,
  installTitle,
} from "./InstallProgress";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  installHint: string;
}

const WandbInstallDialog: React.FC<Props> = ({ open, onOpenChange, installHint }) => {
  const install = useInstallExtra("system/wandb-extra", open);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <InstallTitleIcon state={install.state} />
            {installTitle(install.state, "Weights & Biases not installed")}
          </DialogTitle>
          <DialogDescription className="sr-only">
            Install the wandb package to enable W&amp;B logging.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <InstallProgress
            state={install.state}
            error={install.error}
            logs={install.logs}
            logBoxRef={install.logBoxRef}
            onInstall={install.handleInstall}
            onRetry={install.handleRetry}
            installHint={installHint}
            packageName="wandb"
            idleTitle="Weights &amp; Biases not installed"
            idleDescription={
              <>
                Enabling W&amp;B logging requires the{" "}
                <code className="rounded-sm bg-secondary px-1 py-0.5 font-mono text-info">
                  wandb
                </code>{" "}
                package, which isn't installed in this environment. Install it
                to log this run to W&amp;B.
              </>
            }
            doneDescription={<ReadyInstructions purpose="W&amp;B logging" />}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default WandbInstallDialog;
