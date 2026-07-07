import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useInstallExtra } from "@/hooks/useInstallExtra";
import {
  InstallProgress,
  InstallTitleIcon,
  ReadyInstructions,
  installTitle,
} from "./InstallProgress";

interface Props {
  installHint: string;
}

const TrainingExtraGate: React.FC<Props> = ({ installHint }) => {
  const install = useInstallExtra("system/training-extra");

  return (
    <div className="mx-auto max-w-3xl">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <InstallTitleIcon state={install.state} />
            {installTitle(install.state, "Training extra not installed")}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <InstallProgress
            state={install.state}
            error={install.error}
            logs={install.logs}
            logBoxRef={install.logBoxRef}
            onInstall={install.handleInstall}
            onRetry={install.handleRetry}
            installHint={installHint}
            packageName="accelerate"
            idleTitle="Training extra not installed"
            idleDescription={
              <>
                Training requires the{" "}
                <code className="rounded-sm bg-secondary px-1 py-0.5 font-mono text-info">
                  accelerate
                </code>{" "}
                package, which isn't installed in this environment. Install it
                to enable the Training page.
              </>
            }
            doneDescription={<ReadyInstructions purpose="training" />}
          />
        </CardContent>
      </Card>
    </div>
  );
};

export default TrainingExtraGate;
