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
    <div className="max-w-3xl mx-auto">
      <Card className="bg-card border-border rounded-xl">
        <CardHeader>
          <CardTitle className="flex items-center gap-3 text-foreground">
            <InstallTitleIcon state={install.state} />
            {installTitle(install.state, "Training Extra Not Installed")}
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
            idleTitle="Training Extra Not Installed"
            idleDescription={
              <>
                Training requires the{" "}
                <code className="px-1 py-0.5 rounded bg-muted text-info">
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
