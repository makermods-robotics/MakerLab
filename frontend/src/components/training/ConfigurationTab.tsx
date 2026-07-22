import React from "react";
import EssentialsCard from "./config/EssentialsCard";
import AdvancedCard from "./config/AdvancedCard";
import TargetCard from "./config/TargetCard";
import { ConfigComponentProps } from "./types";
import { RunnerFlavor } from "@/lib/jobsApi";

interface ConfigurationTabProps extends ConfigComponentProps {
  authenticated: boolean;
  flavors: RunnerFlavor[];
  hardwareLoading: boolean;
  /** True when a base skill (fine-tune) or resume seed fixes the policy —
   * the run must train the source checkpoint's architecture. */
  policyLocked?: boolean;
}

const ConfigurationTab: React.FC<ConfigurationTabProps> = ({
  config,
  updateConfig,
  authenticated,
  flavors,
  hardwareLoading,
  policyLocked,
}) => {
  return (
    <div className="space-y-6">
      <TargetCard
        config={config}
        updateConfig={updateConfig}
        authenticated={authenticated}
        flavors={flavors}
        loading={hardwareLoading}
      />
      <EssentialsCard
        config={config}
        updateConfig={updateConfig}
        policyLocked={policyLocked}
      />
      <AdvancedCard config={config} updateConfig={updateConfig} />
    </div>
  );
};

export default ConfigurationTab;
