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
  // On-disk dataset size (bytes) when known, for the cloud timeout suggestion.
  // Null when unknown — the estimator drops the download term.
  datasetSizeBytes: number | null;
}

const ConfigurationTab: React.FC<ConfigurationTabProps> = ({
  config,
  updateConfig,
  authenticated,
  flavors,
  hardwareLoading,
  datasetSizeBytes,
}) => {
  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <TargetCard
        config={config}
        updateConfig={updateConfig}
        authenticated={authenticated}
        flavors={flavors}
        loading={hardwareLoading}
        datasetSizeBytes={datasetSizeBytes}
      />
      <EssentialsCard config={config} updateConfig={updateConfig} />
      <AdvancedCard config={config} updateConfig={updateConfig} />
    </div>
  );
};

export default ConfigurationTab;
