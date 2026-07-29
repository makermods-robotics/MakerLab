import React from "react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ConfigComponentProps, POLICY_TYPE_OPTIONS } from "../types";

/**
 * The policy-architecture picker — the single place a policy type is chosen.
 * Extracted verbatim from EssentialsCard so it can render directly after the
 * Train panel's "Starting point": the two together answer "what is this run
 * built from", which the old ordering split across two eyebrow sections.
 *
 * `policyLocked` disables it when a starting point / resume seed fixes the
 * architecture.
 */
const PolicyField: React.FC<
  ConfigComponentProps & { policyLocked?: boolean }
> = ({ config, updateConfig, policyLocked }) => (
  <div className="space-y-2">
    <Label htmlFor="policy_type">Policy</Label>
    <Select
      value={config.policy_type || undefined}
      onValueChange={(value) => updateConfig("policy_type", value)}
      disabled={policyLocked}
    >
      <SelectTrigger id="policy_type">
        <SelectValue placeholder="Select a policy type" />
      </SelectTrigger>
      <SelectContent>
        {POLICY_TYPE_OPTIONS.map((policy) => (
          <SelectItem key={policy.value} value={policy.value}>
            {policy.display}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
    <p className="text-xs text-muted-foreground">
      {policyLocked
        ? "Set by the starting point — the run trains the same architecture as its source checkpoint."
        : "The network architecture this run trains."}
    </p>
  </div>
);

export default PolicyField;
