import React, { useState } from "react";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ConfigComponentProps, POLICY_TYPE_OPTIONS } from "../types";
import WandbInstallDialog from "../WandbInstallDialog";
import { useApi } from "@/contexts/ApiContext";

/** Run-configuration section of the training form (flat studio-styled
 * section, one design system with the panel around it). Owns the policy
 * select — the single place a policy type is chosen. `policyLocked` disables
 * it when a base skill / resume seed fixes the architecture. */
const EssentialsCard: React.FC<
  ConfigComponentProps & { policyLocked?: boolean }
> = ({ config, updateConfig, policyLocked }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [wandbDialogOpen, setWandbDialogOpen] = useState(false);
  const [wandbInstallHint, setWandbInstallHint] = useState("pip install wandb");

  const handleWandbToggle = async (checked: boolean) => {
    if (!checked) {
      updateConfig("wandb_enable", false);
      return;
    }
    // Check availability before flipping the switch on. If wandb isn't
    // importable in this makerlab process, surface the same install flow used
    // for the training extra (accelerate) instead of letting the user start
    // a run that will fail.
    try {
      const r = await fetchWithHeaders(`${baseUrl}/system/wandb-extra`);
      const data: { available: boolean; install_hint: string } = await r.json();
      if (data.available) {
        updateConfig("wandb_enable", true);
      } else {
        setWandbInstallHint(data.install_hint);
        setWandbDialogOpen(true);
      }
    } catch {
      // Backend unreachable — let the user proceed; training start will
      // surface the real error if wandb is genuinely missing.
      updateConfig("wandb_enable", true);
    }
  };

  return (
    <section className="space-y-4">
      <h3 className="eyebrow">Run configuration</h3>

      <div className="space-y-2">
        <Label htmlFor="job_name">Run name</Label>
        <Input
          id="job_name"
          value={config.job_name || ""}
          onChange={(e) => updateConfig("job_name", e.target.value)}
          placeholder={`${(config.policy_type || "policy").toUpperCase()} · ${
            config.dataset_repo_id || "dataset"
          }`}
        />
        <p className="text-xs text-muted-foreground">
          Optional — shown on the job card and searchable.
        </p>
      </div>

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
            ? "Set by the base skill — the run trains the same architecture as its source checkpoint."
            : "The model architecture this run trains."}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="steps">Training steps</Label>
          <NumberInput
            id="steps"
            value={config.steps}
            onChange={(v) => {
              if (v !== undefined) updateConfig("steps", v);
            }}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="batch_size">Batch size</Label>
          <NumberInput
            id="batch_size"
            value={config.batch_size}
            onChange={(v) => {
              if (v !== undefined) updateConfig("batch_size", v);
            }}
          />
        </div>
      </div>

      <div className="flex items-center gap-3">
        <Switch
          id="wandb_enable"
          checked={config.wandb_enable}
          onCheckedChange={handleWandbToggle}
          className="data-[state=checked]:bg-primary"
        />
        <Label htmlFor="wandb_enable">Enable Weights &amp; Biases</Label>
      </div>

      <WandbInstallDialog
        open={wandbDialogOpen}
        onOpenChange={setWandbDialogOpen}
        installHint={wandbInstallHint}
      />

      {config.wandb_enable && (
        <div className="space-y-4 border-l-2 border-border pl-4">
          <div className="space-y-2">
            <Label htmlFor="wandb_project">W&amp;B project name</Label>
            <Input
              id="wandb_project"
              value={config.wandb_project || ""}
              onChange={(e) =>
                updateConfig("wandb_project", e.target.value || undefined)
              }
              placeholder="my-robotics-project"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="wandb_entity">W&amp;B entity (optional)</Label>
            <Input
              id="wandb_entity"
              value={config.wandb_entity || ""}
              onChange={(e) =>
                updateConfig("wandb_entity", e.target.value || undefined)
              }
              placeholder="your-username"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="wandb_notes">W&amp;B notes (optional)</Label>
            <Input
              id="wandb_notes"
              value={config.wandb_notes || ""}
              onChange={(e) =>
                updateConfig("wandb_notes", e.target.value || undefined)
              }
              placeholder="Training run notes..."
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="wandb_mode">W&amp;B mode</Label>
            <Select
              value={config.wandb_mode || "online"}
              onValueChange={(value) => updateConfig("wandb_mode", value)}
            >
              <SelectTrigger id="wandb_mode">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="online">Online</SelectItem>
                <SelectItem value="offline">Offline</SelectItem>
                <SelectItem value="disabled">Disabled</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center gap-3">
            <Switch
              id="wandb_disable_artifact"
              checked={config.wandb_disable_artifact}
              onCheckedChange={(checked) =>
                updateConfig("wandb_disable_artifact", checked)
              }
              className="data-[state=checked]:bg-primary"
            />
            <Label htmlFor="wandb_disable_artifact">Disable artifacts</Label>
          </div>
        </div>
      )}
    </section>
  );
};

export default EssentialsCard;
