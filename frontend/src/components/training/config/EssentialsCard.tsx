import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Eyebrow } from "@/components/ui/eyebrow";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ConfigComponentProps, policyTypeDisplayName } from "../types";
import WandbInstallDialog from "../WandbInstallDialog";
import { useApi } from "@/contexts/ApiContext";

const EssentialsCard: React.FC<ConfigComponentProps> = ({
  config,
  updateConfig,
}) => {
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
    <Card>
      <CardHeader>
        <CardTitle>Run configuration</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div>
          <Label>Dataset *</Label>
          <div className="mt-1 rounded-sm border border-input bg-secondary px-3 py-2 text-sm">
            {config.dataset_repo_id ? (
              <span className="font-mono text-foreground">
                {config.dataset_repo_id}
              </span>
            ) : (
              <span className="text-muted-foreground">No dataset selected</span>
            )}
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            {config.dataset_repo_id
              ? "Selected on the home page."
              : "Select a dataset on the home page first."}
          </p>
        </div>

        <div>
          <Label htmlFor="job_name">Run name</Label>
          <Input
            id="job_name"
            value={config.job_name || ""}
            onChange={(e) => updateConfig("job_name", e.target.value)}
            placeholder={`${(config.policy_type || "policy").toUpperCase()} · ${
              config.dataset_repo_id || "dataset"
            }`}
            className="mt-1"
          />
          <p className="mt-1 text-xs text-muted-foreground">
            Optional — shown on the job card and searchable.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <Label>Policy</Label>
            {/* Frozen: the model type is chosen on the home page (or inherited
                by the Continue / Fine-tune flows) — read-only here, same
                pattern as the Dataset field above. */}
            <div
              id="policy_type"
              className="mt-1 rounded-sm border border-input bg-secondary px-3 py-2 text-sm text-foreground"
            >
              {policyTypeDisplayName(config.policy_type)}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Model chosen on the home page.
            </p>
          </div>

          <div>
            <Label htmlFor="steps">Training steps</Label>
            <NumberInput
              id="steps"
              value={config.steps}
              onChange={(v) => {
                if (v !== undefined) updateConfig("steps", v);
              }}
              className="mt-1"
            />
          </div>

          <div>
            <Label htmlFor="batch_size">Batch size</Label>
            <NumberInput
              id="batch_size"
              value={config.batch_size}
              onChange={(v) => {
                if (v !== undefined) updateConfig("batch_size", v);
              }}
              className="mt-1"
            />
          </div>

          <div className="flex items-center space-x-3 pt-6">
            <Switch
              id="wandb_enable"
              checked={config.wandb_enable}
              onCheckedChange={handleWandbToggle}
            />
            <Label htmlFor="wandb_enable">Enable Weights & Biases</Label>
          </div>
        </div>

        <WandbInstallDialog
          open={wandbDialogOpen}
          onOpenChange={setWandbDialogOpen}
          installHint={wandbInstallHint}
        />

        {config.wandb_enable && (
          <section className="space-y-4">
            <Eyebrow>Weights & Biases</Eyebrow>
            <div>
              <Label htmlFor="wandb_project">W&B project name</Label>
              <Input
                id="wandb_project"
                value={config.wandb_project || ""}
                onChange={(e) =>
                  updateConfig("wandb_project", e.target.value || undefined)
                }
                placeholder="my-robotics-project"
                className="mt-1"
              />
            </div>
            <div>
              <Label htmlFor="wandb_entity">W&B entity (optional)</Label>
              <Input
                id="wandb_entity"
                value={config.wandb_entity || ""}
                onChange={(e) =>
                  updateConfig("wandb_entity", e.target.value || undefined)
                }
                placeholder="your-username"
                className="mt-1"
              />
            </div>
            <div>
              <Label htmlFor="wandb_notes">W&B notes (optional)</Label>
              <Input
                id="wandb_notes"
                value={config.wandb_notes || ""}
                onChange={(e) =>
                  updateConfig("wandb_notes", e.target.value || undefined)
                }
                placeholder="Training run notes..."
                className="mt-1"
              />
            </div>
            <div>
              <Label htmlFor="wandb_mode">W&B mode</Label>
              <Select
                value={config.wandb_mode || "online"}
                onValueChange={(value) => updateConfig("wandb_mode", value)}
              >
                <SelectTrigger id="wandb_mode" className="mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="online">Online</SelectItem>
                  <SelectItem value="offline">Offline</SelectItem>
                  <SelectItem value="disabled">Disabled</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center space-x-3">
              <Switch
                id="wandb_disable_artifact"
                checked={config.wandb_disable_artifact}
                onCheckedChange={(checked) =>
                  updateConfig("wandb_disable_artifact", checked)
                }
              />
              <Label htmlFor="wandb_disable_artifact">Disable artifacts</Label>
            </div>
          </section>
        )}
      </CardContent>
    </Card>
  );
};

export default EssentialsCard;
