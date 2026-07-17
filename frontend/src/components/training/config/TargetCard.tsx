import React from "react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { ConfigComponentProps } from "../types";
import { RunnerFlavor } from "@/lib/jobsApi";

interface TargetCardProps extends ConfigComponentProps {
  authenticated: boolean;
  flavors: RunnerFlavor[];
  loading: boolean;
}

const formatHourly = (unitCostUsd: number, unitLabel: string): string => {
  const hourly = unitLabel === "minute" ? unitCostUsd * 60 : unitCostUsd;
  return `$${hourly.toFixed(2)}/hr`;
};

const formatFlavorLine = (f: RunnerFlavor): string => {
  const accel = f.accelerator ? f.accelerator : f.cpu;
  return `${f.pretty_name} · ${accel} · ${formatHourly(f.unit_cost_usd, f.unit_label)}`;
};

/** Compute-target section of the training form (flat studio-styled section —
 * the old boxed Card chrome is gone so it reads as one system with the rest
 * of the panel). */
const TargetCard: React.FC<TargetCardProps> = ({
  config,
  updateConfig,
  authenticated,
  flavors,
  loading,
}) => {
  const target = config.target;

  const setRunner = (runner: "local" | "hf_cloud") => {
    if (runner === target.runner) return;
    if (runner === "local") {
      updateConfig("target", { runner: "local" });
    } else {
      // Preserve any previously-chosen flavor (may be undefined until picked).
      updateConfig("target", { runner: "hf_cloud", flavor: target.flavor });
    }
  };

  return (
    <section className="space-y-4">
      <h3 className="eyebrow">Compute target</h3>

      <div className="space-y-2">
        <Label>Run training on</Label>
        <div className="grid grid-cols-2 overflow-hidden rounded-md border border-border text-sm">
          {(["local", "hf_cloud"] as const).map((r) => (
            <button
              key={r}
              type="button"
              onClick={() => setRunner(r)}
              className={cn(
                "px-3 py-1.5 transition-colors",
                target.runner === r
                  ? "bg-primary text-primary-foreground"
                  : "bg-background text-muted-foreground hover:text-foreground",
              )}
            >
              {r === "local" ? "Local — your machine" : "Hugging Face Cloud"}
            </button>
          ))}
        </div>
      </div>

      {target.runner === "local" ? (
        <div className="space-y-2">
          <Label htmlFor="policy_device">Device</Label>
          <Select
            value={config.policy_device === "cpu" ? "cpu" : "auto"}
            onValueChange={(value) => updateConfig("policy_device", value)}
          >
            <SelectTrigger id="policy_device">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">
                Automatic (use GPU if available)
              </SelectItem>
              <SelectItem value="cpu">CPU</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            lerobot auto-detects your GPU (CUDA/MPS); only CPU is forced.
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          <Label>Hardware</Label>
          <Select
            value={target.flavor ?? ""}
            onValueChange={(flavor) =>
              updateConfig("target", { runner: "hf_cloud", flavor })
            }
          >
            <SelectTrigger>
              <SelectValue
                placeholder={loading ? "Loading…" : "Select hardware"}
              />
            </SelectTrigger>
            <SelectContent>
              {flavors.map((f) => (
                <SelectItem
                  key={f.name}
                  value={f.name}
                  disabled={!authenticated}
                >
                  {formatFlavorLine(f)}
                  {!authenticated && (
                    <span className="text-warn ml-2 text-xs">
                      log in to HF
                    </span>
                  )}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Cost shown is per running hour. Final policy uploads to your HF
            account when training completes.
          </p>
        </div>
      )}
    </section>
  );
};

export default TargetCard;
