import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
    <Card>
      <CardHeader>
        <CardTitle>Compute target</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <Label>Run training on</Label>
          <div className="mt-1 flex w-fit overflow-hidden rounded-sm border border-input text-sm">
            {(["local", "hf_cloud"] as const).map((r) => (
              <button
                key={r}
                type="button"
                onClick={() => setRunner(r)}
                className={cn(
                  "px-4 py-1.5 transition-colors",
                  target.runner === r
                    ? "bg-primary text-primary-foreground"
                    : "bg-secondary text-muted-foreground hover:text-foreground",
                )}
              >
                {r === "local" ? "Local — your machine" : "Hugging Face Cloud"}
              </button>
            ))}
          </div>
        </div>

        {target.runner === "local" ? (
          <div>
            <Label htmlFor="policy_device">Device</Label>
            <Select
              value={config.policy_device === "cpu" ? "cpu" : "auto"}
              onValueChange={(value) => updateConfig("policy_device", value)}
            >
              <SelectTrigger id="policy_device" className="mt-1">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">
                  Automatic (use GPU if available)
                </SelectItem>
                <SelectItem value="cpu">CPU</SelectItem>
              </SelectContent>
            </Select>
            <p className="mt-1 text-xs text-muted-foreground">
              lerobot auto-detects your GPU (CUDA/MPS); only CPU is forced.
            </p>
          </div>
        ) : (
          <div>
            <Label>Hardware</Label>
            <Select
              value={target.flavor ?? ""}
              onValueChange={(flavor) =>
                updateConfig("target", { runner: "hf_cloud", flavor })
              }
            >
              <SelectTrigger className="mt-1">
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
                      <span className="ml-2 text-xs text-warn">
                        log in to HF
                      </span>
                    )}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="mt-1 text-xs text-muted-foreground">
              Cost shown is per running hour. Final policy uploads to your HF
              account when training completes.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default TargetCard;
