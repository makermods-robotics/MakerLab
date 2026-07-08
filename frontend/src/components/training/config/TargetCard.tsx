import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ConfigComponentProps } from "../types";
import { RunnerFlavor } from "@/lib/jobsApi";
import { isValidTimeout, suggestedTimeout } from "@/lib/jobTimeout";

interface TargetCardProps extends ConfigComponentProps {
  authenticated: boolean;
  flavors: RunnerFlavor[];
  loading: boolean;
  datasetSizeBytes: number | null;
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
  datasetSizeBytes,
}) => {
  const target = config.target;

  // Cloud-only "Job timeout" state. The raw string the user typed drives both
  // the input and the (mirror-of-backend) inline validity check. The
  // suggestion is a pure recompute from steps/policy/flavor/dataset — it is
  // shown as click-to-apply and NEVER auto-overwrites what the user typed.
  const timeoutValue = config.hf_job_timeout ?? "";
  const timeoutInvalid =
    timeoutValue.trim() !== "" && !isValidTimeout(timeoutValue);
  const suggestion = suggestedTimeout({
    steps: config.steps,
    policyType: config.policy_type,
    flavor: target.flavor,
    datasetSizeBytes,
  });

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
    <Card className="bg-slate-800/50 border-slate-700 rounded-xl">
      <CardHeader>
        <CardTitle className="text-white">Compute target</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <Label className="text-slate-300">Run training on</Label>
          <div className="flex rounded-md border border-slate-600 overflow-hidden text-sm mt-1 w-fit">
            {(["local", "hf_cloud"] as const).map((r) => (
              <button
                key={r}
                type="button"
                onClick={() => setRunner(r)}
                className={`px-4 py-1.5 transition-colors ${
                  target.runner === r
                    ? "bg-blue-600 text-white"
                    : "bg-slate-900 text-slate-400 hover:text-white"
                }`}
              >
                {r === "local" ? "Local — your machine" : "Hugging Face Cloud"}
              </button>
            ))}
          </div>
        </div>

        {target.runner === "local" ? (
          <div>
            <Label htmlFor="policy_device" className="text-slate-300">
              Device
            </Label>
            <Select
              value={config.policy_device === "cpu" ? "cpu" : "auto"}
              onValueChange={(value) => updateConfig("policy_device", value)}
            >
              <SelectTrigger
                id="policy_device"
                className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-slate-800 border-slate-600 text-white">
                <SelectItem value="auto">
                  Automatic (use GPU if available)
                </SelectItem>
                <SelectItem value="cpu">CPU</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-slate-500 mt-1">
              lerobot auto-detects your GPU (CUDA/MPS); only CPU is forced.
            </p>
          </div>
        ) : (
          <div>
            <Label className="text-slate-300">Hardware</Label>
            <Select
              value={target.flavor ?? ""}
              onValueChange={(flavor) =>
                updateConfig("target", { runner: "hf_cloud", flavor })
              }
            >
              <SelectTrigger className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1">
                <SelectValue
                  placeholder={loading ? "Loading…" : "Select hardware"}
                />
              </SelectTrigger>
              <SelectContent className="bg-slate-800 border-slate-600 text-white">
                {flavors.map((f) => (
                  <SelectItem
                    key={f.name}
                    value={f.name}
                    disabled={!authenticated}
                  >
                    {formatFlavorLine(f)}
                    {!authenticated && (
                      <span className="text-amber-300 ml-2 text-xs">
                        log in to HF
                      </span>
                    )}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-slate-500 mt-1">
              Cost shown is per running hour. Final policy uploads to your HF
              account when training completes.
            </p>

            <div className="mt-4">
              <Label htmlFor="hf_job_timeout" className="text-slate-300">
                Job timeout
              </Label>
              <div className="flex items-center gap-2 mt-1">
                <Input
                  id="hf_job_timeout"
                  value={timeoutValue}
                  onChange={(e) =>
                    updateConfig("hf_job_timeout", e.target.value)
                  }
                  placeholder="2h"
                  aria-invalid={timeoutInvalid}
                  className={`bg-slate-900 border-slate-600 text-white rounded-lg w-32 ${
                    timeoutInvalid ? "border-red-500" : ""
                  }`}
                />
                <button
                  type="button"
                  onClick={() =>
                    updateConfig("hf_job_timeout", suggestion.label)
                  }
                  className="text-xs text-blue-400 hover:text-blue-300 underline underline-offset-2"
                  title="Apply the suggested timeout"
                >
                  Suggested: ~{suggestion.label}
                </button>
              </div>
              {timeoutInvalid ? (
                <p className="text-xs text-red-400 mt-1">
                  Use a duration like "2h", "45m", or "3h30m" (units: s, m, h,
                  d).
                </p>
              ) : (
                <p className="text-xs text-slate-500 mt-1">
                  HF Jobs kills the run after this long. Leave blank to use the
                  default (2h). Click the suggestion to apply it.
                </p>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default TargetCard;
