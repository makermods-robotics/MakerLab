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
import { ConfigComponentProps, RESUME_INHERITED_SHORT } from "../types";
import { RunnerFlavor } from "@/lib/jobsApi";

interface TargetCardProps extends ConfigComponentProps {
  authenticated: boolean;
  flavors: RunnerFlavor[];
  loading: boolean;
  /** True on a resume: the runner toggle is pinned to the parent run's runner
   * and disabled (F7). The hardware flavor below it stays live. */
  runnerLocked?: boolean;
}

const formatHourly = (unitCostUsd: number, unitLabel: string): string => {
  const hourly = unitLabel === "minute" ? unitCostUsd * 60 : unitCostUsd;
  return `$${hourly.toFixed(2)}/hr`;
};

const formatFlavorLine = (f: RunnerFlavor): string => {
  const accel = f.accelerator ? f.accelerator : f.cpu;
  return `${f.pretty_name} · ${accel} · ${formatHourly(f.unit_cost_usd, f.unit_label)}`;
};

/** Where the run executes — the runner toggle plus whichever hardware control
 * that runner needs. Flat: the controls carry their own <Label>s and there is
 * no "Compute target" eyebrow above them, which used to restate the "Run
 * training on" label directly beneath it.
 *
 * Which HARDWARE a run rents is genuinely chosen per launch, so the cloud
 * flavor stays live on a resume. WHICH RUNNER does not: a resume can only
 * continue on the parent's runner (`runnerLocked`, F7), so the toggle is pinned
 * and disabled there. `policy_device` is locked too, for a different reason:
 * the resume branch emits no --policy.device, so lerobot uses whatever the
 * checkpoint's train_config.json recorded. */
const TargetCard: React.FC<TargetCardProps> = ({
  config,
  updateConfig,
  authenticated,
  flavors,
  loading,
  resumeLocked,
  runnerLocked,
}) => {
  const target = config.target;

  const setRunner = (runner: "local" | "hf_cloud") => {
    if (runnerLocked) return;
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
      <div className="space-y-2">
        <Label>Compute</Label>
        <div className="grid grid-cols-2 overflow-hidden rounded-md border border-border text-sm">
          {(["local", "hf_cloud"] as const).map((r) => (
            <button
              key={r}
              type="button"
              onClick={() => setRunner(r)}
              disabled={runnerLocked}
              // When pinned, the picked half keeps its fill — the inherited
              // runner should read as the answer, not as disabled chrome —
              // while the unpicked half dims and stops inviting a click it
              // would refuse.
              className={cn(
                "px-3 py-1.5 transition-colors",
                target.runner === r
                  ? "bg-primary text-primary-foreground"
                  : runnerLocked
                    ? "bg-background text-muted-foreground opacity-50 cursor-not-allowed"
                    : "bg-background text-muted-foreground hover:text-foreground",
              )}
            >
              {r === "local" ? "Local — your machine" : "Hugging Face Cloud"}
            </button>
          ))}
        </div>
        {runnerLocked ? (
          <p className="text-xs text-muted-foreground">
            Continues on the parent's runner — cross-runner resume isn't
            supported yet.
          </p>
        ) : null}
      </div>

      {target.runner === "local" ? (
        <div className="space-y-2">
          <Label htmlFor="policy_device">Device</Label>
          <Select
            value={config.policy_device === "cpu" ? "cpu" : "auto"}
            onValueChange={(value) => updateConfig("policy_device", value)}
            disabled={resumeLocked}
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
            {resumeLocked
              ? RESUME_INHERITED_SHORT
              : "lerobot auto-detects your GPU (CUDA/MPS); only CPU is forced."}
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
