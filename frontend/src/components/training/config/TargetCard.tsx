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
import { useApi } from "@/contexts/ApiContext";

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

/** Where the run executes — the runner toggle plus whichever hardware control
 * that runner needs. Flat: the controls carry their own <Label>s and there is
 * no "Compute target" eyebrow above them, which used to restate the "Run
 * training on" label directly beneath it. */
const TargetCard: React.FC<TargetCardProps> = ({
  config,
  updateConfig,
  authenticated,
  flavors,
  loading,
}) => {
  const target = config.target;
  const { hosts, baseUrl, activeHost } = useApi();

  // Training drives no hardware, so a "local" run can execute on either
  // machine — it is only a question of which host receives the job. The
  // picker appears solely when there is more than one, so a single-machine
  // install sees the toggle it always had.
  const trainingHost = target.host ?? baseUrl;
  const multiHost = hosts.length > 1;

  const setRunner = (runner: "local" | "hf_cloud") => {
    if (runner === target.runner) return;
    if (runner === "local") {
      updateConfig("target", { runner: "local", host: target.host });
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
              className={cn(
                "px-3 py-1.5 transition-colors",
                target.runner === r
                  ? "bg-primary text-primary-foreground"
                  : "bg-background text-muted-foreground hover:text-foreground",
              )}
            >
              {r === "local"
                ? multiHost
                  ? "Your own machines"
                  : "Local — your machine"
                : "Hugging Face Cloud"}
            </button>
          ))}
        </div>
      </div>

      {target.runner === "local" && multiHost && (
        <div className="space-y-2">
          <Label htmlFor="training_host">Machine</Label>
          <Select
            value={trainingHost}
            onValueChange={(host) =>
              updateConfig("target", { runner: "local", host })
            }
          >
            <SelectTrigger id="training_host">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {hosts.map((h) => (
                <SelectItem key={h.url} value={h.url}>
                  {h.name}
                  {h.url === activeHost.url ? " (current)" : ""}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            The run happens on this machine, so the dataset must already be
            there — otherwise pull it from the Hub on that machine first.
            {trainingHost !== activeHost.url
              ? " Switch the robot corner to it to watch progress."
              : ""}
          </p>
        </div>
      )}

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
