import React, { useState, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { ConfigComponentProps } from "../types";
import { useApi } from "@/contexts/ApiContext";
import { isValidTimeout } from "@/lib/jobTimeout";

const SectionHeading: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => <h4 className="eyebrow">{children}</h4>;

interface OptimizerDefaults {
  optimizer: string;
  lr: number;
  weight_decay: number;
  grad_clip_norm: number;
}

// Render small floats readably: keep tiny/large magnitudes in exponential
// notation (1e-5, 1e-10) but show human-friendly decimals (0.01, 10) for the
// mid range, trimming any trailing zeros the browser tacks on.
const formatNum = (n: number): string => {
  if (n === 0) return "0";
  const abs = Math.abs(n);
  if (abs < 1e-3 || abs >= 1e6) {
    // toExponential(0) -> "1e-5" style; drop the "+" and leading zeros in exp.
    return n.toExponential().replace(/e\+?(-?)0*(\d)/, "e$1$2");
  }
  return String(Number(n.toPrecision(6)));
};

const OPTIMIZER_LABELS: Record<string, string> = {
  adam: "Adam",
  adamw: "AdamW",
  sgd: "SGD",
  multi_adam: "Multi Adam",
};

/** Advanced-parameters section of the training form — a flat collapsible
 * matching the Collect form's "Advanced parameters" (the old boxed Card
 * chrome is gone). */
const AdvancedCard: React.FC<ConfigComponentProps> = ({
  config,
  updateConfig,
}) => {
  const [expanded, setExpanded] = useState(false);
  const { baseUrl, fetchWithHeaders } = useApi();
  const [policyDefaults, setPolicyDefaults] = useState<
    Record<string, OptimizerDefaults | null>
  >({});

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetchWithHeaders(`${baseUrl}/policy-optimizer-defaults`);
        const data: { defaults: Record<string, OptimizerDefaults | null> } =
          await r.json();
        if (!cancelled) setPolicyDefaults(data.defaults || {});
      } catch {
        // Backend unreachable — fall back to the generic placeholders.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  const d = policyDefaults[config.policy_type] ?? null;
  const lrPlaceholder = d
    ? `${formatNum(d.lr)} (policy default)`
    : "Use policy default";
  const wdPlaceholder = d
    ? `${formatNum(d.weight_decay)} (policy default)`
    : "Use policy default";
  const gradPlaceholder = d
    ? `${formatNum(d.grad_clip_norm)} (policy default)`
    : "Use policy default";
  const defaultOptimizerLabel = d
    ? OPTIMIZER_LABELS[d.optimizer] ?? d.optimizer
    : null;

  // Cloud-only "Job timeout": the raw string drives both the input and the
  // (mirror-of-backend) inline validity check. Blank = HF Jobs default (2h).
  const isCloud = config.target.runner === "hf_cloud";
  const timeoutValue = config.hf_job_timeout ?? "";
  const timeoutInvalid =
    timeoutValue.trim() !== "" && !isValidTimeout(timeoutValue);

  return (
    <Collapsible
      open={expanded}
      onOpenChange={setExpanded}
      className="group space-y-4"
    >
      <CollapsibleTrigger className="flex w-full items-center justify-between border-b border-border pb-2 text-sm font-semibold text-foreground">
        <span>Advanced parameters</span>
        <ChevronDown className="h-4 w-4 transition-transform group-data-[state=open]:rotate-180" />
      </CollapsibleTrigger>

      <CollapsibleContent className="space-y-6">
        {/* Policy */}
        <section className="space-y-3">
          <SectionHeading>Policy</SectionHeading>
          <div className="flex items-center gap-3">
            <Switch
              id="policy_use_amp"
              checked={config.policy_use_amp}
              onCheckedChange={(checked) =>
                updateConfig("policy_use_amp", checked)
              }
              className="data-[state=checked]:bg-primary"
            />
            <Label htmlFor="policy_use_amp">
              Use automatic mixed precision
            </Label>
          </div>
        </section>

        {/* Training */}
        <section className="space-y-3">
          <SectionHeading>Training</SectionHeading>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="seed">Random seed</Label>
              <NumberInput
                id="seed"
                value={config.seed}
                onChange={(v) => updateConfig("seed", v)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="num_workers">Number of workers</Label>
              <NumberInput
                id="num_workers"
                value={config.num_workers}
                onChange={(v) => {
                  if (v !== undefined) updateConfig("num_workers", v);
                }}
              />
            </div>
          </div>
        </section>

        {/* Optimizer */}
        <section className="space-y-3">
          <SectionHeading>Optimizer</SectionHeading>
          <div className="space-y-2">
            <Label htmlFor="optimizer_type">Optimizer</Label>
            <Select
              value={config.optimizer_type || "adam"}
              onValueChange={(value) => updateConfig("optimizer_type", value)}
            >
              <SelectTrigger id="optimizer_type">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="adam">Adam</SelectItem>
                <SelectItem value="adamw">AdamW</SelectItem>
                <SelectItem value="sgd">SGD</SelectItem>
                <SelectItem value="multi_adam">Multi Adam</SelectItem>
              </SelectContent>
            </Select>
            {defaultOptimizerLabel && (
              <p className="text-xs text-muted-foreground">
                Policy default: {defaultOptimizerLabel}
              </p>
            )}
          </div>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="optimizer_lr">Learning rate</Label>
              <NumberInput
                id="optimizer_lr"
                integer={false}
                step="0.0001"
                value={config.optimizer_lr}
                onChange={(v) => updateConfig("optimizer_lr", v)}
                placeholder={lrPlaceholder}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="optimizer_weight_decay">Weight decay</Label>
              <NumberInput
                id="optimizer_weight_decay"
                integer={false}
                step="0.0001"
                value={config.optimizer_weight_decay}
                onChange={(v) => updateConfig("optimizer_weight_decay", v)}
                placeholder={wdPlaceholder}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="optimizer_grad_clip_norm">
                Gradient clipping
              </Label>
              <NumberInput
                id="optimizer_grad_clip_norm"
                integer={false}
                step="0.0001"
                value={config.optimizer_grad_clip_norm}
                onChange={(v) => updateConfig("optimizer_grad_clip_norm", v)}
                placeholder={gradPlaceholder}
              />
            </div>
          </div>
        </section>

        {/* Logging & Checkpointing */}
        <section className="space-y-3">
          <SectionHeading>Logging &amp; checkpointing</SectionHeading>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="log_freq">Log frequency</Label>
              <NumberInput
                id="log_freq"
                value={config.log_freq}
                onChange={(v) => {
                  if (v !== undefined) updateConfig("log_freq", v);
                }}
              />
              {config.steps > 0 && config.log_freq > config.steps && (
                <p className="text-xs text-warn">
                  ⚠ Logging every {config.log_freq} steps exceeds the{" "}
                  {config.steps}-step run — no metrics will be logged.
                </p>
              )}
              <p className="text-xs text-muted-foreground">
                Steps between logged loss/lr points. Lower = higher-resolution
                charts (each point is a window average), but more log volume.
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="save_freq">Save frequency</Label>
              <NumberInput
                id="save_freq"
                value={config.save_freq}
                onChange={(v) => {
                  if (v !== undefined) updateConfig("save_freq", v);
                }}
              />
              {config.steps > 0 && config.save_freq > config.steps && (
                <p className="text-xs text-warn">
                  ⚠ Saving every {config.save_freq} steps exceeds the{" "}
                  {config.steps}-step run — no checkpoint will be saved.
                </p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Switch
              id="save_checkpoint"
              checked={config.save_checkpoint}
              onCheckedChange={(checked) =>
                updateConfig("save_checkpoint", checked)
              }
              className="data-[state=checked]:bg-primary"
            />
            <Label htmlFor="save_checkpoint">Save checkpoints</Label>
          </div>
          <div className="flex items-center gap-3">
            <Switch
              id="resume"
              checked={config.resume}
              onCheckedChange={(checked) => updateConfig("resume", checked)}
              className="data-[state=checked]:bg-primary"
            />
            <Label htmlFor="resume">Resume from checkpoint</Label>
          </div>
        </section>

        {/* Cloud (HF Jobs) */}
        {isCloud && (
          <section className="space-y-3">
            <SectionHeading>Cloud</SectionHeading>
            <div className="space-y-2">
              <Label htmlFor="hf_job_timeout">Job timeout</Label>
              <Input
                id="hf_job_timeout"
                value={timeoutValue}
                onChange={(e) =>
                  updateConfig("hf_job_timeout", e.target.value)
                }
                placeholder="2h (default)"
                aria-invalid={timeoutInvalid}
                className={cn("w-32", timeoutInvalid && "border-destructive")}
              />
              {timeoutInvalid ? (
                <p className="text-xs text-destructive">
                  Use a duration like "2h", "45m", or "3h30m" (units: s, m, h,
                  d).
                </p>
              ) : (
                <p className="text-xs text-muted-foreground">
                  HF Jobs kills the run after this long. Leave blank for the
                  2h default.
                </p>
              )}
            </div>
          </section>
        )}

        {/* Misc */}
        <section className="space-y-3">
          <SectionHeading>Misc</SectionHeading>
          <div className="flex items-center gap-3">
            <Switch
              id="use_policy_training_preset"
              checked={config.use_policy_training_preset}
              onCheckedChange={(checked) =>
                updateConfig("use_policy_training_preset", checked)
              }
              className="data-[state=checked]:bg-primary"
            />
            <Label htmlFor="use_policy_training_preset">
              Use policy training preset
            </Label>
          </div>
        </section>
      </CollapsibleContent>
    </Collapsible>
  );
};

export default AdvancedCard;
