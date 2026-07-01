import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { NumberInput } from '@/components/ui/number-input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { ConfigComponentProps } from '../types';
import { useApi } from '@/contexts/ApiContext';

const SectionHeading: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
    {children}
  </h4>
);

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
  if (n === 0) return '0';
  const abs = Math.abs(n);
  if (abs < 1e-3 || abs >= 1e6) {
    // toExponential(0) -> "1e-5" style; drop the "+" and leading zeros in exp.
    return n
      .toExponential()
      .replace(/e\+?(-?)0*(\d)/, 'e$1$2');
  }
  return String(Number(n.toPrecision(6)));
};

const OPTIMIZER_LABELS: Record<string, string> = {
  adam: 'Adam',
  adamw: 'AdamW',
  sgd: 'SGD',
  multi_adam: 'Multi Adam',
};

const AdvancedCard: React.FC<ConfigComponentProps> = ({ config, updateConfig }) => {
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
  const lrPlaceholder = d ? `${formatNum(d.lr)} (policy default)` : 'Use policy default';
  const wdPlaceholder = d
    ? `${formatNum(d.weight_decay)} (policy default)`
    : 'Use policy default';
  const gradPlaceholder = d
    ? `${formatNum(d.grad_clip_norm)} (policy default)`
    : 'Use policy default';
  const defaultOptimizerLabel = d
    ? OPTIMIZER_LABELS[d.optimizer] ?? d.optimizer
    : null;

  return (
    <Card className="bg-slate-800/50 border-slate-700 rounded-xl">
      <CardHeader
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        onClick={() => setExpanded((v) => !v)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            setExpanded((v) => !v);
          }
        }}
        className="cursor-pointer select-none flex flex-row items-center justify-between"
      >
        <span className="text-white font-semibold">Advanced</span>
        <span className="flex items-center gap-1 text-slate-400 text-sm">
          {expanded ? (
            <ChevronDown className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
          {expanded ? 'Hide' : 'Show'}
        </span>
      </CardHeader>

      {expanded && (
        <CardContent className="space-y-8">
          {/* Policy */}
          <section className="space-y-4">
            <SectionHeading>Policy</SectionHeading>
            <div className="flex items-center space-x-3">
              <Switch
                id="policy_use_amp"
                checked={config.policy_use_amp}
                onCheckedChange={(checked) => updateConfig('policy_use_amp', checked)}
                className="data-[state=checked]:bg-green-500"
              />
              <Label htmlFor="policy_use_amp" className="text-slate-300">
                Use Automatic Mixed Precision
              </Label>
            </div>
          </section>

          <Separator className="bg-slate-700" />

          {/* Training */}
          <section className="space-y-4">
            <SectionHeading>Training</SectionHeading>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="seed" className="text-slate-300">
                  Random Seed
                </Label>
                <NumberInput
                  id="seed"
                  value={config.seed}
                  onChange={(v) => updateConfig('seed', v)}
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
              </div>
              <div>
                <Label htmlFor="num_workers" className="text-slate-300">
                  Number of Workers
                </Label>
                <NumberInput
                  id="num_workers"
                  value={config.num_workers}
                  onChange={(v) => {
                    if (v !== undefined) updateConfig('num_workers', v);
                  }}
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
              </div>
            </div>
          </section>

          <Separator className="bg-slate-700" />

          {/* Optimizer */}
          <section className="space-y-4">
            <SectionHeading>Optimizer</SectionHeading>
            <div>
              <Label htmlFor="optimizer_type" className="text-slate-300">
                Optimizer
              </Label>
              <Select
                value={config.optimizer_type || 'adam'}
                onValueChange={(value) => updateConfig('optimizer_type', value)}
              >
                <SelectTrigger id="optimizer_type" className="bg-slate-900 border-slate-600 text-white rounded-lg">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-slate-800 border-slate-600 text-white">
                  <SelectItem value="adam">Adam</SelectItem>
                  <SelectItem value="adamw">AdamW</SelectItem>
                  <SelectItem value="sgd">SGD</SelectItem>
                  <SelectItem value="multi_adam">Multi Adam</SelectItem>
                </SelectContent>
              </Select>
              {defaultOptimizerLabel && (
                <p className="text-xs text-slate-500 mt-1">
                  Policy default: {defaultOptimizerLabel}
                </p>
              )}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <Label htmlFor="optimizer_lr" className="text-slate-300">
                  Learning Rate
                </Label>
                <NumberInput
                  id="optimizer_lr"
                  integer={false}
                  step="0.0001"
                  value={config.optimizer_lr}
                  onChange={(v) => updateConfig('optimizer_lr', v)}
                  placeholder={lrPlaceholder}
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
              </div>
              <div>
                <Label htmlFor="optimizer_weight_decay" className="text-slate-300">
                  Weight Decay
                </Label>
                <NumberInput
                  id="optimizer_weight_decay"
                  integer={false}
                  step="0.0001"
                  value={config.optimizer_weight_decay}
                  onChange={(v) => updateConfig('optimizer_weight_decay', v)}
                  placeholder={wdPlaceholder}
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
              </div>
              <div>
                <Label htmlFor="optimizer_grad_clip_norm" className="text-slate-300">
                  Gradient Clipping
                </Label>
                <NumberInput
                  id="optimizer_grad_clip_norm"
                  integer={false}
                  step="0.0001"
                  value={config.optimizer_grad_clip_norm}
                  onChange={(v) => updateConfig('optimizer_grad_clip_norm', v)}
                  placeholder={gradPlaceholder}
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
              </div>
            </div>
          </section>

          <Separator className="bg-slate-700" />

          {/* Logging & Checkpointing */}
          <section className="space-y-4">
            <SectionHeading>Logging & Checkpointing</SectionHeading>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="log_freq" className="text-slate-300">
                  Log Frequency
                </Label>
                <NumberInput
                  id="log_freq"
                  value={config.log_freq}
                  onChange={(v) => {
                    if (v !== undefined) updateConfig('log_freq', v);
                  }}
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
                {config.steps > 0 && config.log_freq > config.steps && (
                  <p className="text-xs text-amber-400 mt-1">
                    ⚠ Logging every {config.log_freq} steps exceeds the{' '}
                    {config.steps}-step run — no metrics will be logged.
                  </p>
                )}
                <p className="text-xs text-slate-500 mt-1">
                  Steps between logged loss/lr points. Lower = higher-resolution
                  charts (each point is a window average), but more log volume.
                </p>
              </div>
              <div>
                <Label htmlFor="save_freq" className="text-slate-300">
                  Save Frequency
                </Label>
                <NumberInput
                  id="save_freq"
                  value={config.save_freq}
                  onChange={(v) => {
                    if (v !== undefined) updateConfig('save_freq', v);
                  }}
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
                {config.steps > 0 && config.save_freq > config.steps && (
                  <p className="text-xs text-amber-400 mt-1">
                    ⚠ Saving every {config.save_freq} steps exceeds the{' '}
                    {config.steps}-step run — no checkpoint will be saved.
                  </p>
                )}
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Switch
                id="save_checkpoint"
                checked={config.save_checkpoint}
                onCheckedChange={(checked) => updateConfig('save_checkpoint', checked)}
                className="data-[state=checked]:bg-green-500"
              />
              <Label htmlFor="save_checkpoint" className="text-slate-300">
                Save Checkpoints
              </Label>
            </div>
            <div className="flex items-center space-x-3">
              <Switch
                id="resume"
                checked={config.resume}
                onCheckedChange={(checked) => updateConfig('resume', checked)}
                className="data-[state=checked]:bg-green-500"
              />
              <Label htmlFor="resume" className="text-slate-300">
                Resume from Checkpoint
              </Label>
            </div>
          </section>

          <Separator className="bg-slate-700" />

          {/* Misc */}
          <section className="space-y-4">
            <SectionHeading>Misc</SectionHeading>
            <div className="flex items-center space-x-3">
              <Switch
                id="use_policy_training_preset"
                checked={config.use_policy_training_preset}
                onCheckedChange={(checked) =>
                  updateConfig('use_policy_training_preset', checked)
                }
                className="data-[state=checked]:bg-green-500"
              />
              <Label htmlFor="use_policy_training_preset" className="text-slate-300">
                Use Policy Training Preset
              </Label>
            </div>
          </section>
        </CardContent>
      )}
    </Card>
  );
};

export default AdvancedCard;
