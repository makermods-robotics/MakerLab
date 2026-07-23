export interface TrainingConfig {
  target: { runner: "local" | "hf_cloud"; flavor?: string };

  // Dataset configuration
  dataset_repo_id: string;

  // Policy configuration
  policy_type: string;

  // Optional user-supplied display name for the run.
  job_name?: string;

  // Core training parameters
  steps: number;
  batch_size: number;
  seed?: number;
  num_workers: number;

  // Logging and checkpointing
  log_freq: number;
  save_freq: number;
  save_checkpoint: boolean;

  // Output configuration
  resume: boolean;
  // Set by the "Continue training" flow (source run + checkpoint step).
  resume_from_job_id?: string;
  resume_from_step?: number;
  // Set by the "Fine-tune" flow: fresh run initialized from a source
  // checkpoint's weights (resume stays false).
  finetune_from_job_id?: string;
  finetune_from_step?: number;

  // Weights & Biases
  wandb_enable: boolean;
  wandb_project?: string;
  wandb_entity?: string;
  wandb_notes?: string;
  wandb_mode?: string;
  wandb_disable_artifact: boolean;

  // Policy-specific parameters
  policy_device?: string;
  policy_use_amp: boolean;

  // Optimizer parameters
  optimizer_type?: string;
  optimizer_lr?: number;
  optimizer_weight_decay?: number;
  optimizer_grad_clip_norm?: number;

  // Advanced configuration
  use_policy_training_preset: boolean;

  // HF Cloud only: optional per-run override for the HF Jobs timeout, as a
  // duration string ("2h", "45m", "3h30m"). Undefined/blank ⇒ backend applies
  // its default. Ignored for local runs.
  hf_job_timeout?: string;
}

// The policy types the trainer supports. The model is chosen up-front on the
// landing page's "Create a model" card (one button per type, short `label`);
// the training config then shows it frozen using the full `display` name.
// `stable` = tested on our hardware (user decision 2026-07-03) — a statement
// about what we've validated in MakerLab, not about upstream lerobot quality.
// Unstable types stay selectable; the landing card just subdues them.
export const POLICY_TYPE_OPTIONS: {
  value: string;
  label: string;
  display: string;
  stable: boolean;
}[] = [
  {
    value: "act",
    label: "ACT",
    display: "ACT (Action Chunking Transformer)",
    stable: true,
  },
  { value: "diffusion", label: "Diffusion", display: "Diffusion Policy", stable: false },
  { value: "pi0", label: "PI0", display: "PI0", stable: false },
  { value: "smolvla", label: "SmolVLA", display: "SmolVLA", stable: true },
  { value: "tdmpc", label: "TD-MPC", display: "TD-MPC", stable: false },
  { value: "vqbet", label: "VQ-BeT", display: "VQ-BeT", stable: false },
  { value: "pi0_fast", label: "PI0 Fast", display: "PI0 Fast", stable: false },
  {
    value: "gaussian_actor",
    label: "Gaussian Actor",
    display: "Gaussian Actor",
    stable: false,
  },
  // reward_classifier deliberately absent: it isn't a policy in the pinned
  // lerobot (separate RewardModelConfig registry — scores outcomes, doesn't
  // output actions) so lerobot-train can never construct it. Re-add alongside
  // a dedicated reward-model training pathway if that lands after a pin bump;
  // policyTypeDisplayName's fallback keeps any old records legible meanwhile.
];

// Full display name for a policy type value; falls back to the raw value so
// types coming from older job records still render something legible.
export function policyTypeDisplayName(value: string): string {
  return (
    POLICY_TYPE_OPTIONS.find((o) => o.value === value)?.display ||
    value.toUpperCase()
  );
}

// Short label for a policy type value (the picker-row form: "ACT", "SmolVLA",
// "Diffusion"…) — the same mapping as policyTypeDisplayName but the compact
// `label`. Same raw-value-uppercased fallback for unknown/older types.
export function policyTypeShortLabel(value: string): string {
  return (
    POLICY_TYPE_OPTIONS.find((o) => o.value === value)?.label ||
    value.toUpperCase()
  );
}

export interface TrainingStatus {
  training_active: boolean;
  current_step: number;
  total_steps: number;
  current_loss?: number;
  current_lr?: number;
  grad_norm?: number;
  epoch_time?: number;
  eta_seconds?: number;
  available_controls: {
    stop_training: boolean;
    pause_training: boolean;
    resume_training: boolean;
  };
}

export interface LogEntry {
  timestamp: number;
  message: string;
}

export interface ConfigComponentProps {
  config: TrainingConfig;
  updateConfig: <T extends keyof TrainingConfig>(
    key: T,
    value: TrainingConfig[T],
  ) => void;
}
