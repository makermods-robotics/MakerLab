import { Fetcher, apiRequest } from "./apiClient";

export interface StartInferenceRequest {
  follower_port: string;
  follower_config: string;
  policy_ref: string;
  task: string;
  cameras: Record<string, {
    type: string;
    camera_index?: number;
    width: number;
    height: number;
    fps?: number;
  }>;
  duration_s: number;
  // Raw follower torque limit for the session (0-1000, default 400).
  max_torque_limit?: number;
  // Bimanual: "single" (default) drives one follower; "bimanual" drives two.
  // In bimanual mode follower_port/follower_config above is the LEFT arm and
  // the right_* fields carry the RIGHT arm. Inference has no leader arms.
  mode?: "single" | "bimanual";
  right_follower_port?: string;
  right_follower_config?: string;
  // Robot record name — the BiSO calibration-staging base id (bimanual only).
  robot_name?: string;
  // Checkpoint's flat state width (6 = single arm, 12 = bimanual). Lets the
  // server reject an arm-count mismatch before spawning the rollout subprocess.
  checkpoint_state_dim?: number;
}

export interface InferenceStatus {
  inference_active: boolean;
  started_at: number | null;
  rollout_started_at: number | null;
  elapsed_s: number;
  rollout_elapsed_s: number;
  duration_s: number | null;
  policy_ref: string | null;
  log_path: string | null;
  exited?: boolean;
  exit_code?: number | null;
}

export async function startInference(
  baseUrl: string,
  fetcher: Fetcher,
  request: StartInferenceRequest,
): Promise<{ message: string; log_path: string; warning?: string }> {
  return apiRequest<{ message: string; log_path: string; warning?: string }>(
    baseUrl,
    fetcher,
    "/start-inference",
    { method: "POST", body: request, action: "Start inference" },
  );
}

export async function stopInference(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ message: string }> {
  return apiRequest<{ message: string }>(baseUrl, fetcher, "/stop-inference", {
    method: "POST",
    action: "Stop inference",
  });
}

export async function getInferenceStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<InferenceStatus> {
  return apiRequest<InferenceStatus>(baseUrl, fetcher, "/inference-status", {
    signal,
    action: "Get inference status",
  });
}
