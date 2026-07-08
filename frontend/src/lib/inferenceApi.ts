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
  // Follower torque limit for the session (10-100% of full power).
  motor_power?: number;
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

// Structured startup sub-phase, mirrored from rollout.py's phase constants.
// Names which substep a slow startup is in so the UI can say "Downloading
// model…" / "Connecting to arm…" instead of one opaque spinner. Absent/null
// when no session has seeded a phase yet.
export type InferencePhase =
  | "downloading_model"
  | "starting"
  | "loading_policy"
  | "connecting"
  | "running"
  | "stopping"
  | "stopped"
  | "error";

// How a finished run turned out (present only on the exited status payload):
//   ok               — clean exit.
//   ran_with_warning — the rollout ran but a noisy shutdown/cleanup tripped
//                      (e.g. torque-disable on a gripper still holding an
//                      object). NOT a real failure — render amber, not red.
//   failed           — a real failure (never got going, or crashed mid-run).
export type InferenceOutcome = "ok" | "ran_with_warning" | "failed";

export interface InferenceStatus {
  inference_active: boolean;
  started_at: number | null;
  rollout_started_at: number | null;
  elapsed_s: number;
  rollout_elapsed_s: number;
  duration_s: number | null;
  policy_ref: string | null;
  log_path: string | null;
  phase?: InferencePhase | null;
  exited?: boolean;
  exit_code?: number | null;
  // Present only on the exited payload. `outcome` classifies the run;
  // `error` is a short snippet mined from the log tail; `hint` is a
  // plain-language, actionable diagnosis. All null when not applicable.
  outcome?: InferenceOutcome | null;
  error?: string | null;
  hint?: string | null;
  // Byte progress of the Hub model download, populated only during the
  // `downloading_model` phase (all null outside it / for a local checkpoint).
  // `download_percent` is null while the total is still unknown → the UI shows
  // an indeterminate bar. The total can grow as file sizes are discovered, so
  // the bar may legitimately step backwards — that's honest, not a glitch.
  download_bytes_done?: number | null;
  download_bytes_total?: number | null;
  download_percent?: number | null;
  // Warn-but-allow arm-identity finding, surfaced once the run is up (the
  // preflight now runs server-side in the background, after the POST returned).
  warning?: string | null;
}

// The POST now returns immediately: it only validates the request cheaply, then
// hands the model download + arm preflight + subprocess spawn to a background
// worker. So there's no log_path or warning here yet — the inference page polls
// /inference-status for the download progress, the warn-but-allow finding, and
// any failure. A 4xx still surfaces here (mutex/arm-count/policy-ref shape).
export async function startInference(
  baseUrl: string,
  fetcher: Fetcher,
  request: StartInferenceRequest,
): Promise<{ message: string }> {
  return apiRequest<{ message: string }>(
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

export interface InferenceLog {
  logs: string;
  log_path: string | null;
}

// Tail of the active/most-recent rollout's log file. Read-only + bounded on the
// server (last ~500 lines); empty `logs` (not an error) before output exists.
export async function getInferenceLog(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<InferenceLog> {
  return apiRequest<InferenceLog>(baseUrl, fetcher, "/inference-log", {
    signal,
    action: "Get inference log",
  });
}
