type Fetcher = (url: string, options?: RequestInit) => Promise<Response>;

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

async function expectOk(r: Response, action: string): Promise<Response> {
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      const body = await r.json();
      detail = body.detail || detail;
    } catch {
      // ignore
    }
    throw new Error(`${action} failed: ${detail}`);
  }
  return r;
}

export async function startInference(
  baseUrl: string,
  fetcher: Fetcher,
  request: StartInferenceRequest,
): Promise<{ message: string; log_path: string }> {
  const r = await fetcher(`${baseUrl}/start-inference`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  await expectOk(r, "Start inference");
  return r.json();
}

export async function stopInference(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ message: string }> {
  const r = await fetcher(`${baseUrl}/stop-inference`, { method: "POST" });
  await expectOk(r, "Stop inference");
  return r.json();
}

export async function getInferenceStatus(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<InferenceStatus> {
  const r = await fetcher(`${baseUrl}/inference-status`);
  await expectOk(r, "Get inference status");
  return r.json();
}
