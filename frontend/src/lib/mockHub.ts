import { JobCheckpoint, PolicyConfigSummary } from "./checkpointsApi";
import { HubJob, HubModel, JobRecord, TrainingRequest } from "./jobsApi";
import { ModelItem } from "./modelsApi";

/**
 * Dev-only mock of every Hub-dependent endpoint, so UI work can continue while
 * huggingface.co is down (an outage blanks /jobs, /jobs/hub, /models and flips
 * /hf-auth-status to logged-out — see 2026-07-16).
 *
 * Enable:  visit any page with `?mockHub=1` (sticks for THIS TAB only —
 *          sessionStorage — and a banner shows while it's on)
 * Disable: `?mockHub=0`, the banner's "Turn off" button, or close the tab
 *
 * Only intercepts in a Vite dev build (`import.meta.env.DEV`); a production
 * bundle can never activate it. Reads serve the fixtures below; the common
 * mutations (rename / delete / stop / import / dismiss) mutate them in memory
 * so dialogs and card flows behave end-to-end. Everything else (datasets,
 * robots, cameras, policy-extra…) falls through to the real backend, which
 * works offline.
 */

const FLAG_KEY = "makerlab:mock-hub";

// `?mockHub=1|0` toggles the flag once at module load. sessionStorage, not
// localStorage: mock mode must never outlive the tab that asked for it — an
// earlier localStorage version silently stuck across restarts and had a real
// model "missing from the library" for a day (2026-07-17). The removeItem on
// localStorage purges that legacy sticky flag from existing browsers.
if (import.meta.env.DEV && typeof window !== "undefined") {
  window.localStorage.removeItem(FLAG_KEY);
  const param = new URLSearchParams(window.location.search).get("mockHub");
  if (param === "1") window.sessionStorage.setItem(FLAG_KEY, "1");
  if (param === "0") window.sessionStorage.removeItem(FLAG_KEY);
  if (window.sessionStorage.getItem(FLAG_KEY) === "1") {
    console.info(
      "[mock-hub] Serving mock jobs/models/auth (dev only, this tab only). Disable with ?mockHub=0",
    );
  }
}

export function mockHubEnabled(): boolean {
  return (
    import.meta.env.DEV &&
    typeof window !== "undefined" &&
    window.sessionStorage.getItem(FLAG_KEY) === "1"
  );
}

/** Turn the mock off for this tab and reload so every listing refetches real
 * data. Wired to the MockHubBanner's "Turn off" button. Strips ?mockHub from
 * the URL before reloading — a plain reload would re-arm the flag at module
 * load while the param is still in the address bar. */
export function disableMockHub(): void {
  window.sessionStorage.removeItem(FLAG_KEY);
  const url = new URL(window.location.href);
  url.searchParams.delete("mockHub");
  window.location.replace(url.toString());
}

// ---------------------------------------------------------------------------
// Fixtures — timestamps are relative to page load so "3h ago" stays realistic.
// ---------------------------------------------------------------------------

const NOW = Math.floor(Date.now() / 1000);
const H = 3600;
const D = 24 * H;
const USER = "makermods";

const cfg = (
  dataset: string,
  policy: string,
  steps: number,
  extra: Partial<TrainingRequest> = {},
): TrainingRequest => ({
  dataset_repo_id: dataset,
  policy_type: policy,
  steps,
  batch_size: 8,
  num_workers: 4,
  log_freq: 100,
  save_freq: 2000,
  save_checkpoint: true,
  resume: false,
  wandb_enable: false,
  wandb_disable_artifact: true,
  policy_use_amp: false,
  use_policy_training_preset: true,
  ...extra,
});

const metrics = (current: number, total: number, loss = 0.042) => ({
  current_step: current,
  total_steps: total,
  current_loss: loss,
  current_lr: 1e-5,
  grad_norm: 0.8,
  eta_seconds: total > current ? (total - current) * 0.4 : null,
});

const job = (j: Partial<JobRecord> & Pick<JobRecord, "id" | "name">): JobRecord => ({
  display_name: null,
  state: "done",
  config: cfg(`${USER}/sock_2_only_merged`, "act", 10_000),
  output_dir: `/tmp/makerlab_mock/${j.id}`,
  started_at: NOW - D,
  ended_at: NOW - D + 2 * H,
  exit_code: 0,
  error_message: null,
  metrics: metrics(10_000, 10_000),
  runner: "local",
  hf_job_id: null,
  hf_flavor: null,
  hf_repo_id: null,
  hf_job_url: null,
  wandb_run_url: null,
  checkpoint_count: 0,
  ...j,
});

const hubCkpts = (repo: string, steps: number[]): JobCheckpoint[] =>
  steps.map((step) => ({
    step,
    source: "hub",
    ref: step === 0 ? `${repo}@root` : `${repo}@checkpoints/${step}`,
  }));

/** Mutable registry populated only by mock-mode actions during this tab. */
const jobs: JobRecord[] = [];

const checkpointsByJob: Record<string, JobCheckpoint[]> = {};

const iso = (secAgo: number) => new Date((NOW - secAgo) * 1000).toISOString();

/** Untracked Hub jobs created by mock-mode actions during this tab. */
const hubJobs: HubJob[] = [];

/** Uploaded Hub model repos created by mock-mode actions during this tab. */
const hubModels: HubModel[] = [];

const policyConfig = (policy: string): PolicyConfigSummary => ({
  policy_type: policy,
  image_features: {
    front: { height: 480, width: 640 },
    wrist: { height: 480, width: 640 },
  },
  requires_task: policy === "smolvla" || policy === "pi05",
  state_dim: 6,
  action_dim: 6,
});

/** /models rows derived live from the registry so mutations stay coherent. */
const modelItems = (): ModelItem[] => {
  const fromJobs: ModelItem[] = jobs
    .filter((j) => j.checkpoint_count > 0 && j.state !== "running")
    .map((j) => ({
      id: j.id,
      name: j.display_name?.trim() || j.name,
      policy_type: j.config.policy_type,
      dataset:
        j.config.dataset_repo_id === "(imported)"
          ? null
          : j.config.dataset_repo_id,
      steps: j.config.steps || null,
      path: j.runner === "local" ? j.output_dir : null,
      last_modified: iso(NOW - (j.ended_at ?? j.started_at)),
      hf_repo_id: j.hf_repo_id,
      source: j.runner === "local" ? "local" : j.hf_repo_id ? "both" : "local",
    }));
  const fromHub: ModelItem[] = hubModels.map((m) => ({
    id: m.repo_id,
    name: m.repo_id.split("/").pop() ?? m.repo_id,
    policy_type: null,
    dataset: null,
    steps: null,
    path: null,
    last_modified: m.last_modified,
    hf_repo_id: m.repo_id,
    source: "hub",
    private: m.private,
  }));
  return [...fromJobs, ...fromHub];
};

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

const json = (data: unknown, status = 200): Response =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });

const findJob = (id: string) => jobs.find((j) => j.id === id);

/** Register-or-return-existing for POST /jobs/import (idempotent, like the
 * backend). New repos get a single step-0 "latest" checkpoint. */
const importRepo = (source: string): JobRecord => {
  const existing = jobs.find(
    (j) =>
      j.runner === "imported" &&
      j.hf_repo_id?.toLowerCase() === source.toLowerCase(),
  );
  if (existing) return existing;
  const id = `imported_mock_${jobs.length}_${source.split("/").pop()}`;
  const record = job({
    id,
    name: source.split("/").pop() ?? source,
    config: cfg("(imported)", "act", 0),
    runner: "imported",
    hf_repo_id: source,
    output_dir: `hub:${source}`,
    started_at: NOW,
    ended_at: null,
    checkpoint_count: 1,
  });
  jobs.unshift(record);
  checkpointsByJob[id] = hubCkpts(source, [0]);
  return record;
};

/**
 * Serve a mock response for Hub-dependent endpoints, or null to fall through
 * to the real backend. Called from ApiContext.fetchWithHeaders in dev.
 */
export function mockHubResponse(
  url: string,
  init: RequestInit = {},
): Response | null {
  if (!mockHubEnabled()) return null;
  const method = (init.method ?? "GET").toUpperCase();
  let path: string;
  try {
    path = new URL(url, window.location.origin).pathname;
  } catch {
    return null;
  }

  if (method === "GET" && path === "/hf-auth-status") {
    return json({
      authenticated: true,
      username: USER,
      orgs: [],
      writable_namespaces: [USER],
    });
  }

  if (method === "GET" && path === "/jobs/hub") {
    return json({
      authenticated: true,
      jobs_permission: true,
      jobs: hubJobs,
      models: hubModels,
    });
  }

  const dismiss = path.match(/^\/jobs\/hub\/jobs\/([^/]+)\/dismiss$/);
  if (method === "POST" && dismiss) {
    const idx = hubJobs.findIndex(
      (h) => h.id === decodeURIComponent(dismiss[1]),
    );
    if (idx >= 0) hubJobs.splice(idx, 1);
    return json({ status: "dismissed" });
  }

  const hubModelDelete = path.match(/^\/jobs\/hub\/models\/(.+)$/);
  if (method === "DELETE" && hubModelDelete) {
    const repo = decodeURIComponent(hubModelDelete[1]);
    const idx = hubModels.findIndex((m) => m.repo_id === repo);
    if (idx >= 0) hubModels.splice(idx, 1);
    return json(undefined, 204);
  }

  if (method === "GET" && path === "/models") {
    return json(modelItems());
  }

  if (method === "POST" && path === "/jobs/import") {
    let source = "";
    try {
      source = JSON.parse(String(init.body ?? "{}")).source ?? "";
    } catch {
      /* fall through to 422 below */
    }
    if (!source) return json({ detail: "source required" }, 422);
    const existing = jobs.some(
      (j) => j.hf_repo_id?.toLowerCase() === source.toLowerCase(),
    );
    return json({ ...importRepo(source), already_imported: existing });
  }

  if (method === "GET" && path === "/jobs") {
    return json({ jobs });
  }

  const jobRoute = path.match(/^\/jobs\/([^/]+)(?:\/(.*))?$/);
  if (jobRoute) {
    const id = decodeURIComponent(jobRoute[1]);
    const rest = jobRoute[2] ?? "";
    // Not a job id (POST /jobs/training) or not a fixture — let the real
    // backend answer instead of faking a 404.
    if (id === "training") return null;
    const record = findJob(id);
    if (!record) return null;

    if (method === "GET" && rest === "") return json(record);

    if (method === "GET" && rest === "checkpoints") {
      return json({ checkpoints: checkpointsByJob[id] ?? [] });
    }

    const policyCfg = rest.match(/^checkpoints\/(\d+)\/policy-config$/);
    if (method === "GET" && policyCfg) {
      return json(policyConfig(record.config.policy_type));
    }

    if (method === "POST" && rest === "rename") {
      try {
        record.display_name = JSON.parse(String(init.body ?? "{}")).new_name;
      } catch {
        return json({ detail: "invalid body" }, 400);
      }
      return json(record);
    }

    if (method === "POST" && rest === "stop") {
      record.state = "interrupted";
      record.ended_at = NOW;
      return json(record);
    }

    if (method === "DELETE" && rest === "") {
      const idx = jobs.findIndex((j) => j.id === id);
      if (idx >= 0) jobs.splice(idx, 1);
      delete checkpointsByJob[id];
      return json(undefined, 204);
    }
  }

  return null;
}
