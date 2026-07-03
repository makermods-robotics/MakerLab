import { Fetcher, apiRequest } from "./apiClient";

export type DatasetSource = "local" | "hub" | "both";

export interface DatasetItem {
  repo_id: string;
  last_modified: string | null;
  private: boolean;
  source: DatasetSource;
}

export async function listDatasets(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<DatasetItem[]> {
  return apiRequest<DatasetItem[]>(baseUrl, fetcher, "/datasets", {
    signal,
    action: "List datasets",
  });
}

/** One task string with how many episodes use it (0 = count unavailable). */
export interface DatasetTask {
  task: string;
  num_episodes: number;
}

export interface DatasetInfo {
  repo_id: string;
  total_episodes: number;
  total_frames: number;
  fps: number | null;
  robot_type: string | null;
  cameras: string[];
  tasks: DatasetTask[];
  size_bytes: number;
}

/** Detail view of a locally-cached dataset (404 if it's Hub-only). */
export async function getDatasetInfo(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  signal?: AbortSignal,
): Promise<DatasetInfo> {
  return apiRequest<DatasetInfo>(
    baseUrl,
    fetcher,
    `/datasets/info?repo_id=${encodeURIComponent(repoId)}`,
    { signal, action: "Dataset info" },
  );
}

/** Whether a dataset with this id exists on the Hub. "unknown" is the
 * offline/unauthenticated degrade — the card shows no badge for it. */
export type HubStatusValue = "on_hub" | "local_only" | "unknown";

export interface HubStatus {
  repo_id: string;
  status: HubStatusValue;
  url: string | null;
}

/** Hub existence check, fetched lazily/separately so it never blocks the
 * info card render. */
export async function getDatasetHubStatus(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  signal?: AbortSignal,
): Promise<HubStatus> {
  return apiRequest<HubStatus>(
    baseUrl,
    fetcher,
    `/datasets/hub-status?repo_id=${encodeURIComponent(repoId)}`,
    { signal, action: "Hub status" },
  );
}

export interface UploadResult {
  success: boolean;
  message?: string;
  dataset_url?: string;
  docs_url?: string;
  num_episodes?: number;
}

/** Push a locally-cached dataset to the Hub. Synchronous + slow (datasets are
 * 100+ MB): callers must show an in-flight state and not impose a short client
 * timeout. Returns the endpoint's friendly {success, message, docs_url} shape
 * rather than throwing on a handled auth failure. */
export async function uploadDataset(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  tags: string[],
  isPrivate: boolean,
  signal?: AbortSignal,
): Promise<UploadResult> {
  return apiRequest<UploadResult>(baseUrl, fetcher, "/upload-dataset", {
    method: "POST",
    body: { dataset_repo_id: repoId, tags, private: isPrivate },
    action: "Upload dataset",
    signal,
  });
}

export async function deleteDataset(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
): Promise<{ success: boolean; message?: string }> {
  return apiRequest(baseUrl, fetcher, "/delete-dataset", {
    method: "POST",
    body: { dataset_repo_id: repoId },
    action: "Delete dataset",
  });
}

/**
 * Rename a locally-cached dataset by moving its directory. `newName` is the
 * NAME PART ONLY — the namespace prefix stays fixed (so `ns/old` -> `ns/new`).
 * Returns the new repo_id. Throws ApiError on a rejected rename (invalid name,
 * target exists, dataset in use), with the backend's message in `.detail`.
 */
export async function renameDataset(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  newName: string,
): Promise<{ success: boolean; repo_id: string }> {
  return apiRequest(baseUrl, fetcher, "/datasets/rename", {
    method: "POST",
    body: { repo_id: repoId, new_name: newName },
    action: "Rename dataset",
  });
}

export type MergeState = "idle" | "running" | "done" | "error";

export interface MergeStatus {
  state: MergeState;
  error: string | null;
  output_repo_id: string | null;
  logs: { timestamp: number; message: string }[];
}

export async function startDatasetMerge(
  baseUrl: string,
  fetcher: Fetcher,
  sourceRepoIds: string[],
  outputRepoId: string,
): Promise<{ started: boolean; message: string }> {
  // apiRequest JSON.stringifies `body` itself — pass a raw object, not a string.
  return apiRequest(baseUrl, fetcher, "/datasets/merge", {
    method: "POST",
    body: { source_repo_ids: sourceRepoIds, output_repo_id: outputRepoId },
    action: "Merge datasets",
  });
}

export async function getDatasetMergeStatus(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<MergeStatus> {
  return apiRequest<MergeStatus>(baseUrl, fetcher, "/datasets/merge/status", {
    action: "Merge status",
  });
}
