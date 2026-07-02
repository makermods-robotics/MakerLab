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

export interface DatasetInfo {
  repo_id: string;
  total_episodes: number;
  total_frames: number;
  fps: number | null;
  robot_type: string | null;
  cameras: string[];
  tasks: string[];
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
