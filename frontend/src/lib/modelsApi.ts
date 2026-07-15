import { apiRequest, Fetcher } from "@/lib/apiClient";

/** One model in the library: a Hub repo the user owns, or a model that lives
 * on this machine (local training run, imported local directory). */
export interface UserModel {
  repo_id: string;
  last_modified: string | null;
  created_at: string | null;
  private: boolean;
  /** Carries the lowercase `lerobot` library tag. */
  lerobot: boolean;
  /** Repo name matches MakerLab's cloud-run naming (<policy>_<ns>_<dataset>_<timestamp>). */
  cloud_run: boolean;
  /** Where the model's files live. Local entries carry `job_id`. */
  source: "hub" | "local";
  /** Registry job id — set only for source="local"; run/fine-tune go straight
   * through the job instead of a lazy Hub import. */
  job_id?: string;
}

export interface ModelsResponse {
  status: string;
  authenticated: boolean;
  models: UserModel[];
}

export const listModels = (
  baseUrl: string,
  fetcher: Fetcher,
): Promise<ModelsResponse> =>
  apiRequest<ModelsResponse>(baseUrl, fetcher, "/models", {
    action: "List models",
  });
