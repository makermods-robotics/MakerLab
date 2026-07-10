import { apiRequest, Fetcher } from "@/lib/apiClient";

/** One model repo the user owns on the Hugging Face Hub. */
export interface UserModel {
  repo_id: string;
  last_modified: string | null;
  private: boolean;
  /** Carries the lowercase `lerobot` library tag. */
  lerobot: boolean;
  /** Repo name matches MakerLab's cloud-run naming (<policy>_<ns>_<dataset>_<timestamp>). */
  cloud_run: boolean;
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
