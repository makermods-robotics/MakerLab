import { useHubDownload, UseHubDownloadResult } from "@/hooks/useHubDownload";
import { downloadModel, getModelDownloadStatus } from "@/lib/modelsApi";

interface UseModelDownloadArgs {
  /** The model repo this hook cares about. It only reports "downloading" /
   * fires callbacks when the single global model download is for THIS repo. */
  repoId: string;
  /** Fired once when this model's download finishes. */
  onDone?: () => void;
  /** Fired once when this model's download fails, with the message. */
  onError?: (message: string) => void;
}

/**
 * Drives one model's background Hub download (POST /models/download + polled
 * /models/download-status) into the local models dir. A thin wrapper over the
 * shared useHubDownload state machine — see that hook for the semantics (one
 * download at a time, one-shot callbacks, re-attach across navigation).
 */
export function useModelDownload(
  args: UseModelDownloadArgs,
): UseHubDownloadResult {
  return useHubDownload({
    ...args,
    getStatus: getModelDownloadStatus,
    startDownload: downloadModel,
  });
}
