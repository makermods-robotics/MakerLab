import { useHubDownload, UseHubDownloadResult } from "@/hooks/useHubDownload";
import { downloadDataset, getDatasetDownloadStatus } from "@/lib/replayApi";

interface UseDatasetDownloadArgs {
  /** The dataset this hook cares about. It only reports "downloading" / fires
   * callbacks when the single global download is for THIS repo. */
  repoId: string;
  /** Fired once when this dataset's download finishes. */
  onDone?: () => void;
  /** Fired once when this dataset's download fails, with the message. */
  onError?: (message: string) => void;
}

/**
 * Drives one dataset's background Hub download (POST /datasets/download +
 * polled /datasets/download-status). A thin wrapper over the shared
 * useHubDownload state machine — see that hook for the semantics (one download
 * at a time, one-shot callbacks, re-attach across navigation).
 */
export function useDatasetDownload(
  args: UseDatasetDownloadArgs,
): UseHubDownloadResult {
  return useHubDownload({
    ...args,
    getStatus: getDatasetDownloadStatus,
    startDownload: downloadDataset,
  });
}
