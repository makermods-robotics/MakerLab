import React from "react";
import { AlertTriangle, Loader2, UploadCloud, WifiOff } from "lucide-react";
import VisibilityToggle from "@/components/landing/VisibilityToggle";

interface LocalDatasetCloudNoticeProps {
  /** The local-only dataset the cloud run would train on. */
  repoId: string;
  /** Approximate on-disk size in bytes, if the info endpoint had it cheaply.
   * null when unknown (Hub-only detail, or the fetch hasn't resolved). */
  sizeBytes: number | null;
  /** Backend is in HF_HUB_OFFLINE mode: uploads are disabled, so training
   * can't proceed for this dataset at all. */
  offline: boolean;
  /** True while this dataset's upload is in flight (drives the progress line). */
  uploading: boolean;
  /** Last upload error, shown in-place so the user doesn't lose it to a toast. */
  errorMessage?: string | null;
  /** The visibility this upload will actually use — the user's real, in-flow
   * choice, not a hardcoded claim. Defaults to public (MakerLab's default
   * policy), matching the same toggle UploadDatasetDialog offers for the
   * dataset-library upload flow. */
  isPrivate: boolean;
  /** Flips `isPrivate`. Disabled while `uploading`, same as
   * UploadDatasetDialog's identical toggle. */
  onIsPrivateChange: (isPrivate: boolean) => void;
}

const formatSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(value >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
};

/**
 * Amber notice shown in the training config when a Hugging Face Cloud run is
 * targeted at a dataset that exists only on this machine. HF Jobs trains from
 * the Hub, so the dataset must be uploaded first — the Start button becomes
 * "Upload & start training" and the upload is chained before the job launches
 * (see TrainingConfigurator's handleStart). In offline mode the upload is
 * impossible, so this turns into a hard block instead.
 *
 * Visibility is a real choice made HERE, not an assertion about a hardcoded
 * backend literal: the toggle drives both the explicit upload
 * (useDatasetUpload's `start(tags, isPrivate)`) and, via
 * TrainingConfig.dataset_private / TrainingRequest.dataset_private, the
 * backend's belt-and-braces re-upload fallback in
 * HfCloudJobRunner._ensure_dataset_on_hub — so whichever push site actually
 * fires, it uses what the user picked, not an independently-hardcoded
 * default.
 */
const LocalDatasetCloudNotice: React.FC<LocalDatasetCloudNoticeProps> = ({
  repoId,
  sizeBytes,
  offline,
  uploading,
  errorMessage,
  isPrivate,
  onIsPrivateChange,
}) => {
  const sizeLabel = sizeBytes != null ? formatSize(sizeBytes) : null;

  if (offline) {
    return (
      <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-4 text-sm text-amber-700 dark:text-amber-100">
        <div className="flex items-start gap-2">
          <WifiOff className="w-4 h-4 mt-0.5 shrink-0 text-amber-600 dark:text-amber-300" />
          <div>
            <div className="font-semibold">
              This dataset is only on this machine
            </div>
            <p className="mt-1 text-amber-700/80 dark:text-amber-200/80">
              Hugging Face Cloud trains from the Hub, but the server is in
              offline mode (<code className="text-amber-700 dark:text-amber-100">HF_HUB_OFFLINE</code>
              ), so <span className="font-medium">{repoId}</span> can't be
              uploaded. Switch off offline mode, or run this training locally.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-4 text-sm text-amber-700 dark:text-amber-100">
      <div className="flex items-start gap-2">
        <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0 text-amber-600 dark:text-amber-300" />
        <div className="w-full">
          <div className="font-semibold">
            This dataset is only on this machine
          </div>
          <p className="mt-1 text-amber-700/80 dark:text-amber-200/80">
            Hugging Face Cloud trains from the Hub, so{" "}
            <span className="font-medium">{repoId}</span>
            {sizeLabel ? ` (~${sizeLabel})` : ""} will be uploaded to the Hub
            before training starts.
          </p>
          <div className="mt-3 space-y-1.5">
            <VisibilityToggle
              value={isPrivate}
              onChange={onIsPrivateChange}
              idBase={`cloud-upload-visibility-${repoId}`}
              disabled={uploading}
            />
            <p className="text-amber-700/80 dark:text-amber-200/80">
              {isPrivate
                ? "Only you will be able to see this dataset."
                : "Anyone will be able to see this dataset — including any camera footage it contains."}{" "}
              You can change this later from the dataset's "Visibility &amp;
              tags" settings.
            </p>
          </div>
          {uploading ? (
            <p className="mt-2 flex items-center gap-2 text-amber-700 dark:text-amber-100">
              <Loader2 className="w-4 h-4 animate-spin" />
              Uploading to the Hub… this can take a few minutes for large
              datasets.
            </p>
          ) : errorMessage ? (
            <p className="mt-2 text-red-300">{errorMessage}</p>
          ) : (
            <p className="mt-2 flex items-center gap-2 text-amber-700/70 dark:text-amber-200/70">
              <UploadCloud className="w-4 h-4" />
              Use “Upload &amp; start training” below to upload, then launch.
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default LocalDatasetCloudNotice;
