import React from "react";
import { AlertTriangle, Loader2, UploadCloud, WifiOff } from "lucide-react";

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
 * (see Training.tsx). In offline mode the upload is impossible, so this turns
 * into a hard block instead.
 */
const LocalDatasetCloudNotice: React.FC<LocalDatasetCloudNoticeProps> = ({
  repoId,
  sizeBytes,
  offline,
  uploading,
  errorMessage,
}) => {
  const sizeLabel = sizeBytes != null ? formatSize(sizeBytes) : null;

  if (offline) {
    return (
      <div className="rounded-md border border-warn/40 bg-warn/10 p-4 text-sm text-foreground">
        <div className="flex items-start gap-2">
          <WifiOff className="mt-0.5 h-4 w-4 shrink-0 text-warn" />
          <div>
            <div className="font-display font-semibold">
              This dataset is only on this machine
            </div>
            <p className="mt-1 text-muted-foreground">
              Hugging Face Cloud trains from the Hub, but the server is in
              offline mode (
              <code className="font-mono text-foreground">HF_HUB_OFFLINE</code>
              ), so <span className="font-medium text-foreground">{repoId}</span>{" "}
              can't be uploaded. Switch off offline mode, or run this training
              locally.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-md border border-warn/40 bg-warn/10 p-4 text-sm text-foreground">
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warn" />
        <div className="w-full">
          <div className="font-display font-semibold">
            This dataset is only on this machine
          </div>
          <p className="mt-1 text-muted-foreground">
            Hugging Face Cloud trains from the Hub, so{" "}
            <span className="font-medium text-foreground">{repoId}</span>
            {sizeLabel ? ` (~${sizeLabel})` : ""} will be uploaded as a{" "}
            <span className="font-medium text-foreground">private</span> dataset
            before training starts.
          </p>
          {uploading ? (
            <p className="mt-2 flex items-center gap-2 text-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Uploading to the Hub… this can take a few minutes for large
              datasets.
            </p>
          ) : errorMessage ? (
            <p className="mt-2 text-destructive">{errorMessage}</p>
          ) : (
            <p className="mt-2 flex items-center gap-2 text-muted-foreground">
              <UploadCloud className="h-4 w-4" />
              Use “Upload &amp; start training” below to upload, then launch.
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default LocalDatasetCloudNotice;
