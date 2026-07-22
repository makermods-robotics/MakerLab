/** Human formatting for dataset metadata, shared by the dataset info card and
 * the Collect panel's library cards. */

/** 16723 -> "16.7k", 950 -> "950" */
export const formatCount = (n: number): string => {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(n);
};

/** frames ÷ fps, human-formatted: "~9 min", "~45 s", "~1 h 12 min" */
export const formatDuration = (
  frames: number,
  fps: number | null,
): string | null => {
  if (!fps || fps <= 0 || frames <= 0) return null;
  const seconds = frames / fps;
  if (seconds < 60) return `~${Math.round(seconds)} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `~${minutes} min`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m > 0 ? `~${h} h ${m} min` : `~${h} h`;
};

export const formatBytes = (bytes: number | null | undefined): string => {
  // Null-safe: an unknown size renders nothing rather than "null B". Callers
  // still gate the whole Size row on presence, so this is belt-and-suspenders.
  if (bytes == null) return "";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
};
