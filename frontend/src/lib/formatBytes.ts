/** "0 B", "123 MB", "1.2 GB" — a compact human byte size for download
 * progress. One decimal from GB up (where the step between whole units is big
 * enough to care), whole numbers below. */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let i = 0;
  while (value >= 1000 && i < units.length - 1) {
    value /= 1000;
    i += 1;
  }
  return `${i >= 3 ? value.toFixed(1) : Math.round(value)} ${units[i]}`;
}
