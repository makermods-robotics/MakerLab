import { DatasetItem } from "@/lib/replayApi";
import { sortByNamespaceFirst } from "@/lib/sortByNamespaceFirst";

/**
 * Namespace-first ordering for the dataset picker — a thin wrapper over the
 * shared sortByNamespaceFirst (also used by ModelPicker), keyed and labeled by
 * `repo_id`. See that helper for the exact rules.
 */
export function sortDatasets(
  items: DatasetItem[],
  username: string | null,
): DatasetItem[] {
  return sortByNamespaceFirst(items, username, (d) => d.repo_id);
}
