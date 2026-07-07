import { DatasetItem } from "@/lib/replayApi";

/**
 * Deterministic ordering for the dataset picker:
 *   1. Datasets whose namespace (the part of `repo_id` before the first "/")
 *      equals the logged-in Hugging Face `username` (case-insensitive) come
 *      first, ahead of everything else.
 *   2. Within each group, case-insensitive alphabetical by `repo_id`.
 *
 * Bare ids with no "/" have their whole id treated as the namespace, so a
 * slash-less id only counts as "mine" when it literally equals `username`.
 *
 * When `username` is null/empty (not authenticated or still loading), the
 * namespace-first rule is skipped entirely and the result is plain
 * alphabetical. Pure and stable — does not mutate the input array.
 */
export function sortDatasets(
  items: DatasetItem[],
  username: string | null,
): DatasetItem[] {
  const lowerUser = username ? username.toLowerCase() : null;

  const isMine = (d: DatasetItem): boolean =>
    !!lowerUser && d.repo_id.split("/")[0].toLowerCase() === lowerUser;

  return [...items].sort(
    (a, b) =>
      Number(isMine(b)) - Number(isMine(a)) ||
      a.repo_id.toLowerCase().localeCompare(b.repo_id.toLowerCase()),
  );
}
