/**
 * Deterministic namespace-first ordering shared by the dataset and model
 * pickers:
 *   1. Items whose namespace (the part of `key(item)` before the first "/")
 *      equals the logged-in Hugging Face `username` (case-insensitive) come
 *      first, ahead of everything else.
 *   2. Within each group, case-insensitive alphabetical by `label(item)`
 *      (defaults to the key), so the ordering matches what's shown.
 *
 * Bare keys with no "/" have their whole value treated as the namespace, so a
 * slash-less id only counts as "mine" when it literally equals `username`.
 *
 * When `username` is null/empty (not authenticated or still loading), the
 * namespace-first rule is skipped entirely and the result is plain
 * alphabetical. Pure and stable — does not mutate the input array.
 */
export function sortByNamespaceFirst<T>(
  items: T[],
  username: string | null,
  key: (item: T) => string,
  label: (item: T) => string = key,
): T[] {
  const lowerUser = username ? username.toLowerCase() : null;

  const isMine = (item: T): boolean =>
    !!lowerUser && key(item).split("/")[0].toLowerCase() === lowerUser;

  return [...items].sort(
    (a, b) =>
      Number(isMine(b)) - Number(isMine(a)) ||
      label(a).toLowerCase().localeCompare(label(b).toLowerCase()),
  );
}
