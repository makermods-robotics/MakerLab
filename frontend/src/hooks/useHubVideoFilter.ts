import { useEffect, useMemo, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { DatasetItem, getDatasetInfo } from "@/lib/replayApi";

/** Module-level cache of "does this Hub-only repo have video" — shared across
 * every mount of this hook (the Collect panel's library and the Library sheet
 * both use it) so a repo's one-time /datasets/info check never re-fires per
 * surface. Absent from the map = not yet checked; `null` = check in flight;
 * `true`/`false` = resolved. A failed check is left unset (not cached false)
 * so a transient network error never permanently hides a real dataset. */
const hubHasVideoCache = new Map<string, boolean | null>();

/**
 * Filters `datasets` for library-listing surfaces that open the episode
 * viewer: a `source === "hub"` (Hub-only, no local copy) row is dropped once
 * its Hub video check confirms it has none. `"local"`/`"both"` rows pass
 * through untouched (already-confirmed local content). Pending or failed
 * checks keep a row visible.
 *
 * NOT for the Train picker (TrainPanel/TrainingConfigurator) — training
 * doesn't require video, so a state-only Hub dataset must stay selectable
 * there. Only surfaces that open DatasetDetailDialog should use this.
 */
export function useHubVideoFilter(datasets: DatasetItem[]): DatasetItem[] {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [refreshTick, setRefreshTick] = useState(0);

  const hubOnlyIds = useMemo(
    () => datasets.filter((d) => d.source === "hub").map((d) => d.repo_id),
    [datasets],
  );

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    for (const repoId of hubOnlyIds) {
      if (hubHasVideoCache.has(repoId)) continue;
      hubHasVideoCache.set(repoId, null);
      getDatasetInfo(baseUrl, fetchWithHeaders, repoId, controller.signal)
        .then((info) => {
          hubHasVideoCache.set(repoId, info.cameras.length > 0);
        })
        .catch(() => {
          hubHasVideoCache.delete(repoId);
        })
        .finally(() => {
          if (!cancelled) setRefreshTick((n) => n + 1);
        });
    }
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [hubOnlyIds, baseUrl, fetchWithHeaders]);

  return useMemo(
    () => datasets.filter((d) => d.source !== "hub" || hubHasVideoCache.get(d.repo_id) !== false),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- refreshTick drives recomputation when the module-level cache mutates; datasets itself is the other real dependency.
    [datasets, refreshTick],
  );
}
