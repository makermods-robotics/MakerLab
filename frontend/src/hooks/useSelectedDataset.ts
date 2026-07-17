import { useCallback, useSyncExternalStore } from "react";

const KEY = "makerlab.selectedDataset";

// Module-level store shared by every useSelectedDataset() instance. The studio
// panels (Collect, Train), the launchpad handoff banner, and the library sheet
// mount SIMULTANEOUSLY on the single-page layout, so per-instance useState
// copies drift — localStorage `storage` events don't fire in the tab that
// performed the write. One store, one truth (same pattern as useRobots).
let selected: string | null = (() => {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
})();

const listeners = new Set<() => void>();

const emit = () => listeners.forEach((l) => l());

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

const getSnapshot = () => selected;

const setShared = (id: string | null) => {
  selected = id;
  try {
    if (id) localStorage.setItem(KEY, id);
    else localStorage.removeItem(KEY);
  } catch {
    // storage unavailable (private mode) — in-memory state still works
  }
  emit();
};

// Cross-TAB sync (storage events only fire in other tabs).
if (typeof window !== "undefined") {
  window.addEventListener("storage", (e: StorageEvent) => {
    if (e.key === KEY) {
      selected = e.newValue;
      emit();
    }
  });
}

/**
 * The selected dataset, persisted so it's the single source of truth for
 * training (survives navigation + reload, syncs across mounted instances and
 * across tabs).
 */
export function useSelectedDataset() {
  const selectedDataset = useSyncExternalStore(subscribe, getSnapshot);
  const setSelectedDataset = useCallback(
    (id: string | null) => setShared(id),
    [],
  );
  return { selectedDataset, setSelectedDataset };
}
