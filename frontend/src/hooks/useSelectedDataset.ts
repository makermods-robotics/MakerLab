import { useCallback, useEffect, useState } from "react";

const KEY = "lelab.selectedDataset";

/**
 * The dataset chosen on the home page, persisted so it's the single source of
 * truth for training (survives navigation + reload, and syncs across tabs).
 */
export function useSelectedDataset() {
  const [selectedDataset, setState] = useState<string | null>(() => {
    try {
      return localStorage.getItem(KEY);
    } catch {
      return null;
    }
  });

  const setSelectedDataset = useCallback((id: string | null) => {
    setState(id);
    try {
      if (id) localStorage.setItem(KEY, id);
      else localStorage.removeItem(KEY);
    } catch {
      // storage unavailable (private mode) — in-memory state still works
    }
  }, []);

  // Keep instances in sync if the selection changes in another tab.
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === KEY) setState(e.newValue);
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  return { selectedDataset, setSelectedDataset };
}
