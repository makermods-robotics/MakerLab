import { useCallback, useEffect, useState } from "react";

const KEY = "makerlab.selectedModel";

/**
 * The model chosen on the home page's Models panel, persisted so it survives
 * navigation + reload and syncs across tabs. Mirrors useSelectedDataset. The
 * stored value is a model id (a local run id or a Hub repo id).
 */
export function useSelectedModel() {
  const [selectedModel, setState] = useState<string | null>(() => {
    try {
      return localStorage.getItem(KEY);
    } catch {
      return null;
    }
  });

  const setSelectedModel = useCallback((id: string | null) => {
    setState(id);
    try {
      if (id) localStorage.setItem(KEY, id);
      else localStorage.removeItem(KEY);
    } catch {
      // storage unavailable (private mode) — in-memory state still works
    }
  }, []);

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === KEY) setState(e.newValue);
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  return { selectedModel, setSelectedModel };
}
