import { useCallback, useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { ModelItem, getModels } from "@/lib/modelsApi";

/** The merged /models listing (local runs + Hub repos), with a manual `refresh`
 * so a mutation (upload/delete) can re-pull immediately. Mirrors useDatasets. */
export const useModels = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [models, setModels] = useState<ModelItem[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    setLoading(true);
    getModels(baseUrl, fetchWithHeaders)
      .then(setModels)
      .catch(() => setModels([]))
      .finally(() => setLoading(false));
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { models, loading, refresh };
};
