import { useCallback, useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { listModels, UserModel } from "@/lib/modelsApi";

/** The user's Hugging Face model repos (cloud-trained and uploaded alike). */
export const useModels = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [models, setModels] = useState<UserModel[]>([]);
  const [authenticated, setAuthenticated] = useState(true);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    setLoading(true);
    listModels(baseUrl, fetchWithHeaders)
      .then((res) => {
        setModels(res.models);
        setAuthenticated(res.authenticated);
      })
      .catch(() => setModels([]))
      .finally(() => setLoading(false));
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { models, authenticated, loading, refresh };
};
