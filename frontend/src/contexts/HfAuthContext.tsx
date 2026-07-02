import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  ReactNode,
} from "react";
import { useApi } from "./ApiContext";

export type HfAuthState =
  | { status: "loading" }
  | { status: "authenticated"; username: string; orgs: string[] }
  | { status: "unauthenticated"; loginCommand: string };

interface HfAuthValue {
  auth: HfAuthState;
  refetch: () => Promise<void>;
}

const HfAuthContext = createContext<HfAuthValue | undefined>(undefined);

export const HfAuthProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [auth, setAuth] = useState<HfAuthState>({ status: "loading" });

  const fetchStatus = useCallback(async () => {
    setAuth({ status: "loading" });
    try {
      const response = await fetchWithHeaders(`${baseUrl}/hf-auth-status`);
      const data = await response.json();
      if (data.authenticated) {
        setAuth({
          status: "authenticated",
          username: data.username,
          orgs: data.orgs ?? [],
        });
      } else {
        setAuth({
          status: "unauthenticated",
          loginCommand: data.login_command ?? "hf auth login",
        });
      }
    } catch (err) {
      console.warn("HF auth status fetch failed:", err);
      // Drop to a terminal state so consumers waiting on `status !== "loading"`
      // don't hang forever when the backend is unreachable.
      setAuth({
        status: "unauthenticated",
        loginCommand: "hf auth login",
      });
    }
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const value = useMemo(
    () => ({ auth, refetch: fetchStatus }),
    [auth, fetchStatus]
  );

  return (
    <HfAuthContext.Provider value={value}>
      {children}
    </HfAuthContext.Provider>
  );
};

export const useHfAuth = (): HfAuthValue => {
  const ctx = useContext(HfAuthContext);
  if (ctx === undefined) {
    throw new Error("useHfAuth must be used within an HfAuthProvider");
  }
  return ctx;
};
