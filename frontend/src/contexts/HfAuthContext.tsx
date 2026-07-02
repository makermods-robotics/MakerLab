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
  // `envToken` true means the active identity is pinned by the HF_TOKEN env
  // var; account switching/logout is refused server-side in that case.
  | { status: "authenticated"; username: string; orgs: string[]; envToken: boolean }
  | { status: "unauthenticated"; loginCommand: string };

// Names are HF token displayNames from the machine-global store (shared with
// the `hf` CLI). `active` is the name matching the currently-active token, or
// null when the active identity isn't a stored token (e.g. env-var identity).
export interface HfAccounts {
  accounts: string[];
  active: string | null;
  envToken: boolean;
}

interface HfAuthValue {
  auth: HfAuthState;
  accounts: HfAccounts;
  refetch: () => Promise<void>;
  switchAccount: (name: string) => Promise<void>;
  logout: () => Promise<void>;
}

const HfAuthContext = createContext<HfAuthValue | undefined>(undefined);

const EMPTY_ACCOUNTS: HfAccounts = {
  accounts: [],
  active: null,
  envToken: false,
};

export const HfAuthProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [auth, setAuth] = useState<HfAuthState>({ status: "loading" });
  const [accounts, setAccounts] = useState<HfAccounts>(EMPTY_ACCOUNTS);

  const fetchStatus = useCallback(async () => {
    setAuth({ status: "loading" });
    // Fetch identity and the stored-account list together so the dropdown and
    // the identity chip never disagree about who's signed in.
    try {
      const [statusRes, accountsRes] = await Promise.all([
        fetchWithHeaders(`${baseUrl}/hf-auth-status`),
        fetchWithHeaders(`${baseUrl}/hf-auth/accounts`).catch(() => null),
      ]);
      const data = await statusRes.json();
      let acc: HfAccounts = EMPTY_ACCOUNTS;
      if (accountsRes && accountsRes.ok) {
        const a = await accountsRes.json();
        acc = {
          accounts: a.accounts ?? [],
          active: a.active ?? null,
          envToken: !!a.env_token,
        };
      }
      setAccounts(acc);
      if (data.authenticated) {
        setAuth({
          status: "authenticated",
          username: data.username,
          orgs: data.orgs ?? [],
          envToken: acc.envToken,
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
      setAccounts(EMPTY_ACCOUNTS);
      setAuth({
        status: "unauthenticated",
        loginCommand: "hf auth login",
      });
    }
  }, [baseUrl, fetchWithHeaders]);

  const switchAccount = useCallback(
    async (name: string) => {
      const r = await fetchWithHeaders(`${baseUrl}/hf-auth/switch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${r.status}`);
      }
      // Re-pull the full picture so identity-dependent UI (datasets, runner
      // hardware, etc.) refreshes against the new namespace.
      await fetchStatus();
    },
    [baseUrl, fetchWithHeaders, fetchStatus]
  );

  const logout = useCallback(async () => {
    const r = await fetchWithHeaders(`${baseUrl}/hf-auth/logout`, {
      method: "POST",
    });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${r.status}`);
    }
    await fetchStatus();
  }, [baseUrl, fetchWithHeaders, fetchStatus]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const value = useMemo(
    () => ({ auth, accounts, refetch: fetchStatus, switchAccount, logout }),
    [auth, accounts, fetchStatus, switchAccount, logout]
  );

  return (
    <HfAuthContext.Provider value={value}>{children}</HfAuthContext.Provider>
  );
};

export const useHfAuth = (): HfAuthValue => {
  const ctx = useContext(HfAuthContext);
  if (ctx === undefined) {
    throw new Error("useHfAuth must be used within an HfAuthProvider");
  }
  return ctx;
};
