import React, { createContext, useContext, ReactNode, useState, useCallback, useMemo } from "react";

interface ApiContextType {
  baseUrl: string;
  wsBaseUrl: string;
  fetchWithHeaders: (url: string, options?: RequestInit) => Promise<Response>;
}

const ApiContext = createContext<ApiContextType | undefined>(undefined);

const STORAGE_KEY = "makerlab.apiBaseUrl";
const DEFAULT_LOCALHOST = "http://localhost:8000";

// In production the backend serves the UI, so the page origin is the API. In
// Vite dev the UI is on :8080 while the API stays on :8000, so the origin
// would be wrong there.
const defaultBaseUrl = (): string =>
  import.meta.env.DEV ? DEFAULT_LOCALHOST : window.location.origin;

const httpToWs = (url: string): string => url.replace(/^http(s?):/, "ws$1:");

// Loopback origins are a local `makerlab` / `makerlab --dev` run talking to its
// own backend on this machine.
const isLoopbackHost = (host: string): boolean =>
  host === "localhost" ||
  host === "127.0.0.1" ||
  host === "0.0.0.0" ||
  host === "::1" ||
  host === "[::1]";

const resolveInitialBaseUrl = (): string => {
  if (typeof window === "undefined") return DEFAULT_LOCALHOST;

  // An explicit ?api=<url> always wins and is remembered for later bare loads.
  const fromQuery = new URLSearchParams(window.location.search).get("api");
  if (fromQuery) {
    try {
      new URL(fromQuery);
      const clean = fromQuery.replace(/\/$/, "");
      window.localStorage.setItem(STORAGE_KEY, clean);
      return clean;
    } catch {
      console.warn("Invalid `api` query param, ignoring:", fromQuery);
    }
  }

  const fallback = defaultBaseUrl();

  // A plain localhost run must always talk to its own backend. Reading a
  // persisted override here let a stray `makerlab.apiBaseUrl` (e.g. left over
  // from an earlier ?api= visit, or from a since-moved backend) silently point
  // the UI at a different — often empty or unreachable — API, which then
  // rendered as a confusing "no robots" state indistinguishable from success.
  // So on loopback origins we ignore the stored value; the override only
  // applies to remote deployments where the page origin isn't the API.
  if (isLoopbackHost(window.location.hostname)) return fallback;

  return window.localStorage.getItem(STORAGE_KEY) || fallback;
};

export const ApiProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [baseUrl] = useState<string>(resolveInitialBaseUrl);
  const wsBaseUrl = httpToWs(baseUrl);

  const fetchWithHeaders = useCallback(async (url: string, options: RequestInit = {}): Promise<Response> => {
    return fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });
  }, []);

  const value = useMemo(
    () => ({ baseUrl, wsBaseUrl, fetchWithHeaders }),
    [baseUrl, wsBaseUrl, fetchWithHeaders]
  );

  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>;
};

export const useApi = (): ApiContextType => {
  const context = useContext(ApiContext);
  if (context === undefined) {
    throw new Error("useApi must be used within an ApiProvider");
  }
  return context;
};
