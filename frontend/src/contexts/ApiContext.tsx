import React, { createContext, useContext, ReactNode, useState, useCallback, useEffect, useMemo } from "react";
import { mockHubResponse } from "@/lib/mockHub";
import {
  ApiHost,
  ApiHostEntry,
  HostReachability,
  LOCAL_HOST_NAME,
  httpToWs,
  initialActiveBaseUrl,
  localApiBaseUrl,
  normalizeHostUrl,
  readRemoteHosts,
  writeActiveBaseUrl,
  writeRemoteHosts,
} from "@/lib/apiHosts";

interface ApiContextType {
  /** The ACTIVE host: robots, datasets, training, inference. May be remote. */
  baseUrl: string;
  wsBaseUrl: string;
  fetchWithHeaders: (url: string, options?: RequestInit) => Promise<Response>;

  /** THIS machine, always — regardless of which host is active. The
   * leader-bridge (local leader arm + the MJPEG re-serve of Portal video)
   * lives here and nowhere else. Equal to `baseUrl` when no remote is selected. */
  localBaseUrl: string;
  localWsBaseUrl: string;
  /** True when the active host is a different machine from this one. */
  isRemote: boolean;

  /** Every known host, "This computer" first. */
  hosts: ApiHostEntry[];
  activeHost: ApiHostEntry;
  /** Switch the active host. Unknown URLs are ignored. */
  setActiveHostUrl: (url: string) => void;
  /** Validate + register a remote host. `force` skips the reachability gate so
   * a server that is merely booting can still be saved. */
  addHost: (
    name: string,
    rawUrl: string,
    force?: boolean,
  ) => Promise<{ ok: boolean; message?: string }>;
  /** Forget a remote host. Falls back to this computer if it was active. */
  removeHost: (url: string) => void;
  /** Re-probe every known host (cheap; used when the host menu opens). */
  probeAllHosts: () => void;
}

const ApiContext = createContext<ApiContextType | undefined>(undefined);

const PROBE_TIMEOUT_MS = 4000;
/** While a remote host is active, keep its reachability dot honest. */
const ACTIVE_PROBE_INTERVAL_MS = 15000;

export const ApiProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  // Pure functions of the environment, so a plain constant is enough.
  const localBaseUrl = localApiBaseUrl();
  const localWsBaseUrl = httpToWs(localBaseUrl);

  const [baseUrl, setBaseUrl] = useState<string>(initialActiveBaseUrl);
  const [remoteHosts, setRemoteHosts] = useState<ApiHost[]>(readRemoteHosts);
  const [statuses, setStatuses] = useState<Record<string, HostReachability>>({});
  const wsBaseUrl = httpToWs(baseUrl);
  const isRemote = baseUrl !== localBaseUrl;

  /**
   * Probe a host and record whether it answered.
   *
   * `/portal/info` is the cheapest endpoint that exists on a remote-capable
   * build; ANY http response — including a 404 from a build without it —
   * proves the host is up. Only a network-level failure or the timeout counts
   * as offline, exactly as the old startup probe did.
   *
   * This deliberately does NOT delete the stored host on failure (the old
   * self-heal did). A server that is booting, asleep, or briefly off the
   * network must stay in the list and come back on its own — being forgotten
   * mid-setup is worse than a red dot.
   */
  const probeHost = useCallback(async (url: string): Promise<boolean> => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
    try {
      await fetch(`${url}/portal/info`, { signal: controller.signal });
      setStatuses((prev) => ({ ...prev, [url]: "online" }));
      setRemoteHosts((prev) => {
        if (!prev.some((h) => h.url === url)) return prev;
        const next = prev.map((h) =>
          h.url === url ? { ...h, lastSeen: Date.now() } : h,
        );
        writeRemoteHosts(next);
        return next;
      });
      return true;
    } catch {
      setStatuses((prev) => ({ ...prev, [url]: "offline" }));
      return false;
    } finally {
      clearTimeout(timer);
    }
  }, []);

  // A `?api=` visit (or an install predating the registry) leaves an active
  // URL with no matching entry. Adopt it so the switcher can show it, name it,
  // and — critically — switch away from it.
  useEffect(() => {
    if (baseUrl === localBaseUrl) return;
    setRemoteHosts((prev) => {
      if (prev.some((h) => h.url === baseUrl)) return prev;
      const next = [...prev, { url: baseUrl, name: baseUrl, lastSeen: null }];
      writeRemoteHosts(next);
      return next;
    });
  }, [baseUrl, localBaseUrl]);

  // Keep the active REMOTE host's status current. Nothing runs while the local
  // host is active, so a single-machine install issues no extra requests.
  useEffect(() => {
    if (baseUrl === localBaseUrl) return;
    probeHost(baseUrl);
    const id = setInterval(() => probeHost(baseUrl), ACTIVE_PROBE_INTERVAL_MS);
    return () => clearInterval(id);
  }, [baseUrl, localBaseUrl, probeHost]);

  const setActiveHostUrl = useCallback(
    (url: string) => {
      const clean = normalizeHostUrl(url);
      if (!clean) return;
      writeActiveBaseUrl(clean);
      setBaseUrl(clean);
    },
    [],
  );

  const addHost = useCallback(
    async (
      name: string,
      rawUrl: string,
      force = false,
    ): Promise<{ ok: boolean; message?: string }> => {
      const url = normalizeHostUrl(rawUrl);
      if (!url) {
        return {
          ok: false,
          message: "That doesn't look like an address. Try 192.168.1.20:8000.",
        };
      }
      if (url === localBaseUrl) {
        return {
          ok: false,
          message: "That's this computer — it's already in the list.",
        };
      }
      if (!force) {
        const reachable = await probeHost(url);
        if (!reachable) {
          return {
            ok: false,
            message: `No answer from ${url}. Check the address, that makerlab is running there, and that it was started with --lan.`,
          };
        }
      }
      const label = name.trim() || url;
      setRemoteHosts((prev) => {
        const next = prev.some((h) => h.url === url)
          ? prev.map((h) => (h.url === url ? { ...h, name: label } : h))
          : [...prev, { url, name: label, lastSeen: force ? null : Date.now() }];
        writeRemoteHosts(next);
        return next;
      });
      writeActiveBaseUrl(url);
      setBaseUrl(url);
      return { ok: true };
    },
    [localBaseUrl, probeHost],
  );

  const removeHost = useCallback(
    (url: string) => {
      if (url === localBaseUrl) return; // this computer is not removable
      setRemoteHosts((prev) => {
        const next = prev.filter((h) => h.url !== url);
        writeRemoteHosts(next);
        return next;
      });
      setStatuses((prev) => {
        const { [url]: _dropped, ...rest } = prev;
        return rest;
      });
      setBaseUrl((current) => {
        if (current !== url) return current;
        writeActiveBaseUrl(localBaseUrl);
        return localBaseUrl;
      });
    },
    [localBaseUrl],
  );

  const hosts = useMemo<ApiHostEntry[]>(
    () => [
      {
        url: localBaseUrl,
        name: LOCAL_HOST_NAME,
        lastSeen: null,
        isLocal: true,
        // This machine serves the page, so it answers by definition; a probe
        // may still refine it (Vite dev serves the UI from a different port).
        status: statuses[localBaseUrl] ?? "online",
      },
      ...remoteHosts.map((h) => ({
        ...h,
        isLocal: false,
        status: statuses[h.url] ?? "unknown",
      })),
    ],
    [localBaseUrl, remoteHosts, statuses],
  );

  const activeHost = useMemo<ApiHostEntry>(
    () => hosts.find((h) => h.url === baseUrl) ?? hosts[0],
    [hosts, baseUrl],
  );

  const probeAllHosts = useCallback(() => {
    hosts.forEach((h) => probeHost(h.url));
  }, [hosts, probeHost]);

  const fetchWithHeaders = useCallback(async (url: string, options: RequestInit = {}): Promise<Response> => {
    // Dev-only Hub mock (?mockHub=1): serves canned jobs/models/auth so UI
    // work can continue through a huggingface.co outage. No-op in prod builds.
    if (import.meta.env.DEV) {
      const mocked = mockHubResponse(url, options);
      if (mocked) return mocked;
    }
    return fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });
  }, []);

  const value = useMemo(
    () => ({
      baseUrl,
      wsBaseUrl,
      fetchWithHeaders,
      localBaseUrl,
      localWsBaseUrl,
      isRemote,
      hosts,
      activeHost,
      setActiveHostUrl,
      addHost,
      removeHost,
      probeAllHosts,
    }),
    [
      baseUrl,
      wsBaseUrl,
      fetchWithHeaders,
      localBaseUrl,
      localWsBaseUrl,
      isRemote,
      hosts,
      activeHost,
      setActiveHostUrl,
      addHost,
      removeHost,
      probeAllHosts,
    ]
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
