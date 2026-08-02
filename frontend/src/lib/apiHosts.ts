/**
 * The hosts registry — which MakerLab backend the GUI talks to.
 *
 * MakerLab can run split across two machines (see docs/remote-portal/SPEC.md):
 * a headless "server" that owns the follower arm + cameras, and this laptop
 * (the "client") that owns the leader arm and runs the leader-bridge. The GUI
 * therefore has TWO addresses at once:
 *
 *  - the ACTIVE host — robots, datasets, training, inference. May be remote.
 *  - the LOCAL host  — always this machine, where the leader-bridge lives.
 *
 * With no remote host added they are the same URL and everything behaves
 * exactly as it did before this file existed.
 *
 * Storage keys are read/written here only; ApiContext owns the React state.
 */

/** A host the user has added. The URL is the identity. */
export interface ApiHost {
  /** Normalized origin, no trailing slash (e.g. "http://192.168.1.20:8000"). */
  url: string;
  /** User-supplied label ("Mac Mini"). */
  name: string;
  /** ms epoch of the last successful reach; null if never reached. */
  lastSeen: number | null;
}

/** Whether the last probe of a host answered. "unknown" = not probed yet. */
export type HostReachability = "unknown" | "online" | "offline";

/** A host as the UI consumes it: registry entry + derived flags. */
export interface ApiHostEntry extends ApiHost {
  /** True for the implicit "this computer" entry, which is never stored. */
  isLocal: boolean;
  status: HostReachability;
}

/** Active host. Same key as before hosts existed — an installed user's saved
 * `?api=` override keeps working untouched. */
const ACTIVE_KEY = "makerlab.apiBaseUrl";
/** The added REMOTE hosts. The local host is implicit and never stored. */
const HOSTS_KEY = "makerlab.apiHosts";

const DEFAULT_LOCALHOST = "http://localhost:8000";
/** MakerLab's default port, assumed when the user types a bare IP/hostname. */
const DEFAULT_PORT = 8000;

export const LOCAL_HOST_NAME = "This computer";

export const httpToWs = (url: string): string =>
  url.replace(/^http(s?):/, "ws$1:");

/**
 * This machine's API address, always — independent of which host is selected.
 *
 * In production the backend serves the UI, so the page origin is the API. In
 * Vite dev the UI is on :8080 while the API stays on :8000, so the origin
 * would be wrong there. (Verbatim the old `defaultBaseUrl`.)
 */
export const localApiBaseUrl = (): string => {
  if (typeof window === "undefined") return DEFAULT_LOCALHOST;
  return import.meta.env.DEV ? DEFAULT_LOCALHOST : window.location.origin;
};

/**
 * Accepts what a user actually types — "192.168.1.20", "mac-mini.local:8000",
 * "http://host:8000/" — and returns a bare origin, or null if unparseable.
 * A bare host with no scheme gets http:// and MakerLab's default port; an
 * explicit scheme is trusted as given (an https host on :443 stays :443).
 */
export const normalizeHostUrl = (raw: string): string | null => {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const hadScheme = /^https?:\/\//i.test(trimmed);
  let u: URL;
  try {
    u = new URL(hadScheme ? trimmed : `http://${trimmed}`);
  } catch {
    return null;
  }
  if (!u.hostname) return null;
  if (!hadScheme && !u.port) u.port = String(DEFAULT_PORT);
  // Keep any path (a reverse proxy may mount MakerLab under a prefix), minus
  // the trailing slash, so `${baseUrl}/robots` concatenation stays correct.
  const path = u.pathname.replace(/\/+$/, "");
  return `${u.protocol}//${u.host}${path}`;
};

const safeGet = (key: string): string | null => {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
};

const safeSet = (key: string, value: string) => {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Storage unavailable (private mode, quota) — in-memory state still applies.
  }
};

const safeRemove = (key: string) => {
  try {
    window.localStorage.removeItem(key);
  } catch {
    // See safeSet.
  }
};

export const readRemoteHosts = (): ApiHost[] => {
  if (typeof window === "undefined") return [];
  const raw = safeGet(HOSTS_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((h): ApiHost | null => {
        if (!h || typeof h.url !== "string") return null;
        const url = normalizeHostUrl(h.url);
        if (!url) return null;
        return {
          url,
          name: typeof h.name === "string" && h.name.trim() ? h.name : url,
          lastSeen: typeof h.lastSeen === "number" ? h.lastSeen : null,
        };
      })
      .filter((h): h is ApiHost => h !== null);
  } catch {
    return [];
  }
};

export const writeRemoteHosts = (hosts: ApiHost[]) => {
  if (typeof window === "undefined") return;
  if (hosts.length === 0) safeRemove(HOSTS_KEY);
  else safeSet(HOSTS_KEY, JSON.stringify(hosts));
};

export const writeActiveBaseUrl = (url: string) => {
  if (typeof window === "undefined") return;
  // The local host is the DEFAULT, not an override — clearing the key restores
  // the exact pre-hosts state (origin in prod, :8000 in dev).
  if (url === localApiBaseUrl()) safeRemove(ACTIVE_KEY);
  else safeSet(ACTIVE_KEY, url);
};

/**
 * The host to talk to on this page load: `?api=` → saved override → this
 * machine. Memoized because `?api=` has a persistence side effect and callers
 * outside React (the useRobots store) resolve it at module init.
 */
let cachedInitialActive: string | null = null;
export const initialActiveBaseUrl = (): string => {
  if (cachedInitialActive !== null) return cachedInitialActive;
  if (typeof window === "undefined") {
    cachedInitialActive = DEFAULT_LOCALHOST;
    return cachedInitialActive;
  }
  const fromQuery = new URLSearchParams(window.location.search).get("api");
  if (fromQuery) {
    const clean = normalizeHostUrl(fromQuery);
    if (clean) {
      safeSet(ACTIVE_KEY, clean);
      cachedInitialActive = clean;
      return cachedInitialActive;
    }
    console.warn("Invalid `api` query param, ignoring:", fromQuery);
  }
  cachedInitialActive = safeGet(ACTIVE_KEY) || localApiBaseUrl();
  return cachedInitialActive;
};
