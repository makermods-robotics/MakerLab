export type Fetcher = (
  url: string,
  options?: RequestInit
) => Promise<Response>;

export class ApiError extends Error {
  status: number;
  detail: string | null;
  constructor(message: string, status: number, detail: string | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export interface ApiRequestOptions {
  method?: string;
  body?: unknown;
  signal?: AbortSignal;
  /** Human-readable label for the error message, e.g. "Start training". */
  action?: string;
}

/**
 * Performs a request against the lelab backend and parses the JSON response.
 * Throws ApiError with FastAPI's `detail` field on non-2xx, or on JSON parse
 * failure. Use this in place of ad-hoc `r.ok` / `r.json()` branching.
 */
export async function apiRequest<T = unknown>(
  baseUrl: string,
  fetcher: Fetcher,
  path: string,
  { method = "GET", body, signal, action }: ApiRequestOptions = {}
): Promise<T> {
  const init: RequestInit = { method, signal };
  if (body !== undefined) init.body = JSON.stringify(body);

  const url = `${baseUrl}${path}`;
  const r = await fetcher(url, init);
  if (!r.ok) {
    let detail: string | null = null;
    try {
      const errBody = await r.json();
      const raw = errBody?.detail ?? errBody?.message ?? null;
      // FastAPI 422s put an array of {loc,msg,type} in `detail`; most errors a
      // string. Normalize non-strings so they never render as "[object Object]".
      if (raw == null || typeof raw === "string") {
        detail = raw;
      } else if (Array.isArray(raw)) {
        detail = raw.map((d) => d?.msg ?? JSON.stringify(d)).join("; ");
      } else {
        detail = JSON.stringify(raw);
      }
    } catch {
      // body wasn't JSON
    }
    const label = action || `${method} ${path}`;
    throw new ApiError(
      `${label} failed: ${detail ?? r.status}`,
      r.status,
      detail
    );
  }
  // 204 No Content
  if (r.status === 204) return undefined as T;
  return r.json() as Promise<T>;
}
