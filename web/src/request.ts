/** The one way the web app talks to the gateway: chat, feedback and IELTS all go through `request`. */

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string
  ) {
    super(message);
  }
}

/** What a learner reads when the browser could not reach the gateway at all (status 0). */
export const UNREACHABLE = "Cannot reach the server";

export interface RequestOptions {
  method?: string;
  /** Sent as the JSON body. */
  json?: unknown;
  /** Aborting it rejects with the browser's AbortError, never an ApiError. */
  signal?: AbortSignal;
  keepalive?: boolean;
}

const isAbort = (err: unknown, signal?: AbortSignal) =>
  signal?.aborted === true || (err instanceof Error && err.name === "AbortError");

/** The gateway's `detail` when it sent a plain-text one, else a generic line with the status. */
async function failure(res: Response): Promise<ApiError> {
  let detail = `Request failed (${res.status})`;
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string" && body.detail) detail = body.detail;
  } catch {
    // keep the generic message
  }
  return new ApiError(res.status, detail);
}

/** Fetch `path` and return the response once it is ok; a failure is an `ApiError` with a message fit to show. */
export async function request(path: string, options: RequestOptions = {}): Promise<Response> {
  const { method, json, signal, keepalive } = options;
  let res: Response;
  try {
    res = await fetch(path, {
      method,
      headers: json === undefined ? undefined : { "Content-Type": "application/json" },
      body: json === undefined ? undefined : JSON.stringify(json),
      signal,
      keepalive,
    });
  } catch (err) {
    if (isAbort(err, signal)) throw err;
    throw new ApiError(0, UNREACHABLE);
  }
  if (!res.ok) throw await failure(res);
  return res;
}

/** `request` for an endpoint that answers with JSON. */
export async function requestJson<T>(path: string, options?: RequestOptions): Promise<T> {
  return (await request(path, options)).json() as Promise<T>;
}
