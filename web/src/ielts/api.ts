import type { Attempt, Budget, Report, WritingPrompt } from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string
  ) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(0, "Cannot reach the server");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // keep the generic message
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

const json = (method: string, body?: unknown): RequestInit => ({
  method,
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const getBudget = () => call<Budget>("/ielts/budget");
export const listPrompts = () => call<WritingPrompt[]>("/ielts/prompts");
export const listAttempts = () => call<Attempt[]>("/ielts/attempts");
export const createAttempt = (promptId: string) =>
  call<Attempt>("/ielts/attempts", json("POST", { prompt_id: promptId }));
export const getAttempt = (id: string) => call<Attempt>(`/ielts/attempts/${id}`);
export const startAttempt = (id: string) =>
  call<Attempt>(`/ielts/attempts/${id}/start`, json("POST"));
export const saveDraft = (id: string, text: string) =>
  call<Attempt>(`/ielts/attempts/${id}/draft`, json("PUT", { text }));
export const submitAttempt = (id: string, text: string) =>
  call<Attempt>(`/ielts/attempts/${id}/submit`, json("POST", { text }));
export const regradeAttempt = (id: string) =>
  call<Attempt>(`/ielts/attempts/${id}/regrade`, json("POST"));
export const getReport = (id: string) => call<Report>(`/ielts/attempts/${id}/report`);

/** Last-chance save while the page is closing; the browser may drop it, the 10 s autosave is the net. */
export function saveDraftOnExit(id: string, text: string): void {
  try {
    void fetch(`/ielts/attempts/${id}/draft`, {
      ...json("PUT", { text }),
      headers: { "Content-Type": "application/json" },
      keepalive: true,
    });
  } catch {
    // nothing more can be done while unloading
  }
}
