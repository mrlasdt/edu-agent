import { ApiError, request, requestJson } from "../request";
import type { Attempt, Budget, Report, WritingPrompt } from "./types";

export { ApiError };

export const getBudget = (signal?: AbortSignal) => requestJson<Budget>("/ielts/budget", { signal });
export const listPrompts = (signal?: AbortSignal) =>
  requestJson<WritingPrompt[]>("/ielts/prompts", { signal });
export const listAttempts = (signal?: AbortSignal) =>
  requestJson<Attempt[]>("/ielts/attempts", { signal });
export const createAttempt = (promptId: string) =>
  requestJson<Attempt>("/ielts/attempts", { method: "POST", json: { prompt_id: promptId } });
export const getAttempt = (id: string, signal?: AbortSignal) =>
  requestJson<Attempt>(`/ielts/attempts/${id}`, { signal });
export const startAttempt = (id: string) =>
  requestJson<Attempt>(`/ielts/attempts/${id}/start`, { method: "POST" });
export const saveDraft = (id: string, text: string) =>
  requestJson<Attempt>(`/ielts/attempts/${id}/draft`, { method: "PUT", json: { text } });
export const submitAttempt = (id: string, text: string) =>
  requestJson<Attempt>(`/ielts/attempts/${id}/submit`, { method: "POST", json: { text } });
export const regradeAttempt = (id: string) =>
  requestJson<Attempt>(`/ielts/attempts/${id}/regrade`, { method: "POST" });
export const getReport = (id: string, signal?: AbortSignal) =>
  requestJson<Report>(`/ielts/attempts/${id}/report`, { signal });

/** Last-chance save while the page is closing; the browser may drop it, the 10 s autosave is the net. */
export function saveDraftOnExit(id: string, text: string): void {
  request(`/ielts/attempts/${id}/draft`, { method: "PUT", json: { text }, keepalive: true }).catch(
    () => {
      // nothing more can be done while unloading
    }
  );
}
