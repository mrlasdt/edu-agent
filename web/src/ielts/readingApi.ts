import { requestJson } from "../request";
import type { ReadingAttempt, ReadingInfo, ReadingReport } from "./readingTypes";

const base = "/ielts/reading";

export const readingInfo = (signal?: AbortSignal) => requestJson<ReadingInfo>(`${base}/info`, { signal });
export const createReading = (sample = false) =>
  requestJson<ReadingAttempt>(`${base}/attempts${sample ? "?sample=true" : ""}`, { method: "POST" });
export const getReading = (id: string, signal?: AbortSignal) =>
  requestJson<ReadingAttempt>(`${base}/attempts/${id}`, { signal });
export const startReading = (id: string) =>
  requestJson<ReadingAttempt>(`${base}/attempts/${id}/start`, { method: "POST" });
export const saveReadingAnswers = (id: string, answers: Record<string, string>) =>
  requestJson<ReadingAttempt>(`${base}/attempts/${id}/draft`, { method: "PUT", json: { answers } });
export const submitReading = (id: string, answers: Record<string, string>) =>
  requestJson<ReadingAttempt>(`${base}/attempts/${id}/submit`, { method: "POST", json: { answers } });
export const getReadingReport = (id: string, signal?: AbortSignal) =>
  requestJson<ReadingReport>(`${base}/attempts/${id}/report`, { signal });
