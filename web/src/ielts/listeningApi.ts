import { requestJson } from "../request";
import type { ListeningAttempt, ListeningInfo, ListeningReport } from "./listeningTypes";

const base = "/ielts/listening";

export const listeningInfo = (signal?: AbortSignal) =>
  requestJson<ListeningInfo>(`${base}/info`, { signal });
export const createListening = () => requestJson<ListeningAttempt>(`${base}/attempts`, { method: "POST" });
export const getListening = (id: string, signal?: AbortSignal) =>
  requestJson<ListeningAttempt>(`${base}/attempts/${id}`, { signal });
export const startListening = (id: string) =>
  requestJson<ListeningAttempt>(`${base}/attempts/${id}/start`, { method: "POST" });
export const saveListeningAnswers = (id: string, answers: Record<string, string>) =>
  requestJson<ListeningAttempt>(`${base}/attempts/${id}/draft`, { method: "PUT", json: { answers } });
export const submitListening = (id: string, answers: Record<string, string>) =>
  requestJson<ListeningAttempt>(`${base}/attempts/${id}/submit`, { method: "POST", json: { answers } });
export const getListeningReport = (id: string, signal?: AbortSignal) =>
  requestJson<ListeningReport>(`${base}/attempts/${id}/report`, { signal });
