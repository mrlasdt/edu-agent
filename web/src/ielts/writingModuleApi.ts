import { requestJson } from "../request";
import type {
  ModuleTexts,
  WritingModuleAttempt,
  WritingModuleInfo,
  WritingModuleReport,
} from "./writingModuleTypes";

const base = "/ielts/writing";

export const writingModuleInfo = (signal?: AbortSignal) =>
  requestJson<WritingModuleInfo>(`${base}/info`, { signal });
export const createWritingModule = () =>
  requestJson<WritingModuleAttempt>(`${base}/attempts`, { method: "POST", json: {} });
export const getWritingModule = (id: string, signal?: AbortSignal) =>
  requestJson<WritingModuleAttempt>(`${base}/attempts/${id}`, { signal });
export const startWritingModule = (id: string) =>
  requestJson<WritingModuleAttempt>(`${base}/attempts/${id}/start`, { method: "POST" });
export const saveWritingModuleDraft = (id: string, texts: ModuleTexts) =>
  requestJson<WritingModuleAttempt>(`${base}/attempts/${id}/draft`, { method: "PUT", json: texts });
export const submitWritingModule = (id: string, texts: ModuleTexts) =>
  requestJson<WritingModuleAttempt>(`${base}/attempts/${id}/submit`, { method: "POST", json: texts });
export const regradeWritingModule = (id: string) =>
  requestJson<WritingModuleAttempt>(`${base}/attempts/${id}/regrade`, { method: "POST" });
export const getWritingModuleReport = (id: string, signal?: AbortSignal) =>
  requestJson<WritingModuleReport>(`${base}/attempts/${id}/report`, { signal });
