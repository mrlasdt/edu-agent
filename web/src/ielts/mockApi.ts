import { requestJson } from "../request";

// Mirrors services/gateway/src/gateway/ielts/mock.py (MockReport).
export type ModuleKey = "listening" | "reading" | "writing" | "speaking";
export type ModuleState =
  | "not_assessed"
  | "not_started"
  | "in_progress"
  | "marking"
  | "graded"
  | "partial"
  | "ungraded";

export interface ModuleStatus {
  key: ModuleKey;
  label: string;
  state: ModuleState;
  band: number | null;
  attempt_id: string | null;
  sittable: boolean;
  detail: string;
}

export interface NextModule {
  key: ModuleKey;
  label: string;
  reason: string;
}

export interface MockReport {
  id: string;
  created_at: number;
  modules: ModuleStatus[];
  overall: number | null;
  partial: boolean;
  overall_note: string;
  overall_source_url: string;
  next_module: NextModule | null;
  disclaimer: string;
}

const base = "/ielts/mock";

export const createMock = () => requestJson<MockReport>(base, { method: "POST" });
export const getMock = (id: string, signal?: AbortSignal) =>
  requestJson<MockReport>(`${base}/${id}`, { signal });
export const startMockModule = (id: string, key: ModuleKey) =>
  requestJson<MockReport>(`${base}/${id}/modules/${key}/attempt`, { method: "POST" });
