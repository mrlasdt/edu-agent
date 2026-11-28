// Mirrors services/gateway/src/gateway/ielts_writing_api.py and WritingModuleReport in ielts/models.py.
import type { AttemptStatus, Report, Timing } from "./types";

export const LETTER_MINIMUM_WORDS = 150;
export const ESSAY_MINIMUM_WORDS = 250;

export type TaskKey = "t1" | "t2";

export interface WritingModuleInfo {
  allowed_s: number;
  t1_minutes: number;
  t2_minutes: number;
  t1_min_words: number;
  t2_min_words: number;
  weights_note: string;
  source_urls: string[];
}

export interface TaskText {
  prompt_id: string;
  prompt: string;
  text: string;
  word_count: number;
  band: number | null;
}

export interface WritingModuleAttempt {
  id: string;
  status: AttemptStatus;
  allowed_s: number;
  remaining_s: number | null;
  started_at: number | null;
  submitted_at: number | null;
  late: boolean;
  t1: TaskText;
  t2: TaskText;
  error: string | null;
  retryable: boolean;
  writing_band: number | null;
  disclaimer: string;
  sample?: boolean;
}

export interface WritingModuleReport {
  status: "graded" | "partial";
  module: string;
  t1: Report;
  t2: Report;
  writing_band: number | null;
  weights_note: string;
  weights_source_urls: string[];
  timing: Timing | null;
  disclaimer: string;
  sample?: boolean;
}

export interface ModuleTexts {
  t1: string;
  t2: string;
}
