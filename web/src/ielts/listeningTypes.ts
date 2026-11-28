// Mirrors services/gateway/src/gateway/ielts_listening_api.py and ielts/models.py (ListeningReport).
import type { QuestionResult, TypeScore } from "./readingTypes";

export type { QuestionResult, TypeScore };

export interface Section {
  id: string;
  number: number;
  title: string;
  context: string;
  audio_url: string;
  audio_s: number | null;
}

export interface ListeningQuestion {
  id: string;
  number: number;
  section_id: string;
  task_type: string;
  task_label: string;
  instruction: string;
  prompt: string;
  choices: Record<string, string>;
}

export interface ListeningInfo {
  set_id: string;
  allowed_s: number;
  question_count: number;
  task_types: Record<string, string>;
  timing_source_url: string;
  band_source_url: string;
  band_note: string;
  set_note: string;
  keys_note: string;
}

export interface ListeningAttempt {
  id: string;
  set_id: string;
  status: "created" | "in_progress" | "submitted" | "grading" | "graded" | "ungraded";
  allowed_s: number;
  remaining_s: number | null;
  started_at: number | null;
  submitted_at: number | null;
  late: boolean;
  answers: Record<string, string>;
  answered: number;
  band: number | null;
  raw: number | null;
  scaled_40: number | null;
  paper: { sections: Section[]; questions: ListeningQuestion[] } | null;
  sample?: boolean;
}

export interface SectionTranscript {
  id: string;
  title: string;
  lines: string[];
}

export interface ListeningReport {
  set_id: string;
  raw: number;
  total: number;
  scaled_40: number;
  band: number;
  timing: { allowed_s: number; used_s: number; late: boolean };
  by_type: TypeScore[];
  questions: QuestionResult[];
  transcripts: SectionTranscript[];
  band_note: string;
  set_note: string;
  keys_note: string;
  band_source_url: string;
  timing_source_url: string;
  disclaimer: string;
  sample?: boolean;
}
