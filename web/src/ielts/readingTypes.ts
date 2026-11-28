// Mirrors services/gateway/src/gateway/ielts_reading_api.py and ielts/models.py (ReadingReport).

export interface Passage {
  id: string;
  title: string;
  paragraphs: string[];
}

export interface Question {
  id: string;
  number: number;
  passage_id: string;
  task_type: string;
  task_label: string;
  instruction: string;
  prompt: string;
  paragraph: number | null;
  choices: Record<string, string>;
}

export interface ReadingInfo {
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

export interface ReadingAttempt {
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
  paper: { passages: Passage[]; questions: Question[] } | null;
  sample?: boolean;
}

export interface QuestionResult {
  number: number;
  id: string;
  passage: string;
  task_type: string;
  task_label: string;
  prompt: string;
  paragraph: number | null;
  answer: string;
  answer_label: string;
  key: string;
  key_label: string;
  correct: boolean;
  evidence: string;
}

export interface TypeScore {
  task_type: string;
  label: string;
  correct: number;
  total: number;
}

export interface ReadingReport {
  set_id: string;
  raw: number;
  total: number;
  scaled_40: number;
  band: number;
  timing: { allowed_s: number; used_s: number; late: boolean };
  by_type: TypeScore[];
  questions: QuestionResult[];
  band_note: string;
  set_note: string;
  keys_note: string;
  band_source_url: string;
  timing_source_url: string;
  disclaimer: string;
  sample?: boolean;
}
