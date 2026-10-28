// Mirrors services/agent/src/agent/ielts/models.py and services/gateway/.../ielts_api.py.

export const DISCLAIMER = "Estimated band, not an official IELTS score.";
export const ESSAY_MINIMUM_WORDS = 250;

export type CriterionId = "TR" | "CC" | "LR" | "GRA";

export interface WritingPrompt {
  id: string;
  essay_type: string;
  essay_type_label: string;
  text: string;
}

export type AttemptStatus =
  | "created"
  | "in_progress"
  | "submitted"
  | "grading"
  | "graded"
  | "ungraded";

export interface Attempt {
  id: string;
  prompt_id: string;
  prompt: string;
  status: AttemptStatus;
  allowed_s: number;
  remaining_s: number | null;
  started_at: number | null;
  submitted_at: number | null;
  late: boolean;
  draft: string;
  word_count: number;
  error: string | null;
  retryable: boolean;
  overall: number | null;
  disclaimer: string;
}

export interface Flag {
  code: string;
  severity: "info" | "warn" | "block";
  message: string;
}

export interface Evidence {
  quote: string;
  start: number;
  end: number;
  para: number;
  effect: "supports" | "weakens";
}

export interface Action {
  do: string;
  why: string;
}

export interface ParagraphRole {
  para: number;
  job: string;
  one_central_idea: boolean | null;
}

export interface CriterionReport {
  id: CriterionId;
  name: string;
  band: number;
  sample_bands: number[];
  capped_from: number | null;
  verdict: string;
  evidence: Evidence[];
  holding_back: string;
  to_reach_next_band: string;
  actions: Action[];
  paragraph_map: ParagraphRole[];
  error_table: Record<string, number>;
}

export interface ErrorSpan {
  start: number;
  end: number;
  quote: string;
  type: string;
  note: string;
  suggestion: string;
}

export interface TaskCheck {
  check: string;
  passed: boolean | null;
}

export interface Priority {
  priority: number;
  criterion: CriterionId;
  do: string;
  why: string;
}

export interface Calibration {
  status: "uncalibrated" | "calibrated";
  label: string;
  n: number | null;
  tier: string | null;
  within_half_band: number | null;
  note: string;
}

export interface EstimatedBand {
  overall: number;
  range: [number, number];
  confidence: "low" | "medium" | "high";
  samples: number;
  sample_spread: number;
  calibration: Calibration;
}

export interface Timing {
  allowed_s: number;
  used_s: number;
  late: boolean;
}

export interface PreviousAttempt {
  overall_delta: number;
  criteria_deltas: Record<string, number>;
}

export interface Report {
  status: "graded" | "ungraded";
  task: "T2";
  word_count: number;
  timing: Timing | null;
  estimated_band: EstimatedBand | null;
  criteria: CriterionReport[];
  error_spans: ErrorSpan[];
  task_checks: TaskCheck[];
  top_priorities: Priority[];
  flags: Flag[];
  feedback_status: "complete" | "partial" | "unavailable";
  vs_previous_attempt: PreviousAttempt | null;
  ungraded_reason: string | null;
  disclaimer: string;
}
