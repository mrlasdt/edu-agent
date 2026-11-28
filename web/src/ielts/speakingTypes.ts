// Mirrors services/gateway/src/gateway/ielts_speaking_api.py and ielts/models.py (SpeakingReport).

export interface SpeakingPartPaper {
  part: 1 | 2 | 3;
  title: string;
  questions?: string[];
  topic?: string;
  points?: string[];
  note?: string;
  prep_s?: number;
  talk_s?: number;
}

export interface SpeakingPaper {
  id: string;
  sample: boolean;
  title: string;
  /** The longest recording the server takes for each part, in seconds. */
  max_clip_s: [number, number, number];
  parts: SpeakingPartPaper[];
}

export interface SpeakingInfo {
  allowed_s: number;
  part2_prep_s: number;
  part2_talk_s: number;
  sample: SpeakingPaper;
  not_assessed: Record<string, string>;
  descriptors_source_url: string;
  format_source_url: string;
  model: string;
  set_note: string;
}

export interface SpeakingAttempt {
  id: string;
  set_id: string;
  status: "created" | "in_progress" | "submitted" | "grading" | "graded" | "ungraded";
  allowed_s: number;
  remaining_s: number | null;
  started_at: number | null;
  submitted_at: number | null;
  late: boolean;
  /** The parts that already have a recording on the server. */
  uploaded: number[];
  paper: SpeakingPaper;
  band: number | null;
  partial: boolean | null;
  error: string | null;
  retryable: boolean;
}

export interface SpeakingCriterion {
  id: "FC" | "LR" | "GRA" | "P";
  name: string;
  assessed: boolean;
  band: number | null;
  rationale: string;
  evidence: string[];
  reason: string;
}

export interface SpeakingPartResult {
  part: 1 | 2 | 3;
  title: string;
  transcript: string;
  duration_s: number;
  allowed_s: number | null;
  words: number;
  words_per_minute: number | null;
  fillers: number;
}

export interface SpeakingPriority {
  criterion: "FC" | "LR" | "GRA";
  priority: number;
  do: string;
}

export interface SpeakingReport {
  status: "graded" | "no_speech";
  set_id: string;
  sample: boolean;
  band: number | null;
  partial: boolean;
  partial_note: string;
  criteria: SpeakingCriterion[];
  not_assessed: string[];
  parts: SpeakingPartResult[];
  strengths: string[];
  priorities: SpeakingPriority[];
  flags: string[];
  model: string;
  timing: { allowed_s: number; used_s: number; late: boolean };
  descriptors_source_url: string;
  format_source_url: string;
  disclaimer: string;
}
