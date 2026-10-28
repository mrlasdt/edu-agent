import type { Attempt, CriterionReport, Report } from "../types";

export const ESSAY =
  "Learning to cook at school is useful. Many students has no skills. Another problem is the cost.";

export function attempt(patch: Partial<Attempt> = {}): Attempt {
  return {
    id: "a1",
    prompt_id: "ielts_writing_prompt_1",
    prompt: "Some people think schools should teach cooking. Discuss.",
    status: "in_progress",
    allowed_s: 2400,
    remaining_s: 2400,
    started_at: 1,
    submitted_at: null,
    late: false,
    draft: "",
    word_count: 0,
    error: null,
    retryable: false,
    overall: null,
    disclaimer: "Estimated band, not an official IELTS score.",
    ...patch,
  };
}

const criterion = (id: CriterionReport["id"], name: string, band: number): CriterionReport => ({
  id,
  name,
  band,
  sample_bands: [band, band, band + 0.5],
  capped_from: null,
  verdict: `${id} verdict`,
  evidence: [
    { quote: "Learning to cook at school is useful", start: 0, end: 36, para: 1, effect: "supports" },
  ],
  holding_back: `${id} holding back`,
  to_reach_next_band: `${id} next`,
  actions: [{ do: `${id} do this`, why: "because" }],
  paragraph_map: id === "CC" ? [{ para: 1, job: "introduction", one_central_idea: true }] : [],
  error_table: id === "GRA" ? { agreement: 1 } : {},
});

export function report(patch: Partial<Report> = {}): Report {
  return {
    status: "graded",
    task: "T2",
    word_count: 17,
    timing: { allowed_s: 2400, used_s: 1800, late: false },
    estimated_band: {
      overall: 6.0,
      range: [5.5, 6.5],
      confidence: "medium",
      samples: 3,
      sample_spread: 0.5,
      calibration: {
        status: "uncalibrated",
        label: "Uncalibrated estimate",
        n: null,
        tier: null,
        within_half_band: null,
        note: "No labelled set has passed the calibration gate yet.",
      },
    },
    criteria: [
      criterion("TR", "Task Response", 5.5),
      criterion("CC", "Coherence and Cohesion", 6),
      criterion("LR", "Lexical Resource", 6),
      criterion("GRA", "Grammatical Range and Accuracy", 6.5),
    ],
    error_spans: [
      {
        start: ESSAY.indexOf("students has"),
        end: ESSAY.indexOf("students has") + "students has".length,
        quote: "students has",
        type: "agreement",
        note: "subject-verb agreement",
        suggestion: "students have",
      },
    ],
    task_checks: [
      { check: "at least 250 words", passed: false },
      { check: "clear position in the introduction", passed: null },
    ],
    top_priorities: [{ priority: 1, criterion: "TR", do: "State your position early", why: "It anchors the essay." }],
    flags: [{ code: "under_length", severity: "warn", message: "17 words is well under the 250-word minimum." }],
    feedback_status: "complete",
    vs_previous_attempt: { overall_delta: 0.5, criteria_deltas: { TR: 0.5, CC: 0 } },
    ungraded_reason: null,
    disclaimer: "Estimated band, not an official IELTS score.",
    ...patch,
  };
}
