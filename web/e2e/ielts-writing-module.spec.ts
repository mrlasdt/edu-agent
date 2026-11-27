import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

// The gateway is mocked with a small stateful fake: no agent service or API key is involved.

const LETTER = "Dear Mr Hall, the lift in our building has been broken for three weeks. Please repair it.";
const ESSAY = "Learning to cook at school is useful for young people. Many students has no skills at home.";
const DISCLAIMER = "Mock score: an estimate, not an official IELTS result.";
const SOURCES = [
  "https://ielts.org/take-a-test/test-types/ielts-general-training-test/ielts-general-training-format-writing",
  "https://ielts.org/take-a-test/your-results/ielts-scoring-in-detail",
];
const NOTE = "Task 2 counts twice as much as Task 1 in the Writing band.";

function reportFor(task: "T1" | "T2", text: string, overall: number) {
  const first = task === "T1" ? ["TA", "Task Achievement"] : ["TR", "Task Response"];
  const criterion = (id: string, name: string) => ({
    id, name, band: overall, sample_bands: [overall, overall, overall], capped_from: null,
    verdict: `${name} is adequate.`,
    evidence: [{ quote: text.split(" ").slice(0, 4).join(" "), start: 0, end: 10, para: 1, effect: "supports" }],
    holding_back: "Ideas are thin.", to_reach_next_band: "Develop each idea.",
    actions: [{ do: `Improve ${name}`, why: "It lifts the band." }], paragraph_map: [], error_table: {},
  });
  return {
    status: "graded", task, word_count: text.split(/\s+/).length,
    timing: { allowed_s: 3600, used_s: 120, late: false },
    estimated_band: {
      overall, range: [overall - 0.5, overall + 0.5], confidence: "medium", samples: 3, sample_spread: 0.5,
      calibration: { status: "uncalibrated", label: "Uncalibrated estimate", n: null, tier: null,
        within_half_band: null, note: "No labelled set has passed the calibration gate yet." },
    },
    criteria: [criterion(first[0], first[1]), criterion("CC", "Coherence and Cohesion"),
      criterion("LR", "Lexical Resource"), criterion("GRA", "Grammatical Range and Accuracy")],
    error_spans: [], task_checks: [], top_priorities: [], flags: [], feedback_status: "complete",
    vs_previous_attempt: null, ungraded_reason: null, prompt_versions: null, disclaimer: DISCLAIMER,
  };
}

async function mockGateway(page: Page) {
  const state = { status: "none", t1: "", t2: "", saves: 0, polls: 0 };
  const task = (key: "t1" | "t2") => ({
    prompt_id: `p-${key}`,
    prompt: key === "t1" ? "Write a letter to the building manager." : "Do you agree or disagree?",
    text: state[key], word_count: state[key].split(/\s+/).filter(Boolean).length,
    band: state.status === "graded" ? (key === "t1" ? 6 : 7) : null,
  });
  const view = () => ({
    id: "wm1", status: state.status, allowed_s: 3600,
    remaining_s: state.status === "in_progress" ? 3600 : null, started_at: 1, submitted_at: null,
    late: false, t1: task("t1"), t2: task("t2"), error: null, retryable: false,
    writing_band: state.status === "graded" ? 6.5 : null, disclaimer: DISCLAIMER,
  });
  const json = (body: unknown, status = 200) => ({
    status, contentType: "application/json", body: JSON.stringify(body),
  });

  await page.route(/^https?:\/\/[^/]+\/ielts\/writing\//, async (route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname;
    const method = req.method();
    if (path.endsWith("/info"))
      return route.fulfill(json({
        allowed_s: 3600, t1_minutes: 20, t2_minutes: 40, t1_min_words: 150, t2_min_words: 250,
        weights_note: NOTE, source_urls: SOURCES,
      }));
    if (path.endsWith("/attempts") && method === "POST") {
      state.status = "created";
      return route.fulfill(json(view(), 201));
    }
    if (path.endsWith("/start")) {
      state.status = "in_progress";
      return route.fulfill(json(view()));
    }
    if (path.endsWith("/draft")) {
      Object.assign(state, req.postDataJSON());
      state.saves += 1;
      return route.fulfill(json(view()));
    }
    if (path.endsWith("/submit")) {
      Object.assign(state, req.postDataJSON());
      state.status = "submitted";
      return route.fulfill(json(view()));
    }
    if (path.endsWith("/attempts/wm1")) {
      if (state.status === "submitted" && ++state.polls >= 2) state.status = "graded";
      return route.fulfill(json(view()));
    }
    if (path.endsWith("/report"))
      return route.fulfill(json({
        status: "graded", module: "ielts_writing",
        t1: reportFor("T1", state.t1, 6), t2: reportFor("T2", state.t2, 7), writing_band: 6.5,
        weights_note: NOTE, weights_source_urls: SOURCES,
        timing: { allowed_s: 3600, used_s: 2400, late: false }, disclaimer: DISCLAIMER,
      }));
    return route.fulfill(json({ detail: "not found" }, 404));
  });
  return state;
}

test("write a letter then an essay, submit both, and read one combined Writing band", async ({ page }) => {
  await page.clock.install();
  const server = await mockGateway(page);

  await page.goto("/ielts-mock?module=writing-module");
  await expect(page.getByText(NOTE)).toBeVisible();
  await page.getByRole("button", { name: "Start the 60 minute clock" }).click();
  await expect(page.getByRole("timer")).toContainText("60:00");
  await expect(page.getByText("Write a letter to the building manager.")).toBeVisible();

  await page.getByLabel("Your letter").fill(LETTER);
  await page.getByRole("tab", { name: "Task 2 - Essay" }).click();
  await expect(page.getByText("Do you agree or disagree?")).toBeVisible();
  await page.getByLabel("Your essay").fill(ESSAY);

  await page.clock.runFor(11_000);
  await expect.poll(() => server.t1).toBe(LETTER);
  expect(server.t2).toBe(ESSAY);

  await page.getByRole("button", { name: "Submit both tasks for marking" }).click();
  await expect(page.getByText(/Under the minimum/)).toBeVisible();
  await page.getByRole("button", { name: "Yes, submit" }).click();
  await expect(page.getByRole("heading", { name: "Marking your letter and essay" })).toBeVisible();

  await page.clock.runFor(5_000);
  await expect(page.getByLabel("Mock Writing band 6.5")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Task 1 - Letter" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Task 2 - Essay" })).toBeVisible();
  await expect(page.getByRole("article", { name: "Task Achievement" })).toContainText("Improve Task Achievement");
  await expect(page.getByRole("article", { name: "Task Response" })).toContainText("Improve Task Response");
  await expect(page.getByRole("link", { name: "Discuss this report" })).toHaveAttribute("href", "/?attempt=wm1");
});
