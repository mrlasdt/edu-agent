import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

// The gateway is mocked with a small stateful fake: no agent service or API key is involved.
// The test moves its clock (`elapsed`) by hand. It keeps the clock and the draft on "the server" so a page refresh really resumes.

const PROMPTS = Array.from({ length: 8 }, (_, i) => ({
  id: `ielts_writing_prompt_${i + 1}`,
  essay_type: "opinion",
  essay_type_label: i === 0 ? "Opinion" : "Discuss both views",
  text: `Practice question ${i + 1}: do you agree or disagree?`,
}));

const ESSAY =
  "Learning to cook at school is useful for young people. Many students has no skills at home.";

function reportFor(draft: string) {
  const start = draft.indexOf("students has");
  const criterion = (id: string, name: string, band: number) => ({
    id,
    name,
    band,
    sample_bands: [band, band, band],
    capped_from: null,
    verdict: `${name} is adequate.`,
    evidence: [{ quote: "Learning to cook at school is useful", start: 0, end: 36, para: 1, effect: "supports" }],
    holding_back: "Ideas are thin.",
    to_reach_next_band: "Develop each idea with an example.",
    actions: [{ do: "Add one example per paragraph", why: "It extends the argument." }],
    paragraph_map: [],
    error_table: {},
  });
  return {
    status: "graded",
    task: "T2",
    word_count: draft.split(/\s+/).length,
    timing: { allowed_s: 2400, used_s: 120, late: false },
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
      criterion("TR", "Task Response", 6),
      criterion("CC", "Coherence and Cohesion", 6),
      criterion("LR", "Lexical Resource", 6),
      criterion("GRA", "Grammatical Range and Accuracy", 6),
    ],
    error_spans: [
      { start, end: start + 12, quote: "students has", type: "agreement", note: "agreement", suggestion: "students have" },
    ],
    task_checks: [{ check: "at least 250 words", passed: false }],
    top_priorities: [{ priority: 1, criterion: "TR", do: "State your position early", why: "It anchors the essay." }],
    flags: [],
    feedback_status: "complete",
    vs_previous_attempt: null,
    ungraded_reason: null,
    prompt_versions: null,
    disclaimer: "Estimated band, not an official IELTS score.",
  };
}

async function mockGateway(page: Page) {
  const state = { status: "none", draft: "", startedAt: 0, elapsed: 0, saves: 0, grading: 0 };
  const view = () => ({
    id: "att1",
    prompt_id: PROMPTS[0].id,
    prompt: PROMPTS[0].text,
    status: state.status,
    allowed_s: 2400,
    remaining_s: state.status === "in_progress" ? Math.max(0, 2400 - state.elapsed) : null,
    started_at: state.startedAt || null,
    submitted_at: null,
    late: false,
    draft: state.draft,
    word_count: state.draft.split(/\s+/).filter(Boolean).length,
    error: null,
    retryable: false,
    overall: state.status === "graded" ? 6 : null,
    disclaimer: "Estimated band, not an official IELTS score.",
  });
  const json = (body: unknown, status = 200) => ({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });

  // a regex, not a glob: "**/ielts/**" would also swallow the app's own /src/ielts/ modules
  await page.route(/^https?:\/\/[^/]+\/ielts\//, async (route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname;
    const method = req.method();
    if (path === "/ielts/prompts") return route.fulfill(json(PROMPTS));
    if (path === "/ielts/attempts" && method === "GET") return route.fulfill(json([]));
    if (path === "/ielts/attempts" && method === "POST") {
      state.status = "created";
      return route.fulfill(json(view(), 201));
    }
    if (path === "/ielts/attempts/att1" && method === "GET") {
      if (state.status === "submitted" && ++state.grading >= 2) state.status = "graded";
      return route.fulfill(json(view()));
    }
    if (path.endsWith("/start")) {
      state.status = "in_progress";
      state.startedAt = Date.now();
      return route.fulfill(json(view()));
    }
    if (path.endsWith("/draft")) {
      state.draft = req.postDataJSON().text;
      state.saves += 1;
      return route.fulfill(json(view()));
    }
    if (path.endsWith("/submit")) {
      state.draft = req.postDataJSON().text;
      state.status = "submitted";
      return route.fulfill(json(view()));
    }
    if (path.endsWith("/report")) return route.fulfill(json(reportFor(state.draft)));
    return route.fulfill(json({ detail: "not found" }, 404));
  });
  return state;
}

test("start, autosave, refresh-resume, submit and read the report", async ({ page }) => {
  await page.clock.install();
  const server = await mockGateway(page);

  await page.goto("/ielts-mock");
  await expect(page.getByRole("heading", { name: "Choose a Writing Task 2 question" })).toBeVisible();
  await expect(page.getByRole("button", { name: /Practice question/ })).toHaveCount(8);

  await page.getByRole("button", { name: /Practice question 1/ }).click();
  await expect(page.getByText("Start the 40 minutes")).toBeVisible();
  expect(server.status).toBe("created"); // the clock has not started

  await page.getByRole("button", { name: "Start the 40 minutes" }).click();
  await expect(page.getByRole("timer")).toContainText("40:00");

  await page.getByLabel("Your essay").fill(ESSAY);
  await expect(page.getByLabel("Word count")).toContainText("17 words");

  await page.clock.runFor(11_000);
  await expect.poll(() => server.draft).toBe(ESSAY);
  await expect(page.getByText(/^Saved /)).toBeVisible();

  await page.reload();
  await expect(page.getByLabel("Your essay")).toHaveValue(ESSAY);
  await expect(page.getByRole("timer")).toContainText(/(39|40):\d\d/);

  await page.getByRole("button", { name: "Submit for marking" }).click();
  await expect(page.getByText(/well under the 250 minimum/)).toBeVisible();
  await page.getByRole("button", { name: "Yes, submit" }).click();
  await expect(page.getByRole("heading", { name: "Marking your essay" })).toBeVisible();

  await page.clock.runFor(5_000);
  const hero = page.getByRole("region", { name: "Estimated band" });
  await expect(hero).toContainText("6.0");
  await expect(hero).toContainText("Uncalibrated estimate");
  await expect(hero).toContainText("Estimated band, not an official IELTS score.");
  await expect(page.getByRole("article", { name: "Task Response" })).toContainText("Add one example per paragraph");
  await expect(page.getByRole("region", { name: "Your essay, marked" }).locator("mark")).toHaveText("students has");
  await expect(page.getByRole("region", { name: "Fix these first" })).toContainText("State your position early");
});

test("time running out submits the essay by itself", async ({ page }) => {
  await page.clock.install();
  const server = await mockGateway(page);
  await page.goto("/ielts-mock");
  await page.getByRole("button", { name: /Practice question 1/ }).click();
  await page.getByRole("button", { name: "Start the 40 minutes" }).click();
  await page.getByLabel("Your essay").fill(ESSAY);
  await page.clock.runFor(11_000); // an autosave; the server says 40 minutes left
  server.elapsed = 2399; // the server's clock has moved on: one second left
  await page.getByLabel("Your essay").fill(`${ESSAY} More words.`); // the next autosave syncs the clock
  await page.clock.runFor(11_000);
  await expect.poll(() => server.status).not.toBe("in_progress");
  expect(server.draft).toBe(`${ESSAY} More words.`);
});

test("a long prompt in Earlier attempts is truncated, not scrolled sideways", async ({ page }) => {
  await mockGateway(page);
  const earlier = {
    id: "att0",
    prompt_id: PROMPTS[1].id,
    prompt: "Some people think that governments should spend money on public transport rather than roads. ".repeat(4),
    status: "graded",
    overall: 6.5,
  };
  await page.route(/\/ielts\/attempts$/, (route) =>
    route.request().method() === "GET"
      ? route.fulfill({ contentType: "application/json", body: JSON.stringify([earlier]) })
      : route.fallback(),
  );

  for (const width of [1200, 390]) {
    await page.setViewportSize({ width, height: 800 });
    await page.goto("/ielts-mock");
    await expect(page.getByRole("region", { name: "Earlier attempts" })).toContainText("Estimated band 6.5");
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow, `page scrolls sideways at ${width} px`).toBeLessThanOrEqual(0);
  }
});
