import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

// The quick sample full mock, end to end through the UI: Listening, Reading, then Writing, each
// labelled SAMPLE, then one sample overall band. The gateway is a small stateful fake.

const DISCLAIMER = "Mock score: an estimate, not an official IELTS result.";
const RULE = "https://ielts.org/take-a-test/your-results/ielts-scoring-in-detail";

type Key = "listening" | "reading" | "writing";
type Status = "none" | "created" | "in_progress" | "graded";

function mockGateway(page: Page) {
  const state: Record<Key, Status> = { listening: "none", reading: "none", writing: "none" };
  const created = { sampleQuery: "", writingBody: "" };
  const json = (body: unknown, status = 200) => ({
    status, contentType: "application/json", body: JSON.stringify(body),
  });

  const moduleRow = (key: Key, label: string, band: number) => {
    const s = state[key];
    return {
      key, label, band: s === "graded" ? band : null,
      attempt_id: s === "none" ? null : key,
      state: s === "none" ? "not_started" : s === "in_progress" || s === "created" ? "in_progress" : "graded",
      sittable: s !== "graded", retryable: false, detail: "",
    };
  };
  const mockView = () => {
    const done = state.listening === "graded" && state.reading === "graded" && state.writing === "graded";
    return {
      id: "m1", created_at: 1, sample: true, partial: true, disclaimer: DISCLAIMER,
      overall: done ? 7.5 : null, overall_source_url: RULE,
      overall_note: "SAMPLE: a few items on a short clock. Not a real mock score.",
      next_module: null,
      modules: [
        moduleRow("listening", "Listening", 8),
        moduleRow("reading", "Reading", 7),
        moduleRow("writing", "Writing", 7),
        { key: "speaking", label: "Speaking", band: null, attempt_id: null, state: "not_assessed",
          sittable: false, retryable: false, detail: "" },
      ],
    };
  };

  const attempt = (key: Key) => {
    const base = {
      id: key, set_id: "sample", status: state[key], allowed_s: 120,
      remaining_s: state[key] === "in_progress" ? 120 : null, started_at: 1, submitted_at: null,
      late: false, answers: {}, answered: 0, band: null, raw: null, scaled_40: null, sample: true,
    };
    if (key === "listening")
      return { ...base, paper: state[key] === "in_progress" ? {
        sections: [{ id: "s1", number: 1, title: "Section 1", context: "A call", audio_url: "/none.mp3", audio_s: 137 }],
        questions: [{ id: "q01", number: 1, section_id: "s1", task_type: "form", task_label: "Form",
          instruction: "One word", prompt: "Name", choices: {} }],
      } : null };
    if (key === "reading")
      return { ...base, paper: state[key] === "in_progress" ? {
        passages: [{ id: "p1", title: "City bees", paragraphs: ["Bees live in cities."] }],
        questions: [{ id: "r1", number: 1, passage_id: "p1", task_type: "tfng", task_label: "True / False",
          instruction: "Choose", prompt: "Bees live in cities.", paragraph: 1, choices: {} }],
      } : null };
    return {
      ...base, error: null, retryable: false, writing_band: null, disclaimer: DISCLAIMER,
      t1: { prompt_id: "t1", prompt: "Write a letter.", text: "", word_count: 0, band: null },
      t2: { prompt_id: "t2", prompt: "Give your view.", text: "", word_count: 0, band: null },
    };
  };

  return page.route(/^https?:\/\/[^/]+\/ielts\/(mock|listening|reading|writing)\b/, async (route) => {
    const req = route.request();
    const url = new URL(req.url());
    const path = url.pathname;
    const method = req.method();

    if (path === "/ielts/mock" && method === "POST") {
      created.sampleQuery = url.search;
      return route.fulfill(json(mockView(), 201));
    }
    const mod = path.match(/\/modules\/(listening|reading|writing)\/attempt$/);
    if (mod) {
      state[mod[1] as Key] = "created";
      return route.fulfill(json(mockView(), 201));
    }
    if (path === "/ielts/mock/m1") return route.fulfill(json(mockView()));

    const m = path.match(/^\/ielts\/(listening|reading|writing)\/attempts\/(\w+)(\/\w+)?$/);
    if (m) {
      const key = m[1] as Key;
      if (m[3] === "/start") state[key] = "in_progress";
      if (m[3] === "/submit") {
        if (key === "writing") created.writingBody = req.postData() ?? "";
        state[key] = "graded";
      }
      if (m[3] === "/draft" || m[3] === "/answers") return route.fulfill(json(attempt(key)));
      return route.fulfill(json(attempt(key)));
    }
    return route.fulfill(json({ detail: "not found" }, 404));
  }).then(() => created);
}

test("the quick sample full mock runs Listening, Reading and Writing to a sample overall band", async ({ page }) => {
  const seen = await mockGateway(page);

  await page.goto("/ielts-mock?module=mock");
  await page.getByRole("button", { name: "Quick sample full mock (about 5 minutes)" }).click();
  expect(seen.sampleQuery).toBe("?sample=true");

  // Listening
  await expect(page.getByText("SAMPLE", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Submit answers" }).click();
  await page.getByRole("button", { name: "Yes, submit" }).click();

  // Overview, then Reading
  await page.getByRole("button", { name: "Start Reading" }).click();
  await expect(page.getByText("SAMPLE", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Submit answers" }).click();
  await page.getByRole("button", { name: "Yes, submit" }).click();

  // Writing
  await page.getByRole("button", { name: "Start Writing" }).click();
  await expect(page.getByText("SAMPLE", { exact: true })).toBeVisible();
  const words = Array.from({ length: 60 }, () => "word").join(" ");
  await page.getByLabel("Your letter").fill(words);
  await page.getByRole("tab", { name: "Task 2 - Essay" }).click();
  await page.getByLabel("Your essay").fill(words);
  await page.getByRole("button", { name: "Submit both tasks for marking" }).click();
  await page.getByRole("button", { name: "Yes, submit" }).click();

  // The sample overall band, labelled so it never passes for a real score
  await expect(page.getByLabel("Mock overall band 7.5")).toBeVisible();
  await expect(page.getByText("Sample overall band (not a real mock score)")).toBeVisible();
  await expect(page.getByText("SAMPLE", { exact: true })).toBeVisible();
  expect(seen.writingBody).toContain("word word");
});

test("the real full mock start is unchanged and not a sample", async ({ page }) => {
  const seen = await mockGateway(page);
  await page.goto("/ielts-mock?module=mock");
  await page.getByRole("button", { name: "Start full mock" }).click();
  expect(seen.sampleQuery).toBe("");
});
