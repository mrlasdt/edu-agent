import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ApiError } from "../../request";
import { WritingModule } from "../WritingModule";
import type { Report } from "../types";
import type {
  WritingModuleAttempt,
  WritingModuleInfo,
  WritingModuleReport,
} from "../writingModuleTypes";
import { ESSAY, report } from "./fixtures";

const api = vi.hoisted(() => ({
  writingModuleInfo: vi.fn(),
  createWritingModule: vi.fn(),
  getWritingModule: vi.fn(),
  startWritingModule: vi.fn(),
  saveWritingModuleDraft: vi.fn(),
  submitWritingModule: vi.fn(),
  regradeWritingModule: vi.fn(),
  getWritingModuleReport: vi.fn(),
}));
vi.mock("../writingModuleApi", () => api);

const SOURCES = [
  "https://ielts.org/take-a-test/test-types/ielts-general-training-test/ielts-general-training-format-writing",
  "https://ielts.org/take-a-test/your-results/ielts-scoring-in-detail",
];
const NOTE = "Task 2 counts twice as much as Task 1 in the Writing band.";

const info: WritingModuleInfo = {
  allowed_s: 3600, t1_minutes: 20, t2_minutes: 40, t1_min_words: 150, t2_min_words: 250,
  weights_note: NOTE, source_urls: SOURCES,
};

const task = (key: "t1" | "t2", text = "", band: number | null = null) => ({
  prompt_id: `p-${key}`,
  prompt: key === "t1" ? "Write a letter to your landlord." : "Discuss both views.",
  text, word_count: text.split(/\s+/).filter(Boolean).length, band,
});

const attempt = (over: Partial<WritingModuleAttempt> = {}): WritingModuleAttempt => ({
  id: "m1", status: "in_progress", allowed_s: 3600, remaining_s: 3600, started_at: 1,
  submitted_at: null, late: false, t1: task("t1"), t2: task("t2"), error: null, retryable: false,
  writing_band: null, disclaimer: "Mock score: an estimate, not an official IELTS result.", ...over,
});

const letterReport = (): Report => {
  const base = report();
  return {
    ...base, task: "T1",
    criteria: base.criteria.map((c) => (c.id === "TR" ? { ...c, id: "TA", name: "Task Achievement" } : c)),
    estimated_band: base.estimated_band && { ...base.estimated_band, overall: 6.0 },
  };
};
const moduleReport = (over: Partial<WritingModuleReport> = {}): WritingModuleReport => ({
  status: "graded", module: "ielts_writing", t1: letterReport(),
  t2: report({ estimated_band: { ...report().estimated_band!, overall: 7.0 } }),
  writing_band: 6.5, weights_note: NOTE, weights_source_urls: SOURCES,
  timing: { allowed_s: 3600, used_s: 2400, late: false },
  disclaimer: "Mock score: an estimate, not an official IELTS result.", ...over,
});

const settle = () => act(async () => {});

describe("WritingModule", () => {
  beforeEach(() => {
    window.localStorage.clear();
    Object.values(api).forEach((f) => f.mockReset());
    api.writingModuleInfo.mockResolvedValue(info);
  });

  it("introduces the 20 + 40 minutes, the weighting and its sources", async () => {
    render(<WritingModule />);
    expect(await screen.findByText(/One 60:00 clock covers both tasks/)).toBeInTheDocument();
    expect(screen.getByText(NOTE)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "ielts-scoring-in-detail" })).toHaveAttribute("href", SOURCES[1]);
    expect(screen.getByRole("link", { name: "Writing module" })).toHaveAttribute("aria-current", "page");
  });

  it("shows a short sample target, not the real 150/250 minimums, on a sample attempt", async () => {
    window.localStorage.setItem("ielts.writingmodule.attempt", "m1");
    api.getWritingModule.mockResolvedValue(attempt({ sample: true, allowed_s: 240, remaining_s: 240 }));
    render(<WritingModule />);
    expect(await screen.findByText(/Sample target: about 50 words/)).toBeInTheDocument();
    expect(screen.getByLabelText("Word count")).toHaveTextContent("Letter 0/50 - Essay 0/50 words");
    expect(screen.queryByText(/at least 150/)).not.toBeInTheDocument();
  });

  it("writes a letter then an essay on one clock, submits both, and shows one combined band", async () => {
    api.createWritingModule.mockResolvedValue(attempt({ status: "created", remaining_s: null }));
    api.startWritingModule.mockResolvedValue(attempt());
    api.submitWritingModule.mockResolvedValue(attempt({ status: "grading", remaining_s: null }));
    api.getWritingModule.mockResolvedValue(attempt({ status: "graded", writing_band: 6.5 }));
    api.getWritingModuleReport.mockResolvedValue(moduleReport());
    render(<WritingModule pollMs={20} />);
    await userEvent.click(await screen.findByRole("button", { name: "Start the 60 minute clock" }));
    expect(await screen.findByText("Write a letter to your landlord.")).toBeInTheDocument();
    expect(screen.getByRole("timer")).toHaveTextContent("60:00");

    fireEvent.change(screen.getByLabelText("Your letter"), { target: { value: "Dear Sir, hello there" } });
    await userEvent.click(screen.getByRole("tab", { name: "Task 2 - Essay" }));
    expect(screen.getByText("Discuss both views.")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Your essay"), { target: { value: ESSAY } });
    expect(screen.getByLabelText("Word count")).toHaveTextContent(/Letter 4\/150/);

    await userEvent.click(screen.getByRole("button", { name: "Submit both tasks for marking" }));
    expect(screen.getByText(/Under the minimum/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Yes, submit" }));
    await settle();
    expect(api.submitWritingModule).toHaveBeenCalledWith("m1", { t1: "Dear Sir, hello there", t2: ESSAY });

    expect(await screen.findByLabelText("Mock Writing band 6.5")).toBeInTheDocument();
    expect(screen.getByRole("article", { name: "Task 1 report" })).toBeInTheDocument();
    const t2 = screen.getByRole("article", { name: "Task 2 report" });
    expect(within(t2).getByLabelText("Mock overall score 7.0")).toBeInTheDocument();
    expect(screen.getByText(NOTE, { selector: "p.ielts-hint" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: SOURCES[0] })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /discuss this report/i })).toHaveAttribute("href", "/?attempt=m1");
  });

  it("says plainly that there is no combined band when one task could not be marked", async () => {
    window.localStorage.setItem("ielts.writingmodule.attempt", "m1");
    api.getWritingModule.mockResolvedValue(attempt({ status: "ungraded" }));
    api.getWritingModuleReport.mockResolvedValue(
      moduleReport({
        status: "partial", writing_band: null,
        t1: { ...letterReport(), status: "ungraded", estimated_band: null, criteria: [],
          ungraded_reason: "The letter is empty." },
      })
    );
    render(<WritingModule />);
    expect(await screen.findByText("No combined Writing band")).toBeInTheDocument();
    expect(screen.getByText("The letter is empty.")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /discuss this report/i })).not.toBeInTheDocument();
  });

  it("offers a retry when marking failed and goes back to marking", async () => {
    window.localStorage.setItem("ielts.writingmodule.attempt", "m1");
    api.getWritingModule.mockResolvedValue(attempt({ status: "ungraded", error: "model down", retryable: true }));
    api.getWritingModuleReport.mockRejectedValue(new ApiError(409, "none"));
    api.regradeWritingModule.mockResolvedValue(attempt({ status: "grading" }));
    render(<WritingModule pollMs={1000} />);
    await userEvent.click(await screen.findByRole("button", { name: "Try marking again" }));
    expect(api.regradeWritingModule).toHaveBeenCalledWith("m1");
    expect(await screen.findByText("Marking your letter and essay")).toBeInTheDocument();
  });

  it("resumes an attempt in progress with both texts and the server clock", async () => {
    window.localStorage.setItem("ielts.writingmodule.attempt", "m1");
    api.getWritingModule.mockResolvedValue(
      attempt({ remaining_s: 1800, t1: task("t1", "Dear Sir"), t2: task("t2", "Some essay") })
    );
    render(<WritingModule />);
    expect(await screen.findByRole("timer")).toHaveTextContent("30:00");
    expect(screen.getByLabelText("Your letter")).toHaveValue("Dear Sir");
  });
});
