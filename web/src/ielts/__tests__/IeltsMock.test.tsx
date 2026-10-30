import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { IeltsMock } from "../IeltsMock";
import { ApiError } from "../api";
import { attempt, report } from "./fixtures";

const api = vi.hoisted(() => ({
  getBudget: vi.fn(),
  listPrompts: vi.fn(),
  listAttempts: vi.fn(),
  createAttempt: vi.fn(),
  getAttempt: vi.fn(),
  startAttempt: vi.fn(),
  saveDraft: vi.fn(),
  submitAttempt: vi.fn(),
  regradeAttempt: vi.fn(),
  getReport: vi.fn(),
  saveDraftOnExit: vi.fn(),
}));
vi.mock("../api", async (orig) => ({ ...(await orig<typeof import("../api")>()), ...api }));

const prompts = Array.from({ length: 8 }, (_, i) => ({
  id: `p${i + 1}`,
  essay_type: "opinion",
  essay_type_label: i % 2 ? "Discuss both views" : "Opinion",
  text: `Question number ${i + 1}?`,
}));
const settle = () => act(async () => {});

describe("IeltsMock", () => {
  beforeEach(() => {
    window.localStorage.clear();
    api.getBudget.mockResolvedValue({
      cap_usd: 5,
      spent_usd: 0.2,
      remaining_usd: 4.8,
      exhausted: false,
      message: null,
    });
    api.listPrompts.mockResolvedValue(prompts);
    api.listAttempts.mockResolvedValue([]);
    api.saveDraft.mockResolvedValue(attempt());
  });
  afterEach(() => vi.clearAllMocks());

  it("frames every earlier band as an estimate under the fixed disclaimer", async () => {
    api.listAttempts.mockResolvedValue([
      attempt({ id: "a1", status: "graded", overall: 6.5 }),
      attempt({ id: "a2", status: "graded", overall: 5.0 }),
    ]);
    render(<IeltsMock />);
    await settle();
    const history = screen.getByRole("region", { name: "Earlier attempts" });
    expect(history).toHaveTextContent("Mock score: an estimate, not an official IELTS result.");
    expect(within(history).getAllByText(/^Mock score \d\.\d$/).map((e) => e.textContent)).toEqual([
      "Mock score 6.5",
      "Mock score 5.0",
    ]);
  });

  it("shows the AI spend against the cap", async () => {
    render(<IeltsMock />);
    await settle();
    expect(screen.getByLabelText("AI spend")).toHaveTextContent("$0.20 of the $5.00 cap");
  });

  it("refuses new mocks with the server's message once the spend cap is used up", async () => {
    const message = "The AI spend cap of $5.00 is used up ($5.00 spent), so no new mock can be marked.";
    api.getBudget.mockResolvedValue({
      cap_usd: 5,
      spent_usd: 5,
      remaining_usd: 0,
      exhausted: true,
      message,
    });
    render(<IeltsMock />);
    await settle();
    expect(screen.getByRole("alert")).toHaveTextContent(message);
    for (const card of screen.getAllByRole("button", { name: /Question number/ })) {
      expect(card).toBeDisabled();
    }
  });

  it("offers the eight questions with their essay types", async () => {
    render(<IeltsMock />);
    await settle();
    expect(screen.getAllByRole("button", { name: /Question number/ })).toHaveLength(8);
    expect(screen.getAllByText("Opinion").length).toBeGreaterThan(0);
  });

  it("picking a question creates an attempt and waits for Start; the clock starts on Start", async () => {
    api.createAttempt.mockResolvedValue(attempt({ status: "created", remaining_s: null, started_at: null }));
    api.startAttempt.mockResolvedValue(attempt({ remaining_s: 2400 }));
    render(<IeltsMock />);
    await settle();
    fireEvent.click(screen.getByRole("button", { name: /Question number 3/ }));
    await settle();
    expect(api.createAttempt).toHaveBeenCalledWith("p3");
    expect(api.startAttempt).not.toHaveBeenCalled();
    expect(window.localStorage.getItem("ielts.attempt")).toBe("a1");
    fireEvent.click(screen.getByRole("button", { name: "Start the 40 minutes" }));
    await settle();
    expect(screen.getByRole("timer")).toHaveTextContent("40:00");
  });

  it("resumes an attempt in progress after a refresh, with the saved draft and the remaining time", async () => {
    window.localStorage.setItem("ielts.attempt", "a1");
    api.getAttempt.mockResolvedValue(attempt({ remaining_s: 1500, draft: "Words from before" }));
    render(<IeltsMock />);
    await settle();
    expect(screen.getByLabelText("Your essay")).toHaveValue("Words from before");
    expect(screen.getByRole("timer")).toHaveTextContent("25:00");
  });

  it("prefers a local copy that the server never confirmed", async () => {
    window.localStorage.setItem("ielts.attempt", "a1");
    window.localStorage.setItem("ielts.draft.a1", JSON.stringify({ text: "Newer local words", saved: false }));
    api.getAttempt.mockResolvedValue(attempt({ draft: "Older server words" }));
    render(<IeltsMock />);
    await settle();
    expect(screen.getByLabelText("Your essay")).toHaveValue("Newer local words");
  });

  it("shows the report for a marked attempt", async () => {
    window.localStorage.setItem("ielts.attempt", "a1");
    api.getAttempt.mockResolvedValue(attempt({ status: "graded", draft: "x", overall: 6 }));
    api.getReport.mockResolvedValue(report());
    render(<IeltsMock />);
    await settle();
    expect(screen.getByLabelText("Mock overall score 6.0")).toBeInTheDocument();
  });

  it("polls while marking and then shows the report", async () => {
    vi.useFakeTimers();
    try {
      window.localStorage.setItem("ielts.attempt", "a1");
      api.getAttempt
        .mockResolvedValueOnce(attempt({ status: "grading" }))
        .mockResolvedValue(attempt({ status: "graded", draft: "x" }));
      api.getReport.mockResolvedValue(report());
      render(<IeltsMock pollMs={1000} />);
      await settle();
      expect(screen.getByText("Marking your essay")).toBeInTheDocument();
      await act(async () => void vi.advanceTimersByTime(1100));
      expect(screen.getByLabelText("Mock overall score 6.0")).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("offers a retry when marking failed and the model was unavailable", async () => {
    window.localStorage.setItem("ielts.attempt", "a1");
    api.getAttempt.mockResolvedValue(
      attempt({ status: "ungraded", error: "The marking model is unavailable.", retryable: true, draft: "x" })
    );
    api.getReport.mockRejectedValue(new ApiError(409, "attempt is ungraded; no report yet"));
    api.regradeAttempt.mockResolvedValue(attempt({ status: "grading" }));
    render(<IeltsMock />);
    await settle();
    expect(screen.getByRole("alert")).toHaveTextContent("could not be marked");
    fireEvent.click(screen.getByRole("button", { name: "Try marking again" }));
    await settle();
    expect(api.regradeAttempt).toHaveBeenCalledWith("a1");
    expect(screen.getByText("Marking your essay")).toBeInTheDocument();
  });

  it("forgets an attempt the server no longer knows and shows the questions", async () => {
    window.localStorage.setItem("ielts.attempt", "gone");
    api.getAttempt.mockRejectedValue(new ApiError(404, "attempt not found"));
    render(<IeltsMock />);
    await settle();
    expect(screen.getAllByRole("button", { name: /Question number/ })).toHaveLength(8);
    expect(window.localStorage.getItem("ielts.attempt")).toBeNull();
  });

  it("tells the candidate when the server cannot be reached", async () => {
    api.listPrompts.mockRejectedValue(new ApiError(0, "Cannot reach the server"));
    render(<IeltsMock />);
    await settle();
    expect(screen.getByRole("alert")).toHaveTextContent("Cannot reach the server");
  });
});
