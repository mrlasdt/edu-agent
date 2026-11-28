import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ReadingMock } from "../ReadingMock";
import type { Passage, Question, ReadingAttempt, ReadingInfo, ReadingReport } from "../readingTypes";

const api = vi.hoisted(() => ({
  readingInfo: vi.fn(),
  createReading: vi.fn(),
  getReading: vi.fn(),
  startReading: vi.fn(),
  saveReadingAnswers: vi.fn(),
  submitReading: vi.fn(),
  getReadingReport: vi.fn(),
}));
vi.mock("../readingApi", () => api);

const info: ReadingInfo = {
  set_id: "reading-x",
  allowed_s: 3600,
  question_count: 3,
  task_types: {},
  timing_source_url: "https://ielts.org/timing",
  band_source_url: "https://idp.example/bands",
  band_note: "Half bands are interpolated between published anchors.",
  set_note: "This set has 3 questions, fewer than the 40 of a full section.",
  keys_note: "Answer keys are author-written and not human-checked.",
};

const paper: { passages: Passage[]; questions: Question[] } = {
  passages: [{ id: "p1", title: "The Lighthouse", paragraphs: ["First paragraph.", "Second paragraph."] }],
  questions: [
    {
      id: "q1", number: 1, passage_id: "p1", task_type: "tfng", task_label: "True / False / Not Given",
      instruction: "Choose one.", prompt: "The tower is old.", paragraph: null,
      choices: { TRUE: "True", FALSE: "False", "NOT GIVEN": "Not given" },
    },
    {
      id: "q2", number: 2, passage_id: "p1", task_type: "sentence_completion", task_label: "Sentence completion",
      instruction: "", prompt: "The keeper used a ____.", paragraph: null, choices: {},
    },
    {
      id: "q3", number: 3, passage_id: "p1", task_type: "matching_headings", task_label: "Matching headings",
      instruction: "", prompt: "Choose a heading.", paragraph: 2, choices: { A: "Origins", B: "Decline" },
    },
  ],
};

const attempt = (over: Partial<ReadingAttempt> = {}): ReadingAttempt => ({
  id: "a1", set_id: "reading-x", status: "in_progress", allowed_s: 3600, remaining_s: 3600,
  started_at: 1, submitted_at: null, late: false, answers: {}, answered: 0, band: null, raw: null,
  scaled_40: null, paper, ...over,
});

const report: ReadingReport = {
  set_id: "reading-x", raw: 1, total: 3, scaled_40: 13, band: 3.5,
  timing: { allowed_s: 3600, used_s: 125, late: false },
  by_type: [
    { task_type: "tfng", label: "True / False / Not Given", correct: 1, total: 1 },
    { task_type: "matching_headings", label: "Matching headings", correct: 0, total: 1 },
  ],
  questions: [
    {
      number: 1, id: "q1", passage: "The Lighthouse", task_type: "tfng", task_label: "True / False / Not Given",
      prompt: "The tower is old.", paragraph: null, answer: "TRUE", answer_label: "True", key: "TRUE",
      key_label: "True", correct: true, evidence: "It was built in 1820.",
    },
    {
      number: 3, id: "q3", passage: "The Lighthouse", task_type: "matching_headings", task_label: "Matching headings",
      prompt: "Choose a heading.", paragraph: 2, answer: "B", answer_label: "B: Decline", key: "A",
      key_label: "A: Origins", correct: false, evidence: "The second paragraph traces where it began.",
    },
  ],
  band_note: info.band_note, set_note: info.set_note, keys_note: info.keys_note,
  band_source_url: info.band_source_url, timing_source_url: info.timing_source_url,
  disclaimer: "An unofficial practice estimate.",
};

const settle = () => act(async () => {});

describe("ReadingMock", () => {
  beforeEach(() => {
    window.localStorage.clear();
    Object.values(api).forEach((f) => f.mockReset());
    api.readingInfo.mockResolvedValue(info);
  });
  afterEach(() => vi.useRealTimers());

  it("introduces the 60 minutes, the scaling and the author-key caveat before starting", async () => {
    render(<ReadingMock />);
    expect(await screen.findByText(/You have 60 minutes/)).toBeInTheDocument();
    expect(screen.getByText(info.set_note)).toBeInTheDocument();
    expect(screen.getByText(info.keys_note)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "IELTS.org" })).toHaveAttribute("href", info.timing_source_url);
    expect(screen.getByRole("link", { name: "Writing" })).toHaveAttribute("href", "/ielts-mock");
    expect(screen.getByRole("link", { name: "Reading" })).toHaveAttribute("aria-current", "page");
  });

  it("starts, takes answers, and submits to the report with evidence for the missed question", async () => {
    api.createReading.mockResolvedValue(attempt({ status: "created", paper: null, remaining_s: null }));
    api.startReading.mockResolvedValue(attempt());
    api.submitReading.mockResolvedValue(attempt({ status: "graded", paper: null, band: 3.5, raw: 1, scaled_40: 13 }));
    api.getReadingReport.mockResolvedValue(report);
    render(<ReadingMock />);
    await userEvent.click(await screen.findByRole("button", { name: "Start the 60 minutes" }));
    expect(await screen.findByText("The Lighthouse")).toBeInTheDocument();
    expect(screen.getByRole("timer")).toHaveTextContent("60:00");
    expect(screen.getByText("0 of 3 answered")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("radio", { name: "True" }));
    await userEvent.type(screen.getByLabelText("Answer to question 2"), "lamp");
    expect(screen.getByText("2 of 3 answered")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Submit answers" }));
    expect(screen.getByText(/Submit 2 of 3 answers/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Yes, submit" }));
    await settle();

    expect(api.submitReading).toHaveBeenCalledWith("a1", { q1: "TRUE", q2: "lamp" });
    expect(screen.getByText("3.5")).toBeInTheDocument();
    expect(screen.getByText(/Raw score 1 of 3\. Scaled to 13 of 40/)).toBeInTheDocument();
    expect(screen.getByText("The second paragraph traces where it began.")).toBeInTheDocument();
    expect(screen.queryByText("It was built in 1820.")).not.toBeInTheDocument(); // only misses show evidence
    expect(screen.getByRole("link", { name: /discuss this result/i })).toHaveAttribute(
      "href",
      "/?attempt=a1&subject=reading"
    );
    expect(screen.getByText(info.keys_note)).toBeInTheDocument();
  });

  it("the quick sample test creates a sample attempt on a short clock and labels it SAMPLE", async () => {
    api.createReading.mockResolvedValue(
      attempt({ status: "created", paper: null, remaining_s: null, allowed_s: 180, sample: true })
    );
    api.startReading.mockResolvedValue(attempt({ allowed_s: 180, remaining_s: 180, sample: true }));
    render(<ReadingMock />);
    await userEvent.click(await screen.findByRole("button", { name: "Quick sample test (3 minutes)" }));
    expect(api.createReading).toHaveBeenCalledWith(true);
    expect(await screen.findByText("SAMPLE")).toBeInTheDocument();
    expect(screen.getByRole("timer")).toHaveTextContent("03:00");
  });

  it("the real start is not a sample", async () => {
    api.createReading.mockResolvedValue(attempt({ status: "created", paper: null, remaining_s: null }));
    api.startReading.mockResolvedValue(attempt());
    render(<ReadingMock />);
    await userEvent.click(await screen.findByRole("button", { name: "Start the 60 minutes" }));
    expect(api.createReading).toHaveBeenCalledWith(false);
    expect(await screen.findByText("The Lighthouse")).toBeInTheDocument();
    expect(screen.queryByText("SAMPLE")).not.toBeInTheDocument();
  });

  it("resumes an attempt in progress with its saved answers", async () => {
    window.localStorage.setItem("ielts.reading.attempt", "a1");
    api.getReading.mockResolvedValue(attempt({ remaining_s: 1800, answers: { q3: "A" } }));
    render(<ReadingMock />);
    expect(await screen.findByRole("timer")).toHaveTextContent("30:00");
    expect(screen.getByRole("radio", { name: "A. Origins" })).toBeChecked();
  });

  it("autosaves changed answers on the interval", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    window.localStorage.setItem("ielts.reading.attempt", "a1");
    api.getReading.mockResolvedValue(attempt());
    api.saveReadingAnswers.mockResolvedValue(attempt({ remaining_s: 3590 }));
    render(<ReadingMock autosaveMs={1000} />);
    await settle();
    fireEvent.click(screen.getByRole("radio", { name: "False" }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1100);
    });
    expect(api.saveReadingAnswers).toHaveBeenCalledWith("a1", { q1: "FALSE" });
  });

  it("submits by itself when the clock runs out", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    window.localStorage.setItem("ielts.reading.attempt", "a1");
    api.getReading.mockResolvedValue(attempt({ remaining_s: 2, answers: { q1: "TRUE" } }));
    api.submitReading.mockResolvedValue(attempt({ status: "graded", paper: null }));
    api.getReadingReport.mockResolvedValue(report);
    render(<ReadingMock />);
    await settle();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2600);
    });
    expect(api.submitReading).toHaveBeenCalledTimes(1);
    expect(api.submitReading).toHaveBeenCalledWith("a1", { q1: "TRUE" });
  });

  it("shows the failure and a reload button when the gateway is unreachable", async () => {
    api.readingInfo.mockRejectedValue(new Error("Cannot reach the server"));
    render(<ReadingMock />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Cannot reach the server");
  });
});
