import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FullMock } from "../FullMock";
import type { MockReport, ModuleStatus } from "../mockApi";

const api = vi.hoisted(() => ({ createMock: vi.fn(), getMock: vi.fn(), startMockModule: vi.fn() }));
const reading = vi.hoisted(() => ({ getReading: vi.fn(), startReading: vi.fn() }));
const listening = vi.hoisted(() => ({ getListening: vi.fn(), startListening: vi.fn() }));
const writing = vi.hoisted(() => ({ getWritingModule: vi.fn(),
  startWritingModule: vi.fn(),
  regradeWritingModule: vi.fn(),
}));
const speaking = vi.hoisted(() => ({
  getSpeaking: vi.fn(),
  startSpeaking: vi.fn(),
  regradeSpeaking: vi.fn(),
}));
vi.mock("../speakingApi", () => speaking);
vi.mock("../SpeakingSit", () => ({
  SpeakingSit: ({ onSubmitted }: { onSubmitted: (a: unknown) => void }) => (
    <button onClick={() => onSubmitted({})}>Submit speaking</button>
  ),
}));
vi.mock("../mockApi", () => api);
vi.mock("../readingApi", () => reading);
vi.mock("../listeningApi", () => listening);
vi.mock("../writingModuleApi", () => writing);
// The sit screens have their own tests: here each is a button that "submits".
vi.mock("../ListeningSit", () => ({
  ListeningSit: ({ onSubmitted }: { onSubmitted: (a: unknown) => void }) => (
    <button onClick={() => onSubmitted({})}>Submit listening</button>
  ),
}));
vi.mock("../ReadingSit", () => ({
  ReadingSit: ({ onSubmitted }: { onSubmitted: (a: unknown) => void }) => (
    <button onClick={() => onSubmitted({})}>Submit reading</button>
  ),
}));
vi.mock("../WritingModuleSit", () => ({
  WritingModuleSit: ({ onSubmitted }: { onSubmitted: (a: unknown) => void }) => (
    <button onClick={() => onSubmitted({})}>Submit writing</button>
  ),
}));

const status = (key: ModuleStatus["key"], over: Partial<ModuleStatus> = {}): ModuleStatus => ({
  key, label: key[0].toUpperCase() + key.slice(1), state: "not_started", band: null, attempt_id: null,
  sittable: key !== "speaking", retryable: false, detail: "", ...over,
});

const mock = (over: {
  listening?: Partial<ModuleStatus>;
  reading?: Partial<ModuleStatus>;
  writing?: Partial<ModuleStatus>;
  report?: Partial<MockReport>;
} = {}): MockReport => ({
  id: "m1", created_at: 1,
  modules: [
    status("listening", over.listening),
    status("reading", over.reading),
    status("writing", over.writing),
    status("speaking", { state: "not_assessed", sittable: false }),
  ],
  overall: null, partial: true, overall_note: "No skill has a band yet, so there is no overall band.",
  overall_source_url: "https://ielts.org/take-a-test/your-results/ielts-scoring-in-detail",
  next_module: { key: "listening", label: "Listening", reason: "Sit Listening next: it is the next module." },
  disclaimer: "Mock score: an estimate, not an official IELTS result.",
  ...over.report,
});

const settle = () => act(async () => {});

beforeEach(() => {
  window.localStorage.clear();
  Object.values({ ...api, ...listening, ...reading, ...writing, ...speaking }).forEach((f) => f.mockReset());
});
afterEach(() => window.localStorage.clear());

describe("FullMock", () => {
  it("starts a full mock straight into Listening, then lists the skills", async () => {
    const started = mock({ listening: { attempt_id: "l1", state: "in_progress" } });
    api.createMock.mockResolvedValue(mock());
    api.startMockModule.mockResolvedValue(started);
    listening.getListening.mockResolvedValue({ id: "l1", status: "created" });
    listening.startListening.mockResolvedValue({ id: "l1", status: "in_progress" });
    const done = mock({
      listening: { attempt_id: "l1", state: "graded", band: 7.0, detail: "30 of 40 correct" },
      report: { overall: 7.0, next_module: { key: "reading", label: "Reading", reason: "Sit Reading next." } },
    });
    api.getMock.mockResolvedValue(done);

    render(<FullMock />);
    await settle();
    await userEvent.click(screen.getByRole("button", { name: "Start full mock" }));
    await settle();
    expect(api.startMockModule).toHaveBeenCalledWith("m1", "listening");
    expect(listening.startListening).toHaveBeenCalledWith("l1");

    await userEvent.click(screen.getByRole("button", { name: "Submit listening" }));
    await settle();
    const skills = screen.getByRole("list", { name: "Skill bands" });
    expect(skills).toHaveTextContent("Listening");
    expect(skills).toHaveTextContent("7.0");
    expect(skills).toHaveTextContent("30 of 40 correct");
    expect(screen.getAllByText("Not assessed").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByRole("button", { name: "Start Reading" })).toBeEnabled();
    expect(screen.getByText("Next: Reading")).toBeInTheDocument();
    expect(window.localStorage.getItem("ielts.fullmock.id")).toBe("m1");
  });

  it("opens Reading from the overview and returns to the overview when it is submitted", async () => {
    window.localStorage.setItem("ielts.fullmock.id", "m1");
    const before = mock({ listening: { attempt_id: "l1", state: "graded", band: 7.0 } });
    const after = mock({
      listening: { attempt_id: "l1", state: "graded", band: 7.0 },
      reading: { attempt_id: "r1", state: "graded", band: 6.0 },
      report: { overall: 6.5, next_module: { key: "writing", label: "Writing", reason: "Sit Writing next." } },
    });
    api.getMock.mockResolvedValueOnce(before).mockResolvedValue(after);
    api.startMockModule.mockResolvedValue(
      mock({
        listening: { attempt_id: "l1", state: "graded", band: 7.0 },
        reading: { attempt_id: "r1", state: "in_progress" },
      })
    );
    reading.getReading.mockResolvedValue({ id: "r1", status: "created" });
    reading.startReading.mockResolvedValue({ id: "r1", status: "in_progress" });
    render(<FullMock />);
    await settle();
    await userEvent.click(screen.getByRole("button", { name: "Start Reading" }));
    await settle();
    expect(api.startMockModule).toHaveBeenCalledWith("m1", "reading");
    await userEvent.click(screen.getByRole("button", { name: "Submit reading" }));
    await settle();
    expect(screen.getByLabelText("Mock overall band 6.5")).toBeInTheDocument();
  });

  it("shows the partial overall with the explanation, the source and the coach link", async () => {
    window.localStorage.setItem("ielts.fullmock.id", "m1");
    api.getMock.mockResolvedValue(
      mock({
        reading: { attempt_id: "r1", state: "graded", band: 6.0 },
        writing: { attempt_id: "w1", state: "graded", band: 6.5 },
        report: {
          overall: 6.5, partial: true,
          overall_note: "Partial overall: Speaking not assessed, so this is the mean of 3 of 4 skills.",
          next_module: { key: "reading", label: "Reading", reason: "Reading is your lowest assessed skill." },
        },
      })
    );
    render(<FullMock />);
    await settle();
    expect(screen.getByText("Partial overall band (mock score)")).toBeInTheDocument();
    expect(screen.getByLabelText("Mock overall band 6.5")).toBeInTheDocument();
    expect(screen.getByText(/Speaking not assessed/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "ielts.org" })).toHaveAttribute(
      "href", "https://ielts.org/take-a-test/your-results/ielts-scoring-in-detail"
    );
    expect(screen.getByRole("link", { name: "Discuss with the coach" })).toHaveAttribute(
      "href", "/?attempt=m1&subject=mock"
    );
    expect(screen.getByText("Next: Reading")).toBeInTheDocument();
  });

  it("offers no coach link before any skill has a band", async () => {
    window.localStorage.setItem("ielts.fullmock.id", "m1");
    api.getMock.mockResolvedValue(mock());
    render(<FullMock />);
    await settle();
    expect(screen.queryByRole("link", { name: "Discuss with the coach" })).toBeNull();
  });

  it("forgets a mock the server no longer knows and shows the intro", async () => {
    window.localStorage.setItem("ielts.fullmock.id", "gone");
    const { ApiError } = await import("../../request");
    api.getMock.mockRejectedValue(new ApiError(404, "unknown mock"));
    render(<FullMock />);
    await settle();
    expect(screen.getByRole("button", { name: "Start full mock" })).toBeInTheDocument();
    expect(window.localStorage.getItem("ielts.fullmock.id")).toBeNull();
  });

  it("polls while Writing is being marked and shows the band when it lands", async () => {
    vi.useFakeTimers();
    try {
      window.localStorage.setItem("ielts.fullmock.id", "m1");
      const marking = mock({
        reading: { attempt_id: "r1", state: "graded", band: 6.0 },
        writing: { attempt_id: "w1", state: "marking" },
        report: { overall: 6.0, next_module: null },
      });
      const marked = mock({
        reading: { attempt_id: "r1", state: "graded", band: 6.0 },
        writing: { attempt_id: "w1", state: "graded", band: 6.5 },
        report: { overall: 6.5 },
      });
      api.getMock.mockResolvedValueOnce(marking).mockResolvedValue(marked);
      render(<FullMock pollMs={1000} />);
      await act(async () => {});
      expect(screen.getByText(/Marking your Writing/)).toBeInTheDocument();
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1100);
      });
      expect(screen.getByLabelText("Mock overall band 6.5")).toBeInTheDocument();
      expect(screen.queryByText(/Marking your Writing/)).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it("retries a failed Writing marking on the mock's own attempt, then polls for the band", async () => {
    window.localStorage.setItem("ielts.fullmock.id", "m1");
    const failed = mock({
      reading: { attempt_id: "r1", state: "graded", band: 6.0 },
      writing: { attempt_id: "w1", state: "ungraded", retryable: true, detail: "Marking did not finish." },
      report: { overall: 6.0, next_module: null },
    });
    const again = mock({
      reading: { attempt_id: "r1", state: "graded", band: 6.0 },
      writing: { attempt_id: "w1", state: "marking" },
      report: { overall: 6.0, next_module: null },
    });
    api.getMock.mockResolvedValueOnce(failed).mockResolvedValue(again);
    writing.regradeWritingModule.mockResolvedValue({ id: "w1", status: "grading" });
    render(<FullMock />);
    await settle();
    await userEvent.click(screen.getByRole("button", { name: "Retry marking" }));
    await settle();
    expect(writing.regradeWritingModule).toHaveBeenCalledWith("w1");
    expect(screen.queryByRole("button", { name: "Retry marking" })).toBeNull();
    expect(screen.getByText(/Marking your Writing/)).toBeInTheDocument();
  });

  it("offers no retry when the failure cannot be retried", async () => {
    window.localStorage.setItem("ielts.fullmock.id", "m1");
    api.getMock.mockResolvedValue(
      mock({
        reading: { attempt_id: "r1", state: "graded", band: 6.0 },
        writing: { attempt_id: "w1", state: "ungraded", retryable: false },
      })
    );
    render(<FullMock />);
    await settle();
    expect(screen.queryByRole("button", { name: "Retry marking" })).toBeNull();
  });

  const afterWriting = (speakingOver: Partial<ModuleStatus> = {}) =>
    mock({
      listening: { attempt_id: "l1", state: "graded", band: 7.0 },
      reading: { attempt_id: "r1", state: "graded", band: 6.5 },
      writing: { attempt_id: "w1", state: "graded", band: 6.0 },
      report: {
        modules: [
          status("listening", { attempt_id: "l1", state: "graded", band: 7.0 }),
          status("reading", { attempt_id: "r1", state: "graded", band: 6.5 }),
          status("writing", { attempt_id: "w1", state: "graded", band: 6.0 }),
          status("speaking", speakingOver),
        ],
      },
    });

  it("offers Speaking's quick sample and starts it with the sample flag", async () => {
    window.localStorage.setItem("ielts.fullmock.id", "m1");
    api.getMock.mockResolvedValue(afterWriting({ sittable: true }));
    api.startMockModule.mockResolvedValue(
      afterWriting({ sittable: true, state: "in_progress", attempt_id: "s1" })
    );
    speaking.getSpeaking.mockResolvedValue({ id: "s1", status: "in_progress" });
    render(<FullMock />);
    await settle();
    await userEvent.click(screen.getByRole("button", { name: /Quick sample/ }));
    await settle();
    expect(api.startMockModule).toHaveBeenCalledWith("m1", "speaking", true);
    expect(screen.getByRole("button", { name: "Submit speaking" })).toBeInTheDocument();
  });

  it("shows the Speaking band as partial", async () => {
    window.localStorage.setItem("ielts.fullmock.id", "m1");
    api.getMock.mockResolvedValue(
      afterWriting({ state: "partial", band: 6.5, band_partial: true, attempt_id: "s1" })
    );
    render(<FullMock />);
    await settle();
    const skills = screen.getByRole("list", { name: "Skill bands" });
    expect(skills).toHaveTextContent("6.5");
    expect(skills).toHaveTextContent("partial");
  });

  it("labels a sample-based Speaking band and the overall as not a real mock score", async () => {
    window.localStorage.setItem("ielts.fullmock.id", "m1");
    const sampled = afterWriting({
      state: "graded", band: 6.5, band_partial: true, sample: true, attempt_id: "s1",
    });
    api.getMock.mockResolvedValue({
      ...sampled,
      overall: 6.5,
      sample: true,
      overall_note: "Sample-based: the Speaking band came from the quick sample.",
    });
    render(<FullMock />);
    await settle();
    expect(screen.getByText("Sample-based overall band (not a real mock score)")).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "Skill bands" })).toHaveTextContent("sample");
  });

  it("retries a failed Speaking marking through the Speaking API", async () => {
    window.localStorage.setItem("ielts.fullmock.id", "m1");
    api.getMock.mockResolvedValue(
      afterWriting({ state: "ungraded", attempt_id: "s1", retryable: true })
    );
    speaking.regradeSpeaking.mockResolvedValue({ id: "s1", status: "grading" });
    render(<FullMock />);
    await settle();
    await userEvent.click(screen.getByRole("button", { name: "Retry marking" }));
    await settle();
    expect(speaking.regradeSpeaking).toHaveBeenCalledWith("s1");
    expect(writing.regradeWritingModule).not.toHaveBeenCalled();
  });
});
