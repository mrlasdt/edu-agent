import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FullMock } from "../FullMock";
import type { MockReport, ModuleStatus } from "../mockApi";

const api = vi.hoisted(() => ({ createMock: vi.fn(), getMock: vi.fn(), startMockModule: vi.fn() }));
const reading = vi.hoisted(() => ({ getReading: vi.fn(), startReading: vi.fn() }));
const writing = vi.hoisted(() => ({ getWritingModule: vi.fn(), startWritingModule: vi.fn() }));
vi.mock("../mockApi", () => api);
vi.mock("../readingApi", () => reading);
vi.mock("../writingModuleApi", () => writing);
// The sit screens have their own tests: here each is a button that "submits".
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
  sittable: key === "reading" || key === "writing", detail: "", ...over,
});

const mock = (over: {
  reading?: Partial<ModuleStatus>;
  writing?: Partial<ModuleStatus>;
  report?: Partial<MockReport>;
} = {}): MockReport => ({
  id: "m1", created_at: 1,
  modules: [
    status("listening", { state: "not_assessed", sittable: false }),
    status("reading", over.reading),
    status("writing", over.writing),
    status("speaking", { state: "not_assessed", sittable: false }),
  ],
  overall: null, partial: true, overall_note: "No skill has a band yet, so there is no overall band.",
  overall_source_url: "https://ielts.org/take-a-test/your-results/ielts-scoring-in-detail",
  next_module: { key: "reading", label: "Reading", reason: "Sit Reading next: it is the next module." },
  disclaimer: "Mock score: an estimate, not an official IELTS result.",
  ...over.report,
});

const settle = () => act(async () => {});

beforeEach(() => {
  window.localStorage.clear();
  Object.values({ ...api, ...reading, ...writing }).forEach((f) => f.mockReset());
});
afterEach(() => window.localStorage.clear());

describe("FullMock", () => {
  it("starts a full mock straight into Reading, then lists the skills", async () => {
    const started = mock({ reading: { attempt_id: "r1", state: "in_progress" } });
    api.createMock.mockResolvedValue(mock());
    api.startMockModule.mockResolvedValue(started);
    reading.getReading.mockResolvedValue({ id: "r1", status: "created" });
    reading.startReading.mockResolvedValue({ id: "r1", status: "in_progress" });
    const done = mock({
      reading: { attempt_id: "r1", state: "graded", band: 6.0, detail: "20 of 40 correct" },
      report: { overall: 6.0, next_module: { key: "writing", label: "Writing", reason: "Sit Writing next." } },
    });
    api.getMock.mockResolvedValue(done);

    render(<FullMock />);
    await settle();
    await userEvent.click(screen.getByRole("button", { name: "Start full mock" }));
    await settle();
    expect(api.startMockModule).toHaveBeenCalledWith("m1", "reading");
    expect(reading.startReading).toHaveBeenCalledWith("r1");

    await userEvent.click(screen.getByRole("button", { name: "Submit reading" }));
    await settle();
    const skills = screen.getByRole("list", { name: "Skill bands" });
    expect(skills).toHaveTextContent("Reading");
    expect(skills).toHaveTextContent("6.0");
    expect(screen.getAllByText("Not assessed").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByRole("button", { name: "Start Writing" })).toBeEnabled();
    expect(screen.getByText("Next: Writing")).toBeInTheDocument();
    expect(window.localStorage.getItem("ielts.fullmock.id")).toBe("m1");
  });

  it("shows the partial overall with the explanation, the source and the coach link", async () => {
    window.localStorage.setItem("ielts.fullmock.id", "m1");
    api.getMock.mockResolvedValue(
      mock({
        reading: { attempt_id: "r1", state: "graded", band: 6.0 },
        writing: { attempt_id: "w1", state: "graded", band: 6.5 },
        report: {
          overall: 6.5, partial: true,
          overall_note: "Partial overall: Listening and Speaking not assessed, so this is the mean of 2 of 4 skills.",
          next_module: { key: "reading", label: "Reading", reason: "Reading is your lowest assessed skill." },
        },
      })
    );
    render(<FullMock />);
    await settle();
    expect(screen.getByText("Partial overall band (mock score)")).toBeInTheDocument();
    expect(screen.getByLabelText("Mock overall band 6.5")).toBeInTheDocument();
    expect(screen.getByText(/Listening and Speaking not assessed/)).toBeInTheDocument();
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
});
