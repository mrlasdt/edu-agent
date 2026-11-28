import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SpeakingMock } from "../SpeakingMock";
import type {
  SpeakingAttempt,
  SpeakingInfo,
  SpeakingPaper,
  SpeakingReport,
} from "../speakingTypes";

const api = vi.hoisted(() => ({
  speakingInfo: vi.fn(),
  createSpeaking: vi.fn(),
  getSpeaking: vi.fn(),
  startSpeaking: vi.fn(),
  submitSpeaking: vi.fn(),
  regradeSpeaking: vi.fn(),
  getSpeakingReport: vi.fn(),
  uploadSpeakingPart: vi.fn(),
}));
vi.mock("../speakingApi", () => api);

const paper: SpeakingPaper = {
  id: "sp-sample",
  sample: true,
  title: "Quick sample",
  max_clip_s: [40, 30, 40],
  parts: [
    { part: 1, title: "Part 1: Introduction", questions: ["Do you live in a house or a flat?"] },
    {
      part: 2,
      title: "Part 2: Long turn",
      topic: "Describe a place you like to visit.",
      points: ["where it is", "why you go"],
      prep_s: 5,
      talk_s: 20,
    },
    { part: 3, title: "Part 3: Discussion", questions: ["Why do cities have parks?"] },
  ],
};

const info: SpeakingInfo = {
  allowed_s: 1200,
  part2_prep_s: 60,
  part2_talk_s: 120,
  sample: paper,
  not_assessed: { Pronunciation: "needs a dedicated acoustic model" },
  descriptors_source_url: "https://ielts.org/descriptors",
  format_source_url: "https://ielts.org/format",
  model: "gemini-x",
  set_note: "Original questions, not IELTS material.",
};

const attempt = (over: Partial<SpeakingAttempt> = {}): SpeakingAttempt => ({
  id: "a1",
  set_id: "sp-sample",
  status: "in_progress",
  allowed_s: 1200,
  remaining_s: 1200,
  started_at: 1,
  submitted_at: null,
  late: false,
  uploaded: [],
  paper,
  band: null,
  partial: null,
  error: null,
  retryable: false,
  ...over,
});

const report: SpeakingReport = {
  status: "graded",
  set_id: "sp-sample",
  sample: true,
  band: 6.5,
  partial: true,
  partial_note: "Partial band: Pronunciation is not assessed.",
  criteria: [
    { id: "FC", name: "Fluency and coherence", assessed: true, band: 6.5, rationale: "Steady.", evidence: ["I live in a flat"], reason: "" },
    { id: "LR", name: "Lexical resource", assessed: true, band: 6.5, rationale: "Fine.", evidence: [], reason: "" },
    { id: "GRA", name: "Grammatical range and accuracy", assessed: true, band: 6.5, rationale: "Fine.", evidence: [], reason: "" },
    { id: "P", name: "Pronunciation", assessed: false, band: null, rationale: "", evidence: [], reason: "Needs an acoustic model." },
  ],
  not_assessed: ["P"],
  parts: [
    { part: 1, title: "Part 1", transcript: "I live in a flat.", duration_s: 4, allowed_s: null, words: 5, words_per_minute: 75, fillers: 0 },
  ],
  strengths: ["Clear answers"],
  priorities: [{ criterion: "LR", priority: 1, do: "Use more precise words" }],
  flags: [],
  model: "gemini-x",
  timing: { allowed_s: 1200, used_s: 60, late: false },
  descriptors_source_url: "https://ielts.org/descriptors",
  format_source_url: "https://ielts.org/format",
  disclaimer: "A mock estimate.",
};

class FakeRecorder {
  static instances: FakeRecorder[] = [];
  state: "inactive" | "recording" = "inactive";
  mimeType = "audio/webm;codecs=opus";
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  constructor() {
    FakeRecorder.instances.push(this);
  }
  start() {
    this.state = "recording";
  }
  stop() {
    this.state = "inactive";
    this.ondataavailable?.({ data: new Blob(["voice"], { type: this.mimeType }) });
    this.onstop?.();
  }
}

beforeEach(() => {
  window.localStorage.clear();
  FakeRecorder.instances = [];
  vi.stubGlobal("MediaRecorder", FakeRecorder);
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop: vi.fn() }] }) },
  });
  URL.createObjectURL = vi.fn(() => "blob:x");
  URL.revokeObjectURL = vi.fn();
  Object.values(api).forEach((f) => f.mockReset());
  api.speakingInfo.mockResolvedValue(info);
  api.createSpeaking.mockResolvedValue(attempt({ status: "created" }));
  api.startSpeaking.mockResolvedValue(attempt());
  api.uploadSpeakingPart.mockImplementation(async (_id: string, part: number) =>
    attempt({ uploaded: [1, 2, 3].filter((n) => n <= part) })
  );
});
afterEach(() => vi.unstubAllGlobals());

async function recordPart(user: ReturnType<typeof userEvent.setup>, startLabel: RegExp) {
  await user.click(await screen.findByRole("button", { name: startLabel }));
  await user.click(await screen.findByRole("button", { name: /^stop$/i }));
  await screen.findByText(/^Recorded/);
}

describe("SpeakingMock", () => {
  it("names the unassessed criterion on the intro and offers the quick sample", async () => {
    render(<SpeakingMock />);
    expect(await screen.findByText(/Pronunciation is not assessed/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /quick sample/i })).toBeInTheDocument();
  });

  it("records the three parts, submits, polls and shows the partial report", async () => {
    const user = userEvent.setup();
    api.submitSpeaking.mockResolvedValue(attempt({ status: "grading" }));
    api.getSpeaking.mockResolvedValue(attempt({ status: "graded", band: 6.5, partial: true }));
    api.getSpeakingReport.mockResolvedValue(report);
    render(<SpeakingMock pollMs={10} />);

    await user.click(await screen.findByRole("button", { name: /quick sample/i }));
    expect(api.createSpeaking).toHaveBeenCalledWith(true);
    await user.click(await screen.findByRole("button", { name: /allow the microphone/i }));

    await recordPart(user, /start recording/i);
    await user.click(screen.getByRole("button", { name: /next: part 2/i }));
    expect(screen.getByText("Describe a place you like to visit.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /preparation/i }));
    await user.click(await screen.findByRole("button", { name: /start talking now/i }));
    await user.click(await screen.findByRole("button", { name: /^stop$/i }));
    await screen.findByText(/^Recorded/);
    await user.click(screen.getByRole("button", { name: /next: part 3/i }));
    await recordPart(user, /start recording/i);
    await user.click(screen.getByRole("button", { name: /submit for marking/i }));

    expect(api.submitSpeaking).toHaveBeenCalledWith("a1");
    expect(api.uploadSpeakingPart.mock.calls.map((c) => c[1])).toEqual([1, 2, 3]);
    expect(await screen.findByLabelText("Partial Speaking band 6.5")).toBeInTheDocument();
    expect(screen.getByLabelText("Pronunciation not assessed")).toBeInTheDocument();
  });

  it("skips the preparation and sends the Part 2 recording", async () => {
    const user = userEvent.setup();
    api.createSpeaking.mockResolvedValue(attempt({ status: "created", uploaded: [1] }));
    api.startSpeaking.mockResolvedValue(attempt({ uploaded: [1] }));
    render(<SpeakingMock />);
    await user.click(await screen.findByRole("button", { name: /start speaking/i }));
    await user.click(await screen.findByRole("button", { name: /allow the microphone/i }));
    await user.click(await screen.findByRole("button", { name: /preparation/i }));
    expect(screen.getByRole("timer", { name: /preparation time left/i })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /start talking now/i }));
    await user.click(await screen.findByRole("button", { name: /^stop$/i }));
    await waitFor(() => expect(api.uploadSpeakingPart).toHaveBeenCalled());
    const [id, part, blob, seconds] = api.uploadSpeakingPart.mock.calls[0];
    expect([id, part]).toEqual(["a1", 2]);
    expect(blob).toBeInstanceOf(Blob);
    expect(seconds).toBeGreaterThanOrEqual(0);
  });

  it("shows Pronunciation as not assessed on the report, never as a band", async () => {
    window.localStorage.setItem("ielts.speaking.attempt", "a1");
    api.getSpeaking.mockResolvedValue(attempt({ status: "graded", band: 6.5, partial: true }));
    api.getSpeakingReport.mockResolvedValue(report);
    render(<SpeakingMock />);
    expect(await screen.findByLabelText("Partial Speaking band 6.5")).toBeInTheDocument();
    expect(screen.getByLabelText("Pronunciation not assessed")).toBeInTheDocument();
    expect(screen.getByText("Needs an acoustic model.")).toBeInTheDocument();
    expect(screen.getByText("Use more precise words")).toBeInTheDocument();
    expect(screen.getByText(/quick sample/i)).toBeInTheDocument();
  });

  it("offers a retry when marking failed with the recording kept", async () => {
    const user = userEvent.setup();
    window.localStorage.setItem("ielts.speaking.attempt", "a1");
    api.getSpeaking.mockResolvedValue(
      attempt({ status: "ungraded", error: "The marker was unavailable.", retryable: true })
    );
    api.regradeSpeaking.mockResolvedValue(attempt({ status: "grading" }));
    render(<SpeakingMock pollMs={10000} />);
    expect(await screen.findByText("The marker was unavailable.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /retry marking/i }));
    expect(api.regradeSpeaking).toHaveBeenCalledWith("a1");
    expect(await screen.findByText(/Marking your Speaking/)).toBeInTheDocument();
  });

  it("explains a blocked microphone", async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn().mockRejectedValue(new Error("denied")) },
    });
    render(<SpeakingMock />);
    await user.click(await screen.findByRole("button", { name: /start speaking/i }));
    await user.click(await screen.findByRole("button", { name: /allow the microphone/i }));
    expect(await screen.findByText(/microphone is blocked/i)).toBeInTheDocument();
  });
});
