import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ListeningMock } from "../ListeningMock";
import type {
  ListeningAttempt,
  ListeningInfo,
  ListeningQuestion,
  ListeningReport,
  Section,
} from "../listeningTypes";

const api = vi.hoisted(() => ({
  listeningInfo: vi.fn(),
  createListening: vi.fn(),
  getListening: vi.fn(),
  startListening: vi.fn(),
  saveListeningAnswers: vi.fn(),
  submitListening: vi.fn(),
  getListeningReport: vi.fn(),
}));
vi.mock("../listeningApi", () => api);

const info: ListeningInfo = {
  set_id: "listening-x",
  allowed_s: 1800,
  question_count: 3,
  task_types: {},
  timing_source_url: "https://ielts.org/timing",
  band_source_url: "https://idp.example/listening-bands",
  band_note: "Half bands are interpolated between published anchors.",
  set_note: "This set has 3 questions, fewer than the 40 of a full test.",
  keys_note: "Answer keys are author-written and not human-checked.",
};

const sections: Section[] = [
  {
    id: "s1", number: 1, title: "A booking call", context: "A call to a library.",
    audio_url: "/ielts/listening/audio/s1", audio_s: 60,
  },
  {
    id: "s2", number: 2, title: "A talk", context: "A guide talks to visitors.",
    audio_url: "/ielts/listening/audio/s2", audio_s: 90,
  },
];

const questions: ListeningQuestion[] = [
  {
    id: "q1", number: 1, section_id: "s1", task_type: "form_completion", task_label: "Form completion",
    instruction: "Write ONE WORD.", prompt: "Surname: ____", choices: {},
  },
  {
    id: "q2", number: 2, section_id: "s1", task_type: "mcq", task_label: "Multiple choice",
    instruction: "", prompt: "Which day?", choices: { A: "Monday", B: "Friday" },
  },
  {
    id: "q3", number: 3, section_id: "s2", task_type: "map_labelling", task_label: "Map labelling",
    instruction: "", prompt: "Where is the cafe?", choices: { A: "North", B: "South" },
  },
];

const attempt = (over: Partial<ListeningAttempt> = {}): ListeningAttempt => ({
  id: "a1", set_id: "listening-x", status: "in_progress", allowed_s: 1800, remaining_s: 1800,
  started_at: 1, submitted_at: null, late: false, answers: {}, answered: 0, band: null, raw: null,
  scaled_40: null, paper: { sections, questions }, ...over,
});

const report: ListeningReport = {
  set_id: "listening-x", raw: 2, total: 3, scaled_40: 27, band: 6.0,
  timing: { allowed_s: 1800, used_s: 125, late: false },
  by_type: [
    { task_type: "form_completion", label: "Form completion", correct: 1, total: 1 },
    { task_type: "map_labelling", label: "Map labelling", correct: 0, total: 1 },
  ],
  questions: [
    {
      number: 3, id: "q3", passage: "A talk", task_type: "map_labelling", task_label: "Map labelling",
      prompt: "Where is the cafe?", paragraph: null, answer: "A", answer_label: "A: North", key: "B",
      key_label: "B: South", correct: false, evidence: "The guide says the cafe is by the south gate.",
    },
  ],
  transcripts: [{ id: "s1", title: "Section 1: A booking call", lines: ["Clerk: Good morning."] }],
  band_note: info.band_note, set_note: info.set_note, keys_note: info.keys_note,
  band_source_url: info.band_source_url, timing_source_url: info.timing_source_url,
  disclaimer: "An unofficial practice estimate.",
};

const settle = () => act(async () => {});

describe("ListeningMock", () => {
  beforeEach(() => {
    window.localStorage.clear();
    Object.values(api).forEach((f) => f.mockReset());
    api.listeningInfo.mockResolvedValue(info);
  });
  afterEach(() => vi.useRealTimers());

  it("introduces the 30 minutes, the sound and the author-key caveat before starting", async () => {
    render(<ListeningMock />);
    expect(await screen.findByText(/You have 30 minutes/)).toBeInTheDocument();
    expect(screen.getByText(/you need sound on/)).toBeInTheDocument();
    expect(screen.getByText(info.set_note)).toBeInTheDocument();
    expect(screen.getByText(info.keys_note)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "IELTS.org" })).toHaveAttribute("href", info.timing_source_url);
    expect(screen.getByRole("link", { name: "Listening" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Reading" })).toHaveAttribute(
      "href", "/ielts-mock?module=reading"
    );
  });

  it("starts, plays a recording per section, takes answers, and reports the band with the transcripts", async () => {
    api.createListening.mockResolvedValue(attempt({ status: "created", paper: null, remaining_s: null }));
    api.startListening.mockResolvedValue(attempt());
    api.submitListening.mockResolvedValue(
      attempt({ status: "graded", paper: null, band: 6.0, raw: 2, scaled_40: 27 })
    );
    api.getListeningReport.mockResolvedValue(report);
    render(<ListeningMock />);
    await userEvent.click(await screen.findByRole("button", { name: "Start the 30 minutes" }));
    expect(await screen.findByText("Section 1: A booking call")).toBeInTheDocument();
    expect(screen.getByRole("timer")).toHaveTextContent("30:00");
    expect(screen.getByLabelText("Recording for section 1")).toHaveAttribute("src", "/ielts/listening/audio/s1");
    expect(screen.getByLabelText("Recording for section 2")).toHaveAttribute("src", "/ielts/listening/audio/s2");
    expect(screen.getByText("0 of 3 answered")).toBeInTheDocument();

    await userEvent.type(screen.getByLabelText("Answer to question 1"), "Okafor");
    await userEvent.click(screen.getByRole("radio", { name: "B. Friday" }));
    expect(screen.getByText("2 of 3 answered")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Submit answers" }));
    expect(screen.getByText(/Submit 2 of 3 answers/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Yes, submit" }));
    await settle();

    expect(api.submitListening).toHaveBeenCalledWith("a1", { q1: "Okafor", q2: "B" });
    expect(screen.getByText("6.0")).toBeInTheDocument();
    expect(screen.getByText(/Raw score 2 of 3\. Scaled to 27 of 40/)).toBeInTheDocument();
    expect(screen.getByText("The guide says the cafe is by the south gate.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "IDP Listening scores" })).toHaveAttribute(
      "href", info.band_source_url
    );
    expect(screen.getByText("Section 1: A booking call")).toBeInTheDocument();
    expect(screen.getByText("Clerk: Good morning.")).toBeInTheDocument();
  });

  it("the clock turns urgent only in the last minute of a sample but the last 5 minutes of the real test", async () => {
    window.localStorage.setItem("ielts.listening.attempt", "a1");
    const urgent = () => screen.getByRole("timer").parentElement!.classList.contains("ielts-clock--urgent");
    api.getListening.mockResolvedValue(attempt({ sample: true, remaining_s: 150 }));
    const first = render(<ListeningMock />);
    await screen.findByRole("timer");
    expect(urgent()).toBe(false);
    first.unmount();
    api.getListening.mockResolvedValue(attempt({ sample: true, remaining_s: 50 }));
    const second = render(<ListeningMock />);
    await screen.findByRole("timer");
    expect(urgent()).toBe(true);
    second.unmount();
    api.getListening.mockResolvedValue(attempt({ remaining_s: 250 }));
    render(<ListeningMock />);
    await screen.findByRole("timer");
    expect(urgent()).toBe(true);
  });

  it("resumes an attempt in progress with its saved answers", async () => {
    window.localStorage.setItem("ielts.listening.attempt", "a1");
    api.getListening.mockResolvedValue(attempt({ remaining_s: 900, answers: { q3: "A" } }));
    render(<ListeningMock />);
    expect(await screen.findByRole("timer")).toHaveTextContent("15:00");
    expect(screen.getByRole("radio", { name: "A. North" })).toBeChecked();
  });

  it("shows the report of a graded attempt when the page is reopened", async () => {
    window.localStorage.setItem("ielts.listening.attempt", "a1");
    api.getListening.mockResolvedValue(attempt({ status: "graded", paper: null }));
    api.getListeningReport.mockResolvedValue(report);
    render(<ListeningMock />);
    expect(await screen.findByText("Mock Listening band")).toBeInTheDocument();
  });

  it("autosaves changed answers on the interval", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    window.localStorage.setItem("ielts.listening.attempt", "a1");
    api.getListening.mockResolvedValue(attempt());
    api.saveListeningAnswers.mockResolvedValue(attempt({ remaining_s: 1790 }));
    render(<ListeningMock autosaveMs={1000} />);
    await settle();
    fireEvent.click(screen.getByRole("radio", { name: "A. Monday" }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1100);
    });
    expect(api.saveListeningAnswers).toHaveBeenCalledWith("a1", { q2: "A" });
  });

  it("submits by itself when the clock runs out", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    window.localStorage.setItem("ielts.listening.attempt", "a1");
    api.getListening.mockResolvedValue(attempt({ remaining_s: 2, answers: { q1: "Okafor" } }));
    api.submitListening.mockResolvedValue(attempt({ status: "graded", paper: null }));
    api.getListeningReport.mockResolvedValue(report);
    render(<ListeningMock />);
    await settle();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2600);
    });
    expect(api.submitListening).toHaveBeenCalledTimes(1);
    expect(api.submitListening).toHaveBeenCalledWith("a1", { q1: "Okafor" });
  });

  it("shows the failure and a reload button when the gateway is unreachable", async () => {
    api.listeningInfo.mockRejectedValue(new Error("Cannot reach the server"));
    render(<ListeningMock />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Cannot reach the server");
  });
});
