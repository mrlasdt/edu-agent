import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { WritingScreen } from "../WritingScreen";
import { attempt } from "./fixtures";

const api = vi.hoisted(() => ({
  saveDraft: vi.fn(),
  submitAttempt: vi.fn(),
  saveDraftOnExit: vi.fn(),
}));
vi.mock("../api", async (orig) => ({ ...(await orig<typeof import("../api")>()), ...api }));

const flush = () => act(async () => {});

describe("WritingScreen", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    window.localStorage.clear();
    api.saveDraft.mockResolvedValue(attempt({ remaining_s: 2300 }));
    api.submitAttempt.mockResolvedValue(attempt({ status: "submitted" }));
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("counts down from the server's remaining time", async () => {
    render(<WritingScreen attempt={attempt({ remaining_s: 600 })} initialText="" onSubmitted={vi.fn()} />);
    expect(screen.getByRole("timer")).toHaveTextContent("10:00");
    await act(async () => void vi.advanceTimersByTime(61_000));
    expect(screen.getByRole("timer")).toHaveTextContent("08:59");
  });

  it("warns in the last five minutes", () => {
    render(<WritingScreen attempt={attempt({ remaining_s: 240 })} initialText="" onSubmitted={vi.fn()} />);
    expect(screen.getByRole("timer")).toHaveTextContent("wrap up");
  });

  it("counts words as the candidate types and shows the 250-word minimum", () => {
    render(<WritingScreen attempt={attempt()} initialText="" onSubmitted={vi.fn()} />);
    expect(screen.getByLabelText("Word count")).toHaveTextContent("0 words");
    fireEvent.change(screen.getByLabelText("Your essay"), { target: { value: "one two  three\nfour" } });
    expect(screen.getByLabelText("Word count")).toHaveTextContent("4 words");
    expect(screen.getByLabelText("Word count")).toHaveTextContent("250 minimum");
  });

  it("autosaves changed text about every ten seconds, and not when nothing changed", async () => {
    render(<WritingScreen attempt={attempt()} initialText="" onSubmitted={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Your essay"), { target: { value: "My draft" } });
    await act(async () => void vi.advanceTimersByTime(10_100));
    expect(api.saveDraft).toHaveBeenCalledTimes(1);
    expect(api.saveDraft).toHaveBeenCalledWith("a1", "My draft");
    expect(screen.getByText(/^Saved /)).toBeInTheDocument();
    await act(async () => void vi.advanceTimersByTime(20_000));
    expect(api.saveDraft).toHaveBeenCalledTimes(1);
  });

  it("keeps the words and keeps retrying when a save fails", async () => {
    api.saveDraft.mockRejectedValueOnce(new Error("offline")).mockResolvedValue(attempt());
    render(<WritingScreen attempt={attempt()} initialText="" onSubmitted={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Your essay"), { target: { value: "Keep me" } });
    await act(async () => void vi.advanceTimersByTime(10_100));
    expect(screen.getByText(/Not saved/)).toBeInTheDocument();
    expect(screen.getByLabelText("Your essay")).toHaveValue("Keep me");
    expect(JSON.parse(window.localStorage.getItem("ielts.draft.a1")!)).toEqual({
      text: "Keep me",
      saved: false,
    });
    await act(async () => void vi.advanceTimersByTime(10_000));
    expect(api.saveDraft).toHaveBeenCalledTimes(2);
    expect(screen.getByText(/^Saved /)).toBeInTheDocument();
  });

  it("asks before submitting, warns about a very short essay, then submits the text", async () => {
    const onSubmitted = vi.fn();
    render(<WritingScreen attempt={attempt()} initialText="Short essay here." onSubmitted={onSubmitted} />);
    fireEvent.click(screen.getByRole("button", { name: "Submit for marking" }));
    expect(screen.getByText(/well under the 250 minimum/)).toBeInTheDocument();
    expect(api.submitAttempt).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Keep writing" }));
    expect(screen.getByRole("button", { name: "Submit for marking" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Submit for marking" }));
    fireEvent.click(screen.getByRole("button", { name: "Yes, submit" }));
    await flush();
    expect(api.submitAttempt).toHaveBeenCalledWith("a1", "Short essay here.");
    expect(onSubmitted).toHaveBeenCalled();
  });

  it("submits by itself when time runs out", async () => {
    const onSubmitted = vi.fn();
    render(
      <WritingScreen attempt={attempt({ remaining_s: 3 })} initialText="Whatever I wrote" onSubmitted={onSubmitted} />
    );
    await act(async () => void vi.advanceTimersByTime(3_600));
    expect(api.submitAttempt).toHaveBeenCalledTimes(1);
    expect(api.submitAttempt).toHaveBeenCalledWith("a1", "Whatever I wrote");
    expect(onSubmitted).toHaveBeenCalled();
  });

  it("keeps the essay on screen and lets the candidate retry when submit fails", async () => {
    api.submitAttempt.mockRejectedValueOnce(new Error("Cannot reach the server"));
    const onSubmitted = vi.fn();
    render(<WritingScreen attempt={attempt()} initialText="Precious words" onSubmitted={onSubmitted} />);
    fireEvent.click(screen.getByRole("button", { name: "Submit for marking" }));
    fireEvent.click(screen.getByRole("button", { name: "Yes, submit" }));
    await flush();
    expect(screen.getByRole("alert")).toHaveTextContent("Your essay is still here");
    expect(screen.getByLabelText("Your essay")).toHaveValue("Precious words");
    fireEvent.click(screen.getByRole("button", { name: "Submit for marking" }));
    fireEvent.click(screen.getByRole("button", { name: "Yes, submit" }));
    await flush();
    expect(onSubmitted).toHaveBeenCalledTimes(1);
  });

  it("makes a last save when the page is hidden with unsaved words", () => {
    render(<WritingScreen attempt={attempt()} initialText="" onSubmitted={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Your essay"), { target: { value: "Unsaved" } });
    window.dispatchEvent(new Event("pagehide"));
    expect(api.saveDraftOnExit).toHaveBeenCalledWith("a1", "Unsaved");
  });
});
