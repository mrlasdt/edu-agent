import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MessageBubble } from "../components/MessageBubble";
import { ESSAY_DISCLAIMER } from "../types";

const baseMsg = { id: "m1", role: "agent" as const, content: "Think about it." };

describe("MessageBubble — plain content", () => {
  it("renders the message text", () => {
    render(<MessageBubble message={baseMsg} />);
    expect(screen.getByText(/think about it/i)).toBeInTheDocument();
  });
});

describe("MessageBubble — citations", () => {
  it("renders [N] as a clickable citation span", () => {
    const msg = { ...baseMsg, content: "See reference [1] for details." };
    render(<MessageBubble message={msg} onCitationClick={vi.fn()} />);
    expect(screen.getByRole("button", { name: "[1]" })).toBeInTheDocument();
  });

  it("calls onCitationClick with the citation index", async () => {
    const onCitationClick = vi.fn();
    const msg = { ...baseMsg, content: "Evidence in [2]." };
    render(<MessageBubble message={msg} onCitationClick={onCitationClick} />);
    await userEvent.click(screen.getByRole("button", { name: "[2]" }));
    expect(onCitationClick).toHaveBeenCalledWith(2);
  });

  it("renders multiple citations independently", () => {
    const msg = { ...baseMsg, content: "Points from [1] and [3]." };
    render(<MessageBubble message={msg} onCitationClick={vi.fn()} />);
    expect(screen.getByRole("button", { name: "[1]" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "[3]" })).toBeInTheDocument();
  });
});

describe("MessageBubble — disclaimer", () => {
  it("renders disclaimer text in a distinct disclaimer element", () => {
    const msg = { ...baseMsg, content: `Great essay.\n\n${ESSAY_DISCLAIMER}` };
    render(<MessageBubble message={msg} />);
    const disclaimer = screen.getByTestId("essay-disclaimer");
    expect(disclaimer).toBeInTheDocument();
    expect(disclaimer).toHaveTextContent(/study purposes only/i);
  });
});

describe("MessageBubble — feedback", () => {
  it("renders thumbs up and down buttons for agent messages", () => {
    render(<MessageBubble message={baseMsg} onFeedback={vi.fn()} />);
    expect(screen.getByRole("button", { name: /thumbs up/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /thumbs down/i })).toBeInTheDocument();
  });

  it("does not render feedback for candidate messages", () => {
    const msg = { id: "m2", role: "candidate" as const, content: "My question." };
    render(<MessageBubble message={msg} onFeedback={vi.fn()} />);
    expect(screen.queryByRole("button", { name: /thumbs up/i })).not.toBeInTheDocument();
  });

  it("calls onFeedback with 'up' on thumbs up click", async () => {
    const onFeedback = vi.fn();
    render(<MessageBubble message={baseMsg} onFeedback={onFeedback} />);
    await userEvent.click(screen.getByRole("button", { name: /thumbs up/i }));
    expect(onFeedback).toHaveBeenCalledWith("up");
  });
});

describe("MessageBubble — status lines", () => {
  it("renders status line when status is set", () => {
    const msg = { ...baseMsg, status: "Checking your answer…" };
    render(<MessageBubble message={msg} />);
    expect(screen.getByText(/checking your answer/i)).toBeInTheDocument();
  });
});
