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

describe("MessageBubble — markdown", () => {
  const renderAgent = (content: string) =>
    render(<MessageBubble message={{ ...baseMsg, content }} onCitationClick={vi.fn()} />);

  it("renders headings and bold without literal markers", () => {
    const { container } = renderAgent("### Step 1\nFactor **carefully** first.");
    expect(screen.getByRole("heading", { name: "Step 1" })).toBeInTheDocument();
    expect(container.querySelector("strong")).toHaveTextContent("carefully");
    expect(container.textContent).not.toContain("###");
    expect(container.textContent).not.toContain("**");
  });

  it("renders lists, italics and code", () => {
    const { container } = renderAgent(
      "1. *First* step\n2. Use `x^2`\n\n- a\n- b\n\n```\nx = 2\n```"
    );
    expect(container.querySelectorAll("ol > li")).toHaveLength(2);
    expect(container.querySelectorAll("ul > li")).toHaveLength(2);
    expect(container.querySelector("em")).toHaveTextContent("First");
    expect(container.querySelector("li code")).toHaveTextContent("x^2");
    expect(container.querySelector("pre code")).toHaveTextContent("x = 2");
  });

  it("keeps multiplication asterisks literal", () => {
    const { container } = renderAgent("So 2*3*4 = 24 and 2 * 3 = 6.");
    expect(container.querySelector("em")).toBeNull();
    expect(container.textContent).toBe("So 2*3*4 = 24 and 2 * 3 = 6.");
  });

  it("renders HTML in the text as inert text", () => {
    const { container } = renderAgent(
      "<script>alert(1)</script>\n\n**<img src=x onerror=alert(1)>**"
    );
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain("<script>alert(1)</script>");
    expect(container.textContent).toContain("<img src=x onerror=alert(1)>");
  });

  it("tolerates unfinished markup while streaming", () => {
    const { container, rerender } = renderAgent("Start **bol");
    expect(container.textContent).toBe("Start **bol");
    rerender(<MessageBubble message={{ ...baseMsg, content: "Start **bold**\n```\ncode" }} />);
    expect(container.querySelector("strong")).toHaveTextContent("bold");
    expect(container.querySelector("pre code")).toHaveTextContent("code");
  });

  it("keeps citation markers clickable inside formatted text", async () => {
    const onCitationClick = vi.fn();
    render(
      <MessageBubble
        message={{ ...baseMsg, content: "- **See [2]** for the rule" }}
        onCitationClick={onCitationClick}
      />
    );
    await userEvent.click(screen.getByRole("button", { name: "[2]" }));
    expect(onCitationClick).toHaveBeenCalledWith(2);
  });

  it("renders the answer line as bold text", () => {
    const { container } = renderAgent("Work shown.\n\n**Answer: 6**");
    expect(container.querySelector("strong")).toHaveTextContent("Answer: 6");
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

describe("MessageBubble — markers without a handler", () => {
  it("renders [N] as plain text, not a button that does nothing", () => {
    const { container } = render(
      <MessageBubble message={{ ...baseMsg, content: "See reference [1] for details." }} />
    );
    expect(screen.queryByRole("button", { name: "[1]" })).not.toBeInTheDocument();
    expect(container.querySelector("sup.citation-marker")).toHaveTextContent("[1]");
  });
});

describe("MessageBubble — math", () => {
  const text = (content: string) =>
    render(<MessageBubble message={{ ...baseMsg, content }} />).container;

  it("typesets exponents as superscripts", () => {
    const container = text("Factor x^2 - 5x + 6 = 0, then 2^(n+1) and 10^-3.");
    const sups = [...container.querySelectorAll("sup")].map((s) => s.textContent);
    expect(sups).toEqual(["2", "n+1", "-3"]);
    expect(container.textContent).toBe("Factor x2 - 5x + 6 = 0, then 2n+1 and 10-3.");
  });

  it("replaces ASCII operators with symbols", () => {
    const container = text("x^2 = 4 => x = 2 or x <= 3, y >= 1, z != 0, a -> b, sqrt(2), 2 +- 1");
    expect(container.textContent).toContain("\u21d2 x = 2");
    for (const symbol of ["\u2264", "\u2265", "\u2260", "\u2192", "\u221a(2)", "\u00b1"]) {
      expect(container.textContent).toContain(symbol);
    }
  });

  it("leaves code spans and fences literal", () => {
    const container = text("Use `x^2 => y` here\n\n```\nx^2 <= 3\n```");
    expect(container.querySelector("sup")).toBeNull();
    expect(container.querySelector("p code")).toHaveTextContent("x^2 => y");
    expect(container.querySelector("pre code")).toHaveTextContent("x^2 <= 3");
  });
});

describe("MessageBubble — math verification", () => {
  it("says when the math verifier checked the final answer", () => {
    render(<MessageBubble message={{ ...baseMsg, content: "**Answer: 2**", mathVerified: true }} />);
    expect(screen.getByRole("note")).toHaveTextContent(/checked by the math verifier/i);
  });

  it("warns when the math verifier could not confirm it", () => {
    render(<MessageBubble message={{ ...baseMsg, content: "Best effort.", mathVerified: false }} />);
    expect(screen.getByRole("note")).toHaveTextContent(/could not confirm/i);
  });

  it("shows nothing when the verifier did not run", () => {
    render(<MessageBubble message={baseMsg} />);
    expect(screen.queryByRole("note")).not.toBeInTheDocument();
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
