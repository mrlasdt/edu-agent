import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ModeSwitch } from "../components/ModeSwitch";

describe("ModeSwitch", () => {
  it("shows 'Show me the answer' in tutor mode", () => {
    render(<ModeSwitch mode="tutor" onToggle={vi.fn()} />);
    expect(screen.getByRole("button", { name: /show me the answer/i })).toBeInTheDocument();
  });

  it("shows 'Back to hints' in solve mode", () => {
    render(<ModeSwitch mode="solve" onToggle={vi.fn()} />);
    expect(screen.getByRole("button", { name: /back to hints/i })).toBeInTheDocument();
  });

  it("calls onToggle when button clicked", async () => {
    const onToggle = vi.fn();
    render(<ModeSwitch mode="tutor" onToggle={onToggle} />);
    await userEvent.click(screen.getByRole("button", { name: /show me the answer/i }));
    expect(onToggle).toHaveBeenCalledOnce();
  });
});
