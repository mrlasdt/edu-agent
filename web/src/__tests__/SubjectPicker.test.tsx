import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SubjectPicker } from "../components/SubjectPicker";

describe("SubjectPicker", () => {
  it("renders Quant and AW options", () => {
    render(<SubjectPicker onSelect={vi.fn()} />);
    expect(screen.getByRole("button", { name: /quant/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /aw/i })).toBeInTheDocument();
  });

  it("calls onSelect with 'quant' when Quant clicked", async () => {
    const onSelect = vi.fn();
    render(<SubjectPicker onSelect={onSelect} />);
    await userEvent.click(screen.getByRole("button", { name: /quant/i }));
    expect(onSelect).toHaveBeenCalledWith("quant");
  });

  it("calls onSelect with 'aw' when AW clicked", async () => {
    const onSelect = vi.fn();
    render(<SubjectPicker onSelect={onSelect} />);
    await userEvent.click(screen.getByRole("button", { name: /aw/i }));
    expect(onSelect).toHaveBeenCalledWith("aw");
  });
});
