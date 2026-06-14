import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PreferencesPanel } from "../components/PreferencesPanel";

describe("PreferencesPanel — visibility", () => {
  it("renders when isOpen is true", () => {
    render(
      <PreferencesPanel
        isOpen={true}
        onClose={vi.fn()}
        personalStyleEnabled={false}
        hasEnoughEssays={true}
        onStyleToggle={vi.fn()}
      />
    );
    expect(screen.getByRole("complementary")).toBeInTheDocument();
  });

  it("does not render when isOpen is false", () => {
    render(
      <PreferencesPanel
        isOpen={false}
        onClose={vi.fn()}
        personalStyleEnabled={false}
        hasEnoughEssays={true}
        onStyleToggle={vi.fn()}
      />
    );
    expect(screen.queryByRole("complementary")).not.toBeInTheDocument();
  });
});

describe("PreferencesPanel — style toggle visibility", () => {
  it("shows style toggle when hasEnoughEssays is true", () => {
    render(
      <PreferencesPanel
        isOpen={true}
        onClose={vi.fn()}
        personalStyleEnabled={false}
        hasEnoughEssays={true}
        onStyleToggle={vi.fn()}
      />
    );
    expect(screen.getByRole("checkbox", { name: /match my writing style/i })).toBeInTheDocument();
  });

  it("hides style toggle (not just disables) when hasEnoughEssays is false", () => {
    render(
      <PreferencesPanel
        isOpen={true}
        onClose={vi.fn()}
        personalStyleEnabled={false}
        hasEnoughEssays={false}
        onStyleToggle={vi.fn()}
      />
    );
    expect(
      screen.queryByRole("checkbox", { name: /match my writing style/i })
    ).not.toBeInTheDocument();
  });

  it("shows cold-start hint when hasEnoughEssays is false", () => {
    render(
      <PreferencesPanel
        isOpen={true}
        onClose={vi.fn()}
        personalStyleEnabled={false}
        hasEnoughEssays={false}
        onStyleToggle={vi.fn()}
      />
    );
    expect(screen.getByText(/upload.*essays.*enable/i)).toBeInTheDocument();
  });
});

describe("PreferencesPanel — style toggle interaction", () => {
  it("reflects personalStyleEnabled=true as checked", () => {
    render(
      <PreferencesPanel
        isOpen={true}
        onClose={vi.fn()}
        personalStyleEnabled={true}
        hasEnoughEssays={true}
        onStyleToggle={vi.fn()}
      />
    );
    expect(screen.getByRole("checkbox", { name: /match my writing style/i })).toBeChecked();
  });

  it("calls onStyleToggle when toggled", async () => {
    const onStyleToggle = vi.fn();
    render(
      <PreferencesPanel
        isOpen={true}
        onClose={vi.fn()}
        personalStyleEnabled={false}
        hasEnoughEssays={true}
        onStyleToggle={onStyleToggle}
      />
    );
    await userEvent.click(screen.getByRole("checkbox", { name: /match my writing style/i }));
    expect(onStyleToggle).toHaveBeenCalledOnce();
  });
});

describe("PreferencesPanel — close", () => {
  it("calls onClose when close button clicked", async () => {
    const onClose = vi.fn();
    render(
      <PreferencesPanel
        isOpen={true}
        onClose={onClose}
        personalStyleEnabled={false}
        hasEnoughEssays={true}
        onStyleToggle={vi.fn()}
      />
    );
    await userEvent.click(screen.getByRole("button", { name: /close/i }));
    expect(onClose).toHaveBeenCalledOnce();
  });
});
