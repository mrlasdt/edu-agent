import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SourcesPanel } from "../components/SourcesPanel";
import type { Citation } from "../types";

const mockCitations: Citation[] = [
  { index: 1, text: "The GRE tests analytical reasoning.", source_uri: "ets-guide.pdf", page_or_section: "Introduction", tier: "global" },
  { index: 2, text: "Issue essays require a clear thesis.", source_uri: "ets-guide.pdf", page_or_section: "Writing Tips", tier: "global" },
];

describe("SourcesPanel", () => {
  it("does not render when isOpen is false", () => {
    render(<SourcesPanel citations={mockCitations} isOpen={false} onClose={vi.fn()} />);
    expect(screen.queryByText(/ets-guide/i)).not.toBeInTheDocument();
  });

  it("renders citation source URIs when open", () => {
    render(<SourcesPanel citations={mockCitations} isOpen={true} onClose={vi.fn()} />);
    expect(screen.getAllByText(/ets-guide\.pdf/i).length).toBeGreaterThan(0);
  });

  it("renders citation text content when open", () => {
    render(<SourcesPanel citations={mockCitations} isOpen={true} onClose={vi.fn()} />);
    expect(screen.getByText(/analytical reasoning/i)).toBeInTheDocument();
  });

  it("calls onClose when close button clicked", async () => {
    const onClose = vi.fn();
    render(<SourcesPanel citations={mockCitations} isOpen={true} onClose={onClose} />);
    await userEvent.click(screen.getByRole("button", { name: /close/i }));
    expect(onClose).toHaveBeenCalledOnce();
  });
});
