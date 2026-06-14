import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { JobsTable } from "../components/JobsTable";
import type { IngestionJob } from "../types";

const mockJobs: IngestionJob[] = [
  {
    job_id: "job-1",
    filename: "ets-guide.pdf",
    test_type: "gre",
    status: "completed",
    created_at: "2026-06-14T10:00:00Z",
    error_stage: null,
  },
  {
    job_id: "job-2",
    filename: "practice.docx",
    test_type: "gre",
    status: "processing",
    created_at: "2026-06-14T10:05:00Z",
    error_stage: null,
  },
  {
    job_id: "job-3",
    filename: "broken.pdf",
    test_type: "gre",
    status: "failed",
    created_at: "2026-06-14T10:10:00Z",
    error_stage: "embed",
  },
];

describe("JobsTable — renders rows", () => {
  it("renders a row per job", () => {
    render(<JobsTable jobs={mockJobs} dlqCount={0} onViewDlq={vi.fn()} />);
    expect(screen.getByText("ets-guide.pdf")).toBeInTheDocument();
    expect(screen.getByText("practice.docx")).toBeInTheDocument();
    expect(screen.getByText("broken.pdf")).toBeInTheDocument();
  });

  it("shows status for each job", () => {
    render(<JobsTable jobs={mockJobs} dlqCount={0} onViewDlq={vi.fn()} />);
    expect(screen.getByText("completed")).toBeInTheDocument();
    expect(screen.getByText("processing")).toBeInTheDocument();
    expect(screen.getByText("failed")).toBeInTheDocument();
  });

  it("shows test_type in each row", () => {
    render(<JobsTable jobs={mockJobs} dlqCount={0} onViewDlq={vi.fn()} />);
    expect(screen.getAllByText("gre").length).toBeGreaterThan(0);
  });
});

describe("JobsTable — failed jobs", () => {
  it("shows error_stage for failed jobs", () => {
    render(<JobsTable jobs={mockJobs} dlqCount={0} onViewDlq={vi.fn()} />);
    expect(screen.getByText(/embed/i)).toBeInTheDocument();
  });

  it("does not show error_stage for successful jobs", () => {
    const completedOnly: IngestionJob[] = [mockJobs[0]];
    render(<JobsTable jobs={completedOnly} dlqCount={0} onViewDlq={vi.fn()} />);
    expect(screen.queryByText(/embed|parse|chunk|index/i)).not.toBeInTheDocument();
  });
});

describe("JobsTable — DLQ badge", () => {
  it("shows DLQ count badge when dlqCount > 0", () => {
    render(<JobsTable jobs={mockJobs} dlqCount={3} onViewDlq={vi.fn()} />);
    expect(screen.getByText(/3/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /dead.letter|dlq/i })).toBeInTheDocument();
  });

  it("calls onViewDlq when DLQ button clicked", async () => {
    const onViewDlq = vi.fn();
    render(<JobsTable jobs={mockJobs} dlqCount={2} onViewDlq={onViewDlq} />);
    await userEvent.click(screen.getByRole("button", { name: /dead.letter|dlq/i }));
    expect(onViewDlq).toHaveBeenCalledOnce();
  });

  it("does not show DLQ button when dlqCount is 0", () => {
    render(<JobsTable jobs={mockJobs} dlqCount={0} onViewDlq={vi.fn()} />);
    expect(screen.queryByRole("button", { name: /dead.letter|dlq/i })).not.toBeInTheDocument();
  });
});

describe("JobsTable — empty state", () => {
  it("shows empty message when no jobs", () => {
    render(<JobsTable jobs={[]} dlqCount={0} onViewDlq={vi.fn()} />);
    expect(screen.getByText(/no jobs/i)).toBeInTheDocument();
  });
});
