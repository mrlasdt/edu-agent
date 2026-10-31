import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ReportView } from "../ReportView";
import { ESSAY, report } from "./fixtures";

describe("ReportView", () => {
  it("leads with the mock score, its range, and the fixed disclaimer", () => {
    render(<ReportView report={report()} essay={ESSAY} onAnother={vi.fn()} />);
    const hero = screen.getByRole("region", { name: "Mock score" });
    expect(within(hero).getByLabelText("Mock overall score 6.0")).toBeInTheDocument();
    expect(hero).toHaveTextContent("between 5.5 and 6.5");
    expect(hero).toHaveTextContent("Mock score: an estimate, not an official IELTS result.");
  });

  it("says 'Uncalibrated estimate' until the calibration gate has passed", () => {
    render(<ReportView report={report()} essay={ESSAY} onAnother={vi.fn()} />);
    expect(screen.getByText("Uncalibrated estimate")).toBeInTheDocument();
  });

  it("shows the calibration evidence once it has passed", () => {
    const r = report();
    r.estimated_band!.calibration = {
      status: "calibrated",
      label: "Calibrated on 60 essays (captain_blind), 85% within half a band.",
      n: 60,
      tier: "captain_blind",
      within_half_band: 0.85,
      note: "",
    };
    render(<ReportView report={r} essay={ESSAY} onAnother={vi.fn()} />);
    expect(screen.queryByText("Uncalibrated estimate")).not.toBeInTheDocument();
    expect(screen.getByText(/Calibrated on 60 essays/)).toBeInTheDocument();
  });

  it("gives each criterion a card with evidence, a next step and actions", () => {
    render(<ReportView report={report()} essay={ESSAY} onAnother={vi.fn()} />);
    const tr = screen.getByRole("article", { name: "Task Response" });
    expect(tr).toHaveTextContent("5.5");
    expect(tr).toHaveTextContent("To reach 6.0.");
    expect(within(tr).getByText("Learning to cook at school is useful")).toBeInTheDocument();
    expect(within(tr).getByText(/TR do this/)).toBeInTheDocument();
    const gra = screen.getByRole("article", { name: "Grammatical Range and Accuracy" });
    expect(gra).toHaveTextContent("Agreement x1");
  });

  it("labels a strong quote as a strength, not as support for the band", () => {
    const r = report();
    r.criteria[0].band = 4;
    r.criteria[0].capped_from = 7;
    render(<ReportView report={r} essay={ESSAY} onAnother={vi.fn()} />);
    const tr = screen.getByRole("article", { name: "Task Response" });
    expect(tr).toHaveTextContent("a strength - paragraph 1");
    expect(tr).not.toHaveTextContent("supports the band");
  });

  it("shows an ellipsis after a quote that was shortened", () => {
    const r = report();
    r.criteria[0].evidence[0] = { ...r.criteria[0].evidence[0], cut: true };
    render(<ReportView report={r} essay={ESSAY} onAnother={vi.fn()} />);
    const tr = screen.getByRole("article", { name: "Task Response" });
    expect(within(tr).getByText("Learning to cook at school is useful...")).toBeInTheDocument();
  });

  it("explains low confidence as an override by the automatic checks", () => {
    const r = report();
    r.estimated_band!.confidence = "low";
    render(<ReportView report={r} essay={ESSAY} onAnother={vi.fn()} />);
    expect(screen.getByText(/automatic check on the essay overrode them/)).toBeInTheDocument();
  });

  it("lists the top priorities, warnings, the previous-attempt delta and the checklist", () => {
    render(<ReportView report={report()} essay={ESSAY} onAnother={vi.fn()} />);
    expect(screen.getByRole("region", { name: "Fix these first" })).toHaveTextContent(
      "State your position early"
    );
    expect(screen.getByText(/well under the 250-word minimum/)).toBeInTheDocument();
    expect(screen.getByText(/overall \+0.5/)).toHaveTextContent(
      "Change in mock score since your last marked attempt: overall +0.5",
    );
    expect(screen.getByText("at least 250 words")).toBeInTheDocument();
    expect(screen.getByText(/not assessed/)).toBeInTheDocument();
  });

  it("marks the located error spans in the candidate's own essay", () => {
    render(<ReportView report={report()} essay={ESSAY} onAnother={vi.fn()} />);
    const essay = screen.getByRole("region", { name: "Your essay, marked" });
    expect(within(essay).getAllByText("students has", { exact: false })[0].tagName).toBe("MARK");
  });

  it("does not highlight a span whose text no longer matches the essay", () => {
    const r = report();
    r.error_spans[0].quote = "something else";
    render(<ReportView report={r} essay={ESSAY} onAnother={vi.fn()} />);
    expect(document.querySelector("mark")).toBeNull();
  });

  it("explains an ungraded response instead of showing a band", () => {
    const r = report({
      status: "ungraded",
      estimated_band: null,
      criteria: [],
      error_spans: [],
      task_checks: [],
      top_priorities: [],
      vs_previous_attempt: null,
      ungraded_reason: "The response is too short to mark (11 words).",
    });
    render(<ReportView report={r} essay="Cooking is good." onAnother={vi.fn()} />);
    expect(screen.getByText(/too short to mark/)).toBeInTheDocument();
    expect(screen.queryByLabelText(/Mock overall score/)).not.toBeInTheDocument();
    expect(screen.getByText("Mock score: an estimate, not an official IELTS result.")).toBeInTheDocument();
  });

  it("says when the written feedback is unavailable and notes a late submission", () => {
    const r = report({
      feedback_status: "unavailable",
      criteria: [],
      timing: { allowed_s: 2400, used_s: 2400, late: true },
    });
    render(<ReportView report={r} essay={ESSAY} onAnother={vi.fn()} />);
    expect(screen.getByText(/written feedback was unavailable/)).toBeInTheDocument();
    expect(screen.getByText(/after time was up, so the last autosave at the deadline was marked/)).toBeInTheDocument();
  });

  it("starts another essay", async () => {
    const onAnother = vi.fn();
    render(<ReportView report={report()} essay={ESSAY} onAnother={onAnother} />);
    await userEvent.click(screen.getByRole("button", { name: "Write another essay" }));
    expect(onAnother).toHaveBeenCalled();
  });
});
