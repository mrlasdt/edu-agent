import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ShowWorking } from "../components/ShowWorking";
import { CalculatorSwitch } from "../components/CalculatorSwitch";
import type { TurnView } from "../types";

const base: TurnView = {
  trace_id: "t1",
  agent: "quant",
  mode: "solve",
  llm_calls: 3,
  input_tokens: 120,
  output_tokens: 40,
  estimated: false,
  cost_usd: 0.0021,
  total_ms: 2400,
  calls: [
    { source: "guardrails", model: "m", input_tokens: 10, output_tokens: 1, estimated: false, cost_usd: 0, ms: 50 },
  ],
  math_tools: false,
  tool_calls: [],
  tools_hidden: false,
  calc_check: null,
};

const withCalculator: TurnView = {
  ...base,
  math_tools: true,
  tool_calls: [
    {
      name: "math_evaluate",
      ok: true,
      ms: 5,
      round: 1,
      arguments: '{"expression":"(1+15/100)*(1-20/100)"}',
      result: '{"value":"0.92"}',
    },
  ],
  calc_check: { verdict: "agrees", answer: "-8%", tool_value: "0.92", tool: "math_evaluate" },
};

function serve(...bodies: Array<TurnView | null>) {
  const fetchMock = vi.fn();
  for (const body of bodies) {
    fetchMock.mockResolvedValueOnce(
      body
        ? { ok: true, status: 200, json: async () => body }
        : { ok: false, status: 404, json: async () => ({ detail: "turn not recorded yet" }) }
    );
  }
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

async function open() {
  await userEvent.click(screen.getByText("Show working"));
}

afterEach(() => vi.unstubAllGlobals());

describe("ShowWorking", () => {
  it("shows calls, tokens, cost and time, and says the calculator was not used", async () => {
    serve(base);
    render(<ShowWorking traceId="t1" />);
    await open();
    expect(await screen.findByText("Calculator not used.")).toBeInTheDocument();
    expect(screen.getByText("3 (1 safety check)")).toBeInTheDocument();
    expect(screen.getByText("120 in, 40 out")).toBeInTheDocument();
    expect(screen.getByText("$0.00210")).toBeInTheDocument();
    expect(screen.getByText("2.4 s")).toBeInTheDocument();
    expect(screen.getByText("Check: not calculator-checked")).toBeInTheDocument();
  });

  it("lists the calculator calls and the verdict", async () => {
    serve(withCalculator);
    render(<ShowWorking traceId="t1" />);
    await open();
    expect(await screen.findByText("math_evaluate (1+15/100)*(1-20/100) -> 0.92")).toBeInTheDocument();
    expect(screen.getByText("Calculator agrees: answer -8% = 0.92")).toBeInTheDocument();
  });

  it("says when the calculator differs", async () => {
    serve({ ...withCalculator, calc_check: { verdict: "differs", answer: "-5%", tool_value: "0.92", tool: "math_evaluate" } });
    render(<ShowWorking traceId="t1" />);
    await open();
    expect(await screen.findByText(/Calculator differs: answer -5%, calculator 0.92/)).toBeInTheDocument();
  });

  it("does not fetch until it is opened", () => {
    const fetchMock = serve(base);
    render(<ShowWorking traceId="t1" />);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("asks again when the record is not written yet", async () => {
    const fetchMock = serve(null, null, base);
    render(<ShowWorking traceId="t1" />);
    await open();
    expect(await screen.findByText("Calculator not used.", {}, { timeout: 3000 })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("hides tutor results until Reveal, then asks for them", async () => {
    const hidden: TurnView = {
      ...withCalculator,
      mode: "tutor",
      tools_hidden: true,
      calc_check: null,
      tool_calls: [{ name: "math_solve", ok: true, ms: 5, round: 1, arguments: null, result: null }],
    };
    const revealed: TurnView = {
      ...hidden,
      tools_hidden: false,
      tool_calls: [
        { name: "math_solve", ok: true, ms: 5, round: 1, arguments: '{"equation":"x^2-3x-4=0"}', result: '{"roots":["-1","4"]}' },
      ],
    };
    const fetchMock = serve(hidden, revealed);
    render(<ShowWorking traceId="t1" />);
    await open();
    expect(await screen.findByText(/Calculator: used 1 time\. Result hidden in tutor mode/)).toBeInTheDocument();
    expect(screen.queryByText(/roots/)).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Reveal" }));
    expect(await screen.findByText("math_solve x^2-3x-4=0 -> roots -1, 4")).toBeInTheDocument();
    expect(fetchMock.mock.calls[1][0]).toBe("/turns/t1?reveal=1");
  });

  it("says so when the record never arrives", async () => {
    serve(null, null, null);
    render(<ShowWorking traceId="t1" />);
    await open();
    await waitFor(() => expect(screen.getByText(/not available/)).toBeInTheDocument(), { timeout: 3000 });
  });
});

describe("CalculatorSwitch", () => {
  it("shows its state and toggles", async () => {
    const onToggle = vi.fn();
    render(<CalculatorSwitch on={false} onToggle={onToggle} />);
    const button = screen.getByRole("switch", { name: "Calculator: off" });
    expect(button).toHaveAttribute("aria-checked", "false");
    await userEvent.click(button);
    expect(onToggle).toHaveBeenCalledOnce();
  });
});
