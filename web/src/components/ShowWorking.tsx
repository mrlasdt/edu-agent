import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { fetchTurn } from "../chat";
import type { TurnToolCall, TurnView } from "../types";

// The panel is a list of sections; a later slice adds one by appending to `buildSections`.
interface Section {
  id: string;
  title: string;
  body: ReactNode;
}

const money = (usd: number) => `$${usd < 0.01 ? usd.toFixed(5) : usd.toFixed(3)}`;
const seconds = (ms: number | null) => (ms === null ? "n/a" : `${(ms / 1000).toFixed(1)} s`);

function UsageSection({ turn }: { turn: TurnView }) {
  const guardrails = turn.calls.filter((c) => c.source === "guardrails").length;
  return (
    <dl className="working-rows">
      <dt>Model calls</dt>
      <dd>
        {turn.llm_calls}
        {guardrails > 0 && ` (${guardrails} safety check${guardrails === 1 ? "" : "s"})`}
      </dd>
      <dt>Tokens</dt>
      <dd>
        {turn.input_tokens} in, {turn.output_tokens} out{turn.estimated ? " (estimated)" : ""}
      </dd>
      <dt>Cost</dt>
      <dd>{money(turn.cost_usd)}</dd>
      <dt>Time</dt>
      <dd>{seconds(turn.total_ms)}</dd>
    </dl>
  );
}

// A tool call as one line: the model's input and what the tool returned.
function parsed(text: string | null): unknown {
  try {
    return text === null ? null : JSON.parse(text);
  } catch {
    return text;
  }
}

function firstString(value: unknown): string | null {
  if (typeof value === "string") return value;
  if (typeof value === "object" && value !== null) {
    for (const v of Object.values(value)) if (typeof v === "string") return v;
  }
  return null;
}

function resultText(result: string | null): string {
  const value = parsed(result);
  if (typeof value !== "object" || value === null) return result ?? "";
  const data = value as Record<string, unknown>;
  if (typeof data.error === "string") return `error: ${data.error}`;
  if (Array.isArray(data.roots) && data.roots.length) return `roots ${data.roots.join(", ")}`;
  if (typeof data.value === "string") return data.value;
  if (typeof data.verdict === "string") return `verdict ${data.verdict}`;
  return result ?? "";
}

function toolLine(call: TurnToolCall): string {
  const input = firstString(parsed(call.arguments)) ?? call.arguments ?? "";
  return `${call.name} ${input} -> ${call.ok ? resultText(call.result) : "failed"}`;
}

function CalculatorSection({
  turn,
  onReveal,
  revealing,
}: {
  turn: TurnView;
  onReveal: () => void;
  revealing: boolean;
}) {
  const n = turn.tool_calls.length;
  if (!turn.math_tools) return <p>Calculator not used.</p>;
  if (n === 0) return <p>Calculator on, but it was not needed for this reply.</p>;
  if (turn.tools_hidden) {
    return (
      <p>
        Calculator: used {n} time{n === 1 ? "" : "s"}. Result hidden in tutor mode so it cannot
        spoil the hint.{" "}
        <button type="button" className="reveal-btn" onClick={onReveal} disabled={revealing}>
          Reveal
        </button>
      </p>
    );
  }
  return (
    <ul className="working-tools">
      {turn.tool_calls.map((c, i) => (
        <li key={i}>
          <code>{toolLine(c)}</code>
        </li>
      ))}
    </ul>
  );
}

function CheckSection({ turn }: { turn: TurnView }) {
  const check = turn.calc_check;
  if (check?.verdict === "agrees") {
    return (
      <p className="calc-check calc-check--agrees">
        Calculator agrees: answer {check.answer} = {check.tool_value}
      </p>
    );
  }
  if (check?.verdict === "differs") {
    return (
      <p className="calc-check calc-check--differs">
        Calculator differs: answer {check.answer}, calculator {check.tool_value}. Check the working
        yourself.
      </p>
    );
  }
  return <p className="calc-check calc-check--none">Check: not calculator-checked</p>;
}

export function buildSections(
  turn: TurnView,
  reveal: { onReveal: () => void; revealing: boolean }
): Section[] {
  return [
    { id: "usage", title: "This reply", body: <UsageSection turn={turn} /> },
    {
      id: "calculator",
      title: "Calculator",
      body: <CalculatorSection turn={turn} {...reveal} />,
    },
    { id: "check", title: "Check", body: <CheckSection turn={turn} /> },
  ];
}

/** A collapsed "Show working" panel under an agent reply; it loads the turn's record on first open. */
export function ShowWorking({ traceId }: { traceId: string }) {
  const [open, setOpen] = useState(false);
  const [turn, setTurn] = useState<TurnView | null>(null);
  const [failed, setFailed] = useState(false);
  const [revealing, setRevealing] = useState(false);

  useEffect(() => {
    if (!open || turn || failed) return;
    const controller = new AbortController();
    fetchTurn(traceId, { signal: controller.signal })
      .then((t) => (t ? setTurn(t) : setFailed(true)))
      .catch(() => undefined);
    return () => controller.abort();
  }, [open, turn, failed, traceId]);

  const reveal = async () => {
    setRevealing(true);
    const revealed = await fetchTurn(traceId, { reveal: true }).catch(() => null);
    if (revealed) setTurn(revealed);
    setRevealing(false);
  };

  return (
    <details
      className="show-working"
      onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}
    >
      <summary>Show working</summary>
      {failed && <p>The working for this reply is not available.</p>}
      {!turn && !failed && open && <p role="status">Loading...</p>}
      {turn &&
        buildSections(turn, { onReveal: () => void reveal(), revealing }).map((s) => (
          <section key={s.id} className={`working-section working-section--${s.id}`}>
            <h3>{s.title}</h3>
            {s.body}
          </section>
        ))}
    </details>
  );
}
