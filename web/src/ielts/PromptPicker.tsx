import { band } from "./format";
import { DISCLAIMER } from "./types";
import type { Attempt, Budget, WritingPrompt } from "./types";

interface Props {
  prompts: WritingPrompt[];
  history: Attempt[];
  budget: Budget | null;
  busy: boolean;
  onPick: (prompt: WritingPrompt) => void;
  onOpen: (attempt: Attempt) => void;
}

const STATUS_TEXT: Record<Attempt["status"], string> = {
  created: "Not started",
  in_progress: "In progress",
  submitted: "Marking",
  grading: "Marking",
  graded: "Marked",
  ungraded: "Not marked",
};

const usd = (n: number) => `$${n.toFixed(2)}`;

export function PromptPicker({ prompts, history, budget, busy, onPick, onOpen }: Props) {
  const exhausted = budget?.exhausted ?? false;
  return (
    <div className="ielts-picker">
      <h2>Choose a Writing Task 2 question</h2>
      <p className="ielts-hint">
        General Training Writing Task 2: 40 minutes, at least 250 words, as on the computer-delivered
        test. The clock starts when you press Start on the next screen.
      </p>
      {budget && (
        <p className="ielts-hint" aria-label="AI spend">
          AI spend so far: {usd(budget.spent_usd)} of the {usd(budget.cap_usd)} cap.
        </p>
      )}
      {exhausted && budget?.message && (
        <p className="ielts-error" role="alert">
          {budget.message}
        </p>
      )}
      <ul className="ielts-prompt-list">
        {prompts.map((p) => (
          <li key={p.id}>
            <button
              className="ielts-prompt-card"
              onClick={() => onPick(p)}
              disabled={busy || exhausted}
              aria-label={`${p.essay_type_label}: ${p.text}`}
            >
              <span className="ielts-chip">{p.essay_type_label}</span>
              <span className="ielts-prompt-text">{p.text}</span>
            </button>
          </li>
        ))}
      </ul>
      {history.length > 0 && (
        <section aria-label="Earlier attempts">
          <h3>Earlier attempts</h3>
          <p className="ielts-disclaimer">{DISCLAIMER}</p>
          <ul className="ielts-history">
            {history.map((a) => (
              <li key={a.id}>
                <button onClick={() => onOpen(a)}>
                  <span className="ielts-history-band">
                    {a.overall !== null ? `Mock score ${band(a.overall)}` : "-"}
                  </span>
                  <span className="ielts-history-text">{a.prompt}</span>
                  <span className="ielts-hint">{STATUS_TEXT[a.status]}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
