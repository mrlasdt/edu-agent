import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError } from "../request";
import { saveReadingAnswers, submitReading } from "./readingApi";
import type { Passage, Question, ReadingAttempt } from "./readingTypes";
import { clock } from "./format";
import { SampleBadge } from "./SampleBadge";
import { useCountdown } from "./useCountdown";

export const READING_AUTOSAVE_MS = 10_000;

interface Props {
  attempt: ReadingAttempt;
  onSubmitted: (attempt: ReadingAttempt) => void;
  autosaveMs?: number;
}

type Save = "idle" | "saving" | "saved" | "error";

/** The input for one question: a radio group when it has choices, else a short text box. */
export function Field({
  q,
  value,
  onChange,
}: {
  q: Pick<Question, "id" | "number" | "task_type" | "choices">;
  value: string;
  onChange: (v: string) => void;
}) {
  const choices = Object.entries(q.choices);
  if (choices.length === 0) {
    return (
      <input
        aria-label={`Answer to question ${q.number}`}
        value={value}
        maxLength={80}
        autoComplete="off"
        onChange={(e) => onChange(e.target.value)}
      />
    );
  }
  return (
    <div role="radiogroup" aria-label={`Answer to question ${q.number}`} className="reading-choices">
      {choices.map(([key, text]) => (
        <label key={key}>
          <input type="radio" name={q.id} value={key} checked={value === key} onChange={() => onChange(key)} />{" "}
          {q.task_type === "tfng" ? text : `${key}. ${text}`}
        </label>
      ))}
    </div>
  );
}

function PassageBlock({ passage }: { passage: Passage }) {
  return (
    <article className="reading-passage">
      <h2>{passage.title}</h2>
      {passage.paragraphs.map((p, i) => (
        <p key={i}>
          <strong className="reading-para">{i + 1}</strong> {p}
        </p>
      ))}
    </article>
  );
}

export function ReadingSit({ attempt, onSubmitted, autosaveMs = READING_AUTOSAVE_MS }: Props) {
  const paper = attempt.paper!;
  const [answers, setAnswers] = useState<Record<string, string>>(attempt.answers);
  const [save, setSave] = useState<Save>("idle");
  const [confirming, setConfirming] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const latest = useRef(answers);
  latest.current = answers;
  const saved = useRef(JSON.stringify(attempt.answers));
  const sent = useRef(false);

  const submit = async () => {
    if (sent.current) return;
    sent.current = true;
    setSubmitting(true);
    setError(null);
    try {
      onSubmitted(await submitReading(attempt.id, latest.current));
    } catch (err) {
      sent.current = false;
      setSubmitting(false);
      setError(err instanceof Error ? err.message : "Could not submit");
    }
  };

  const { left, sync } = useCountdown(attempt.remaining_s ?? attempt.allowed_s, () => void submit());

  useEffect(() => {
    const id = window.setInterval(async () => {
      const body = JSON.stringify(latest.current);
      if (body === saved.current || sent.current) return;
      setSave("saving");
      try {
        const view = await saveReadingAnswers(attempt.id, latest.current);
        saved.current = body;
        sync(view.remaining_s);
        setSave("saved");
      } catch (err) {
        // a 409 means the time is up and the countdown is about to submit
        setSave(err instanceof ApiError && err.status === 409 ? "idle" : "error");
      }
    }, autosaveMs);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attempt.id, autosaveMs]);

  const answered = Object.values(answers).filter((v) => v.trim()).length;
  const total = paper.questions.length;
  const byPassage = useMemo(
    () => paper.passages.map((p) => ({ p, qs: paper.questions.filter((q) => q.passage_id === p.id) })),
    [paper]
  );
  const set = (id: string, v: string) => setAnswers((a) => ({ ...a, [id]: v }));

  return (
    <div className="ielts-writing reading-sit">
      <div className="ielts-bar">
        <div className={`ielts-clock${left <= 300 ? " ielts-clock--urgent" : ""}`}>
          <span className="ielts-clock-digits" role="timer" aria-label="Time left">
            {clock(left)}
          </span>
        </div>
        <div className="ielts-meta">
          {attempt.sample && <SampleBadge />}
          <span className="ielts-words">
            {answered} of {total} answered
          </span>
          <span className="ielts-save" data-state={save} role="status">
            {save === "saving" && "Saving..."}
            {save === "saved" && "Saved"}
            {save === "error" && "Could not save - will retry"}
          </span>
        </div>
      </div>
      {left === 0 && <p className="ielts-timeup">Time is up - submitting your answers.</p>}
      {byPassage.map(({ p, qs }) => (
        <section key={p.id} className="reading-block" aria-label={p.title}>
          <PassageBlock passage={p} />
          <ol className="reading-questions">
            {qs.map((q, i) => (
              <li key={q.id} value={q.number}>
                {(i === 0 || qs[i - 1].task_type !== q.task_type) && (
                  <p className="ielts-eyebrow">
                    {q.task_label}
                    {q.instruction ? ` - ${q.instruction}` : ""}
                  </p>
                )}
                <p>
                  {q.paragraph ? <em>Paragraph {q.paragraph}. </em> : null}
                  {q.prompt}
                </p>
                <Field q={q} value={answers[q.id] ?? ""} onChange={(v) => set(q.id, v)} />
              </li>
            ))}
          </ol>
        </section>
      ))}
      {error && (
        <p className="ielts-error" role="alert">
          {error}
        </p>
      )}
      <div className="ielts-actions">
        {confirming ? (
          <>
            <span className="ielts-confirm">
              Submit {answered} of {total} answers? You cannot change them afterwards.
            </span>
            <button className="ielts-primary" onClick={() => void submit()} disabled={submitting}>
              {submitting ? "Marking..." : "Yes, submit"}
            </button>
            <button onClick={() => setConfirming(false)} disabled={submitting}>
              Keep working
            </button>
          </>
        ) : (
          <button className="ielts-primary" onClick={() => setConfirming(true)} disabled={submitting}>
            Submit answers
          </button>
        )}
      </div>
    </div>
  );
}
