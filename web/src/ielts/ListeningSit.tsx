import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError } from "../request";
import { saveListeningAnswers, submitListening } from "./listeningApi";
import type { ListeningAttempt, Section } from "./listeningTypes";
import { clock, clockWarnAtS } from "./format";
import { Field } from "./ReadingSit";
import { SampleBadge } from "./SampleBadge";
import { useCountdown } from "./useCountdown";

export const LISTENING_AUTOSAVE_MS = 10_000;

interface Props {
  attempt: ListeningAttempt;
  onSubmitted: (attempt: ListeningAttempt) => void;
  autosaveMs?: number;
}

type Save = "idle" | "saving" | "saved" | "error";

function Recording({ section }: { section: Section }) {
  return (
    <div className="listening-recording">
      <h2>
        Section {section.number}: {section.title}
      </h2>
      <p className="ielts-hint">{section.context}</p>
      <audio
        controls
        preload="metadata"
        src={section.audio_url}
        aria-label={`Recording for section ${section.number}`}
      />
    </div>
  );
}

export function ListeningSit({ attempt, onSubmitted, autosaveMs = LISTENING_AUTOSAVE_MS }: Props) {
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
      onSubmitted(await submitListening(attempt.id, latest.current));
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
        const view = await saveListeningAnswers(attempt.id, latest.current);
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
  const bySection = useMemo(
    () => paper.sections.map((s) => ({ s, qs: paper.questions.filter((q) => q.section_id === s.id) })),
    [paper]
  );
  const set = (id: string, v: string) => setAnswers((a) => ({ ...a, [id]: v }));

  return (
    <div className="ielts-writing reading-sit listening-sit">
      <div className="ielts-bar">
        <div className={`ielts-clock${left <= clockWarnAtS(attempt.sample) ? " ielts-clock--urgent" : ""}`}>
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
      {bySection.map(({ s, qs }) => (
        <section key={s.id} className="reading-block" aria-label={`Section ${s.number}`}>
          <Recording section={s} />
          <ol className="reading-questions">
            {qs.map((q, i) => (
              <li key={q.id} value={q.number}>
                {(i === 0 || qs[i - 1].task_type !== q.task_type) && (
                  <p className="ielts-eyebrow">
                    {q.task_label}
                    {q.instruction ? ` - ${q.instruction}` : ""}
                  </p>
                )}
                <p>{q.prompt}</p>
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
