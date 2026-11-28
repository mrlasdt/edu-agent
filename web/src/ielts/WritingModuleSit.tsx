import { useEffect, useRef, useState } from "react";
import { ApiError } from "../request";
import { clock, clockWarnAtS, countWords } from "./format";
import { SampleBadge } from "./SampleBadge";
import { saveWritingModuleDraft, submitWritingModule } from "./writingModuleApi";
import {
  ESSAY_MINIMUM_WORDS,
  LETTER_MINIMUM_WORDS,
  SAMPLE_MINIMUM_WORDS,
  type ModuleTexts,
  type TaskKey,
  type WritingModuleAttempt,
} from "./writingModuleTypes";
import { useCountdown } from "./useCountdown";

interface Props {
  attempt: WritingModuleAttempt;
  onSubmitted: (attempt: WritingModuleAttempt) => void;
  autosaveMs?: number;
}

const SAVE_MS = 10_000;
const TASKS: { key: TaskKey; tab: string; label: string; min: number; minutes: number }[] = [
  { key: "t1", tab: "Task 1 - Letter", label: "Your letter", min: LETTER_MINIMUM_WORDS, minutes: 20 },
  { key: "t2", tab: "Task 2 - Essay", label: "Your essay", min: ESSAY_MINIMUM_WORDS, minutes: 40 },
];

/** Both tasks on one clock: switch between the letter and the essay, autosave both texts. */
export function WritingModuleSit({ attempt, onSubmitted, autosaveMs = SAVE_MS }: Props) {
  const [texts, setTexts] = useState<ModuleTexts>({ t1: attempt.t1.text, t2: attempt.t2.text });
  const [active, setActive] = useState<TaskKey>("t1");
  const [confirming, setConfirming] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveState, setSaveState] = useState("Autosave on");
  const latest = useRef(texts);
  latest.current = texts;
  const saved = useRef(JSON.stringify(texts));
  const submitted = useRef(false);

  const countdown = useCountdown(attempt.remaining_s ?? attempt.allowed_s, () => void submit(true));

  async function flush() {
    const body = latest.current;
    const key = JSON.stringify(body);
    if (key === saved.current || submitted.current) return;
    setSaveState("Saving...");
    try {
      const view = await saveWritingModuleDraft(attempt.id, body);
      saved.current = key;
      countdown.sync(view.remaining_s);
      setSaveState(`Saved ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`);
    } catch (err) {
      setSaveState(`Not saved: ${err instanceof ApiError ? err.message : "could not save"}. Retrying`);
    }
  }
  const flushRef = useRef(flush);
  flushRef.current = flush;
  useEffect(() => {
    const id = window.setInterval(() => void flushRef.current(), autosaveMs);
    return () => window.clearInterval(id);
  }, [autosaveMs]);

  async function submit(expired = false) {
    if (submitted.current) return;
    submitted.current = true;
    setSubmitting(true);
    setError(null);
    try {
      onSubmitted(await submitWritingModule(attempt.id, latest.current));
    } catch (err) {
      submitted.current = false;
      setSubmitting(false);
      setConfirming(false);
      const reason = err instanceof Error ? err.message : "Could not submit";
      setError(
        expired
          ? `${reason}. Press Submit to send your last autosave at the deadline for marking.`
          : `${reason}. Your writing is still here: try again.`
      );
    }
  }

  const task = TASKS.find((t) => t.key === active)!;
  const view = active === "t1" ? attempt.t1 : attempt.t2;
  const words = { t1: countWords(texts.t1), t2: countWords(texts.t2) };
  const urgent = countdown.left <= clockWarnAtS(attempt.sample);
  const timeUp = countdown.left <= 0;
  const minOf = (t: (typeof TASKS)[number]) => (attempt.sample ? SAMPLE_MINIMUM_WORDS : t.min);
  const short = TASKS.filter((t) => words[t.key] < minOf(t));

  return (
    <section className="ielts-writing" aria-label="Writing module">
      <header className="ielts-bar">
        <div className={`ielts-clock${urgent ? " ielts-clock--urgent" : ""}`} role="timer" aria-label="Time left">
          <span className="ielts-clock-digits">{clock(countdown.left)}</span>
          <span className="ielts-clock-label">{urgent ? "left - wrap up" : "left"}</span>
        </div>
        <div className="ielts-meta">
          {attempt.sample && <SampleBadge />}
          <span className="ielts-words" aria-label="Word count">
            Letter {words.t1}/{minOf(TASKS[0])} - Essay {words.t2}/{minOf(TASKS[1])} words
          </span>
          <span className="ielts-save">{saveState}</span>
        </div>
      </header>

      <div className="ielts-chips" role="tablist" aria-label="Writing tasks">
        {TASKS.map((t) => (
          <button
            key={t.key}
            role="tab"
            className="ielts-chip"
            aria-selected={active === t.key}
            aria-current={active === t.key ? "true" : undefined}
            onClick={() => setActive(t.key)}
          >
            {t.tab}
          </button>
        ))}
      </div>

      <div className="ielts-prompt" aria-label="Question">
        <p className="ielts-eyebrow">
          Writing {task.tab.split(" - ")[0]}{attempt.sample ? " (sample)" : ` (about ${task.minutes} minutes)`}
        </p>
        <p style={{ whiteSpace: "pre-line" }}>{view.prompt}</p>
        <p className="ielts-hint">{attempt.sample ? `Sample target: about ${SAMPLE_MINIMUM_WORDS} words (a short answer is enough to be marked).` : `Write at least ${task.min} words.`} One {clock(attempt.allowed_s)} clock covers both tasks.</p>
      </div>

      {timeUp && (
        <p className="ielts-timeup">
          Time is up. The last autosave at the deadline is what will be marked; words not yet autosaved are not.
        </p>
      )}

      <textarea
        key={active}
        className="ielts-editor"
        aria-label={task.label}
        value={texts[active]}
        onChange={(e) => setTexts({ ...texts, [active]: e.target.value })}
        readOnly={submitting || timeUp}
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        placeholder="Start writing here. Spell-check is off, as in the exam."
      />

      {error && (
        <p className="ielts-error" role="alert">
          {error}
        </p>
      )}

      <footer className="ielts-actions">
        {confirming ? (
          <>
            <span className="ielts-confirm">
              {short.length > 0
                ? `Under the minimum: ${short.map((t) => `${t.key === "t1" ? "letter" : "essay"} ${words[t.key]}/${t.min}`).join(", ")}. Submit anyway?`
                : "Submit both tasks for marking? You cannot edit them afterwards."}
            </span>
            <button className="ielts-primary" onClick={() => void submit()} disabled={submitting}>
              {submitting ? "Submitting..." : "Yes, submit"}
            </button>
            <button onClick={() => setConfirming(false)} disabled={submitting}>
              Keep writing
            </button>
          </>
        ) : (
          <button
            className="ielts-primary"
            onClick={() => setConfirming(true)}
            disabled={submitting || words.t1 + words.t2 === 0}
          >
            Submit both tasks for marking
          </button>
        )}
      </footer>
    </section>
  );
}
