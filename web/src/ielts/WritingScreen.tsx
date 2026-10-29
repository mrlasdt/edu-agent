import { useEffect, useRef, useState } from "react";
import { submitAttempt } from "./api";
import { clock, countWords } from "./format";
import { ESSAY_MINIMUM_WORDS } from "./types";
import type { Attempt } from "./types";
import { useAutosave } from "./useAutosave";
import type { SaveState } from "./useAutosave";
import { useCountdown } from "./useCountdown";

interface Props {
  attempt: Attempt;
  initialText: string;
  onSubmitted: (attempt: Attempt) => void;
  autosaveMs?: number;
}

const LOW_WORDS = 150;
const WARN_AT_S = 5 * 60;
const ANNOUNCE_AT_S = [600, 300, 60];
const TIME_UP_NOTICE =
  "Time is up. The last autosave at the deadline is what will be marked; words not yet autosaved are not.";

function saveLabel(state: SaveState): string {
  switch (state.kind) {
    case "idle":
      return "Autosave on";
    case "saving":
      return "Saving...";
    case "saved":
      return `Saved ${state.at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
    case "error":
      return `Not saved: ${state.message}. Retrying`;
  }
}

export function WritingScreen({ attempt, initialText, onSubmitted, autosaveMs }: Props) {
  const [text, setText] = useState(initialText);
  const [confirming, setConfirming] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [announcement, setAnnouncement] = useState("");
  const textRef = useRef(text);
  textRef.current = text;
  const submitted = useRef(false);

  const countdown = useCountdown(attempt.remaining_s ?? attempt.allowed_s, () => {
    void submit(true);
  });
  const autosave = useAutosave(
    attempt.id,
    text,
    attempt.draft,
    (view) => countdown.sync(view.remaining_s),
    autosaveMs
  );

  async function submit(expired = false) {
    if (submitted.current) return;
    submitted.current = true;
    setSubmitting(true);
    setError(null);
    try {
      onSubmitted(await submitAttempt(attempt.id, textRef.current));
    } catch (err) {
      submitted.current = false;
      setSubmitting(false);
      setConfirming(false);
      const reason = err instanceof Error ? err.message : "Could not submit";
      setError(
        expired || countdown.left <= 0
          ? `${reason}. Press Submit to send your last autosave at the deadline for marking.`
          : `${reason}. Your essay is still here: try again.`
      );
    }
  }

  useEffect(() => {
    if (ANNOUNCE_AT_S.includes(countdown.left)) {
      setAnnouncement(`${countdown.left / 60} ${countdown.left === 60 ? "minute" : "minutes"} left`);
    }
  }, [countdown.left]);

  const words = countWords(text);
  const low = words < LOW_WORDS;
  const urgent = countdown.left <= WARN_AT_S;
  const timeUp = countdown.left <= 0;

  return (
    <section className="ielts-writing" aria-label="Writing task">
      <header className="ielts-bar">
        <div
          className={`ielts-clock${urgent ? " ielts-clock--urgent" : ""}`}
          role="timer"
          aria-label="Time left"
        >
          <span className="ielts-clock-digits">{clock(countdown.left)}</span>
          <span className="ielts-clock-label">{urgent ? "left - wrap up" : "left"}</span>
        </div>
        <div className="ielts-meta">
          <span className="ielts-words" aria-label="Word count">
            {words} words
            {words < ESSAY_MINIMUM_WORDS && <span className="ielts-hint"> / {ESSAY_MINIMUM_WORDS} minimum</span>}
          </span>
          <span className="ielts-save" data-state={autosave.state.kind}>
            {saveLabel(autosave.state)}
          </span>
        </div>
      </header>
      <p className="ielts-sr" role="status" aria-live="polite">
        {announcement}
      </p>

      <div className="ielts-prompt" aria-label="Question">
        <p className="ielts-eyebrow">Writing Task 2</p>
        <p>{attempt.prompt}</p>
        <p className="ielts-hint">Write at least {ESSAY_MINIMUM_WORDS} words. Give reasons and relevant examples.</p>
      </div>

      {timeUp && <p className="ielts-timeup">{TIME_UP_NOTICE}</p>}

      <textarea
        className="ielts-editor"
        aria-label="Your essay"
        value={text}
        onChange={(e) => setText(e.target.value)}
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
              {low
                ? `Only ${words} words - well under the ${ESSAY_MINIMUM_WORDS} minimum. Submit anyway?`
                : "Submit your essay for marking? You cannot edit it afterwards."}
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
            disabled={submitting || words === 0}
          >
            Submit for marking
          </button>
        )}
      </footer>
    </section>
  );
}
