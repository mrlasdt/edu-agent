import type { Attempt } from "./types";

interface Props {
  attempt: Attempt;
  busy: boolean;
  error: string | null;
  onStart: () => void;
  onBack: () => void;
}

export function ReadyScreen({ attempt, busy, error, onStart, onBack }: Props) {
  return (
    <div className="ielts-ready">
      <p className="ielts-eyebrow">Writing Task 2</p>
      <blockquote className="ielts-prompt-quote">{attempt.prompt}</blockquote>
      <ul className="ielts-rules">
        <li>You have 40 minutes. The clock runs on the server, so refreshing the page does not reset it.</li>
        <li>Write at least 250 words. Your essay autosaves about every 10 seconds.</li>
        <li>When time is up, what you have written is submitted for you.</li>
      </ul>
      {error && (
        <p className="ielts-error" role="alert">
          {error}
        </p>
      )}
      <div className="ielts-actions">
        <button className="ielts-primary" onClick={onStart} disabled={busy}>
          {busy ? "Starting..." : "Start the 40 minutes"}
        </button>
        <button onClick={onBack} disabled={busy}>
          Choose another question
        </button>
      </div>
    </div>
  );
}
