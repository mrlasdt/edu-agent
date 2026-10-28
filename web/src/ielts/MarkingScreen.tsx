import type { Attempt } from "./types";

interface Props {
  attempt: Attempt;
  retrying: boolean;
  onRetry: () => void;
  onBack: () => void;
}

export function MarkingScreen({ attempt, retrying, onRetry, onBack }: Props) {
  if (attempt.status === "ungraded") {
    return (
      <div className="ielts-marking" role="alert">
        <h2>Your essay could not be marked</h2>
        <p>{attempt.error ?? "Marking did not finish."}</p>
        <p className="ielts-hint">Your essay is saved. {attempt.retryable ? "Marking can be retried." : ""}</p>
        <div className="ielts-actions">
          {attempt.retryable && (
            <button className="ielts-primary" onClick={onRetry} disabled={retrying}>
              {retrying ? "Retrying..." : "Try marking again"}
            </button>
          )}
          <button onClick={onBack}>Choose another question</button>
        </div>
      </div>
    );
  }
  return (
    <div className="ielts-marking" role="status" aria-live="polite">
      <div className="ielts-spinner" aria-hidden="true" />
      <h2>Marking your essay</h2>
      <p className="ielts-hint">
        Three independent marking passes and a written review. This usually takes under a minute.
      </p>
      {attempt.late && (
        <p className="ielts-hint">Submitted after the time limit; the report will say so.</p>
      )}
    </div>
  );
}
