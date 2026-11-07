import type { IndependentCheck, Message } from "../types";
import { ESSAY_DISCLAIMER } from "../types";
import { renderMarkdown } from "../markdown";

interface Props {
  message: Message;
  onCitationClick?: (index: number) => void;
  onFeedback?: (vote: "up" | "down") => void;
  onRegenerate?: () => void;
}

function renderContentWithCitations(
  content: string,
  hasDisclaimer: boolean,
  onCitationClick?: (index: number) => void
) {
  return (
    <>
      {renderMarkdown(content, (idx, key) =>
        // The web app has no sources panel yet, so a marker is plain text unless something
        // handles the click: a button that does nothing is worse than no button.
        onCitationClick ? (
          <button
            key={key}
            className="citation-marker"
            aria-label={`[${idx}]`}
            onClick={() => onCitationClick(idx)}
          >
            [{idx}]
          </button>
        ) : (
          <sup key={key} className="citation-marker citation-marker--static">
            [{idx}]
          </sup>
        )
      )}
      {hasDisclaimer && (
        <aside
          data-testid="essay-disclaimer"
          className="essay-disclaimer"
          aria-label="Disclaimer"
        >
          {ESSAY_DISCLAIMER}
        </aside>
      )}
    </>
  );
}

// The independent check covers only questions whose answer is a number a solver can recompute
// (ADR-0012), so every verdict says so: "agrees" is not "the problem was read correctly".
function IndependentCheckNote({ check }: { check: IndependentCheck }) {
  return (
    <p className={`math-check independent-check independent-check--${check.verdict}`} role="note">
      <strong>Independent check: </strong>
      {check.verdict === "agrees" &&
        "agrees. A solver computed the problem from a separately written equation and got the same answer. Numeric answers only."}
      {check.verdict === "disagrees" &&
        `disagrees${check.solver ? ` (solver: ${check.solver})` : ""}. Numeric answers only.`}
      {check.verdict === "not_checkable" &&
        "not checkable. It covers only questions with a numeric answer."}
    </p>
  );
}

export function MessageBubble({ message, onCitationClick, onFeedback, onRegenerate }: Props) {
  const isAgent = message.role === "agent";

  return (
    <div className={`message-bubble message-bubble--${message.role}`}>
      {message.status && (
        <p className="status-line" role="status">
          {message.status}
        </p>
      )}
      <div className="message-content">
        {renderContentWithCitations(message.content, message.disclaimer === "essay", onCitationClick)}
      </div>
      {message.mathVerified !== undefined && (
        <p
          className={`math-check math-check--${message.mathVerified ? "ok" : "fail"}`}
          role="note"
        >
          {message.mathVerified
            ? "Final answer checked by the math verifier. It confirms the answer is well-formed, not that the working is right."
            : "The math verifier could not confirm the final answer. Check the working yourself."}
        </p>
      )}
      {message.independentCheck && <IndependentCheckNote check={message.independentCheck} />}
      {isAgent && onRegenerate && (
        <div className="message-actions">
          <button type="button" className="regenerate-btn" onClick={onRegenerate}>
            Regenerate
          </button>
        </div>
      )}
      {isAgent && onFeedback && (
        <div className="feedback-buttons" role="group" aria-label="feedback">
          <button
            aria-label="Thumbs up"
            aria-pressed={message.vote === "up"}
            className="feedback-btn feedback-btn--up"
            onClick={() => onFeedback("up")}
          >
            👍
          </button>
          <button
            aria-label="Thumbs down"
            aria-pressed={message.vote === "down"}
            className="feedback-btn feedback-btn--down"
            onClick={() => onFeedback("down")}
          >
            👎
          </button>
        </div>
      )}
      {isAgent && message.feedbackError && (
        <p className="feedback-error" role="status">
          {message.feedbackError}
        </p>
      )}
    </div>
  );
}
