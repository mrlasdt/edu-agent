import type { Message } from "../types";
import { ESSAY_DISCLAIMER } from "../types";
import { renderMarkdown } from "../markdown";

interface Props {
  message: Message;
  onCitationClick?: (index: number) => void;
  onFeedback?: (vote: "up" | "down") => void;
}

function renderContentWithCitations(
  content: string,
  onCitationClick?: (index: number) => void
) {
  // Split content at the disclaimer boundary
  const parts = content.split(ESSAY_DISCLAIMER);
  const mainContent = parts[0];
  const hasDisclaimer = parts.length > 1;

  return (
    <>
      {renderMarkdown(mainContent, (idx, key) =>
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

export function MessageBubble({ message, onCitationClick, onFeedback }: Props) {
  const isAgent = message.role === "agent";

  return (
    <div className={`message-bubble message-bubble--${message.role}`}>
      {message.status && (
        <p className="status-line" role="status">
          {message.status}
        </p>
      )}
      <div className="message-content">
        {renderContentWithCitations(message.content, onCitationClick)}
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
      {isAgent && onFeedback && (
        <div className="feedback-buttons" role="group" aria-label="feedback">
          <button
            aria-label="Thumbs up"
            className="feedback-btn feedback-btn--up"
            onClick={() => onFeedback("up")}
          >
            👍
          </button>
          <button
            aria-label="Thumbs down"
            className="feedback-btn feedback-btn--down"
            onClick={() => onFeedback("down")}
          >
            👎
          </button>
        </div>
      )}
    </div>
  );
}
