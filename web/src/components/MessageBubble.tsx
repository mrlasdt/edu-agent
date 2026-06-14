import { Fragment } from "react";
import type { Message } from "../types";
import { ESSAY_DISCLAIMER } from "../types";

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

  // Parse [N] citation markers in the main content
  const tokens = mainContent.split(/(\[\d+\])/g);

  return (
    <>
      {tokens.map((token, i) => {
        const match = token.match(/^\[(\d+)\]$/);
        if (match) {
          const idx = parseInt(match[1], 10);
          return (
            <button
              key={i}
              className="citation-marker"
              aria-label={`[${idx}]`}
              onClick={() => onCitationClick?.(idx)}
            >
              [{idx}]
            </button>
          );
        }
        return <Fragment key={i}>{token}</Fragment>;
      })}
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
