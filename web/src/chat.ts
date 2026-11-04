import type { Message, Mode, Subject } from "./types";

export interface ChatTurn {
  candidateId: string;
  subject: Subject;
  mode: Mode;
  message: string;
  history: Message[];
}

// Mirror the gateway's /chat caps (services/gateway/src/gateway/main.py); it rejects bigger requests.
export const MAX_MESSAGE_CHARS = 4000;
export const MAX_HISTORY_ITEMS = 20;
export const MAX_HISTORY_ITEM_CHARS = 4000;

const clip = (text: string, limit: number) => Array.from(text).slice(0, limit).join("");

export function toHistoryPayload(history: Message[]) {
  return history
    .filter((m) => m.content)
    .slice(-MAX_HISTORY_ITEMS)
    .map((m) => ({ role: m.role, content: clip(m.content, MAX_HISTORY_ITEM_CHARS) }));
}

export type ChatEvent =
  | { kind: "text"; text: string }
  | { kind: "meta"; mathVerified?: boolean };

// The gateway ends a solved answer with a record separator and a JSON object (see
// shared/trailer.py). It is metadata about the answer, never part of its text.
const TRAILER_MARK = "\x1e";

function parseTrailer(raw: string): ChatEvent | null {
  try {
    const value: unknown = JSON.parse(raw);
    if (typeof value !== "object" || value === null) return null;
    const verified = (value as { math_verified?: unknown }).math_verified;
    return { kind: "meta", mathVerified: typeof verified === "boolean" ? verified : undefined };
  } catch {
    return null;
  }
}

/**
 * POST a turn to the gateway and yield the streamed plain-text answer chunk by
 * chunk, then the answer's metadata if the gateway sent any. The gateway
 * streams `text/plain`; clarifications and refusals arrive the same way as
 * answers.
 */
export async function* streamChat(
  turn: ChatTurn,
  signal?: AbortSignal
): AsyncGenerator<ChatEvent> {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      candidate_id: turn.candidateId,
      test_type: "gre",
      section: turn.subject,
      message: turn.message,
      mode: turn.mode,
      history: toHistoryPayload(turn.history),
    }),
  });
  if (!res.ok || !res.body) {
    throw new Error(`Chat request failed (${res.status})`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let trailer: string | null = null;
  const route = function* (text: string): Generator<ChatEvent> {
    if (trailer !== null) {
      trailer += text;
      return;
    }
    const at = text.indexOf(TRAILER_MARK);
    if (at === -1) {
      if (text) yield { kind: "text", text };
      return;
    }
    if (at > 0) yield { kind: "text", text: text.slice(0, at) };
    trailer = text.slice(at + 1);
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    yield* route(decoder.decode(value, { stream: true }));
  }
  yield* route(decoder.decode());
  const meta = trailer === null ? null : parseTrailer(trailer);
  if (meta) yield meta;
}
