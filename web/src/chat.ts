import { ApiError, request, requestJson } from "./request";
import type { IndependentCheck, Message, Mode, Subject, TurnView } from "./types";

export interface ChatTurn {
  candidateId: string;
  subject: Subject;
  mode: Mode;
  message: string;
  history: Message[];
  // The IELTS coaches only: the graded mock attempt to coach from; absent means the latest one.
  attemptId?: string;
  // Quant only: the Calculator switch. Absent means the server's own setting.
  mathTools?: boolean;
}

// The gateway routes on (test_type, section); the picker's subjects map onto them.
export function chatTarget(subject: Subject): { test_type: string; section: string } {
  if (subject === "ielts_writing") return { test_type: "ielts", section: "writing" };
  if (subject === "ielts_reading") return { test_type: "ielts", section: "reading" };
  if (subject === "ielts_mock") return { test_type: "ielts", section: "mock" };
  return { test_type: "gre", section: subject };
}

// Mirror the gateway's /chat caps (services/gateway/src/gateway/main.py); it rejects bigger requests.
export const MAX_MESSAGE_CHARS = 4000;
export const MAX_HISTORY_ITEMS = 20;
export const MAX_HISTORY_ITEM_CHARS = 4000;

const clip = (text: string, limit: number) => Array.from(text).slice(0, limit).join("");

// Past the cap the opening message is kept with the latest ones: it states the problem, and the
// agent pins it (services/agent/src/agent/context.py) while it folds the turns in between.
export function toHistoryPayload(history: Message[]) {
  const sendable = history.filter((m) => m.content);
  const kept =
    sendable.length > MAX_HISTORY_ITEMS
      ? [sendable[0], ...sendable.slice(-(MAX_HISTORY_ITEMS - 1))]
      : sendable;
  return kept.map((m) => ({ role: m.role, content: clip(m.content, MAX_HISTORY_ITEM_CHARS) }));
}

export type ChatEvent =
  | { kind: "trace"; traceId: string }
  | { kind: "text"; text: string }
  | {
      kind: "meta";
      mathVerified?: boolean;
      independentCheck?: IndependentCheck;
      // Set for a sample essay: the app shows its own notice, the text carries none.
      disclaimer?: "essay";
    };

// The gateway names each turn in this response header; feedback on the reply refers to it.
export const TRACE_ID_HEADER = "X-Trace-Id";

// The gateway ends a solved answer with a record separator and a JSON object (see
// shared/trailer.py). It is metadata about the answer, never part of its text.
const TRAILER_MARK = "\x1e";

const VERDICTS = ["agrees", "disagrees", "not_checkable"];

function parseIndependentCheck(value: unknown): IndependentCheck | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const { verdict, solver } = value as { verdict?: unknown; solver?: unknown };
  if (typeof verdict !== "string" || !VERDICTS.includes(verdict)) return undefined;
  return {
    verdict: verdict as IndependentCheck["verdict"],
    ...(typeof solver === "string" && solver ? { solver } : {}),
  };
}

function parseTrailer(raw: string): ChatEvent | null {
  try {
    const value: unknown = JSON.parse(raw);
    if (typeof value !== "object" || value === null) return null;
    const {
      math_verified: verified,
      independent_check: independent,
      disclaimer,
    } = value as {
      math_verified?: unknown;
      independent_check?: unknown;
      disclaimer?: unknown;
    };
    return {
      kind: "meta",
      mathVerified: typeof verified === "boolean" ? verified : undefined,
      independentCheck: parseIndependentCheck(independent),
      ...(disclaimer === "essay" ? { disclaimer } : {}),
    };
  } catch {
    return null;
  }
}

/**
 * POST a turn to the gateway and yield the streamed plain-text answer chunk by
 * chunk, then the answer's metadata if the gateway sent any. The turn's trace id, when the
 * gateway sends one, comes first. The gateway
 * streams `text/plain`; clarifications and refusals arrive the same way as
 * answers.
 */
export async function* streamChat(
  turn: ChatTurn,
  signal?: AbortSignal
): AsyncGenerator<ChatEvent> {
  const res = await request("/chat", {
    method: "POST",
    signal,
    json: {
      candidate_id: turn.candidateId,
      ...chatTarget(turn.subject),
      message: turn.message,
      mode: turn.mode,
      history: toHistoryPayload(turn.history),
      ...(turn.attemptId ? { attempt_id: turn.attemptId } : {}),
      ...(turn.mathTools === undefined ? {} : { math_tools: turn.mathTools }),
    },
  });
  if (!res.body) throw new ApiError(res.status, "The server sent no reply");
  const traceId = res.headers?.get(TRACE_ID_HEADER);
  if (traceId) yield { kind: "trace", traceId };
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

export type Vote = "up" | "down";

/** Send a thumbs up or down for the reply of the turn `traceId` names. */
export async function sendFeedback(traceId: string, vote: Vote): Promise<void> {
  await request("/feedback", { method: "POST", json: { trace_id: traceId, vote } });
}

export const TURN_FETCH_ATTEMPTS = 3;
export const TURN_FETCH_GAP_MS = 500;

/**
 * The record of the turn `traceId` names, for the "Show working" panel. The gateway writes it
 * after the reply stream closes, so the first ask can come too early (404): it asks up to
 * three times, half a second apart, then gives up with `null`. `reveal` asks for the tool results
 * a tutor-mode turn hides; the gateway, not this app, decides whether they are shown.
 */
export async function fetchTurn(
  traceId: string,
  options: { reveal?: boolean; gapMs?: number; signal?: AbortSignal } = {}
): Promise<TurnView | null> {
  const { reveal = false, gapMs = TURN_FETCH_GAP_MS, signal } = options;
  const path = `/turns/${encodeURIComponent(traceId)}${reveal ? "?reveal=1" : ""}`;
  for (let attempt = 1; attempt <= TURN_FETCH_ATTEMPTS; attempt++) {
    try {
      return await requestJson<TurnView>(path, { signal });
    } catch (err) {
      if (signal?.aborted) throw err;
      if (attempt === TURN_FETCH_ATTEMPTS) return null;
      await new Promise((resolve) => setTimeout(resolve, gapMs));
    }
  }
  return null;
}
