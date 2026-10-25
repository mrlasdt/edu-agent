import type { Message, Mode, Subject } from "./types";

export interface ChatTurn {
  candidateId: string;
  subject: Subject;
  mode: Mode;
  message: string;
  history: Message[];
}

/**
 * POST a turn to the gateway and yield the streamed plain-text answer chunk by
 * chunk. The gateway streams `text/plain`; clarifications and refusals arrive
 * the same way as answers.
 */
export async function* streamChat(
  turn: ChatTurn,
  signal?: AbortSignal
): AsyncGenerator<string> {
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
      history: turn.history.map((m) => ({
        role: m.role,
        content: m.content,
      })),
    }),
  });
  if (!res.ok || !res.body) {
    throw new Error(`Chat request failed (${res.status})`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    yield decoder.decode(value, { stream: true });
  }
  const tail = decoder.decode();
  if (tail) yield tail;
}
