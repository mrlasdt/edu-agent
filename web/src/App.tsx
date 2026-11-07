import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { MessageBubble } from "./components/MessageBubble";
import { ModeSwitch } from "./components/ModeSwitch";
import { SubjectPicker } from "./components/SubjectPicker";
import { newId } from "./newId";
import { MAX_MESSAGE_CHARS, sendFeedback, streamChat } from "./chat";
import type { Vote } from "./chat";
import type { Message, Mode, Subject } from "./types";

const STOPPED = "Stopped";

const SUBJECT_LABEL: Record<Subject, string> = { quant: "Quant", aw: "AW" };

export function App() {
  const [subject, setSubject] = useState<Subject | null>(null);
  const [mode, setMode] = useState<Mode>("tutor");
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [candidateId] = useState(newId); // lazy: runs once, not on every render
  const nextId = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  // Leaving the page ends the stream; the signal also tells the gateway to stop the model call.
  useEffect(() => () => abortRef.current?.abort(), []);

  if (!subject) {
    return (
      <main className="app">
        <SubjectPicker onSelect={setSubject} />
      </main>
    );
  }

  const patchMessage = (id: string, patch: (m: Message) => Message) =>
    setMessages((prev) => prev.map((m) => (m.id === id ? patch(m) : m)));

  async function rate(message: Message, vote: Vote) {
    if (!message.traceId) return;
    const traceId = message.traceId;
    const previous = message.vote;
    patchMessage(message.id, (m) => ({ ...m, vote, feedbackError: undefined }));
    try {
      await sendFeedback(traceId, vote);
    } catch {
      patchMessage(message.id, (m) => ({
        ...m,
        vote: previous,
        feedbackError: "Could not send feedback. Try again.",
      }));
    }
  }

  // Streams the agent's reply to `text` into a new bubble. `history` is what came before it.
  async function runTurn(text: string, history: Message[], shown: Message[]) {
    if (!subject) return;
    const agentId = `m${nextId.current++}`;
    const controller = new AbortController();
    abortRef.current = controller;
    setMessages([...shown, { id: agentId, role: "agent", content: "", status: "thinking..." }]);
    setBusy(true);
    try {
      for await (const event of streamChat(
        { candidateId, subject, mode, message: text, history },
        controller.signal
      )) {
        patchMessage(agentId, (m) => {
          if (event.kind === "trace") return { ...m, traceId: event.traceId };
          if (event.kind === "text") {
            return { ...m, content: m.content + event.text, status: undefined };
          }
          return {
            ...m,
            mathVerified: event.mathVerified,
            independentCheck: event.independentCheck,
          };
        });
      }
      patchMessage(agentId, (m) => ({
        ...m,
        status: controller.signal.aborted
          ? STOPPED
          : m.content
            ? undefined
            : "No response received",
      }));
    } catch (err) {
      patchMessage(agentId, (m) => ({
        ...m,
        status: controller.signal.aborted
          ? STOPPED
          : err instanceof Error
            ? err.message
            : "Something went wrong",
      }));
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setBusy(false);
    }
  }

  async function send(e: FormEvent) {
    e.preventDefault();
    const text = draft.trim();
    if (!text || busy || !subject) return;

    const candidate: Message = {
      id: `m${nextId.current++}`,
      role: "candidate",
      content: text,
    };
    setDraft("");
    await runTurn(text, messages, [...messages, candidate]);
  }

  const stop = () => abortRef.current?.abort();

  // Asks again: the last question is resent with the history that preceded it, and the new
  // reply replaces the old one.
  async function regenerate() {
    if (busy) return;
    const at = messages.map((m) => m.role).lastIndexOf("candidate");
    if (at === -1) return;
    await runTurn(messages[at].content, messages.slice(0, at), messages.slice(0, at + 1));
  }

  const lastReply = messages[messages.length - 1];
  const canRegenerate = !busy && lastReply?.role === "agent";

  return (
    <main className="app">
      <header className="app-header">
        <h1>Edu Agent: GRE {SUBJECT_LABEL[subject]}</h1>
        <ModeSwitch
          mode={mode}
          onToggle={() => setMode((m) => (m === "tutor" ? "solve" : "tutor"))}
        />
      </header>
      <section className="messages" aria-label="Conversation" aria-live="polite">
        {messages.map((m, i) => {
          // Rate a reply once it is complete and the gateway has named its turn.
          const streaming = busy && i === messages.length - 1;
          const canRate = m.role === "agent" && m.traceId && m.content && !streaming;
          return (
            <MessageBubble
              key={m.id}
              message={m}
              onFeedback={canRate ? (vote) => void rate(m, vote) : undefined}
              onRegenerate={canRegenerate && m === lastReply ? regenerate : undefined}
            />
          );
        })}
      </section>
      <form className="composer" onSubmit={send}>
        <input
          aria-label="Your message"
          value={draft}
          maxLength={MAX_MESSAGE_CHARS}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask a question"
          disabled={busy}
        />
        {busy ? (
          <button type="button" onClick={stop}>
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!draft.trim()}>
            Send
          </button>
        )}
      </form>
    </main>
  );
}
