import { useState } from "react";
import type { FormEvent } from "react";
import { MessageBubble } from "./components/MessageBubble";
import { ModeSwitch } from "./components/ModeSwitch";
import { SubjectPicker } from "./components/SubjectPicker";
import { MAX_MESSAGE_CHARS } from "./chat";
import { useChat } from "./useChat";
import type { Mode, Subject } from "./types";

const SUBJECT_LABEL: Record<Subject, string> = { quant: "Quant", aw: "AW" };

export function App() {
  const [subject, setSubject] = useState<Subject | null>(null);
  if (!subject) {
    return (
      <main className="app">
        <SubjectPicker onSelect={setSubject} />
      </main>
    );
  }
  return <Chat subject={subject} />;
}

function Chat({ subject }: { subject: Subject }) {
  const [mode, setMode] = useState<Mode>("tutor");
  const [draft, setDraft] = useState("");
  const { messages, busy, canRegenerate, send, regenerate, stop, rate } = useChat(subject, mode);

  async function submit(e: FormEvent) {
    e.preventDefault();
    const text = draft.trim();
    if (!text || busy) return;
    setDraft("");
    await send(text);
  }

  const lastReply = messages[messages.length - 1];

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
        {messages.map((m) => {
          // Rate a reply once it is complete and the gateway has named its turn.
          const canRate = m.role === "agent" && m.traceId && m.content && m.status !== "streaming";
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
      <form className="composer" onSubmit={submit}>
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
