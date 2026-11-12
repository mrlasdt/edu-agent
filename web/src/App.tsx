import { useState } from "react";
import type { FormEvent } from "react";
import { MessageBubble } from "./components/MessageBubble";
import { ModeSwitch } from "./components/ModeSwitch";
import { SubjectPicker } from "./components/SubjectPicker";
import { MAX_MESSAGE_CHARS } from "./chat";
import { useChat } from "./useChat";
import type { Mode, Subject } from "./types";

const SUBJECT_TITLE: Record<Subject, string> = {
  quant: "GRE Quant",
  aw: "GRE AW",
  ielts_writing: "IELTS Writing coach",
};

// The mock report's "Discuss this report" link opens the coach on that attempt: /?attempt=<id>.
function attemptFromUrl(): string | undefined {
  return new URLSearchParams(window.location.search).get("attempt") || undefined;
}

export function App() {
  const attemptId = attemptFromUrl();
  const [subject, setSubject] = useState<Subject | null>(attemptId ? "ielts_writing" : null);
  if (!subject) {
    return (
      <main className="app">
        <SubjectPicker onSelect={setSubject} />
      </main>
    );
  }
  return <Chat subject={subject} attemptId={subject === "ielts_writing" ? attemptId : undefined} />;
}

function Chat({ subject, attemptId }: { subject: Subject; attemptId?: string }) {
  const [mode, setMode] = useState<Mode>("tutor");
  const [draft, setDraft] = useState("");
  const { messages, busy, canRegenerate, send, regenerate, stop, rate } = useChat(subject, mode, attemptId);

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
        <h1>Edu Agent: {SUBJECT_TITLE[subject]}</h1>
        {/* The coach has one mode: it never writes the essay, so there is no solve mode to switch to. */}
        {subject !== "ielts_writing" && (
          <ModeSwitch
            mode={mode}
            onToggle={() => setMode((m) => (m === "tutor" ? "solve" : "tutor"))}
          />
        )}
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
          placeholder={
            subject === "ielts_writing" ? "Ask about your report, e.g. what should I fix first?" : "Ask a question"
          }
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
