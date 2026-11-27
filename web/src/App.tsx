import { useState } from "react";
import type { FormEvent } from "react";
import { CalculatorSwitch } from "./components/CalculatorSwitch";
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
  ielts_reading: "IELTS Reading coach",
  ielts_mock: "IELTS full mock coach",
};

const isCoach = (subject: Subject) =>
  subject === "ielts_writing" || subject === "ielts_reading" || subject === "ielts_mock";

// A mock report's "Discuss" link opens the coach on that attempt: /?attempt=<id>, plus
// &subject=reading for a Reading attempt or &subject=mock for a full mock (Writing is the default).
function attemptFromUrl(): string | undefined {
  return new URLSearchParams(window.location.search).get("attempt") || undefined;
}

const SUBJECT_FROM_URL: Record<string, Subject> = {
  reading: "ielts_reading",
  mock: "ielts_mock",
};

export function App() {
  const attemptId = attemptFromUrl();
  const urlSubject = new URLSearchParams(window.location.search).get("subject") ?? "";
  const [subject, setSubject] = useState<Subject | null>(
    attemptId ? (SUBJECT_FROM_URL[urlSubject] ?? "ielts_writing") : null
  );
  if (!subject) {
    return (
      <main className="app">
        <SubjectPicker onSelect={setSubject} />
      </main>
    );
  }
  return <Chat subject={subject} attemptId={isCoach(subject) ? attemptId : undefined} />;
}

function Chat({ subject, attemptId }: { subject: Subject; attemptId?: string }) {
  const [mode, setMode] = useState<Mode>("tutor");
  const [draft, setDraft] = useState("");
  // Quant only: the math tools are the agent's calculator. Off until the learner turns it on.
  const [calculator, setCalculator] = useState(false);
  const mathTools = subject === "quant" ? calculator : undefined;
  const { messages, busy, canRegenerate, send, regenerate, stop, rate } = useChat(
    subject,
    mode,
    attemptId,
    mathTools
  );

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
        {!isCoach(subject) && (
          <div className="header-actions">
            {subject === "quant" && (
              <CalculatorSwitch on={calculator} onToggle={() => setCalculator((c) => !c)} />
            )}
            <ModeSwitch
              mode={mode}
              onToggle={() => setMode((m) => (m === "tutor" ? "solve" : "tutor"))}
            />
          </div>
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
            subject === "ielts_writing"
              ? "Ask about your report, e.g. what should I fix first?"
              : subject === "ielts_reading"
                ? "Ask about your result, e.g. why did I miss the matching headings?"
                : subject === "ielts_mock"
                  ? "Ask about your mock, e.g. which skill should I practise next?"
                  : "Ask a question"
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
