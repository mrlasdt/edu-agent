import { useRef, useState } from "react";
import type { FormEvent } from "react";
import { MessageBubble } from "./components/MessageBubble";
import { ModeSwitch } from "./components/ModeSwitch";
import { SubjectPicker } from "./components/SubjectPicker";
import { streamChat } from "./chat";
import type { Message, Mode, Subject } from "./types";

const SUBJECT_LABEL: Record<Subject, string> = { quant: "Quant", aw: "AW" };

export function App() {
  const [subject, setSubject] = useState<Subject | null>(null);
  const [mode, setMode] = useState<Mode>("tutor");
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const candidateId = useRef(crypto.randomUUID());
  const nextId = useRef(0);

  if (!subject) {
    return (
      <main className="app">
        <SubjectPicker onSelect={setSubject} />
      </main>
    );
  }

  const patchMessage = (id: string, patch: (m: Message) => Message) =>
    setMessages((prev) => prev.map((m) => (m.id === id ? patch(m) : m)));

  async function send(e: FormEvent) {
    e.preventDefault();
    const text = draft.trim();
    if (!text || busy || !subject) return;

    const history = messages;
    const agentId = `m${nextId.current + 1}`;
    const candidate: Message = {
      id: `m${nextId.current}`,
      role: "candidate",
      content: text,
    };
    nextId.current += 2;
    setMessages([
      ...history,
      candidate,
      { id: agentId, role: "agent", content: "", status: "thinking..." },
    ]);
    setDraft("");
    setBusy(true);
    try {
      for await (const event of streamChat({
        candidateId: candidateId.current,
        subject,
        mode,
        message: text,
        history,
      })) {
        patchMessage(agentId, (m) =>
          event.kind === "text"
            ? { ...m, content: m.content + event.text, status: undefined }
            : { ...m, mathVerified: event.mathVerified }
        );
      }
      patchMessage(agentId, (m) => ({
        ...m,
        status: m.content ? undefined : "No response received",
      }));
    } catch (err) {
      patchMessage(agentId, (m) => ({
        ...m,
        status: err instanceof Error ? err.message : "Something went wrong",
      }));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app">
      <header className="app-header">
        <h1>GRE Tutor: {SUBJECT_LABEL[subject]}</h1>
        <ModeSwitch
          mode={mode}
          onToggle={() => setMode((m) => (m === "tutor" ? "solve" : "tutor"))}
        />
      </header>
      <section className="messages" aria-label="Conversation" aria-live="polite">
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
      </section>
      <form className="composer" onSubmit={send}>
        <input
          aria-label="Your message"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask a question"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !draft.trim()}>
          Send
        </button>
      </form>
    </main>
  );
}
