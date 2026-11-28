import type { Subject } from "../types";

interface Props {
  onSelect: (subject: Subject) => void;
}

const CHOICES: { subject: Subject; label: string; hint: string }[] = [
  { subject: "quant", label: "Quant", hint: "Work through GRE maths problems step by step." },
  { subject: "aw", label: "AW", hint: "Get feedback on your GRE analytical writing." },
  { subject: "ielts_writing", label: "IELTS Writing", hint: "Talk through your marked essay and what to fix first." },
  { subject: "ielts_reading", label: "IELTS Reading", hint: "Review your Reading result and the questions you missed." },
];

export function SubjectPicker({ onSelect }: Props) {
  return (
    <div className="subject-picker" role="dialog" aria-label="Choose a subject">
      <h2>What are you working on today?</h2>
      <p className="subject-lead">Pick a subject to start. Your tutor is ready when you are.</p>
      <div className="subject-buttons">
        {CHOICES.map((c) => (
          <button key={c.subject} onClick={() => onSelect(c.subject)} aria-label={c.label}>
            <span className="subject-name">{c.label}</span>
            <span className="subject-desc">{c.hint}</span>
          </button>
        ))}
      </div>
      <p className="subject-other">
        Want a timed practice test? <a href="/ielts-mock">Take an IELTS Writing mock</a>
        {" or "}
        <a href="/ielts-mock?module=reading">an IELTS Reading mock</a>
        {". The IELTS coaches work from a marked mock."}
      </p>
    </div>
  );
}
