import type { Subject } from "../types";

interface Props {
  onSelect: (subject: Subject) => void;
}

export function SubjectPicker({ onSelect }: Props) {
  return (
    <div className="subject-picker" role="dialog" aria-label="Choose a subject">
      <h2>What are you working on today?</h2>
      <div className="subject-buttons">
        <button onClick={() => onSelect("quant")} aria-label="Quant">
          Quant
        </button>
        <button onClick={() => onSelect("aw")} aria-label="AW">
          AW
        </button>
        <button onClick={() => onSelect("ielts_writing")} aria-label="IELTS Writing">
          IELTS Writing
        </button>
      </div>
      <p className="subject-other">
        IELTS Writing coaches you from a marked mock. <a href="/ielts-mock">Take an IELTS Writing mock</a>
      </p>
    </div>
  );
}
