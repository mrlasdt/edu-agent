import { band, clock, encourage } from "./format";
import type { ListeningReport } from "./listeningTypes";

interface Props {
  report: ListeningReport;
  onAnother: () => void;
}

export function ListeningReportView({ report, onAnother }: Props) {
  const missed = report.questions.filter((q) => !q.correct);
  return (
    <div className="ielts-report">
      <section className="ielts-hero" aria-label="Result">
        <p className="ielts-eyebrow">Mock Listening band</p>
        <p className="ielts-hero-band">{band(report.band)}</p>
        <p className="ielts-encourage">{encourage(report.band)}</p>
        <p>
          Raw score {report.raw} of {report.total}.
          {report.total !== 40 && ` Scaled to ${report.scaled_40} of 40 for the band table.`}
        </p>
        <p className="ielts-hint">
          Time used {clock(report.timing.used_s)} of {clock(report.timing.allowed_s)}
          {report.timing.late ? " (submitted late; your last autosave was marked)" : ""}.
        </p>
        <p className="ielts-disclaimer">{report.disclaimer}</p>
      </section>
      <ul className="ielts-flags" aria-label="Notes on this result">
        <li>{report.set_note}</li>
        <li>{report.keys_note}</li>
        <li>
          {report.band_note} Source: <a href={report.band_source_url}>IDP Listening scores</a>. Timing
          source: <a href={report.timing_source_url}>IELTS.org</a>.
        </li>
      </ul>
      <h2>By question type</h2>
      <ul className="ielts-chips" aria-label="Score by question type">
        {report.by_type.map((t) => (
          <li key={t.task_type} className="ielts-chip">
            {t.label}: {t.correct}/{t.total}
          </li>
        ))}
      </ul>
      <h2>{missed.length ? `Missed questions (${missed.length})` : "No missed questions"}</h2>
      <ol className="reading-results">
        {missed.map((q) => (
          <li key={q.id} value={q.number} className="reading-miss">
            <p>
              <span className="ielts-chip">{q.task_label}</span> {q.passage}
            </p>
            <p>{q.prompt}</p>
            <p>
              Your answer: <strong>{q.answer_label || "(blank)"}</strong>. Key: <strong>{q.key_label}</strong>.
            </p>
            {q.evidence && (
              <blockquote className="ielts-prompt-quote">
                <span className="ielts-hint">What the recording says (author-written key):</span> {q.evidence}
              </blockquote>
            )}
          </li>
        ))}
      </ol>
      <h2>Transcripts</h2>
      {report.transcripts.map((t) => (
        <details key={t.id} className="listening-transcript">
          <summary>{t.title}</summary>
          {t.lines.map((line, i) => (
            <p key={i}>{line}</p>
          ))}
        </details>
      ))}
      <div className="ielts-actions">
        <button onClick={onAnother}>Sit it again</button>
      </div>
    </div>
  );
}
