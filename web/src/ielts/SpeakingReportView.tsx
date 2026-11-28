import { band, clock } from "./format";
import type { SpeakingCriterion, SpeakingReport } from "./speakingTypes";

interface Props {
  report: SpeakingReport;
  onAnother: () => void;
}

function CriterionCard({ c }: { c: SpeakingCriterion }) {
  if (!c.assessed || c.band === null) {
    return (
      <article className="ielts-criterion" aria-label={c.name} data-assessed="false">
        <header>
          <h3>{c.name}</h3>
          <span className="ielts-band-pill" aria-label={`${c.name} not assessed`}>
            Not assessed
          </span>
        </header>
        <p>{c.reason}</p>
      </article>
    );
  }
  return (
    <article className="ielts-criterion" aria-label={c.name}>
      <header>
        <h3>{c.name}</h3>
        <span className="ielts-band-pill" aria-label={`${c.name} band ${band(c.band)}`}>
          {band(c.band)}
        </span>
      </header>
      <div className="ielts-meter" aria-hidden="true">
        <span style={{ width: `${(c.band / 9) * 100}%` }} />
      </div>
      {c.rationale && <p>{c.rationale}</p>}
      {c.evidence.length > 0 && (
        <ul className="ielts-evidence" aria-label={`Evidence for ${c.name}`}>
          {c.evidence.map((e, i) => (
            <li key={i}>
              <q>{e}</q>
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}

export function SpeakingReportView({ report, onAnother }: Props) {
  return (
    <article className="ielts-report" aria-label="Speaking report">
      <section className="ielts-hero" aria-label="Speaking result">
        <p className="ielts-eyebrow">
          {report.partial ? "Partial Speaking band (mock score)" : "Speaking band (mock score)"}
        </p>
        {report.band !== null ? (
          <p className="ielts-hero-band" aria-label={`Partial Speaking band ${band(report.band)}`}>
            {band(report.band)}
          </p>
        ) : (
          <p className="ielts-hint">No band was given.</p>
        )}
        <p>{report.partial_note}</p>
        {report.not_assessed.length > 0 && (
          <p className="ielts-hint">
            Not assessed:{" "}
            {report.criteria
              .filter((c) => !c.assessed)
              .map((c) => c.name)
              .join(", ")}
          </p>
        )}
        <p className="ielts-disclaimer">{report.disclaimer}</p>
      </section>
      {report.sample && (
        <p className="ielts-flags">
          This was the quick sample (one short question per part). It checks that the marking works
          end to end; its band says little about your level.
        </p>
      )}
      {report.flags.length > 0 && (
        <ul className="ielts-flags" aria-label="Things to know about this marking">
          {report.flags.map((f) => (
            <li key={f}>{f}</li>
          ))}
        </ul>
      )}
      <p className="ielts-hint ielts-facts">
        Time used {clock(report.timing.used_s)} of {clock(report.timing.allowed_s)}
        {report.timing.late ? " (submitted late)" : ""}. Marked by {report.model}.
      </p>

      {report.priorities.length > 0 && (
        <section aria-label="Fix these first">
          <h2>Fix these first</h2>
          <ol className="ielts-priorities">
            {report.priorities.map((p) => (
              <li key={p.priority}>
                <span className="ielts-chip">{p.criterion}</span> <strong>{p.do}</strong>
              </li>
            ))}
          </ol>
        </section>
      )}
      {report.strengths.length > 0 && (
        <section aria-label="Strengths">
          <h2>What worked</h2>
          <ul>
            {report.strengths.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </section>
      )}

      <section aria-label="Criteria">
        <h2>By criterion</h2>
        {report.criteria.map((c) => (
          <CriterionCard key={c.id} c={c} />
        ))}
      </section>

      <section aria-label="What you said">
        <h2>What you said</h2>
        {report.parts.map((p) => (
          <details key={p.part} className="speaking-transcript" open={report.parts.length === 1}>
            <summary>
              {p.title}
              <span className="ielts-hint">
                {" "}
                - {p.words} words, {clock(p.duration_s)}
                {p.words_per_minute !== null ? `, ${Math.round(p.words_per_minute)} words a minute` : ""}
                {p.fillers > 0 ? `, ${p.fillers} filler${p.fillers === 1 ? "" : "s"}` : ""}
              </span>
            </summary>
            <p>{p.transcript || "(nothing heard)"}</p>
          </details>
        ))}
        <p className="ielts-hint">
          The transcript is the model&apos;s reading of your audio and can contain mistakes. The
          counts above are made in code from it.
        </p>
      </section>

      <p className="ielts-hint">
        Sources:{" "}
        <a href={report.descriptors_source_url} target="_blank" rel="noreferrer">
          IELTS Speaking band descriptors
        </a>
        ,{" "}
        <a href={report.format_source_url} target="_blank" rel="noreferrer">
          test format
        </a>
        .
      </p>
      <div className="ielts-actions">
        <button onClick={onAnother}>Sit Speaking again</button>
      </div>
    </article>
  );
}
