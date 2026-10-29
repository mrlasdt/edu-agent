import { Fragment } from "react";
import { band, clock, signed } from "./format";
import { DISCLAIMER } from "./types";
import type { CriterionReport, ErrorSpan, Report } from "./types";

interface Props {
  report: Report;
  essay: string;
  onAnother: () => void;
}

const CONFIDENCE_TEXT = {
  high: "High confidence: the marking passes agreed closely.",
  medium: "Medium confidence: the marking passes differed a little.",
  low: "Low confidence: the marking passes disagreed or the essay has issues, so treat this band loosely.",
} as const;

const ERROR_LABEL: Record<string, string> = {
  agreement: "Agreement",
  tense: "Tense",
  article: "Articles",
  preposition: "Prepositions",
  punctuation: "Punctuation",
  spelling: "Spelling",
  collocation: "Collocation",
  word_form: "Word form",
  linking: "Linking",
  other: "Other",
};

function Segments({ essay, spans }: { essay: string; spans: ErrorSpan[] }) {
  const ordered = [...spans]
    .filter((s) => essay.slice(s.start, s.end) === s.quote)
    .sort((a, b) => a.start - b.start);
  const parts: JSX.Element[] = [];
  let at = 0;
  ordered.forEach((s, i) => {
    if (s.start < at) return; // overlapping spans: keep the first
    if (s.start > at) parts.push(<Fragment key={`t${i}`}>{essay.slice(at, s.start)}</Fragment>);
    parts.push(
      <mark key={`m${i}`} data-type={s.type} title={`${ERROR_LABEL[s.type] ?? s.type}: ${s.note}`}>
        {essay.slice(s.start, s.end)}
      </mark>
    );
    at = s.end;
  });
  parts.push(<Fragment key="end">{essay.slice(at)}</Fragment>);
  return <>{parts}</>;
}

function CriterionCard({ c }: { c: CriterionReport }) {
  const next = Math.min(9, c.band + 0.5);
  const errors = Object.entries(c.error_table);
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
      {c.capped_from !== null && (
        <p className="ielts-hint">
          Held to {band(c.band)} by an automatic check (the marking passes gave {band(c.capped_from)}).
        </p>
      )}
      {c.sample_bands.length > 1 && (
        <p className="ielts-hint">Marking passes: {c.sample_bands.map(band).join(", ")}</p>
      )}
      {c.verdict && <p>{c.verdict}</p>}
      {c.holding_back && (
        <p>
          <strong>Holding it back.</strong> {c.holding_back}
        </p>
      )}
      {c.to_reach_next_band && (
        <p>
          <strong>To reach {band(next)}.</strong> {c.to_reach_next_band}
        </p>
      )}
      {c.evidence.length > 0 && <p className="ielts-sub">From your essay</p>}
      {c.evidence.length > 0 && (
        <ul className="ielts-evidence" aria-label={`Evidence for ${c.name}`}>
          {c.evidence.map((e, i) => (
            <li key={i} data-effect={e.effect}>
              <q>{e.quote}</q>
              <span className="ielts-hint">
                {e.effect === "supports" ? "supports the band" : "holds the band back"} - paragraph {e.para}
              </span>
            </li>
          ))}
        </ul>
      )}
      {c.paragraph_map.length > 0 && <p className="ielts-sub">What each paragraph does</p>}
      {c.paragraph_map.length > 0 && (
        <ol className="ielts-paragraph-map" aria-label="What each paragraph does">
          {c.paragraph_map.map((p) => (
            <li key={p.para}>
              Paragraph {p.para}: {p.job}
              {p.one_central_idea === false && <span className="ielts-hint"> (more than one main idea)</span>}
            </li>
          ))}
        </ol>
      )}
      {errors.length > 0 && (
        <p className="ielts-chips" aria-label="Errors found">
          {errors.map(([type, n]) => (
            <span className="ielts-chip" key={type}>
              {ERROR_LABEL[type] ?? type} x{n}
            </span>
          ))}
        </p>
      )}
      {c.actions.length > 0 && <p className="ielts-sub">What to do</p>}
      {c.actions.length > 0 && (
        <ol className="ielts-actions-list" aria-label="What to do">
          {c.actions.map((a, i) => (
            <li key={i}>
              <strong>{a.do}</strong> <span className="ielts-hint">{a.why}</span>
            </li>
          ))}
        </ol>
      )}
    </article>
  );
}

export function ReportView({ report, essay, onAnother }: Props) {
  const flags = report.flags.filter((f) => f.severity !== "info");
  const est = report.estimated_band;

  return (
    <article className="ielts-report" aria-label="Report">
      {est ? (
        <section className="ielts-hero" aria-label="Mock score">
          <p className="ielts-eyebrow">Mock score</p>
          <p className="ielts-hero-band" aria-label={`Mock overall score ${band(est.overall)}`}>
            {band(est.overall)}
          </p>
          <p>
            Most likely between {band(est.range[0])} and {band(est.range[1])}.
          </p>
          <p className="ielts-hint">{CONFIDENCE_TEXT[est.confidence]}</p>
          <p className="ielts-calibration" data-status={est.calibration.status}>
            <strong>{est.calibration.label}</strong>
            {est.calibration.note && <span> - {est.calibration.note}</span>}
          </p>
          <p className="ielts-disclaimer">{DISCLAIMER}</p>
        </section>
      ) : (
        <section className="ielts-hero" aria-label="Not marked">
          <h2>This response could not be given a band</h2>
          <p>{report.ungraded_reason}</p>
          <p className="ielts-disclaimer">{DISCLAIMER}</p>
        </section>
      )}

      <p className="ielts-hint ielts-facts">
        {report.word_count} words
        {report.timing && ` - ${clock(report.timing.used_s)} of ${clock(report.timing.allowed_s)} used`}
        {report.timing?.late && " - submitted after time was up, so the last autosave at the deadline was marked"}
      </p>

      {flags.length > 0 && (
        <ul className="ielts-flags" aria-label="Things to know about this marking">
          {flags.map((f) => (
            <li key={f.code} data-severity={f.severity}>
              {f.message}
            </li>
          ))}
        </ul>
      )}

      {report.vs_previous_attempt && (
        <p className="ielts-delta">
          Change in mock score since your last marked attempt: overall{" "}
          {signed(report.vs_previous_attempt.overall_delta)}
          {Object.entries(report.vs_previous_attempt.criteria_deltas).map(([k, v]) => (
            <span key={k}>
              {" "}
              | {k} {signed(v)}
            </span>
          ))}
        </p>
      )}

      {report.feedback_status !== "complete" && est && (
        <p className="ielts-flags" role="note">
          {report.feedback_status === "unavailable"
            ? "The written feedback was unavailable this time, so only the bands are shown."
            : "Some of the written feedback could not be checked against your essay, so it is incomplete."}
        </p>
      )}

      {report.top_priorities.length > 0 && (
        <section aria-label="Fix these first">
          <h2>Fix these first</h2>
          <ol className="ielts-priorities">
            {report.top_priorities.map((p) => (
              <li key={p.priority}>
                <span className="ielts-chip">{p.criterion}</span> <strong>{p.do}</strong>
                <span className="ielts-hint"> {p.why}</span>
              </li>
            ))}
          </ol>
        </section>
      )}

      {report.criteria.length > 0 && (
        <section aria-label="Criteria">
          <h2>By criterion</h2>
          <div className="ielts-criteria">
            {report.criteria.map((c) => (
              <CriterionCard key={c.id} c={c} />
            ))}
          </div>
        </section>
      )}

      {report.task_checks.length > 0 && (
        <section aria-label="Task checks">
          <h2>Task checklist</h2>
          <ul className="ielts-checks">
            {report.task_checks.map((t) => (
              <li key={t.check} data-passed={String(t.passed)}>
                <span aria-hidden="true">{t.passed === null ? "?" : t.passed ? "✓" : "✗"}</span>{" "}
                {t.check}
                <span className="ielts-sr">
                  {t.passed === null ? " - not assessed" : t.passed ? " - yes" : " - no"}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {report.error_spans.length > 0 && (
        <section aria-label="Your essay, marked">
          <h2>Your essay, marked</h2>
          <p className="ielts-essay">
            <Segments essay={essay} spans={report.error_spans} />
          </p>
          <table className="ielts-errors">
            <caption className="ielts-sr">Errors found in your essay</caption>
            <thead>
              <tr>
                <th scope="col">You wrote</th>
                <th scope="col">Type</th>
                <th scope="col">Why</th>
                <th scope="col">Try</th>
              </tr>
            </thead>
            <tbody>
              {report.error_spans.map((e, i) => (
                <tr key={i}>
                  <td>
                    <q>{e.quote}</q>
                  </td>
                  <td>{ERROR_LABEL[e.type] ?? e.type}</td>
                  <td>{e.note}</td>
                  <td>{e.suggestion}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <footer className="ielts-report-foot">
        <p className="ielts-hint">
          An IELTS mock. It is good for spotting patterns and tracking change across attempts; it
          cannot replace an examiner's mark. Use the criterion feedback more than the exact number.
        </p>
        <button className="ielts-primary" onClick={onAnother}>
          Write another essay
        </button>
      </footer>
    </article>
  );
}
