import { useCallback, useEffect, useState } from "react";
import { ApiError } from "../request";
import { band } from "./format";
import { createMock, getMock, startMockModule } from "./mockApi";
import type { MockReport, ModuleKey, ModuleStatus } from "./mockApi";
import { ModuleNav } from "./ModuleNav";
import { getListening, startListening } from "./listeningApi";
import type { ListeningAttempt } from "./listeningTypes";
import { ListeningSit } from "./ListeningSit";
import { ReadingSit } from "./ReadingSit";
import { getReading, startReading } from "./readingApi";
import type { ReadingAttempt } from "./readingTypes";
import { WritingModuleSit } from "./WritingModuleSit";
import { getWritingModule, regradeWritingModule, startWritingModule } from "./writingModuleApi";
import type { WritingModuleAttempt } from "./writingModuleTypes";

const KEY = "ielts.fullmock.id";
const remember = (id: string | null) => {
  try {
    if (id) window.localStorage.setItem(KEY, id);
    else window.localStorage.removeItem(KEY);
  } catch {
    // storage unavailable: the learner just starts from the intro
  }
};
const remembered = () => {
  try {
    return window.localStorage.getItem(KEY);
  } catch {
    return null;
  }
};

type View =
  | { kind: "loading" }
  | { kind: "failed"; message: string }
  | { kind: "intro" }
  | { kind: "overview"; report: MockReport }
  | { kind: "listening"; report: MockReport; attempt: ListeningAttempt }
  | { kind: "reading"; report: MockReport; attempt: ReadingAttempt }
  | { kind: "writing"; report: MockReport; attempt: WritingModuleAttempt };

const message = (err: unknown) => (err instanceof Error ? err.message : "Something went wrong");
const POLL_MS = 2000;

const STATE_TEXT: Record<ModuleStatus["state"], string> = {
  not_assessed: "Not assessed",
  not_started: "Not started",
  in_progress: "In progress",
  marking: "Being marked",
  graded: "Marked",
  partial: "No band (one task not marked)",
  ungraded: "Marking failed",
};

export interface FullMockProps {
  pollMs?: number;
  autosaveMs?: number;
}

/** The full mock: Listening, Reading, then Writing, one combined result. The modules are the standalone ones. */
export function FullMock({ pollMs = POLL_MS, autosaveMs }: FullMockProps) {
  const [view, setView] = useState<View>({ kind: "loading" });
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    document.title = "IELTS full mock";
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    (async () => {
      try {
        const id = remembered();
        if (id) {
          try {
            const report = await getMock(id, controller.signal);
            if (!cancelled) setView({ kind: "overview", report });
            return;
          } catch (err) {
            if (!(err instanceof ApiError && err.status === 404)) throw err;
          }
          remember(null);
        }
        if (!cancelled) setView({ kind: "intro" });
      } catch (err) {
        if (!cancelled) setView({ kind: "failed", message: message(err) });
      }
    })();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, []);

  // While a module is being marked, ask the server until it has a band.
  const marking =
    view.kind === "overview" && view.report.modules.some((m) => m.state === "marking")
      ? view.report.id
      : null;
  useEffect(() => {
    if (!marking) return;
    const controller = new AbortController();
    const timer = window.setInterval(async () => {
      try {
        const report = await getMock(marking, controller.signal);
        setView((cur) => (cur.kind === "overview" && cur.report.id === marking ? { kind: "overview", report } : cur));
      } catch {
        // a blip while polling: try again on the next tick
      }
    }, pollMs);
    return () => {
      window.clearInterval(timer);
      controller.abort();
    };
  }, [marking, pollMs]);

  async function run(work: () => Promise<void>) {
    setBusy(true);
    setNotice(null);
    try {
      await work();
    } catch (err) {
      setNotice(message(err));
    } finally {
      setBusy(false);
    }
  }

  const refresh = useCallback(async (id: string) => {
    setView({ kind: "overview", report: await getMock(id) });
  }, []);

  /** Open a module's page: create its attempt if it has none, start it if not started, and sit it. */
  const openModule = (report: MockReport, key: ModuleKey) =>
    run(async () => {
      let current = report;
      let status = current.modules.find((m) => m.key === key);
      if (status && !status.attempt_id) {
        current = await startMockModule(report.id, key);
        status = current.modules.find((m) => m.key === key);
      }
      const attemptId = status?.attempt_id;
      if (!attemptId) throw new Error("This module could not be started.");
      if (key === "listening") {
        let attempt = await getListening(attemptId);
        if (attempt.status === "created") attempt = await startListening(attemptId);
        if (attempt.status === "in_progress") return setView({ kind: "listening", report: current, attempt });
      } else if (key === "reading") {
        let attempt = await getReading(attemptId);
        if (attempt.status === "created") attempt = await startReading(attemptId);
        if (attempt.status === "in_progress") return setView({ kind: "reading", report: current, attempt });
      } else {
        let attempt = await getWritingModule(attemptId);
        if (attempt.status === "created") attempt = await startWritingModule(attemptId);
        if (attempt.status === "in_progress") return setView({ kind: "writing", report: current, attempt });
      }
      await refresh(report.id);
    });

  const retryMarking = (report: MockReport, attemptId: string) =>
    run(async () => {
      await regradeWritingModule(attemptId);
      await refresh(report.id);
    });

  const begin = () =>
    run(async () => {
      const report = await createMock();
      remember(report.id);
      await openModule(report, "listening");
    });

  const another = () => {
    remember(null);
    setNotice(null);
    setView({ kind: "intro" });
  };

  return (
    <main className="app ielts" data-view={view.kind}>
      <header className="app-header">
        <h1>IELTS full mock</h1>
        <a className="ielts-home" href="/">
          Tutor chat
        </a>
      </header>
      <ModuleNav current="mock" />
      {notice && (
        <p className="ielts-error" role="alert">
          {notice}
        </p>
      )}
      {view.kind === "loading" && <p role="status">Loading...</p>}
      {view.kind === "failed" && (
        <div role="alert" className="ielts-error">
          <p>{view.message}</p>
          <button onClick={() => window.location.reload()}>Reload</button>
        </div>
      )}
      {view.kind === "intro" && (
        <section className="ielts-ready" aria-label="Full mock">
          <h2>Full mock: Listening, Reading, then Writing</h2>
          <ul className="ielts-rules">
            <li>
              You sit the modules in order: Listening (30 minutes, with sound on), Reading (60 minutes),
              then Writing (60 minutes).
            </li>
            <li>
              Each skill gets its own band, and the program combines them into one overall band. Speaking
              is not part of the mock yet, so the overall band is partial.
            </li>
            <li>When you finish, the coach can tell you which skill to practise next.</li>
          </ul>
          <div className="ielts-actions">
            <button className="ielts-primary" onClick={() => void begin()} disabled={busy}>
              {busy ? "Starting..." : "Start full mock"}
            </button>
          </div>
        </section>
      )}
      {view.kind === "listening" && (
        <ListeningSit
          key={view.attempt.id}
          attempt={view.attempt}
          autosaveMs={autosaveMs}
          onSubmitted={() => void run(() => refresh(view.report.id))}
        />
      )}
      {view.kind === "reading" && (
        <ReadingSit
          key={view.attempt.id}
          attempt={view.attempt}
          autosaveMs={autosaveMs}
          onSubmitted={() => void run(() => refresh(view.report.id))}
        />
      )}
      {view.kind === "writing" && (
        <WritingModuleSit
          key={view.attempt.id}
          attempt={view.attempt}
          autosaveMs={autosaveMs}
          onSubmitted={() => void run(() => refresh(view.report.id))}
        />
      )}
      {view.kind === "overview" && (
        <Overview
          report={view.report}
          busy={busy}
          onOpen={(key) => void openModule(view.report, key)}
          onRetry={(attemptId) => void retryMarking(view.report, attemptId)}
          onAnother={another}
        />
      )}
    </main>
  );
}

function Overview({
  report,
  busy,
  onOpen,
  onRetry,
  onAnother,
}: {
  report: MockReport;
  busy: boolean;
  onOpen: (key: ModuleKey) => void;
  onRetry: (attemptId: string) => void;
  onAnother: () => void;
}) {
  const anyMarking = report.modules.some((m) => m.state === "marking");
  const next = report.next_module;
  return (
    <article className="ielts-report" aria-label="Full mock result">
      <section className="ielts-hero" aria-label="Overall band">
        <p className="ielts-eyebrow">
          {report.partial ? "Partial overall band (mock score)" : "Overall band (mock score)"}
        </p>
        {report.overall !== null ? (
          <p className="ielts-hero-band" aria-label={`Mock overall band ${band(report.overall)}`}>
            {band(report.overall)}
          </p>
        ) : (
          <p className="ielts-hint">No skill has a band yet.</p>
        )}
        <p>{report.overall_note}</p>
        <p className="ielts-hint">
          Rounding rule:{" "}
          <a href={report.overall_source_url} target="_blank" rel="noreferrer">
            ielts.org
          </a>
        </p>
        <p className="ielts-disclaimer">{report.disclaimer}</p>
      </section>
      <h2>Skills</h2>
      <ul className="ielts-skills" aria-label="Skill bands">
        {report.modules.map((m) => (
          <li key={m.key} className="ielts-skill" data-state={m.state}>
            <div>
              <strong>{m.label}</strong>
              <span className="ielts-hint"> {STATE_TEXT[m.state]}</span>
              {m.detail && <div className="ielts-hint">{m.detail}</div>}
            </div>
            <div className="ielts-skill-band">
              {m.band !== null ? band(m.band) : m.state === "not_assessed" ? "Not assessed" : "-"}
            </div>
            {m.sittable && (m.state === "not_started" || m.state === "in_progress") && (
              <button className="ielts-primary" onClick={() => onOpen(m.key)} disabled={busy}>
                {m.state === "in_progress" ? `Continue ${m.label}` : `Start ${m.label}`}
              </button>
            )}
            {m.retryable && m.attempt_id && (
              <button
                className="ielts-primary"
                onClick={() => onRetry(m.attempt_id as string)}
                disabled={busy}
              >
                Retry marking
              </button>
            )}
          </li>
        ))}
      </ul>
      {anyMarking && (
        <p className="ielts-hint" role="status" aria-live="polite">
          Marking your Writing. This usually takes under a minute.
        </p>
      )}
      {next && (
        <section className="ielts-next" aria-label="Next module">
          <h2>Next: {next.label}</h2>
          <p>{next.reason}</p>
        </section>
      )}
      <footer className="ielts-report-foot">
        {report.overall !== null && (
          <a
            className="ielts-primary"
            href={`/?attempt=${encodeURIComponent(report.id)}&subject=mock`}
          >
            Discuss with the coach
          </a>
        )}
        <button onClick={onAnother}>Start another full mock</button>
      </footer>
    </article>
  );
}
