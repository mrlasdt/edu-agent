import { useEffect, useState } from "react";
import { ApiError } from "../request";
import { ModuleNav } from "./ModuleNav";
import { ReadingReportView } from "./ReadingReportView";
import { ReadingSit } from "./ReadingSit";
import {
  createReading,
  getReading,
  getReadingReport,
  readingInfo,
  startReading,
} from "./readingApi";
import type { ReadingAttempt, ReadingInfo, ReadingReport } from "./readingTypes";

const KEY = "ielts.reading.attempt";
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
  | { kind: "intro"; info: ReadingInfo }
  | { kind: "sit"; attempt: ReadingAttempt }
  | { kind: "report"; attempt: ReadingAttempt; report: ReadingReport };

const message = (err: unknown) => (err instanceof Error ? err.message : "Something went wrong");

export interface ReadingMockProps {
  autosaveMs?: number;
}

export function ReadingMock({ autosaveMs }: ReadingMockProps) {
  const [view, setView] = useState<View>({ kind: "loading" });
  const [info, setInfo] = useState<ReadingInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    document.title = "IELTS Reading mock";
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    let cancelled = false;
    (async () => {
      try {
        const i = await readingInfo(signal);
        if (cancelled) return;
        setInfo(i);
        const id = remembered();
        if (id) {
          try {
            const a = await getReading(id, signal);
            if (cancelled) return;
            if (a.status === "in_progress" && a.paper) return setView({ kind: "sit", attempt: a });
            if (a.status === "graded") {
              const report = await getReadingReport(id, signal);
              if (!cancelled) setView({ kind: "report", attempt: a, report });
              return;
            }
          } catch (err) {
            if (!(err instanceof ApiError && err.status === 404)) throw err;
          }
          remember(null);
        }
        if (!cancelled) setView({ kind: "intro", info: i });
      } catch (err) {
        if (!cancelled) setView({ kind: "failed", message: message(err) });
      }
    })();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, []);

  async function begin() {
    setBusy(true);
    setNotice(null);
    try {
      const created = await createReading();
      remember(created.id);
      setView({ kind: "sit", attempt: await startReading(created.id) });
    } catch (err) {
      setNotice(message(err));
    } finally {
      setBusy(false);
    }
  }

  async function submitted(attempt: ReadingAttempt) {
    try {
      const report = await getReadingReport(attempt.id);
      setView({ kind: "report", attempt, report });
    } catch (err) {
      setView({ kind: "failed", message: message(err) });
    }
  }

  function again() {
    remember(null);
    if (info) setView({ kind: "intro", info });
    else window.location.reload();
  }

  return (
    <main className="app ielts" data-view={view.kind}>
      <header className="app-header">
        <h1>IELTS Reading mock</h1>
        <a className="ielts-home" href="/">
          Tutor chat
        </a>
      </header>
      <ModuleNav current="reading" />
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
        <div className="ielts-ready">
          <p className="ielts-eyebrow">General Training Reading</p>
          <ul className="ielts-rules">
            <li>
              You have {view.info.allowed_s / 60} minutes, the official General Training Reading time (
              <a href={view.info.timing_source_url}>IELTS.org</a>). The clock runs on the server, so
              refreshing the page does not reset it.
            </li>
            <li>
              {view.info.question_count} questions on short passages. Your answers autosave about every 10
              seconds, and are submitted for you when time is up.
            </li>
            <li>It is marked in code against the answer key. You see the evidence sentence for every miss.</li>
          </ul>
          <p className="ielts-flags">{view.info.set_note}</p>
          <p className="ielts-flags">{view.info.keys_note}</p>
          <p className="ielts-hint">{view.info.band_note}</p>
          <div className="ielts-actions">
            <button className="ielts-primary" onClick={() => void begin()} disabled={busy}>
              {busy ? "Starting..." : "Start the 60 minutes"}
            </button>
          </div>
        </div>
      )}
      {view.kind === "sit" && (
        <ReadingSit
          key={view.attempt.id}
          attempt={view.attempt}
          autosaveMs={autosaveMs}
          onSubmitted={(a) => void submitted(a)}
        />
      )}
      {view.kind === "report" && (
        <ReadingReportView attemptId={view.attempt.id} report={view.report} onAnother={again} />
      )}
    </main>
  );
}
