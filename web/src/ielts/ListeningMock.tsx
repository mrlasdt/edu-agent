import { useEffect, useState } from "react";
import { ApiError } from "../request";
import { ModuleNav } from "./ModuleNav";
import { SAMPLE_HINT } from "./SampleBadge";
import { ListeningReportView } from "./ListeningReportView";
import { ListeningSit } from "./ListeningSit";
import {
  createListening,
  getListening,
  getListeningReport,
  listeningInfo,
  startListening,
} from "./listeningApi";
import type { ListeningAttempt, ListeningInfo, ListeningReport } from "./listeningTypes";

const KEY = "ielts.listening.attempt";
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
  | { kind: "intro"; info: ListeningInfo }
  | { kind: "sit"; attempt: ListeningAttempt }
  | { kind: "report"; attempt: ListeningAttempt; report: ListeningReport };

const message = (err: unknown) => (err instanceof Error ? err.message : "Something went wrong");

export interface ListeningMockProps {
  autosaveMs?: number;
}

export function ListeningMock({ autosaveMs }: ListeningMockProps) {
  const [view, setView] = useState<View>({ kind: "loading" });
  const [info, setInfo] = useState<ListeningInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    document.title = "IELTS Listening mock";
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    let cancelled = false;
    (async () => {
      try {
        const i = await listeningInfo(signal);
        if (cancelled) return;
        setInfo(i);
        const id = remembered();
        if (id) {
          try {
            const a = await getListening(id, signal);
            if (cancelled) return;
            if (a.status === "in_progress" && a.paper) return setView({ kind: "sit", attempt: a });
            if (a.status === "graded") {
              const report = await getListeningReport(id, signal);
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

  async function begin(sample = false) {
    setBusy(true);
    setNotice(null);
    try {
      const created = await createListening(sample);
      remember(created.id);
      setView({ kind: "sit", attempt: await startListening(created.id) });
    } catch (err) {
      setNotice(message(err));
    } finally {
      setBusy(false);
    }
  }

  async function submitted(attempt: ListeningAttempt) {
    try {
      const report = await getListeningReport(attempt.id);
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
        <h1>IELTS Listening mock</h1>
        <a className="ielts-home" href="/">
          Tutor chat
        </a>
      </header>
      <ModuleNav current="listening" />
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
          <p className="ielts-eyebrow">General Training Listening</p>
          <ul className="ielts-rules">
            <li>
              You have {view.info.allowed_s / 60} minutes, the official General Training Listening time (
              <a href={view.info.timing_source_url}>IELTS.org</a>). The clock runs on the server, so
              refreshing the page does not reset it.
            </li>
            <li>
              {view.info.question_count} questions over four recordings. Press play on each recording
              (you need sound on). Your answers autosave about every 10 seconds, and are submitted for you
              when time is up.
            </li>
            <li>
              It is marked in code against the answer key. For every miss you see what the recording says,
              and the transcripts.
            </li>
          </ul>
          <p className="ielts-flags">{view.info.set_note}</p>
          <p className="ielts-flags">{view.info.keys_note}</p>
          <p className="ielts-hint">{view.info.band_note}</p>
          <div className="ielts-actions">
            <button className="ielts-primary" onClick={() => void begin()} disabled={busy}>
              {busy ? "Starting..." : `Start the ${view.info.allowed_s / 60} minutes`}
            </button>
            <button onClick={() => void begin(true)} disabled={busy} title={SAMPLE_HINT}>
              Quick sample test (about 4 minutes)
            </button>
          </div>
        </div>
      )}
      {view.kind === "sit" && (
        <ListeningSit
          key={view.attempt.id}
          attempt={view.attempt}
          autosaveMs={autosaveMs}
          onSubmitted={(a) => void submitted(a)}
        />
      )}
      {view.kind === "report" && <ListeningReportView report={view.report} onAnother={again} />}
    </main>
  );
}
