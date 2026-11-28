import { useCallback, useEffect, useState } from "react";
import { ApiError } from "../request";
import { band, clock } from "./format";
import { ModuleNav } from "./ModuleNav";
import { SAMPLE_HINT, SampleBadge } from "./SampleBadge";
import { ReportView } from "./ReportView";
import { DISCLAIMER } from "./types";
import { WritingModuleSit } from "./WritingModuleSit";
import {
  createWritingModule,
  getWritingModule,
  getWritingModuleReport,
  regradeWritingModule,
  startWritingModule,
  writingModuleInfo,
} from "./writingModuleApi";
import type {
  WritingModuleAttempt,
  WritingModuleInfo,
  WritingModuleReport,
} from "./writingModuleTypes";

const KEY = "ielts.writingmodule.attempt";
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
  | { kind: "sit"; attempt: WritingModuleAttempt }
  | { kind: "marking"; attempt: WritingModuleAttempt }
  | { kind: "report"; attempt: WritingModuleAttempt; report: WritingModuleReport };

const message = (err: unknown) => (err instanceof Error ? err.message : "Something went wrong");
const POLL_MS = 2000;

export interface WritingModuleProps {
  pollMs?: number;
  autosaveMs?: number;
}

export function WritingModule({ pollMs = POLL_MS, autosaveMs }: WritingModuleProps) {
  const [view, setView] = useState<View>({ kind: "loading" });
  const [info, setInfo] = useState<WritingModuleInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    document.title = "IELTS Writing module mock";
  }, []);

  const route = useCallback(async (a: WritingModuleAttempt, signal?: AbortSignal): Promise<View> => {
    switch (a.status) {
      case "created":
        return { kind: "intro" };
      case "in_progress":
        return { kind: "sit", attempt: a };
      case "submitted":
      case "grading":
        return { kind: "marking", attempt: a };
      case "graded":
      case "ungraded":
        try {
          return { kind: "report", attempt: a, report: await getWritingModuleReport(a.id, signal) };
        } catch (err) {
          if (err instanceof ApiError && err.status === 409) return { kind: "marking", attempt: a };
          throw err;
        }
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    let cancelled = false;
    (async () => {
      try {
        const i = await writingModuleInfo(signal);
        if (cancelled) return;
        setInfo(i);
        const id = remembered();
        if (id) {
          try {
            const next = await route(await getWritingModule(id, signal), signal);
            if (!cancelled) setView(next);
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
  }, [route]);

  // While marking runs, ask the server until it has a report (or says marking failed).
  const waitingFor =
    view.kind === "marking" && view.attempt.status !== "ungraded" ? view.attempt.id : null;
  useEffect(() => {
    if (!waitingFor) return;
    const controller = new AbortController();
    const id = window.setInterval(async () => {
      try {
        const next = await route(await getWritingModule(waitingFor, controller.signal), controller.signal);
        setView((cur) => (cur.kind === "marking" && cur.attempt.id === waitingFor ? next : cur));
      } catch {
        // a blip while polling: try again on the next tick
      }
    }, pollMs);
    return () => {
      window.clearInterval(id);
      controller.abort();
    };
  }, [waitingFor, route, pollMs]);

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

  const begin = (sample = false) =>
    run(async () => {
      const created = await createWritingModule(sample);
      remember(created.id);
      setView({ kind: "sit", attempt: await startWritingModule(created.id) });
    });

  const retry = (a: WritingModuleAttempt) =>
    run(async () => setView({ kind: "marking", attempt: await regradeWritingModule(a.id) }));

  const another = () => {
    remember(null);
    setNotice(null);
    setView({ kind: "intro" });
  };

  return (
    <main className="app ielts" data-view={view.kind}>
      <header className="app-header">
        <h1>IELTS Writing module mock</h1>
        <a className="ielts-home" href="/">
          Tutor chat
        </a>
      </header>
      <ModuleNav current="writing-module" />
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
      {view.kind === "intro" && info && (
        <section className="ielts-ready" aria-label="Writing module">
          <h2>Writing module: a letter, then an essay</h2>
          <p>
            One {clock(info.allowed_s)} clock covers both tasks: a letter of at least {info.t1_min_words}{" "}
            words (about {info.t1_minutes} minutes) and an essay of at least {info.t2_min_words} words
            (about {info.t2_minutes} minutes). You can move between the two at any time. Each is marked
            separately, and you get one combined Writing band.
          </p>
          <p className="ielts-hint">{info.weights_note}</p>
          <p className="ielts-hint">
            Sources:{" "}
            {info.source_urls.map((u, i) => (
              <span key={u}>
                {i > 0 && ", "}
                <a href={u} target="_blank" rel="noreferrer">
                  {new URL(u).pathname.split("/").filter(Boolean).pop()}
                </a>
              </span>
            ))}
          </p>
          <p className="ielts-disclaimer">{DISCLAIMER}</p>
          <div className="ielts-actions">
            <button className="ielts-primary" onClick={() => void begin()} disabled={busy}>
              {busy ? "Starting..." : "Start the 60 minute clock"}
            </button>
            <button onClick={() => void begin(true)} disabled={busy} title={SAMPLE_HINT}>
              Quick sample test (3 minutes)
            </button>
          </div>
        </section>
      )}
      {view.kind === "sit" && (
        <WritingModuleSit
          key={view.attempt.id}
          attempt={view.attempt}
          autosaveMs={autosaveMs}
          onSubmitted={(a) => setView({ kind: "marking", attempt: a })}
        />
      )}
      {view.kind === "marking" &&
        (view.attempt.status === "ungraded" ? (
          <div className="ielts-marking" role="alert">
            <h2>Your writing could not be marked</h2>
            <p>{view.attempt.error ?? "Marking did not finish."}</p>
            <p className="ielts-hint">
              Your letter and essay are saved. {view.attempt.retryable ? "Marking can be retried." : ""}
            </p>
            <div className="ielts-actions">
              {view.attempt.retryable && (
                <button className="ielts-primary" onClick={() => void retry(view.attempt)} disabled={busy}>
                  {busy ? "Retrying..." : "Try marking again"}
                </button>
              )}
              <button onClick={another}>Start over</button>
            </div>
          </div>
        ) : (
          <div className="ielts-marking" role="status" aria-live="polite">
            <div className="ielts-spinner" aria-hidden="true" />
            <h2>Marking your letter and essay</h2>
            <p className="ielts-hint">
              Both tasks are marked at the same time, three independent passes each. This usually takes under
              a minute.
            </p>
          </div>
        ))}
      {view.kind === "report" && (
        <article className="ielts-report" aria-label="Writing module report">
          {view.report.writing_band !== null ? (
            <section className="ielts-hero" aria-label="Combined Writing band">
              <p className="ielts-eyebrow">
                {view.report.sample && <SampleBadge />}{" "}
                {view.report.sample ? "Sample Writing band (not a real mock score)" : "Combined Writing band (mock score)"}
              </p>
              <p className="ielts-hero-band" aria-label={`Mock Writing band ${band(view.report.writing_band)}`}>
                {band(view.report.writing_band)}
              </p>
              <p>
                Task 1 {band(view.report.t1.estimated_band?.overall ?? 0)} and Task 2{" "}
                {band(view.report.t2.estimated_band?.overall ?? 0)}.
              </p>
              <p className="ielts-hint">{view.report.weights_note}</p>
              <p className="ielts-disclaimer">{DISCLAIMER}</p>
            </section>
          ) : (
            <section className="ielts-hero" aria-label="No combined band">
              <h2>No combined Writing band</h2>
              <p>
                One of the two tasks could not be marked, so there is no combined band. Each task's own
                report is below.
              </p>
              <p className="ielts-disclaimer">{DISCLAIMER}</p>
            </section>
          )}
          <p className="ielts-hint">
            Sources for the weighting:{" "}
            {view.report.weights_source_urls.map((u, i) => (
              <span key={u}>
                {i > 0 && ", "}
                <a href={u} target="_blank" rel="noreferrer">
                  {u}
                </a>
              </span>
            ))}
          </p>
          <h2>Task 1 - Letter</h2>
          <ReportView report={view.report.t1} essay={view.attempt.t1.text} onAnother={another} embedded />
          <h2>Task 2 - Essay</h2>
          <ReportView report={view.report.t2} essay={view.attempt.t2.text} onAnother={another} embedded />
          <footer className="ielts-report-foot">
            {view.report.status === "graded" && (
              <a className="ielts-primary" href={`/?attempt=${encodeURIComponent(view.attempt.id)}`}>
                Discuss this report
              </a>
            )}
            <button className="ielts-primary" onClick={another}>
              Sit another Writing module
            </button>
          </footer>
        </article>
      )}
    </main>
  );
}
