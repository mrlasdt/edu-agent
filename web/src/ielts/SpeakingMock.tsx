import { useEffect, useState } from "react";
import { ApiError } from "../request";
import { ModuleNav } from "./ModuleNav";
import { SpeakingReportView } from "./SpeakingReportView";
import { SpeakingSit } from "./SpeakingSit";
import {
  createSpeaking,
  getSpeaking,
  getSpeakingReport,
  regradeSpeaking,
  speakingInfo,
  startSpeaking,
} from "./speakingApi";
import type { SpeakingAttempt, SpeakingInfo, SpeakingReport } from "./speakingTypes";

const KEY = "ielts.speaking.attempt";
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
  | { kind: "intro"; info: SpeakingInfo }
  | { kind: "sit"; attempt: SpeakingAttempt }
  | { kind: "marking"; attempt: SpeakingAttempt }
  | { kind: "report"; attempt: SpeakingAttempt; report: SpeakingReport };

const message = (err: unknown) => (err instanceof Error ? err.message : "Something went wrong");
const POLL_MS = 2000;

export interface SpeakingMockProps {
  pollMs?: number;
}

export function SpeakingMock({ pollMs = POLL_MS }: SpeakingMockProps) {
  const [view, setView] = useState<View>({ kind: "loading" });
  const [info, setInfo] = useState<SpeakingInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    document.title = "IELTS Speaking mock";
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    let cancelled = false;
    (async () => {
      try {
        const i = await speakingInfo(signal);
        if (cancelled) return;
        setInfo(i);
        const id = remembered();
        if (id) {
          try {
            const a = await getSpeaking(id, signal);
            if (cancelled) return;
            if (a.status === "in_progress") return setView({ kind: "sit", attempt: a });
            if (a.status === "submitted" || a.status === "grading" || a.status === "ungraded") {
              return setView({ kind: "marking", attempt: a });
            }
            if (a.status === "graded") {
              const report = await getSpeakingReport(id, signal);
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

  // After the recordings are sent, ask the server until the report is ready.
  const waiting = view.kind === "marking" && view.attempt.status !== "ungraded" ? view.attempt.id : null;
  useEffect(() => {
    if (!waiting) return;
    const controller = new AbortController();
    const timer = window.setInterval(async () => {
      try {
        const a = await getSpeaking(waiting, controller.signal);
        if (a.status === "graded") {
          const report = await getSpeakingReport(waiting, controller.signal);
          setView({ kind: "report", attempt: a, report });
        } else if (a.status === "ungraded") {
          setView({ kind: "marking", attempt: a });
        }
      } catch {
        // a blip while polling: try again on the next tick
      }
    }, pollMs);
    return () => {
      window.clearInterval(timer);
      controller.abort();
    };
  }, [waiting, pollMs]);

  async function begin(sample: boolean) {
    setBusy(true);
    setNotice(null);
    try {
      const created = await createSpeaking(sample);
      remember(created.id);
      setView({ kind: "sit", attempt: await startSpeaking(created.id) });
    } catch (err) {
      setNotice(message(err));
    } finally {
      setBusy(false);
    }
  }

  async function retry(attempt: SpeakingAttempt) {
    setBusy(true);
    setNotice(null);
    try {
      setView({ kind: "marking", attempt: await regradeSpeaking(attempt.id) });
    } catch (err) {
      setNotice(message(err));
    } finally {
      setBusy(false);
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
        <h1>IELTS Speaking mock</h1>
        <a className="ielts-home" href="/">
          Tutor chat
        </a>
      </header>
      <ModuleNav current="speaking" />
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
          <p className="ielts-eyebrow">Speaking</p>
          <ul className="ielts-rules">
            <li>
              Three parts, recorded in the browser: Part 1 questions, Part 2 a cue card with{" "}
              {view.info.part2_prep_s} seconds to prepare and up to {view.info.part2_talk_s / 60}{" "}
              minutes of talking, then Part 3 questions (
              <a href={view.info.format_source_url}>format on IELTS.org</a>).
            </li>
            <li>
              A Gemini audio model listens, writes the transcript and marks Fluency and coherence,
              Lexical resource and Grammatical range and accuracy against the{" "}
              <a href={view.info.descriptors_source_url}>public band descriptors</a>.
            </li>
            <li>
              The band is partial. Pronunciation is not assessed:{" "}
              {Object.values(view.info.not_assessed).join(" ")}
            </li>
          </ul>
          <p className="ielts-flags">{view.info.set_note}</p>
          <div className="ielts-actions">
            <button className="ielts-primary" onClick={() => void begin(false)} disabled={busy}>
              {busy ? "Starting..." : "Start Speaking"}
            </button>
            <button onClick={() => void begin(true)} disabled={busy}>
              Quick sample (about 2 minutes)
            </button>
          </div>
        </div>
      )}
      {view.kind === "sit" && (
        <SpeakingSit
          key={view.attempt.id}
          attempt={view.attempt}
          onSubmitted={(a) => setView({ kind: "marking", attempt: a })}
        />
      )}
      {view.kind === "marking" && view.attempt.status !== "ungraded" && (
        <p className="ielts-marking" role="status" aria-live="polite">
          <span className="ielts-spinner" aria-hidden="true" /> Marking your Speaking. This usually
          takes under a minute.
        </p>
      )}
      {view.kind === "marking" && view.attempt.status === "ungraded" && (
        <div role="alert" className="ielts-error">
          <p>{view.attempt.error ?? "Marking failed."}</p>
          {view.attempt.retryable ? (
            <button
              className="ielts-primary"
              onClick={() => void retry(view.attempt)}
              disabled={busy}
            >
              Retry marking
            </button>
          ) : (
            <button onClick={again}>Start again</button>
          )}
        </div>
      )}
      {view.kind === "report" && <SpeakingReportView report={view.report} onAnother={again} />}
    </main>
  );
}
