import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  createAttempt,
  getAttempt,
  getReport,
  listAttempts,
  listPrompts,
  regradeAttempt,
  startAttempt,
} from "./api";
import { MarkingScreen } from "./MarkingScreen";
import { PromptPicker } from "./PromptPicker";
import { ReadyScreen } from "./ReadyScreen";
import { ReportView } from "./ReportView";
import { WritingScreen } from "./WritingScreen";
import { forgetDraft, rememberAttempt, rememberedAttempt, unsavedDraft } from "./storage";
import type { Attempt, Report, WritingPrompt } from "./types";

type View =
  | { kind: "loading" }
  | { kind: "failed"; message: string }
  | { kind: "picker"; prompts: WritingPrompt[]; history: Attempt[] }
  | { kind: "ready"; attempt: Attempt }
  | { kind: "writing"; attempt: Attempt; initialText: string }
  | { kind: "marking"; attempt: Attempt }
  | { kind: "report"; attempt: Attempt; report: Report };

const POLL_MS = 2000;
const message = (err: unknown) => (err instanceof Error ? err.message : "Something went wrong");

export interface IeltsMockProps {
  pollMs?: number;
  autosaveMs?: number;
}

export function IeltsMock({ pollMs = POLL_MS, autosaveMs }: IeltsMockProps) {
  const [view, setView] = useState<View>({ kind: "loading" });
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const prompts = useRef<WritingPrompt[]>([]);

  useEffect(() => {
    document.title = "IELTS-style Writing mock";
  }, []);

  /** Where an attempt belongs on screen, given what the server says about it. */
  const route = useCallback(async (attempt: Attempt): Promise<View> => {
    switch (attempt.status) {
      case "created":
        return { kind: "ready", attempt };
      case "in_progress":
        return {
          kind: "writing",
          attempt,
          initialText: unsavedDraft(attempt.id) ?? attempt.draft,
        };
      case "submitted":
      case "grading":
        return { kind: "marking", attempt };
      case "graded":
      case "ungraded":
        try {
          return { kind: "report", attempt, report: await getReport(attempt.id) };
        } catch (err) {
          if (err instanceof ApiError && err.status === 409) return { kind: "marking", attempt };
          throw err;
        }
    }
  }, []);

  const showPicker = useCallback(async () => {
    rememberAttempt(null);
    setNotice(null);
    const [list, history] = await Promise.all([
      prompts.current.length ? Promise.resolve(prompts.current) : listPrompts(),
      listAttempts().catch(() => [] as Attempt[]),
    ]);
    prompts.current = list;
    setView({ kind: "picker", prompts: list, history: history.slice(0, 5) });
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const id = rememberedAttempt();
        if (id) {
          try {
            const next = await route(await getAttempt(id));
            if (!cancelled) setView(next);
            return;
          } catch (err) {
            if (!(err instanceof ApiError && err.status === 404)) throw err;
          }
        }
        if (!cancelled) await showPicker();
      } catch (err) {
        if (!cancelled) setView({ kind: "failed", message: message(err) });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [route, showPicker]);

  const waitingFor = view.kind === "marking" && view.attempt.status !== "ungraded" ? view.attempt.id : null;
  useEffect(() => {
    if (!waitingFor) return;
    const id = window.setInterval(async () => {
      try {
        const next = await route(await getAttempt(waitingFor));
        setView((current) =>
          current.kind === "marking" && current.attempt.id === waitingFor ? next : current
        );
      } catch {
        // a blip while polling: try again on the next tick
      }
    }, pollMs);
    return () => window.clearInterval(id);
  }, [waitingFor, route, pollMs]);

  async function pick(prompt: WritingPrompt) {
    setBusy(true);
    setNotice(null);
    try {
      const attempt = await createAttempt(prompt.id);
      rememberAttempt(attempt.id);
      setView({ kind: "ready", attempt });
    } catch (err) {
      setNotice(message(err));
    } finally {
      setBusy(false);
    }
  }

  async function start(attempt: Attempt) {
    setBusy(true);
    setNotice(null);
    try {
      const started = await startAttempt(attempt.id);
      setView({ kind: "writing", attempt: started, initialText: started.draft });
    } catch (err) {
      setNotice(message(err));
    } finally {
      setBusy(false);
    }
  }

  async function retry(attempt: Attempt) {
    setBusy(true);
    try {
      setView({ kind: "marking", attempt: await regradeAttempt(attempt.id) });
    } catch (err) {
      setNotice(message(err));
    } finally {
      setBusy(false);
    }
  }

  async function another() {
    if (view.kind === "report" || view.kind === "marking") forgetDraft(view.attempt.id);
    setBusy(true);
    try {
      await showPicker();
    } catch (err) {
      setView({ kind: "failed", message: message(err) });
    } finally {
      setBusy(false);
    }
  }

  async function open(attempt: Attempt) {
    try {
      rememberAttempt(attempt.id);
      setView(await route(attempt));
    } catch (err) {
      setNotice(message(err));
    }
  }

  return (
    <main className="app ielts" data-view={view.kind}>
      <header className="app-header">
        <h1>IELTS-style Writing mock</h1>
        <a className="ielts-home" href="/">
          GRE tutor
        </a>
      </header>
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
      {view.kind === "picker" && (
        <PromptPicker
          prompts={view.prompts}
          history={view.history}
          busy={busy}
          onPick={(p) => void pick(p)}
          onOpen={(a) => void open(a)}
        />
      )}
      {view.kind === "ready" && (
        <ReadyScreen
          attempt={view.attempt}
          busy={busy}
          error={notice}
          onStart={() => void start(view.attempt)}
          onBack={() => void another()}
        />
      )}
      {view.kind === "writing" && (
        <WritingScreen
          key={view.attempt.id}
          attempt={view.attempt}
          initialText={view.initialText}
          autosaveMs={autosaveMs}
          onSubmitted={(a) => {
            setView({ kind: "marking", attempt: a });
          }}
        />
      )}
      {view.kind === "marking" && (
        <MarkingScreen
          attempt={view.attempt}
          retrying={busy}
          onRetry={() => void retry(view.attempt)}
          onBack={() => void another()}
        />
      )}
      {view.kind === "report" && (
        <ReportView report={view.report} essay={view.attempt.draft} onAnother={() => void another()} />
      )}
    </main>
  );
}
