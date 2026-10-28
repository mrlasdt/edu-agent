import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, saveDraft, saveDraftOnExit } from "./api";
import { backupDraft } from "./storage";
import type { Attempt } from "./types";

export type SaveState =
  | { kind: "idle" }
  | { kind: "saving" }
  | { kind: "saved"; at: Date }
  | { kind: "error"; message: string };

export const AUTOSAVE_MS = 10_000;

/**
 * Saves `text` to the server about every `intervalMs` while it has changed, keeps a local
 * copy so a dropped connection never loses words, and tries once more when the page closes.
 * `serverText` is the draft the server already holds, so nothing is saved until it changes.
 * `flush` saves immediately and resolves with the server's view (or null if it failed).
 */
export function useAutosave(
  attemptId: string,
  text: string,
  serverText: string,
  onServerView: (a: Attempt) => void,
  intervalMs: number = AUTOSAVE_MS
) {
  const [state, setState] = useState<SaveState>({ kind: "idle" });
  const latest = useRef(text);
  const saved = useRef<string>(serverText); // what the server already holds
  const inFlight = useRef<Promise<Attempt | null> | null>(null);
  const onView = useRef(onServerView);
  onView.current = onServerView;
  latest.current = text;

  const flush = useCallback(async (): Promise<Attempt | null> => {
    if (inFlight.current) await inFlight.current;
    const body = latest.current;
    if (saved.current === body) return null;
    setState({ kind: "saving" });
    const run = (async () => {
      try {
        const view = await saveDraft(attemptId, body);
        saved.current = body;
        backupDraft(attemptId, body, latest.current === body);
        setState({ kind: "saved", at: new Date() });
        onView.current(view);
        return view;
      } catch (err) {
        const message = err instanceof ApiError ? err.message : "Could not save";
        // a 409 means the time is up: the page is about to submit; anything else is retried
        setState({ kind: "error", message });
        return null;
      } finally {
        inFlight.current = null;
      }
    })();
    inFlight.current = run;
    return run;
  }, [attemptId]);

  useEffect(() => {
    backupDraft(attemptId, text, saved.current === text);
  }, [attemptId, text]);

  useEffect(() => {
    const id = window.setInterval(() => void flush(), intervalMs);
    return () => window.clearInterval(id);
  }, [flush, intervalMs]);

  useEffect(() => {
    const onHide = () => {
      if (saved.current !== latest.current) saveDraftOnExit(attemptId, latest.current);
    };
    window.addEventListener("pagehide", onHide);
    return () => window.removeEventListener("pagehide", onHide);
  }, [attemptId]);

  return { state, flush };
}
