import { useEffect, useRef, useState } from "react";
import "./speaking.css";
import { clock } from "./format";
import { submitSpeaking, uploadSpeakingPart } from "./speakingApi";
import type { SpeakingAttempt, SpeakingPartPaper } from "./speakingTypes";

interface Props {
  attempt: SpeakingAttempt;
  onSubmitted: (attempt: SpeakingAttempt) => void;
}

type Phase = "ready" | "prep" | "recording" | "uploading" | "done";
type Part = 1 | 2 | 3;

/** Parts 1 and 3 stop a few seconds before the server's cap, so the measured length always fits. */
const CAP_MARGIN_S = 5;
const TICK_MS = 200;

const supported = () =>
  typeof navigator !== "undefined" &&
  !!navigator.mediaDevices?.getUserMedia &&
  typeof MediaRecorder !== "undefined";

const errorText = (err: unknown) => (err instanceof Error ? err.message : "Something went wrong");

/** Seconds since `since` (a performance.now() stamp), refreshed while `active`. */
function useElapsed(since: number | null) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (since === null) return;
    const id = window.setInterval(() => setNow(performance.now()), TICK_MS);
    return () => window.clearInterval(id);
  }, [since]);
  return since === null ? 0 : Math.max(0, (now - since) / 1000);
}

function CueCard({ part, prepLeft }: { part: SpeakingPartPaper; prepLeft: number | null }) {
  return (
    <div className="speaking-card" aria-label="Cue card">
      <p>
        <strong>{part.topic}</strong>
      </p>
      <p className="ielts-hint">You should say:</p>
      <ul>
        {part.points?.map((p) => (
          <li key={p}>{p}</li>
        ))}
      </ul>
      {part.note && <p>{part.note}</p>}
      {prepLeft !== null && (
        <p role="timer" aria-label="Preparation time left" className="speaking-prep">
          Prepare: {clock(prepLeft)}
        </p>
      )}
    </div>
  );
}

export function SpeakingSit({ attempt, onSubmitted }: Props) {
  const paper = attempt.paper;
  const firstMissing = ([1, 2, 3] as Part[]).find((n) => !attempt.uploaded.includes(n)) ?? 3;
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [part, setPart] = useState<Part>(firstMissing);
  const [phase, setPhase] = useState<Phase>(attempt.uploaded.includes(firstMissing) ? "done" : "ready");
  const [uploaded, setUploaded] = useState<number[]>(attempt.uploaded);
  const [heard, setHeard] = useState<{ url: string; seconds: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  streamRef.current = stream;

  const paperPart = paper.parts.find((p) => p.part === part)!;
  const prepS = paperPart.prep_s ?? 0;
  const talkS = paperPart.talk_s ?? 0;
  // the longest this part is allowed to run
  const capS = part === 2 ? talkS : Math.max(5, paper.max_clip_s[part - 1] - CAP_MARGIN_S);
  const elapsed = useElapsed(phase === "prep" || phase === "recording" ? startedAt : null);
  const prepLeft = phase === "prep" ? Math.max(0, Math.ceil(prepS - elapsed)) : null;

  // release the microphone and the playback URL when the page goes away
  useEffect(
    () => () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    },
    []
  );
  useEffect(() => () => (heard ? URL.revokeObjectURL(heard.url) : undefined), [heard]);

  async function allowMic() {
    setError(null);
    try {
      setStream(await navigator.mediaDevices.getUserMedia({ audio: true }));
    } catch {
      setError("The microphone is blocked. Allow it for this page in the browser, then try again.");
    }
  }

  function beginRecording() {
    if (!stream) return;
    setError(null);
    chunks.current = [];
    const rec = new MediaRecorder(stream);
    rec.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.current.push(e.data);
    };
    const began = performance.now();
    rec.onstop = () => {
      const seconds = Math.min((performance.now() - began) / 1000, paper.max_clip_s[part - 1]);
      void upload(new Blob(chunks.current, { type: rec.mimeType || "audio/webm" }), seconds);
    };
    recorder.current = rec;
    rec.start();
    setStartedAt(began);
    setPhase("recording");
  }

  async function upload(blob: Blob, seconds: number) {
    setPhase("uploading");
    try {
      const view = await uploadSpeakingPart(attempt.id, part, blob, seconds);
      setUploaded(view.uploaded);
      setHeard({ url: URL.createObjectURL(blob), seconds });
      setPhase("done");
    } catch (err) {
      setError(`${errorText(err)} Record this part again.`);
      setPhase("ready");
    }
  }

  const stop = () => {
    if (recorder.current && recorder.current.state !== "inactive") recorder.current.stop();
  };

  function beginPart() {
    if (part === 2 && prepS > 0) {
      setStartedAt(performance.now());
      setPhase("prep");
    } else {
      beginRecording();
    }
  }

  // the preparation ends by itself, and a long turn stops at its limit
  useEffect(() => {
    if (phase === "prep" && prepLeft === 0) beginRecording();
    if (phase === "recording" && elapsed >= capS) stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, prepLeft, elapsed, capS]);

  function next() {
    setHeard(null);
    setStartedAt(null);
    const following = (part + 1) as Part;
    setPart(following);
    setPhase(uploaded.includes(following) ? "done" : "ready");
  }

  async function submit() {
    setSubmitting(true);
    setError(null);
    try {
      onSubmitted(await submitSpeaking(attempt.id));
    } catch (err) {
      setSubmitting(false);
      setError(errorText(err));
    }
  }

  if (!supported()) {
    return (
      <p className="ielts-error" role="alert">
        This browser cannot record audio. Use a recent Chrome, Edge, Firefox or Safari, on https or
        localhost.
      </p>
    );
  }

  if (!stream) {
    return (
      <div className="ielts-ready">
        <h2>Allow the microphone</h2>
        <p>
          You will record three parts. Each recording is sent to Google&apos;s Gemini model to be
          transcribed and marked, then the audio is deleted.
        </p>
        {error && (
          <p className="ielts-error" role="alert">
            {error}
          </p>
        )}
        <div className="ielts-actions">
          <button className="ielts-primary" onClick={() => void allowMic()}>
            Allow the microphone
          </button>
        </div>
      </div>
    );
  }

  const last = part === 3;
  const recorded = uploaded.includes(part);
  return (
    <div className="ielts-writing speaking-sit">
      <ol className="speaking-steps" aria-label="Parts">
        {([1, 2, 3] as Part[]).map((n) => (
          <li
            key={n}
            aria-current={n === part ? "step" : undefined}
            data-done={uploaded.includes(n) || undefined}
          >
            Part {n}
          </li>
        ))}
      </ol>
      {paper.sample && (
        <p className="ielts-flags">
          Quick sample: one short question per part and short timings, marked the same way as the
          full test. Its band says little about your level.
        </p>
      )}
      <section className="ielts-prompt" aria-label={`Part ${part}`}>
        <p className="ielts-eyebrow">{paperPart.title}</p>
        {part === 2 ? (
          <CueCard part={paperPart} prepLeft={prepLeft} />
        ) : (
          <ul className="speaking-questions">
            {paperPart.questions?.map((q) => (
              <li key={q}>{q}</li>
            ))}
          </ul>
        )}
      </section>
      {phase === "recording" && (
        <p className="speaking-live" role="status">
          <span className="speaking-dot" aria-hidden="true" /> Recording{" "}
          <span role="timer" aria-label="Recording time">
            {clock(elapsed)}
          </span>{" "}
          <span className="ielts-hint">of up to {clock(capS)}</span>
        </p>
      )}
      {phase === "uploading" && (
        <p className="ielts-hint" role="status">
          Saving the recording...
        </p>
      )}
      {phase === "done" && recorded && (
        <p className="ielts-hint" role="status">
          Recorded{heard ? ` ${clock(heard.seconds)}` : ""}. Listen to it, or record it again.
          {heard && <audio controls src={heard.url} aria-label={`Part ${part} recording`} />}
        </p>
      )}
      {error && (
        <p className="ielts-error" role="alert">
          {error}
        </p>
      )}
      <div className="ielts-actions">
        {phase === "ready" && (
          <button className="ielts-primary" onClick={beginPart}>
            {part === 2 ? `Start ${prepS} seconds of preparation` : "Start recording"}
          </button>
        )}
        {phase === "prep" && (
          <button className="ielts-primary" onClick={beginRecording}>
            Start talking now
          </button>
        )}
        {phase === "recording" && (
          <button className="ielts-primary" onClick={stop}>
            Stop
          </button>
        )}
        {phase === "done" && (
          <>
            {last ? (
              <button className="ielts-primary" onClick={() => void submit()} disabled={submitting}>
                {submitting ? "Sending..." : "Submit for marking"}
              </button>
            ) : (
              <button className="ielts-primary" onClick={next}>
                Next: Part {part + 1}
              </button>
            )}
            <button onClick={() => setPhase("ready")} disabled={submitting}>
              Record again
            </button>
          </>
        )}
      </div>
    </div>
  );
}
