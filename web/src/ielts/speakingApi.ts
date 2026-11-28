import { ApiError, UNREACHABLE, requestJson } from "../request";
import type { SpeakingAttempt, SpeakingInfo, SpeakingReport } from "./speakingTypes";

const base = "/ielts/speaking";

export const speakingInfo = (signal?: AbortSignal) =>
  requestJson<SpeakingInfo>(`${base}/info`, { signal });
/** `sample` makes the quick sample set: one question per part and short timings. */
export const createSpeaking = (sample = false) =>
  requestJson<SpeakingAttempt>(`${base}/attempts${sample ? "?sample=true" : ""}`, { method: "POST" });
export const getSpeaking = (id: string, signal?: AbortSignal) =>
  requestJson<SpeakingAttempt>(`${base}/attempts/${id}`, { signal });
export const startSpeaking = (id: string) =>
  requestJson<SpeakingAttempt>(`${base}/attempts/${id}/start`, { method: "POST" });
export const submitSpeaking = (id: string) =>
  requestJson<SpeakingAttempt>(`${base}/attempts/${id}/submit`, { method: "POST" });
export const regradeSpeaking = (id: string) =>
  requestJson<SpeakingAttempt>(`${base}/attempts/${id}/regrade`, { method: "POST" });
export const getSpeakingReport = (id: string, signal?: AbortSignal) =>
  requestJson<SpeakingReport>(`${base}/attempts/${id}/report`, { signal });

/** The recorder's mime type ("audio/webm;codecs=opus") without the codec list the server does not need. */
export const bareMime = (mime: string) => mime.split(";")[0].trim().toLowerCase();

/** Send one part's recording as the raw body; the browser-measured length goes in `X-Duration-S`. */
export async function uploadSpeakingPart(
  id: string,
  part: number,
  audio: Blob,
  durationS: number
): Promise<SpeakingAttempt> {
  let res: Response;
  try {
    res = await fetch(`${base}/attempts/${id}/parts/${part}/audio`, {
      method: "PUT",
      headers: {
        "Content-Type": bareMime(audio.type),
        "X-Duration-S": Math.max(0.1, durationS).toFixed(1),
      },
      body: audio,
    });
  } catch {
    throw new ApiError(0, UNREACHABLE);
  }
  if (!res.ok) {
    let detail = `Upload failed (${res.status})`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string" && body.detail) detail = body.detail;
    } catch {
      // keep the generic message
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as SpeakingAttempt;
}
