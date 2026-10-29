// Per-viewer conveniences only: which attempt to resume, and a local copy of the draft
// that is used when the server could not be reached. Every access can throw (private mode).
const ATTEMPT_KEY = "ielts.attempt";
const draftKey = (id: string) => `ielts.draft.${id}`;

interface DraftBackup {
  text: string;
  saved: boolean;
}

function read(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string | null): void {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    // storage unavailable: the server copy is the only copy
  }
}

export const rememberAttempt = (id: string | null) => write(ATTEMPT_KEY, id);
export const rememberedAttempt = () => read(ATTEMPT_KEY);

export function backupDraft(id: string, text: string, saved: boolean): void {
  write(draftKey(id), JSON.stringify({ text, saved } satisfies DraftBackup));
}

export function forgetDraft(id: string): void {
  write(draftKey(id), null);
}

/** The local copy, only if the server never confirmed it. */
export function unsavedDraft(id: string): string | null {
  const raw = read(draftKey(id));
  if (!raw) return null;
  try {
    const backup = JSON.parse(raw) as DraftBackup;
    return backup.saved ? null : backup.text;
  } catch {
    return null;
  }
}
