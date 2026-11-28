import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { listAttempts } from "./api";
import { band } from "./format";
import { listListenings } from "./listeningApi";
import { listMocks } from "./mockApi";
import type { MockReport } from "./mockApi";
import { ModuleNav } from "./ModuleNav";
import { listReadings } from "./readingApi";
import { listSpeakings } from "./speakingApi";
import { listWritingModules } from "./writingModuleApi";

type SkillKey = "listening" | "reading" | "writing-module" | "writing" | "speaking" | "mock";

interface Card {
  key: SkillKey;
  name: string;
  minutes: string;
  blurb: string;
  href: string;
  cta: string;
  icon: ReactNode;
}

const svg = (children: ReactNode) => (
  <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {children}
  </svg>
);

const CARDS: Card[] = [
  {
    key: "listening",
    name: "Listening",
    minutes: "30 min",
    blurb: "Four recordings, 40 questions. Sound on.",
    href: "/ielts-mock?module=listening",
    cta: "Start Listening",
    icon: svg(
      <>
        <path d="M4 14v-2a8 8 0 0 1 16 0v2" />
        <rect x="3" y="14" width="4" height="6" rx="1.5" />
        <rect x="17" y="14" width="4" height="6" rx="1.5" />
      </>
    ),
  },
  {
    key: "reading",
    name: "Reading",
    minutes: "60 min",
    blurb: "Three passages with the questions beside them.",
    href: "/ielts-mock?module=reading",
    cta: "Start Reading",
    icon: svg(
      <>
        <path d="M4 5.5A1.5 1.5 0 0 1 5.5 4H11v15H5.5A1.5 1.5 0 0 0 4 20.5z" />
        <path d="M20 5.5A1.5 1.5 0 0 0 18.5 4H13v15h5.5a1.5 1.5 0 0 1 1.5 1.5z" />
      </>
    ),
  },
  {
    key: "writing-module",
    name: "Writing test",
    minutes: "60 min",
    blurb: "A letter and an essay, marked with a band for each.",
    href: "/ielts-mock?module=writing-module",
    cta: "Start Writing test",
    icon: svg(
      <>
        <path d="M4 20l4-1 11-11a2.1 2.1 0 0 0-3-3L5 16z" />
        <path d="M14 7l3 3" />
      </>
    ),
  },
  {
    key: "writing",
    name: "Writing practice",
    minutes: "40 min",
    blurb: "One essay on a question you pick. A quick way to practise.",
    href: "/ielts-mock",
    cta: "Practise an essay",
    icon: svg(
      <>
        <path d="M6 3h9l4 4v14H6z" />
        <path d="M14 3v5h5M9 13h7M9 17h5" />
      </>
    ),
  },
  {
    key: "speaking",
    name: "Speaking",
    minutes: "Recorded",
    blurb: "Answer the examiner's questions aloud with your microphone on.",
    href: "/ielts-mock?module=speaking",
    cta: "Start Speaking",
    icon: svg(
      <>
        <rect x="9" y="3" width="6" height="11" rx="3" />
        <path d="M5 11a7 7 0 0 0 14 0M12 18v3M9 21h6" />
      </>
    ),
  },
  {
    key: "mock",
    name: "Full mock",
    minutes: "2 h 30 min+",
    blurb: "Listening, Reading, Writing, then Speaking, with one overall band at the end.",
    href: "/ielts-mock?module=mock",
    cta: "Open the full mock",
    icon: svg(
      <>
        <circle cx="12" cy="13" r="8" />
        <path d="M12 9v4l2.5 2M9.5 2.5h5" />
      </>
    ),
  },
];

type Bands = Partial<Record<SkillKey, number>>;
interface SpeakingNote {
  sample: boolean;
  partial: boolean;
}
interface HomeData {
  bands: Bands;
  mock: MockReport | null;
  speakingNote: SpeakingNote | null;
}

/** The learner's latest band per skill: the newest marked attempt in each existing list API. */
async function loadBands(signal: AbortSignal): Promise<HomeData> {
  const quiet = <T,>(p: Promise<T[]>) => p.catch(() => [] as T[]);
  const [writing, reading, listening, writingModule, speaking, mocks] = await Promise.all([
    quiet(listAttempts(signal)),
    quiet(listReadings(signal)),
    quiet(listListenings(signal)),
    quiet(listWritingModules(signal)),
    quiet(listSpeakings(signal)),
    quiet(listMocks(signal)),
  ]);
  const mock = mocks[0] ?? null;
  const fromMocks = (key: string) =>
    mocks.map((m) => m.modules.find((s) => s.key === key)?.band).find((b) => b != null) ?? undefined;
  const bands: Bands = {};
  bands.writing = writing.find((a) => a.overall !== null)?.overall ?? undefined;
  bands.reading = reading.find((a) => a.band !== null)?.band ?? fromMocks("reading");
  bands.listening = listening.find((a) => a.band !== null)?.band ?? fromMocks("listening");
  bands["writing-module"] =
    writingModule.find((a) => a.writing_band !== null)?.writing_band ?? fromMocks("writing");
  const marked = speaking.find((a) => a.band !== null);
  bands.speaking = marked?.band ?? fromMocks("speaking");
  const speakingNote = marked ? { sample: marked.paper.sample, partial: marked.partial === true } : null;
  bands.mock = mocks.find((m) => m.overall !== null)?.overall ?? undefined;
  return { bands, mock, speakingNote };
}

/** The first skill with no band yet, so a new learner always has one obvious next step. */
function recommend(bands: Bands, mock: MockReport | null): { key: SkillKey; reason: string } {
  const next = mock?.next_module;
  if (next) {
    const key = next.key === "writing" ? "writing-module" : next.key;
    if (key === "listening" || key === "reading" || key === "writing-module" || key === "speaking") {
      return { key, reason: next.reason };
    }
  }
  if (bands.listening === undefined) return { key: "listening", reason: "A good place to begin: you have no Listening band yet." };
  if (bands.reading === undefined) return { key: "reading", reason: "Next, get a Reading band so you can see how you compare." };
  if (bands["writing-module"] === undefined) return { key: "writing-module", reason: "Finish with Writing to complete your picture." };
  if (bands.speaking === undefined) return { key: "speaking", reason: "Last skill to try: record your Speaking answers." };
  const lowest = (["listening", "reading", "writing-module", "speaking"] as const).reduce((a, b) =>
    (bands[a] ?? 0) <= (bands[b] ?? 0) ? a : b
  );
  return { key: lowest, reason: "This is your lowest band right now, so a retake pays off most." };
}

export function IeltsHome() {
  const [data, setData] = useState<HomeData | null>(null);

  useEffect(() => {
    document.title = "IELTS practice";
    const controller = new AbortController();
    loadBands(controller.signal)
      .then(setData)
      .catch(() => setData({ bands: {}, mock: null, speakingNote: null }));
    return () => controller.abort();
  }, []);

  const rec = data ? recommend(data.bands, data.mock) : null;
  const recCard = rec ? CARDS.find((c) => c.key === rec.key) : undefined;
  const tried = data ? Object.values(data.bands).filter((b) => b !== undefined).length : 0;

  return (
    <main className="app ielts" data-view="home">
      <header className="app-header">
        <h1>IELTS practice</h1>
        <a className="ielts-home" href="/">
          Tutor chat
        </a>
      </header>
      <ModuleNav current="home" />
      {!data && <p role="status">Loading...</p>}
      {data && recCard && rec && (
        <>
          <section className="home-hero" aria-label="Your next step">
            <p className="ielts-eyebrow">{tried === 0 ? "Welcome" : "Your next step"}</p>
            <h2>
              {tried === 0
                ? "Start with one test. It takes under an hour."
                : `Keep going: ${recCard.name} is next.`}
            </h2>
            <p>{rec.reason}</p>
            <div className="ielts-actions">
              <a className="ielts-primary" href={recCard.href}>
                {recCard.cta}
              </a>
            </div>
          </section>
          <ul className="home-cards" aria-label="Practice tests">
            {CARDS.map((c) => {
              const b = data.bands[c.key];
              const note = c.key === "speaking" ? data.speakingNote : null;
              return (
                <li key={c.key} className="home-card" data-recommended={c.key === rec.key ? "true" : undefined}>
                  <div className="home-card-top">
                    <span className="home-icon">{c.icon}</span>
                    {c.key === rec.key && <span className="home-badge">Recommended</span>}
                  </div>
                  <h3>{c.name}</h3>
                  <p className="home-meta">{c.minutes}</p>
                  <p className="home-blurb">{c.blurb}</p>
                  <p className="home-band" data-tried={b !== undefined ? "true" : "false"}>
                    {b !== undefined ? (
                      <>
                        <span>{note?.sample ? "Latest band (quick sample)" : "Latest band"}</span>
                        <strong>{band(b)}</strong>
                      </>
                    ) : (
                      <span>No band yet</span>
                    )}
                  </p>
                  {note?.partial && <p className="home-meta">Partial: Pronunciation not assessed</p>}
                  <a className="home-cta" href={c.href}>
                    {b !== undefined ? "Open" : c.cta}
                  </a>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </main>
  );
}
