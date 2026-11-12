import { afterEach, describe, expect, it, vi } from "vitest";
import { MAX_HISTORY_ITEMS, sendFeedback, streamChat, toHistoryPayload } from "../chat";
import type { ChatEvent } from "../chat";

const turn = { candidateId: "c1", subject: "quant" as const, mode: "solve" as const, message: "x", history: [] };

function respondWith(...chunks: string[]) {
  respondWithHeaders({}, ...chunks);
}

function respondWithHeaders(headers: Record<string, string>, ...chunks: string[]) {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      chunks.forEach((c) => controller.enqueue(encoder.encode(c)));
      controller.close();
    },
  });
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({ ok: true, status: 200, body, headers: new Headers(headers) })
  );
}

async function collect(): Promise<ChatEvent[]> {
  const events: ChatEvent[] = [];
  for await (const e of streamChat(turn)) events.push(e);
  return events;
}

const text = (events: ChatEvent[]) =>
  events.map((e) => (e.kind === "text" ? e.text : "")).join("");

afterEach(() => vi.unstubAllGlobals());

describe("streamChat", () => {
  it("yields the answer text as it arrives", async () => {
    respondWith("Try ", "factoring.");
    const events = await collect();
    expect(text(events)).toBe("Try factoring.");
    expect(events.every((e) => e.kind === "text")).toBe(true);
  });

  it("keeps the verifier trailer out of the text and reports it as metadata", async () => {
    respondWith("Answer: 2", '\x1e{"math_ver', 'ified":true}');
    const events = await collect();
    expect(text(events)).toBe("Answer: 2");
    expect(events[events.length - 1]).toEqual({ kind: "meta", mathVerified: true });
  });

  it("splits a trailer that shares a chunk with the end of the answer", async () => {
    respondWith('Best effort.\x1e{"math_verified":false}');
    const events = await collect();
    expect(text(events)).toBe("Best effort.");
    expect(events[events.length - 1]).toEqual({ kind: "meta", mathVerified: false });
  });

  it("reads the independent check from the trailer", async () => {
    respondWith(
      'Best effort.\x1e{"math_verified":false,"independent_check":{"verdict":"disagrees","solver":"2, 3"}}'
    );
    const events = await collect();
    expect(events[events.length - 1]).toEqual({
      kind: "meta",
      mathVerified: false,
      independentCheck: { verdict: "disagrees", solver: "2, 3" },
    });
  });

  it("drops an independent check it does not recognise", async () => {
    respondWith('Fine.\x1e{"math_verified":true,"independent_check":{"verdict":"maybe"}}');
    const events = await collect();
    expect(events[events.length - 1]).toEqual({ kind: "meta", mathVerified: true });
  });

  it("reports the essay disclaimer flag the gateway sends after an essay", async () => {
    respondWith('A sample essay.\x1e{"disclaimer":"essay"}');
    const events = await collect();
    expect(text(events)).toBe("A sample essay.");
    expect(events[events.length - 1]).toEqual({ kind: "meta", disclaimer: "essay" });
  });

  it("ignores a disclaimer value it does not recognise", async () => {
    respondWith('Fine.\x1e{"math_verified":true,"disclaimer":"other"}');
    const events = await collect();
    expect(events[events.length - 1]).toEqual({ kind: "meta", mathVerified: true });
  });

  it("ignores a trailer it cannot read", async () => {
    respondWith("Fine.\x1enot json");
    const events = await collect();
    expect(text(events)).toBe("Fine.");
    expect(events.some((e) => e.kind === "meta")).toBe(false);
  });
});

describe("streamChat trace id", () => {
  it("yields the turn's trace id from the X-Trace-Id header before any text", async () => {
    respondWithHeaders({ "X-Trace-Id": "t-123" }, "Hi.");
    const events = await collect();
    expect(events[0]).toEqual({ kind: "trace", traceId: "t-123" });
    expect(text(events)).toBe("Hi.");
  });

  it("yields no trace event when the gateway sent no header", async () => {
    respondWith("Hi.");
    expect((await collect()).some((e) => e.kind === "trace")).toBe(false);
  });
});

describe("sendFeedback", () => {
  it("posts the trace id and vote to /feedback", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200 });
    vi.stubGlobal("fetch", fetchMock);
    await sendFeedback("t-123", "down");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/feedback");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ trace_id: "t-123", vote: "down" });
  });

  it("throws when the gateway refuses", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 404 }));
    await expect(sendFeedback("nope", "up")).rejects.toThrow("404");
  });
});

describe("toHistoryPayload", () => {
  const exchange = (i: number) => [
    { id: `c${i}`, role: "candidate" as const, content: `question ${i}` },
    { id: `a${i}`, role: "agent" as const, content: `answer ${i}` },
  ];
  const conversation = (turns: number) => Array.from({ length: turns }, (_, i) => exchange(i)).flat();

  it("sends a conversation under the cap whole", () => {
    const sent = toHistoryPayload(conversation(5));
    expect(sent).toHaveLength(10);
    expect(sent[0]).toEqual({ role: "candidate", content: "question 0" });
  });

  it("keeps the opening message, which states the problem, with the latest ones past the cap", () => {
    const sent = toHistoryPayload(conversation(MAX_HISTORY_ITEMS / 2 + 5));
    expect(sent).toHaveLength(MAX_HISTORY_ITEMS);
    expect(sent[0]).toEqual({ role: "candidate", content: "question 0" });
    expect(sent[1]).toEqual({ role: "agent", content: "answer 5" });
    expect(sent[sent.length - 1]).toEqual({ role: "agent", content: `answer ${MAX_HISTORY_ITEMS / 2 + 4}` });
  });

  it("leaves out empty messages and sends only role and content", () => {
    const sent = toHistoryPayload([...exchange(0), { id: "x", role: "agent" as const, content: "" }]);
    expect(sent).toEqual([
      { role: "candidate", content: "question 0" },
      { role: "agent", content: "answer 0" },
    ]);
  });
});
