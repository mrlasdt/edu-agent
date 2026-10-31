import { afterEach, describe, expect, it, vi } from "vitest";
import { streamChat } from "../chat";
import type { ChatEvent } from "../chat";

const turn = { candidateId: "c1", subject: "quant" as const, mode: "solve" as const, message: "x", history: [] };

function respondWith(...chunks: string[]) {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      chunks.forEach((c) => controller.enqueue(encoder.encode(c)));
      controller.close();
    },
  });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, status: 200, body }));
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

  it("ignores a trailer it cannot read", async () => {
    respondWith("Fine.\x1enot json");
    const events = await collect();
    expect(text(events)).toBe("Fine.");
    expect(events.some((e) => e.kind === "meta")).toBe(false);
  });
});
