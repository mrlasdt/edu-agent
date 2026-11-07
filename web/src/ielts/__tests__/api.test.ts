import { afterEach, describe, expect, it, vi } from "vitest";
import { ATTEMPT_NOT_FOUND, getAttempt, startAttempt } from "../api";

afterEach(() => vi.unstubAllGlobals());

const reply = (status: number, detail: string) =>
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail }), { status })));

describe("ielts api errors", () => {
  it("says a missing attempt in words a learner can act on, not the server's terse detail", async () => {
    reply(404, "attempt not found");
    await expect(startAttempt("a1")).rejects.toMatchObject({ status: 404, message: ATTEMPT_NOT_FOUND });
    expect(ATTEMPT_NOT_FOUND).toMatch(/^This attempt could not be found\./);
  });

  it("keeps the server's detail for every other error", async () => {
    reply(409, "attempt is graded; nothing to retry");
    await expect(getAttempt("a1")).rejects.toMatchObject({ status: 409, message: "attempt is graded; nothing to retry" });
  });
});
