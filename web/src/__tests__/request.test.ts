import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, UNREACHABLE, request, requestJson } from "../request";

afterEach(() => vi.unstubAllGlobals());

describe("request", () => {
  it("maps a network failure to a message a learner can read", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const err = await request("/x").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 0, message: UNREACHABLE });
  });

  it("uses the gateway's detail, or a generic line with the status when there is none", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ detail: "No budget" }), { status: 402 }))
    );
    await expect(request("/x")).rejects.toMatchObject({ status: 402, message: "No budget" });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(new Response("<html>", { status: 502 })));
    await expect(request("/x")).rejects.toMatchObject({ status: 502, message: "Request failed (502)" });
  });

  it("ignores a detail that is not text, such as a validation error list", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: [{ msg: "bad" }] }), { status: 422 }))
    );
    await expect(request("/x")).rejects.toMatchObject({ message: "Request failed (422)" });
  });

  it("lets an abort through as the browser's AbortError, not as 'cannot reach the server'", async () => {
    const controller = new AbortController();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(async () => {
        controller.abort();
        throw new DOMException("aborted", "AbortError");
      })
    );
    const err = await request("/x", { signal: controller.signal }).catch((e: unknown) => e);
    expect(err).not.toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ name: "AbortError" });
  });

  it("sends json bodies with a content type and passes the signal on", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response("{}"));
    vi.stubGlobal("fetch", fetchMock);
    const { signal } = new AbortController();
    await requestJson("/x", { method: "PUT", json: { a: 1 }, signal });
    const init = fetchMock.mock.calls[0][1];
    expect(init).toMatchObject({ method: "PUT", body: '{"a":1}', headers: { "Content-Type": "application/json" } });
    expect(init.signal).toBe(signal);
  });
});
