import { afterEach, describe, expect, it, vi } from "vitest";
import { newId } from "../newId";

const V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

afterEach(() => vi.unstubAllGlobals());

describe("newId", () => {
  it("uses crypto.randomUUID when the browser provides it", () => {
    const randomUUID = vi.fn().mockReturnValue("from-random-uuid");
    vi.stubGlobal("crypto", { randomUUID, getRandomValues: vi.fn() });
    expect(newId()).toBe("from-random-uuid");
  });

  it("falls back to a v4 UUID from getRandomValues on insecure origins", () => {
    // randomUUID is undefined outside secure contexts (plain HTTP, non-localhost).
    vi.stubGlobal("crypto", {
      getRandomValues: (bytes: Uint8Array) => bytes.fill(0xff),
    });
    expect(newId()).toMatch(V4);
  });

  it("produces distinct ids in the fallback", () => {
    vi.stubGlobal("crypto", {
      getRandomValues: (bytes: Uint8Array) => {
        for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
        return bytes;
      },
    });
    expect(newId()).not.toBe(newId());
  });
});
