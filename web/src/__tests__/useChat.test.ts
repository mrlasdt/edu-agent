import { describe, expect, it } from "vitest";
import { chatReducer, initialChat, isBusy } from "../useChat";
import type { ChatAction } from "../useChat";
import type { Message } from "../types";

const question: Message = { id: "m0", role: "candidate", content: "2+2?" };
const run = (...actions: ChatAction[]) => actions.reduce(chatReducer, initialChat);
const started: ChatAction = { type: "turn_started", shown: [question], replyId: "m1" };
const text = (t: string): ChatAction => ({ type: "event", replyId: "m1", event: { kind: "text", text: t } });

describe("chatReducer", () => {
  it("is busy from the start of a turn until its reply ends, and from nothing else", () => {
    expect(isBusy(initialChat)).toBe(false);
    expect(isBusy(run(started))).toBe(true);
    expect(isBusy(run(started, text("4")))).toBe(true);
    expect(isBusy(run(started, text("4"), { type: "turn_ended", replyId: "m1", stopped: false }))).toBe(false);
    expect(isBusy(run(started, { type: "turn_failed", replyId: "m1", stopped: false, error: "x" }))).toBe(false);
  });

  it("ends a reply by how it went", () => {
    const end = (stopped: boolean, ...before: ChatAction[]) =>
      run(started, ...before, { type: "turn_ended", replyId: "m1", stopped }).messages[1];
    expect(end(false, text("4"))).toMatchObject({ content: "4", status: undefined });
    expect(end(false)).toMatchObject({ status: "empty" });
    expect(end(true, text("4"))).toMatchObject({ content: "4", status: "stopped" });
  });

  it("records a failure as a status and a reason, keeping the partial text", () => {
    const failed = run(started, text("4"), { type: "turn_failed", replyId: "m1", stopped: false, error: "Cannot reach the server" });
    expect(failed.messages[1]).toMatchObject({ content: "4", status: "failed", error: "Cannot reach the server" });
    const stopped = run(started, { type: "turn_failed", replyId: "m1", stopped: true, error: "aborted" });
    expect(stopped.messages[1]).toMatchObject({ status: "stopped" });
  });
});
