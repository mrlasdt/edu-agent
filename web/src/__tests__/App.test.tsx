import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";
import { MAX_HISTORY_ITEMS, MAX_HISTORY_ITEM_CHARS, MAX_MESSAGE_CHARS } from "../chat";

function stubGateway(reply = "ok") {
  const fetchMock = vi.fn().mockImplementation(async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(reply));
        controller.close();
      },
    });
    return { ok: true, status: 200, body };
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const lastRequest = (fetchMock: ReturnType<typeof stubGateway>) =>
  JSON.parse(fetchMock.mock.calls[fetchMock.mock.calls.length - 1][1].body as string);

async function openQuant() {
  const user = userEvent.setup();
  render(<App />);
  await user.click(screen.getByRole("button", { name: "Quant" }));
  return user;
}

async function send(user: ReturnType<typeof userEvent.setup>, text: string) {
  const box = screen.getByLabelText("Your message");
  await user.click(box);
  await user.paste(text);
  await user.click(screen.getByRole("button", { name: "Send" }));
  await screen.findByDisplayValue("");
}

afterEach(() => vi.unstubAllGlobals());

describe("App /chat request limits", () => {
  it("caps the composer at the gateway's message limit", async () => {
    stubGateway();
    await openQuant();
    expect(screen.getByLabelText("Your message")).toHaveAttribute(
      "maxlength",
      String(MAX_MESSAGE_CHARS)
    );
  });

  it("sends only the latest history items once the conversation outgrows the cap", async () => {
    const fetchMock = stubGateway();
    const user = await openQuant();
    const turns = MAX_HISTORY_ITEMS / 2 + 3;
    for (let i = 0; i < turns; i++) {
      await send(user, `question ${i}`);
      await screen.findAllByText("ok").then((all) => expect(all).toHaveLength(i + 1));
    }
    await send(user, "final question");
    const { history } = lastRequest(fetchMock);
    expect(history).toHaveLength(MAX_HISTORY_ITEMS);
    expect(history[history.length - 1]).toEqual({ role: "agent", content: "ok" });
    expect(history[history.length - 2]).toEqual({ role: "candidate", content: `question ${turns - 1}` });
    expect(history[0]).toEqual({ role: "candidate", content: "question 3" });
  }, 30000);

  it("truncates an oversized replayed reply to the gateway's per-item limit", async () => {
    const fetchMock = stubGateway("a".repeat(MAX_HISTORY_ITEM_CHARS + 500));
    const user = await openQuant();
    await send(user, "first");
    await screen.findByText("a".repeat(MAX_HISTORY_ITEM_CHARS + 500));
    await send(user, "second");
    const { history } = lastRequest(fetchMock);
    expect(history[1].content).toHaveLength(MAX_HISTORY_ITEM_CHARS);
  });
});
