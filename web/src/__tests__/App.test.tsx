import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";
import { MAX_HISTORY_ITEMS, MAX_HISTORY_ITEM_CHARS, MAX_MESSAGE_CHARS } from "../chat";

function stubGateway(reply = "ok", traceId?: string, feedbackStatus = 200) {
  const fetchMock = vi.fn().mockImplementation(async (url: string) => {
    if (url === "/feedback") return { ok: feedbackStatus === 200, status: feedbackStatus };
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(reply));
        controller.close();
      },
    });
    const headers = new Headers(traceId ? { "X-Trace-Id": traceId } : {});
    return { ok: true, status: 200, body, headers };
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

describe("App candidate id", () => {
  it("renders on a plain-HTTP origin where crypto.randomUUID is undefined", async () => {
    // Browsers expose randomUUID only in secure contexts (HTTPS or localhost).
    const realCrypto = globalThis.crypto;
    vi.stubGlobal("crypto", {
      getRandomValues: (bytes: Uint8Array) => realCrypto.getRandomValues(bytes),
    });
    const fetchMock = stubGateway();
    const user = await openQuant();
    await send(user, "hello");
    expect(lastRequest(fetchMock).candidate_id).toMatch(/^[0-9a-f-]{36}$/);
  });

  it("keeps one candidate id across renders and requests", async () => {
    const randomUUID = vi.fn(() => "11111111-1111-4111-8111-111111111111");
    vi.stubGlobal("crypto", {
      randomUUID,
      getRandomValues: globalThis.crypto.getRandomValues.bind(globalThis.crypto),
    });
    const fetchMock = stubGateway();
    const user = await openQuant();
    await send(user, "one");
    await send(user, "two");
    expect(randomUUID).toHaveBeenCalledTimes(1);
    expect(lastRequest(fetchMock).candidate_id).toBe("11111111-1111-4111-8111-111111111111");
  });
});

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

function stubReplies(...replies: string[]) {
  const fetchMock = vi.fn();
  for (const reply of replies) {
    fetchMock.mockImplementationOnce(async () => {
      const encoder = new TextEncoder();
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode(reply));
          controller.close();
        },
      });
      return { ok: true, status: 200, body };
    });
  }
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

// A reply that sends one chunk and then waits, until the request's signal aborts (as a real
// fetch does: the pending read rejects with an AbortError).
function stubStalledGateway(firstChunk: string) {
  const fetchMock = vi.fn().mockImplementation(async (_url: string, init: RequestInit) => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(firstChunk));
        init.signal?.addEventListener("abort", () =>
          controller.error(new DOMException("The operation was aborted.", "AbortError"))
        );
      },
    });
    return { ok: true, status: 200, body };
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("App stop", () => {
  it("aborts the request, keeps the partial reply and lets the learner send again", async () => {
    const fetchMock = stubStalledGateway("Let's start by");
    const user = await openQuant();
    await user.click(screen.getByLabelText("Your message"));
    await user.paste("Help me");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await screen.findByText("Let's start by");
    const signal = fetchMock.mock.calls[0][1].signal as AbortSignal;
    expect(signal.aborted).toBe(false);
    expect(screen.queryByRole("button", { name: "Send" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Stop" }));

    expect(signal.aborted).toBe(true);
    expect(await screen.findByText("Stopped")).toBeInTheDocument();
    expect(screen.getByText("Let's start by")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument();
    expect(screen.getByLabelText("Your message")).toBeEnabled();
  });

  it("shows no Stop button when no reply is streaming", async () => {
    stubGateway();
    await openQuant();
    expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument();
  });

  it("aborts a reply still streaming when the page goes away", async () => {
    const fetchMock = stubStalledGateway("Partial");
    const user = userEvent.setup();
    const { unmount } = render(<App />);
    await user.click(screen.getByRole("button", { name: "Quant" }));
    await user.click(screen.getByLabelText("Your message"));
    await user.paste("Help me");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByText("Partial");

    unmount();

    expect((fetchMock.mock.calls[0][1].signal as AbortSignal).aborted).toBe(true);
  });
});

describe("App regenerate", () => {
  it("offers Regenerate on the last reply only, once it has finished", async () => {
    stubReplies("first answer", "second answer");
    const user = await openQuant();
    await send(user, "one");
    await screen.findByText("first answer");
    expect(screen.getAllByRole("button", { name: "Regenerate" })).toHaveLength(1);

    await send(user, "two");
    await screen.findByText("second answer");
    expect(screen.getAllByRole("button", { name: "Regenerate" })).toHaveLength(1);
    expect(screen.queryByRole("button", { name: "Regenerate" })?.closest(".message-bubble"))
      .toHaveTextContent("second answer");
  });

  it("resends the last message with the earlier history and replaces the reply", async () => {
    const fetchMock = stubReplies("first answer", "old answer", "new answer");
    const user = await openQuant();
    await send(user, "one");
    await screen.findByText("first answer");
    await send(user, "two");
    await screen.findByText("old answer");

    await user.click(screen.getByRole("button", { name: "Regenerate" }));

    await screen.findByText("new answer");
    expect(screen.queryByText("old answer")).not.toBeInTheDocument();
    expect(screen.getAllByText("two")).toHaveLength(1);
    const [first, second, third] = fetchMock.mock.calls.map(
      (c) => JSON.parse(c[1].body as string)
    );
    expect(third.message).toBe("two");
    expect(third.history).toEqual(second.history);
    expect(third.history).toEqual([
      { role: "candidate", content: "one" },
      { role: "agent", content: "first answer" },
    ]);
    expect(first.history).toEqual([]);
  });

  it("regenerates a stopped reply", async () => {
    const stalled = stubStalledGateway("Half an ans");
    const user = await openQuant();
    await user.click(screen.getByLabelText("Your message"));
    await user.paste("Help me");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByText("Half an ans");
    await user.click(screen.getByRole("button", { name: "Stop" }));
    await screen.findByText("Stopped");

    const fetchMock = stubReplies("Full answer");
    await user.click(screen.getByRole("button", { name: "Regenerate" }));

    await screen.findByText("Full answer");
    expect(screen.queryByText("Half an ans")).not.toBeInTheDocument();
    expect(lastRequest(fetchMock)).toMatchObject({ message: "Help me", history: [] });
    expect(stalled).toHaveBeenCalledTimes(1);
  });
});

describe("App feedback", () => {
  const feedbackCalls = (fetchMock: ReturnType<typeof stubGateway>) =>
    fetchMock.mock.calls.filter(([url]) => url === "/feedback");

  it("files a thumbs-down against the trace id the gateway sent", async () => {
    const fetchMock = stubGateway("ok", "trace-abc");
    const user = await openQuant();
    await send(user, "question");
    await user.click(await screen.findByRole("button", { name: "Thumbs down" }));
    expect(feedbackCalls(fetchMock)).toHaveLength(1);
    expect(JSON.parse(feedbackCalls(fetchMock)[0][1].body as string)).toEqual({
      trace_id: "trace-abc",
      vote: "down",
    });
    expect(screen.getByRole("button", { name: "Thumbs down" })).toHaveAttribute(
      "aria-pressed",
      "true"
    );
  });

  it("offers no thumbs when the gateway sent no trace id", async () => {
    stubGateway("ok");
    const user = await openQuant();
    await send(user, "question");
    await screen.findByText("ok");
    expect(screen.queryByRole("button", { name: "Thumbs down" })).not.toBeInTheDocument();
  });

  it("takes the vote back and says so when feedback could not be sent", async () => {
    stubGateway("ok", "trace-abc", 404);
    const user = await openQuant();
    await send(user, "question");
    await user.click(await screen.findByRole("button", { name: "Thumbs up" }));
    expect(await screen.findByText(/could not send feedback/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Thumbs up" })).toHaveAttribute(
      "aria-pressed",
      "false"
    );
  });
});

describe("App essay disclaimer", () => {
  it("shows the notice once and no stray rule or underscore above it", async () => {
    // The gateway's real reply for a sample essay: the essay text, then the trailer (see
    // tests/test_chat_endpoint.py); the web app's own copy of the sentence is not in the stream.
    stubGateway('A sample essay about technology.\x1e{"disclaimer":"essay"}');
    const user = userEvent.setup();
    const { container } = render(<App />);
    await user.click(screen.getByRole("button", { name: "AW" }));
    await send(user, "Write an essay");

    const notice = await screen.findByTestId("essay-disclaimer");
    expect(notice).toHaveTextContent(/study purposes only/i);
    const content = container.querySelector(".message-bubble--agent .message-content");
    expect(content?.querySelector("hr")).toBeNull();
    expect(content?.textContent).not.toContain("_");
    expect(content?.textContent).not.toContain("---");
    expect(content?.textContent).not.toContain("\x1e");
  });
});
