import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { sendFeedback, streamChat } from "./chat";
import type { ChatEvent, Vote } from "./chat";
import { newId } from "./newId";
import type { Message, Mode, Subject } from "./types";

export interface ChatState {
  messages: Message[];
}

export type ChatAction =
  /** A new agent reply begins; `shown` is the conversation up to and including the question. */
  | { type: "turn_started"; shown: Message[]; replyId: string }
  | { type: "event"; replyId: string; event: ChatEvent }
  | { type: "turn_ended"; replyId: string; stopped: boolean }
  | { type: "turn_failed"; replyId: string; stopped: boolean; error: string }
  | { type: "vote"; messageId: string; vote: Vote | undefined; error?: string };

export const initialChat: ChatState = { messages: [] };

/** A turn is running while its reply has not ended: nothing else says so, so nothing can disagree. */
export const isBusy = ({ messages }: ChatState) => {
  const status = messages[messages.length - 1]?.status;
  return status === "thinking" || status === "streaming";
};

const patch = (state: ChatState, id: string, change: (m: Message) => Message): ChatState => ({
  messages: state.messages.map((m) => (m.id === id ? change(m) : m)),
});

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "turn_started":
      return {
        messages: [
          ...action.shown,
          { id: action.replyId, role: "agent", content: "", status: "thinking" },
        ],
      };
    case "event":
      return patch(state, action.replyId, (m) => {
        const { event } = action;
        if (event.kind === "trace") return { ...m, traceId: event.traceId };
        if (event.kind === "text") {
          return { ...m, content: m.content + event.text, status: "streaming" };
        }
        return {
          ...m,
          mathVerified: event.mathVerified,
          independentCheck: event.independentCheck,
          disclaimer: event.disclaimer,
        };
      });
    case "turn_ended":
      return patch(state, action.replyId, (m) => ({
        ...m,
        status: action.stopped ? "stopped" : m.content ? undefined : "empty",
      }));
    case "turn_failed":
      return patch(state, action.replyId, (m) =>
        action.stopped ? { ...m, status: "stopped" } : { ...m, status: "failed", error: action.error }
      );
    case "vote":
      return patch(state, action.messageId, (m) => ({
        ...m,
        vote: action.vote,
        feedbackError: action.error,
      }));
  }
}

/**
 * The conversation with the agent: its messages, whether a reply is running, and the actions on it.
 * One reducer owns every change, so a reply's status is the only record of whether it is still running.
 */
export function useChat(subject: Subject, mode: Mode, attemptId?: string) {
  const [state, dispatch] = useReducer(chatReducer, initialChat);
  const [candidateId] = useState(newId); // lazy: runs once, not on every render
  const nextId = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const { messages } = state;
  const busy = isBusy(state);

  // Leaving the page ends the stream; the signal also tells the gateway to stop the model call.
  useEffect(() => () => abortRef.current?.abort(), []);

  // Streams the agent's reply to `text` into a new bubble. `history` is what came before it.
  const runTurn = async (text: string, history: Message[], shown: Message[]) => {
    const replyId = `m${nextId.current++}`;
    const controller = new AbortController();
    abortRef.current = controller;
    dispatch({ type: "turn_started", shown, replyId });
    try {
      for await (const event of streamChat(
        { candidateId, subject, mode, message: text, history, attemptId },
        controller.signal
      )) {
        dispatch({ type: "event", replyId, event });
      }
      dispatch({ type: "turn_ended", replyId, stopped: controller.signal.aborted });
    } catch (err) {
      dispatch({
        type: "turn_failed",
        replyId,
        stopped: controller.signal.aborted,
        error: err instanceof Error ? err.message : "Something went wrong",
      });
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  };

  const send = async (text: string) => {
    if (!text || busy) return;
    const question: Message = { id: `m${nextId.current++}`, role: "candidate", content: text };
    await runTurn(text, messages, [...messages, question]);
  };

  // Asks again: the last question is resent with the history that preceded it, and the new
  // reply replaces the old one.
  const regenerate = async () => {
    if (busy) return;
    const at = messages.map((m) => m.role).lastIndexOf("candidate");
    if (at === -1) return;
    await runTurn(messages[at].content, messages.slice(0, at), messages.slice(0, at + 1));
  };

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const rate = async (message: Message, vote: Vote) => {
    if (!message.traceId) return;
    const previous = message.vote;
    dispatch({ type: "vote", messageId: message.id, vote });
    try {
      await sendFeedback(message.traceId, vote);
    } catch {
      dispatch({
        type: "vote",
        messageId: message.id,
        vote: previous,
        error: "Could not send feedback. Try again.",
      });
    }
  };

  const lastReply = messages[messages.length - 1];
  return {
    messages,
    busy,
    canRegenerate: !busy && lastReply?.role === "agent",
    send,
    regenerate,
    stop,
    rate,
  };
}
