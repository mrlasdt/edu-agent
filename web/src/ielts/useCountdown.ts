import { useEffect, useRef, useState } from "react";

/**
 * Seconds left, counted down against a deadline derived from the server's `remaining_s`.
 * The server stays authoritative: call `sync` with each fresh `remaining_s` it returns.
 * Using a wall-clock deadline (not a decrementing counter) keeps the display right after
 * the tab was throttled or the laptop slept.
 */
export function useCountdown(initialSeconds: number, onExpire: () => void) {
  const deadline = useRef(Date.now() + initialSeconds * 1000);
  const [left, setLeft] = useState(initialSeconds);
  const expired = useRef(false);
  const onExpireRef = useRef(onExpire);
  onExpireRef.current = onExpire;

  useEffect(() => {
    const tick = () => {
      const remaining = Math.max(0, Math.ceil((deadline.current - Date.now()) / 1000));
      setLeft(remaining);
      if (remaining === 0 && !expired.current) {
        expired.current = true;
        onExpireRef.current();
      }
    };
    tick();
    const id = window.setInterval(tick, 500);
    return () => window.clearInterval(id);
  }, []);

  const sync = (remainingSeconds: number | null) => {
    if (remainingSeconds !== null) deadline.current = Date.now() + remainingSeconds * 1000;
  };
  return { left, sync };
}
