export function countWords(text: string): number {
  return text.match(/[A-Za-z0-9][A-Za-z0-9'’-]*/g)?.length ?? 0;
}

export function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const mm = String(Math.floor(s / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

export function band(value: number): string {
  return value.toFixed(1);
}

export function signed(delta: number): string {
  if (delta === 0) return "no change";
  return `${delta > 0 ? "+" : "-"}${Math.abs(delta).toFixed(1)}`;
}

/** One plain, encouraging line for a band. It never claims more than the number says. */
export function encourage(value: number): string {
  if (value >= 8) return "Excellent result. Keep your routine going and stay sharp.";
  if (value >= 7) return "A strong result. A few targeted fixes can lift it further.";
  if (value >= 6) return "A solid base. Your next fixes are the quickest way up.";
  if (value >= 5) return "You are getting there. Steady practice moves this band.";
  return "A start to build on. Every test shows you exactly what to practise next.";
}
