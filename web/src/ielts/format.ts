export function countWords(text: string): number {
  return text.match(/\S+/g)?.length ?? 0;
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
