import "@testing-library/jest-dom";

// Node 25 ships a global `localStorage` that shadows jsdom's and has no methods unless a file
// is configured. Tests get a plain in-memory Storage instead.
const memory = new Map<string, string>();
Object.defineProperty(window, "localStorage", {
  configurable: true,
  value: {
    getItem: (k: string) => memory.get(k) ?? null,
    setItem: (k: string, v: string) => void memory.set(k, String(v)),
    removeItem: (k: string) => void memory.delete(k),
    clear: () => memory.clear(),
    key: (i: number) => [...memory.keys()][i] ?? null,
    get length() {
      return memory.size;
    },
  } satisfies Storage,
});
