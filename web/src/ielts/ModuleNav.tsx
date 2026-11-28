import { useEffect, useRef } from "react";

/** The mock modules at /ielts-mock; the page has no router, so the others are `?module=...`. */
type Current = "writing" | "writing-module" | "listening" | "reading" | "mock";

const LINKS: { key: Current; label: string; href: string }[] = [
  { key: "mock", label: "Full mock", href: "/ielts-mock?module=mock" },
  { key: "writing", label: "Writing", href: "/ielts-mock" },
  { key: "writing-module", label: "Writing module", href: "/ielts-mock?module=writing-module" },
  { key: "listening", label: "Listening", href: "/ielts-mock?module=listening" },
  { key: "reading", label: "Reading", href: "/ielts-mock?module=reading" },
];

export function ModuleNav({ current }: { current: Current }) {
  const nav = useRef<HTMLElement>(null);
  // On a phone the strip scrolls sideways; bring the current module into view (sideways only).
  useEffect(() => {
    const el = nav.current;
    const here = el?.querySelector<HTMLElement>('[aria-current="page"]');
    if (!el || !here) return;
    const chip = here.getBoundingClientRect();
    const strip = el.getBoundingClientRect();
    el.scrollLeft += chip.left - strip.left - (strip.width - chip.width) / 2;
  }, [current]);
  return (
    <nav ref={nav} className="ielts-chips" aria-label="Mock module">
      {LINKS.map((l) => (
        <a
          key={l.key}
          className="ielts-chip"
          href={l.href}
          aria-current={current === l.key ? "page" : undefined}
        >
          {l.label}
        </a>
      ))}
    </nav>
  );
}
