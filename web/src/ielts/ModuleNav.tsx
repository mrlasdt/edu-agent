/** The mock modules at /ielts-mock; the page has no router, so the others are `?module=...`. */
type Current = "writing" | "writing-module" | "reading" | "mock";

const LINKS: { key: Current; label: string; href: string }[] = [
  { key: "mock", label: "Full mock", href: "/ielts-mock?module=mock" },
  { key: "writing", label: "Writing", href: "/ielts-mock" },
  { key: "writing-module", label: "Writing module", href: "/ielts-mock?module=writing-module" },
  { key: "reading", label: "Reading", href: "/ielts-mock?module=reading" },
];

export function ModuleNav({ current }: { current: Current }) {
  return (
    <nav className="ielts-chips" aria-label="Mock module">
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
