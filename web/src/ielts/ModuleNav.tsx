/** The two mock modules at /ielts-mock; the page has no router, so Reading is `?module=reading`. */
export function ModuleNav({ current }: { current: "writing" | "reading" }) {
  return (
    <nav className="ielts-chips" aria-label="Mock module">
      <a
        className="ielts-chip"
        href="/ielts-mock"
        aria-current={current === "writing" ? "page" : undefined}
      >
        Writing
      </a>
      <a
        className="ielts-chip"
        href="/ielts-mock?module=reading"
        aria-current={current === "reading" ? "page" : undefined}
      >
        Reading
      </a>
    </nav>
  );
}
