import type { Mode } from "../types";

interface Props {
  mode: Mode;
  onToggle: () => void;
}

export function ModeSwitch({ mode, onToggle }: Props) {
  const label = mode === "tutor" ? "Show me the answer" : "Back to hints";
  return (
    <button
      className={`mode-switch mode-switch--${mode}`}
      onClick={onToggle}
      aria-label={label}
    >
      {label}
    </button>
  );
}
