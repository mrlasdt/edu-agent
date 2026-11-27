interface Props {
  on: boolean;
  onToggle: () => void;
}

/** The per-chat Calculator switch: lets the agent use the math tools for this chat. */
export function CalculatorSwitch({ on, onToggle }: Props) {
  return (
    <button
      type="button"
      className={`calculator-switch calculator-switch--${on ? "on" : "off"}`}
      role="switch"
      aria-checked={on}
      onClick={onToggle}
    >
      Calculator: {on ? "on" : "off"}
    </button>
  );
}
