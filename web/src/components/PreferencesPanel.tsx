interface Props {
  isOpen: boolean;
  onClose: () => void;
  personalStyleEnabled: boolean;
  hasEnoughEssays: boolean;
  onStyleToggle: () => void;
}

export function PreferencesPanel({
  isOpen,
  onClose,
  personalStyleEnabled,
  hasEnoughEssays,
  onStyleToggle,
}: Props) {
  if (!isOpen) return null;

  return (
    <aside className="preferences-panel" aria-label="Preferences">
      <div className="preferences-panel__header">
        <h3>Preferences</h3>
        <button onClick={onClose} aria-label="Close preferences panel">
          Close ✕
        </button>
      </div>

      <section className="preferences-section">
        <h4>Writing style</h4>

        {hasEnoughEssays ? (
          <label className="style-toggle-label">
            <input
              type="checkbox"
              aria-label="Match my writing style"
              checked={personalStyleEnabled}
              onChange={onStyleToggle}
              className="style-toggle-input"
            />
            <span>Match my writing style</span>
          </label>
        ) : (
          <p className="style-toggle-hint">
            Upload your essays to enable style matching. You need at least 2 essays in your
            Candidate corpus.
          </p>
        )}
      </section>
    </aside>
  );
}
