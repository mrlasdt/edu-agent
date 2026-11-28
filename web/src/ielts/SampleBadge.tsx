// The quick sample test is never a real mock score, so every screen it shows carries this label.
export const SAMPLE_HINT =
  "Quick sample test: a few of the same items on a short clock, to check the flow. Not a real mock score.";

export function SampleBadge() {
  return (
    <span className="ielts-sample-badge" role="note" aria-label="Sample test, not a real mock score">
      SAMPLE
    </span>
  );
}
