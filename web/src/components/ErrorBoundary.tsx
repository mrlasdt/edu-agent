import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

interface State {
  failed: boolean;
}

/** Turns a render crash into a visible message; without it React unmounts the root and the page is blank. */
export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { failed: false };

  static getDerivedStateFromError(): State {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Render crash:", error, info.componentStack);
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <main className="app">
        <div role="alert" className="ielts-error">
          <p>Something went wrong. Reload the page to try again.</p>
          <button type="button" onClick={() => window.location.reload()}>
            Reload
          </button>
        </div>
      </main>
    );
  }
}
