import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { IeltsMock } from "./ielts/IeltsMock";
import { ReadingMock } from "./ielts/ReadingMock";
import "./styles.css";

// Two pages and no router: the mocks live at /ielts-mock (Writing, or Reading with ?module=reading), everything else is the tutor.
const isIeltsMock = window.location.pathname.replace(/\/+$/, "") === "/ielts-mock";

const isReading = new URLSearchParams(window.location.search).get("module") === "reading";
const page = !isIeltsMock ? <App /> : isReading ? <ReadingMock /> : <IeltsMock />;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>{page}</ErrorBoundary>
  </StrictMode>
);
