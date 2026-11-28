import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { FullMock } from "./ielts/FullMock";
import { IeltsHome } from "./ielts/IeltsHome";
import { IeltsMock } from "./ielts/IeltsMock";
import { ListeningMock } from "./ielts/ListeningMock";
import { ReadingMock } from "./ielts/ReadingMock";
import { SpeakingMock } from "./ielts/SpeakingMock";
import { WritingModule } from "./ielts/WritingModule";
import "./styles.css";

// Two pages and no router: the mocks live at /ielts-mock (Writing, or Listening, Reading, the letter + essay module and the full mock with ?module=home|listening|reading|writing-module|speaking|mock), everything else is the tutor.
const isIeltsMock = window.location.pathname.replace(/\/+$/, "") === "/ielts-mock";

const mockModule = new URLSearchParams(window.location.search).get("module");
const mock =
  mockModule === "home" ? (
    <IeltsHome />
  ) : mockModule === "mock" ? (
    <FullMock />
  ) : mockModule === "listening" ? (
    <ListeningMock />
  ) : mockModule === "reading" ? (
    <ReadingMock />
  ) : mockModule === "speaking" ? (
    <SpeakingMock />
  ) : mockModule === "writing-module" ? (
    <WritingModule />
  ) : (
    <IeltsMock />
  );
const page = isIeltsMock ? mock : <App />;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>{page}</ErrorBoundary>
  </StrictMode>
);
