import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { IeltsMock } from "./ielts/IeltsMock";
import "./styles.css";

// Two pages and no router: the mock lives at /ielts-mock, everything else is the tutor.
const isIeltsMock = window.location.pathname.replace(/\/+$/, "") === "/ielts-mock";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>{isIeltsMock ? <IeltsMock /> : <App />}</ErrorBoundary>
  </StrictMode>
);
