import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { IeltsMock } from "./ielts/IeltsMock";
import "./styles.css";

// Two pages and no router: the mock lives at /ielts-mock, everything else is the tutor.
const isIeltsMock = window.location.pathname.replace(/\/+$/, "") === "/ielts-mock";

createRoot(document.getElementById("root")!).render(
  <StrictMode>{isIeltsMock ? <IeltsMock /> : <App />}</StrictMode>
);
