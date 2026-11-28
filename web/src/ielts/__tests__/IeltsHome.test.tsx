import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { IeltsHome } from "../IeltsHome";
import { encourage } from "../format";

const apis = vi.hoisted(() => ({
  listAttempts: vi.fn(),
  listReadings: vi.fn(),
  listListenings: vi.fn(),
  listMocks: vi.fn(),
  listWritingModules: vi.fn(),
}));
vi.mock("../api", () => ({ listAttempts: apis.listAttempts }));
vi.mock("../readingApi", () => ({ listReadings: apis.listReadings }));
vi.mock("../listeningApi", () => ({ listListenings: apis.listListenings }));
vi.mock("../mockApi", () => ({ listMocks: apis.listMocks }));
vi.mock("../writingModuleApi", () => ({ listWritingModules: apis.listWritingModules }));

beforeEach(() => {
  localStorage.clear();
  Object.values(apis).forEach((f) => f.mockReset());
  Object.values(apis).forEach((f) => f.mockResolvedValue([]));
});

describe("IeltsHome", () => {
  it("welcomes a new learner, marks every test not tried and recommends Listening", async () => {
    render(<IeltsHome />);
    expect(await screen.findByRole("heading", { name: /Start with one test/ })).toBeInTheDocument();
    expect(screen.getAllByText("No band yet").length).toBe(5);
    for (const link of screen.getAllByRole("link", { name: "Start Listening" })) {
      expect(link).toHaveAttribute("href", "/ielts-mock?module=listening");
    }
    expect(screen.getByText("Recommended")).toBeInTheDocument();
  });

  it("shows the latest Reading band and moves the recommendation on", async () => {
    apis.listReadings.mockResolvedValue([{ band: 6.5 }]);
    apis.listListenings.mockResolvedValue([{ band: 7 }]);
    render(<IeltsHome />);
    expect(await screen.findByRole("heading", { name: /Writing test is next/ })).toBeInTheDocument();
    expect(screen.getByText("6.5")).toBeInTheDocument();
  });

  it("keeps the last marked band when a newer attempt has none yet", async () => {
    apis.listReadings.mockResolvedValue([{ band: null }, { band: 7 }]);
    apis.listListenings.mockResolvedValue([{ band: 6.5 }]);
    render(<IeltsHome />);
    expect(await screen.findByRole("heading", { name: /Writing test is next/ })).toBeInTheDocument();
    expect(screen.getByText("7.0")).toBeInTheDocument();
    expect(screen.getAllByText("No band yet").length).toBe(3);
  });
});

describe("encourage", () => {
  it("gives a different plain line per band range", () => {
    const lines = [4, 5, 6, 7, 8].map(encourage);
    expect(new Set(lines).size).toBe(5);
  });
});
