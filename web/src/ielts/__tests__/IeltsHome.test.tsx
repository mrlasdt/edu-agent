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
  listSpeakings: vi.fn(),
}));
vi.mock("../api", () => ({ listAttempts: apis.listAttempts }));
vi.mock("../readingApi", () => ({ listReadings: apis.listReadings }));
vi.mock("../listeningApi", () => ({ listListenings: apis.listListenings }));
vi.mock("../mockApi", () => ({ listMocks: apis.listMocks }));
vi.mock("../speakingApi", () => ({ listSpeakings: apis.listSpeakings }));
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
    expect(screen.getAllByText("No band yet").length).toBe(6);
    expect(screen.getByRole("link", { name: "Start Speaking" })).toHaveAttribute("href", "/ielts-mock?module=speaking");
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
    expect(screen.getAllByText("No band yet").length).toBe(4);
  });

  it("shows the latest Speaking band from the full mocks", async () => {
    apis.listMocks.mockResolvedValue([
      { overall: null, next_module: null, modules: [{ key: "speaking", band: 6.5 }] },
    ]);
    render(<IeltsHome />);
    expect(await screen.findByText("6.5")).toBeInTheDocument();
    expect(screen.getAllByText("No band yet").length).toBe(5);
  });
});

describe("IeltsHome Speaking card", () => {
  it("uses the newest marked Speaking attempt over the full mocks", async () => {
    apis.listSpeakings.mockResolvedValue([
      { band: null, partial: null, paper: { sample: false } },
      { band: 6, partial: false, paper: { sample: false } },
    ]);
    apis.listMocks.mockResolvedValue([
      { overall: null, next_module: null, modules: [{ key: "speaking", band: 5 }] },
    ]);
    render(<IeltsHome />);
    expect(await screen.findByText("6.0")).toBeInTheDocument();
    expect(screen.queryByText("5.0")).not.toBeInTheDocument();
    expect(screen.queryByText(/quick sample/)).not.toBeInTheDocument();
  });

  it("labels a quick sample band as a sample and keeps partial visible", async () => {
    apis.listSpeakings.mockResolvedValue([{ band: 6.5, partial: true, paper: { sample: true } }]);
    render(<IeltsHome />);
    expect(await screen.findByText("Latest band (quick sample)")).toBeInTheDocument();
    expect(screen.getByText("6.5")).toBeInTheDocument();
    expect(screen.getByText("Partial: Pronunciation not assessed")).toBeInTheDocument();
  });

  it("does not let a quick sample band count as a real Speaking band", async () => {
    apis.listListenings.mockResolvedValue([{ band: 6 }]);
    apis.listReadings.mockResolvedValue([{ band: 6 }]);
    apis.listWritingModules.mockResolvedValue([{ writing_band: 6 }]);
    apis.listSpeakings.mockResolvedValue([{ band: 8, partial: true, paper: { sample: true } }]);
    render(<IeltsHome />);
    expect(await screen.findByRole("heading", { name: /Speaking is next/ })).toBeInTheDocument();
    expect(screen.getByText("Latest band (quick sample)")).toBeInTheDocument();
    expect(screen.getByText(/record your Speaking answers/)).toBeInTheDocument();
  });

  it("ignores a sample band when picking the lowest real band", async () => {
    apis.listListenings.mockResolvedValue([{ band: 7 }]);
    apis.listReadings.mockResolvedValue([{ band: 6 }]);
    apis.listWritingModules.mockResolvedValue([{ writing_band: 7 }]);
    apis.listSpeakings.mockResolvedValue([{ band: 4, partial: false, paper: { sample: true } }]);
    render(<IeltsHome />);
    expect(await screen.findByRole("heading", { name: /Speaking is next/ })).toBeInTheDocument();
  });

  it("stops recommending Speaking once a Speaking attempt is marked", async () => {
    apis.listListenings.mockResolvedValue([{ band: 6 }]);
    apis.listReadings.mockResolvedValue([{ band: 6 }]);
    apis.listWritingModules.mockResolvedValue([{ writing_band: 6 }]);
    apis.listSpeakings.mockResolvedValue([{ band: 6, partial: false, paper: { sample: false } }]);
    render(<IeltsHome />);
    expect(await screen.findByText(/lowest band/)).toBeInTheDocument();
  });
});

describe("encourage", () => {
  it("gives a different plain line per band range", () => {
    const lines = [4, 5, 6, 7, 8].map(encourage);
    expect(new Set(lines).size).toBe(5);
  });
});
