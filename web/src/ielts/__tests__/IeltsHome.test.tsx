import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { IeltsHome } from "../IeltsHome";
import { encourage } from "../format";

const apis = vi.hoisted(() => ({
  listAttempts: vi.fn(),
  getReading: vi.fn(),
  getListening: vi.fn(),
  getMock: vi.fn(),
  getWritingModule: vi.fn(),
}));
vi.mock("../api", () => ({ listAttempts: apis.listAttempts }));
vi.mock("../readingApi", () => ({ getReading: apis.getReading }));
vi.mock("../listeningApi", () => ({ getListening: apis.getListening }));
vi.mock("../mockApi", () => ({ getMock: apis.getMock }));
vi.mock("../writingModuleApi", () => ({ getWritingModule: apis.getWritingModule }));

beforeEach(() => {
  localStorage.clear();
  Object.values(apis).forEach((f) => f.mockReset());
  apis.listAttempts.mockResolvedValue([]);
});

describe("IeltsHome", () => {
  it("welcomes a new learner, marks every test not tried and recommends Listening", async () => {
    render(<IeltsHome />);
    expect(await screen.findByRole("heading", { name: /Start with one test/ })).toBeInTheDocument();
    expect(screen.getAllByText("Not tried yet").length).toBe(5);
    for (const link of screen.getAllByRole("link", { name: "Start Listening" })) {
      expect(link).toHaveAttribute("href", "/ielts-mock?module=listening");
    }
    expect(screen.getByText("Recommended")).toBeInTheDocument();
  });

  it("shows the latest Reading band and moves the recommendation on", async () => {
    localStorage.setItem("ielts.reading.attempt", "r1");
    localStorage.setItem("ielts.listening.attempt", "l1");
    apis.getReading.mockResolvedValue({ band: 6.5 });
    apis.getListening.mockResolvedValue({ band: 7 });
    render(<IeltsHome />);
    expect(await screen.findByRole("heading", { name: /Writing test is next/ })).toBeInTheDocument();
    expect(screen.getByText("6.5")).toBeInTheDocument();
  });
});

describe("encourage", () => {
  it("gives a different plain line per band range", () => {
    const lines = [4, 5, 6, 7, 8].map(encourage);
    expect(new Set(lines).size).toBe(5);
  });
});
