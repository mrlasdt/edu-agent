import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { UploadOverlay } from "../components/UploadOverlay";

function uploadFile(input: HTMLElement, file: File) {
  // fireEvent.change bypasses jsdom's interactability checks and directly
  // fires the change event — appropriate for testing validation logic
  // (not the browser's file-picker UI).
  fireEvent.change(input, { target: { files: [file] } });
}

const defaultProps = {
  isOpen: true,
  onClose: vi.fn(),
  onUpload: vi.fn(),
  uploading: false,
  piiDetected: false,
};

describe("UploadOverlay — visibility", () => {
  it("renders when isOpen is true", () => {
    render(<UploadOverlay {...defaultProps} />);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("does not render when isOpen is false", () => {
    render(<UploadOverlay {...defaultProps} isOpen={false} />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});

describe("UploadOverlay — accepted types shown", () => {
  it("shows PDF, DOCX, TXT as accepted types", () => {
    render(<UploadOverlay {...defaultProps} />);
    const text = screen.getByRole("dialog").textContent ?? "";
    expect(text).toMatch(/pdf/i);
    expect(text).toMatch(/docx/i);
    expect(text).toMatch(/txt/i);
  });
});

describe("UploadOverlay — client-side validation", () => {
  it("rejects unsupported file type without calling onUpload", () => {
    const onUpload = vi.fn();
    render(<UploadOverlay {...defaultProps} onUpload={onUpload} />);
    const input = screen.getByTestId("file-input");
    const badFile = new File(["data"], "photo.jpg", { type: "image/jpeg" });
    uploadFile(input, badFile);
    expect(onUpload).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(/not supported/i);
  });

  it("rejects files over 10 MB without calling onUpload", () => {
    const onUpload = vi.fn();
    render(<UploadOverlay {...defaultProps} onUpload={onUpload} />);
    const input = screen.getByTestId("file-input");
    const bigContent = new Uint8Array(11 * 1024 * 1024); // 11 MB
    const bigFile = new File([bigContent], "big.txt", { type: "text/plain" });
    uploadFile(input, bigFile);
    expect(onUpload).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent(/10 mb/i);
  });

  it("calls onUpload for a valid small file", () => {
    const onUpload = vi.fn();
    render(<UploadOverlay {...defaultProps} onUpload={onUpload} />);
    const input = screen.getByTestId("file-input");
    const file = new File(["hello world"], "notes.txt", { type: "text/plain" });
    uploadFile(input, file);
    expect(onUpload).toHaveBeenCalledWith(file);
  });
});

describe("UploadOverlay — uploading state", () => {
  it("shows progress indicator when uploading", () => {
    render(<UploadOverlay {...defaultProps} uploading={true} />);
    expect(screen.getByRole("progressbar")).toBeInTheDocument();
  });

  it("does not show progressbar when not uploading", () => {
    render(<UploadOverlay {...defaultProps} uploading={false} />);
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });
});

describe("UploadOverlay — PII warning", () => {
  it("shows PII warning banner when piiDetected is true", () => {
    render(<UploadOverlay {...defaultProps} piiDetected={true} />);
    expect(screen.getByRole("alert")).toHaveTextContent(/pii|personal information|redact/i);
  });

  it("does not show PII warning when piiDetected is false", () => {
    render(<UploadOverlay {...defaultProps} piiDetected={false} />);
    // No alert unless there's a validation error
    expect(screen.queryByText(/redact/i)).not.toBeInTheDocument();
  });
});
