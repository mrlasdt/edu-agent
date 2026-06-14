import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AdminUploadForm } from "../components/AdminUploadForm";

const defaultProps = {
  onSubmit: vi.fn(),
  uploading: false,
};

describe("AdminUploadForm — form elements", () => {
  it("renders a file input", () => {
    render(<AdminUploadForm {...defaultProps} />);
    expect(screen.getByTestId("admin-file-input")).toBeInTheDocument();
  });

  it("renders a test_type dropdown with gre as an option", () => {
    render(<AdminUploadForm {...defaultProps} />);
    const select = screen.getByRole("combobox", { name: /test type/i });
    expect(select).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /gre/i })).toBeInTheDocument();
  });

  it("renders a submit button", () => {
    render(<AdminUploadForm {...defaultProps} />);
    expect(screen.getByRole("button", { name: /upload/i })).toBeInTheDocument();
  });
});

describe("AdminUploadForm — test_type selection", () => {
  it("defaults to gre", () => {
    render(<AdminUploadForm {...defaultProps} />);
    const select = screen.getByRole("combobox", { name: /test type/i }) as HTMLSelectElement;
    expect(select.value).toBe("gre");
  });

  it("allows changing test_type", async () => {
    render(<AdminUploadForm {...defaultProps} />);
    const select = screen.getByRole("combobox", { name: /test type/i });
    await userEvent.selectOptions(select, "gre");
    expect((select as HTMLSelectElement).value).toBe("gre");
  });
});

describe("AdminUploadForm — submission", () => {
  it("calls onSubmit with file and test_type when submitted", async () => {
    const onSubmit = vi.fn();
    render(<AdminUploadForm {...defaultProps} onSubmit={onSubmit} />);

    const fileInput = screen.getByTestId("admin-file-input");
    const file = new File(["content"], "guide.pdf", { type: "application/pdf" });
    fireEvent.change(fileInput, { target: { files: [file] } });

    await userEvent.click(screen.getByRole("button", { name: /upload/i }));
    expect(onSubmit).toHaveBeenCalledWith(file, "gre");
  });

  it("does not call onSubmit when no file selected", async () => {
    const onSubmit = vi.fn();
    render(<AdminUploadForm {...defaultProps} onSubmit={onSubmit} />);
    await userEvent.click(screen.getByRole("button", { name: /upload/i }));
    expect(onSubmit).not.toHaveBeenCalled();
  });
});

describe("AdminUploadForm — uploading state", () => {
  it("disables submit button while uploading", () => {
    render(<AdminUploadForm {...defaultProps} uploading={true} />);
    expect(screen.getByRole("button", { name: /uploading/i })).toBeDisabled();
  });

  it("shows enabled button when not uploading", () => {
    render(<AdminUploadForm {...defaultProps} uploading={false} />);
    expect(screen.getByRole("button", { name: /upload/i })).not.toBeDisabled();
  });
});
