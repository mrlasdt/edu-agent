import { useRef, useState } from "react";

const ACCEPTED_EXTENSIONS = [".pdf", ".docx", ".txt"];
const ACCEPTED_TYPES = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"];
const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onUpload: (file: File) => void;
  uploading: boolean;
  piiDetected: boolean;
}

function isAccepted(file: File): boolean {
  const name = file.name.toLowerCase();
  return (
    ACCEPTED_EXTENSIONS.some((ext) => name.endsWith(ext)) ||
    ACCEPTED_TYPES.includes(file.type)
  );
}

export function UploadOverlay({ isOpen, onClose, onUpload, uploading, piiDetected }: Props) {
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  function handleFile(file: File) {
    setError(null);
    if (!isAccepted(file)) {
      setError(`File type not supported. Please upload PDF, DOCX, or TXT.`);
      return;
    }
    if (file.size > MAX_SIZE_BYTES) {
      setError(`File exceeds the 10 MB limit. Please upload a smaller file.`);
      return;
    }
    onUpload(file);
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  return (
    <div
      role="dialog"
      aria-label="Upload document"
      aria-modal="true"
      className="upload-overlay"
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
    >
      <div className="upload-overlay__content">
        <h2>Upload your notes</h2>
        <p className="upload-accepted-types">
          Accepted formats: <strong>PDF</strong>, <strong>DOCX</strong>, <strong>TXT</strong>
        </p>

        <div
          className="upload-drop-zone"
          onClick={() => inputRef.current?.click()}
          aria-label="Drop zone — click or drag a file here"
        >
          {uploading ? (
            <div role="progressbar" aria-label="Uploading…" className="upload-progress">
              Uploading…
            </div>
          ) : (
            <span>Click or drag a file here</span>
          )}
        </div>

        <input
          ref={inputRef}
          data-testid="file-input"
          type="file"
          accept=".pdf,.docx,.txt"
          style={{ position: "absolute", width: 0, height: 0, opacity: 0 }}
          onChange={handleChange}
        />

        {error && (
          <div role="alert" className="upload-error">
            {error}
          </div>
        )}

        {piiDetected && !error && (
          <div role="alert" className="upload-pii-warning">
            Personal information (PII) was detected in this document. Please redact
            sensitive details before uploading.
          </div>
        )}

        <button onClick={onClose} aria-label="Close upload dialog" className="upload-close">
          Cancel
        </button>
      </div>
    </div>
  );
}
