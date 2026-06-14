import { useRef, useState } from "react";

const TEST_TYPES = ["gre"]; // extend list as new test types ship

interface Props {
  onSubmit: (file: File, testType: string) => void;
  uploading: boolean;
}

export function AdminUploadForm({ onSubmit, uploading }: Props) {
  const [testType, setTestType] = useState("gre");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    setSelectedFile(e.target.files?.[0] ?? null);
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedFile) return;
    onSubmit(selectedFile, testType);
  }

  return (
    <form onSubmit={handleSubmit} className="admin-upload-form">
      <div className="form-field">
        <label htmlFor="admin-file-input">Document</label>
        <input
          id="admin-file-input"
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          onChange={handleFileChange}
          data-testid="admin-file-input"
        />
      </div>

      <div className="form-field">
        <label htmlFor="test-type-select">Test type</label>
        <select
          id="test-type-select"
          value={testType}
          onChange={(e) => setTestType(e.target.value)}
          aria-label="Test type"
        >
          {TEST_TYPES.map((t) => (
            <option key={t} value={t}>
              {t.toUpperCase()}
            </option>
          ))}
        </select>
      </div>

      <button
        type="submit"
        disabled={uploading}
        aria-label={uploading ? "Uploading…" : "Upload document"}
      >
        {uploading ? "Uploading…" : "Upload"}
      </button>
    </form>
  );
}
