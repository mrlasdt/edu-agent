import type { IngestionJob } from "../types";

interface Props {
  jobs: IngestionJob[];
  dlqCount: number;
  onViewDlq: () => void;
}

const STATUS_CLASS: Record<string, string> = {
  completed: "status--completed",
  processing: "status--processing",
  queued: "status--queued",
  failed: "status--failed",
};

export function JobsTable({ jobs, dlqCount, onViewDlq }: Props) {
  return (
    <div className="jobs-table-container">
      <div className="jobs-table-header">
        <h2>Ingestion jobs</h2>
        {dlqCount > 0 && (
          <button
            className="dlq-badge"
            onClick={onViewDlq}
            aria-label={`Dead-letter queue — ${dlqCount} items`}
          >
            Dead-letter queue <span className="dlq-count">{dlqCount}</span>
          </button>
        )}
      </div>

      {jobs.length === 0 ? (
        <p className="jobs-empty">No jobs yet. Upload a document to get started.</p>
      ) : (
        <table className="jobs-table" aria-label="Ingestion jobs">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Test type</th>
              <th>Status</th>
              <th>Error stage</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.job_id} className={`job-row job-row--${job.status}`}>
                <td>{job.filename}</td>
                <td>{job.test_type}</td>
                <td>
                  <span className={`job-status ${STATUS_CLASS[job.status] ?? ""}`}>
                    {job.status}
                  </span>
                </td>
                <td>{job.error_stage ?? "—"}</td>
                <td>{new Date(job.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
