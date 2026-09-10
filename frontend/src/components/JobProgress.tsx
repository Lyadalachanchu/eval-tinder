import { useEffect, useRef, useState } from 'react'
import { api, errorMessage } from '../api'
import { formatDate } from '../format'
import { isTerminalJob, useJob } from '../hooks/usePolling'
import type { JobOut } from '../types'
import { StateBadge } from './Badge'
import { JsonBlock } from './JsonBlock'
import { ErrorBox } from './Status'

/**
 * Polls a job every 1.5s and shows its state, progress, result and errors.
 * `onFinished` fires once when the job reaches a terminal state.
 */
export function JobProgress({
  jobId,
  title = 'Job',
  onFinished,
  compact = false,
}: {
  jobId: string | null | undefined
  title?: string
  onFinished?: (job: JobOut) => void
  compact?: boolean
}) {
  const { data: job, error } = useJob(jobId)
  const [cancelError, setCancelError] = useState<unknown>(null)
  const notified = useRef<string | null>(null)
  const onFinishedRef = useRef(onFinished)
  onFinishedRef.current = onFinished

  useEffect(() => {
    if (job && isTerminalJob(job) && notified.current !== job.id) {
      notified.current = job.id
      onFinishedRef.current?.(job)
    }
  }, [job])

  if (!jobId) return null
  if (error) return <ErrorBox error={error} prefix={`${title} status`} />
  if (!job) return <p className="muted">Polling job {jobId}…</p>

  const progressEntries = Object.entries(job.progress ?? {})
  const cancel = async () => {
    setCancelError(null)
    try {
      await api.cancelJob(job.id)
    } catch (e) {
      setCancelError(e)
    }
  }

  return (
    <div className="job-progress">
      <div className="row gap">
        <strong>{title}</strong>
        <StateBadge state={job.state} />
        <span className="muted small">
          {job.kind} · {job.id}
        </span>
        {!isTerminalJob(job) ? (
          <button type="button" className="btn btn-small" onClick={cancel} disabled={job.cancel_requested}>
            {job.cancel_requested ? 'Cancel requested' : 'Cancel'}
          </button>
        ) : null}
      </div>
      {progressEntries.length > 0 ? (
        <ul className="kv-list">
          {progressEntries.map(([k, v]) => (
            <li key={k}>
              <span className="kv-key">{k}</span>
              <span className="kv-val">{typeof v === 'object' ? JSON.stringify(v) : String(v)}</span>
            </li>
          ))}
        </ul>
      ) : !isTerminalJob(job) ? (
        <p className="muted small">No progress reported yet.</p>
      ) : null}
      {job.error ? <ErrorBox error={new Error(job.error)} prefix="Job failed" /> : null}
      {cancelError ? <ErrorBox error={new Error(errorMessage(cancelError))} prefix="Cancel" /> : null}
      {!compact && isTerminalJob(job) && Object.keys(job.result ?? {}).length > 0 ? (
        <details>
          <summary>Result</summary>
          <JsonBlock value={job.result} maxHeight={280} />
        </details>
      ) : null}
      {!compact ? (
        <p className="muted small">
          Started {formatDate(job.started_at)} · Finished {formatDate(job.finished_at)} · attempts {job.attempts}
        </p>
      ) : null}
    </div>
  )
}
