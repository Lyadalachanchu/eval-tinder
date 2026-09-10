import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, buildUrl, fetchBlob, isNotAvailable } from '../api'
import { StateBadge } from '../components/Badge'
import { JobProgress } from '../components/JobProgress'
import { ProjectNav } from '../components/ProjectNav'
import { EmptyState, ErrorBox, Loading, NotAvailable } from '../components/Status'
import { formatDate, shortId } from '../format'
import { useAsync } from '../hooks/useAsync'
import { useIdempotencyKey } from '../hooks/useIdempotencyKey'
import type { ExportOut } from '../types'

const STORAGE_PREFIX = 'eval-tinder.exports.'

function loadKnownIds(projectId: string): string[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_PREFIX + projectId)
    const parsed = raw ? (JSON.parse(raw) as unknown) : []
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === 'string') : []
  } catch {
    return []
  }
}

function rememberId(projectId: string, id: string): void {
  try {
    const ids = loadKnownIds(projectId)
    if (!ids.includes(id)) window.localStorage.setItem(STORAGE_PREFIX + projectId, JSON.stringify([id, ...ids].slice(0, 50)))
  } catch {
    // ignore
  }
}

function DownloadButton({ exp }: { exp: ExportOut }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const path = api.exportDownloadPath(exp.id)
  const download = async () => {
    setBusy(true)
    setError(null)
    try {
      const blob = await fetchBlob(path)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `eval-tinder-export-${exp.kind.toLowerCase()}-${exp.id}.zip`
      document.body.appendChild(a)
      a.click()
      a.remove()
      setTimeout(() => URL.revokeObjectURL(url), 10_000)
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="stack">
      <div className="row gap">
        <button type="button" className="btn btn-primary btn-small" onClick={download} disabled={busy}>
          {busy ? 'Downloading…' : exp.kind === 'GRADER' ? 'Download grader bundle (zip)' : 'Download full export (zip)'}
        </button>
        <span className="muted small mono">{exp.download_url || buildUrl(path)}</span>
      </div>
      <ErrorBox error={error} />
    </div>
  )
}

function ExportRow({ exp, onFinished }: { exp: ExportOut; onFinished: () => void }) {
  const done = exp.state === 'SUCCEEDED' || exp.state === 'COMPLETE' || exp.state === 'READY'
  const extraLinks = Object.entries(exp).filter(([k, v]) => k !== 'download_url' && k.endsWith('_url') && typeof v === 'string' && v)
  return (
    <div className="card">
      <div className="row gap">
        <strong>{exp.kind}</strong>
        <StateBadge state={exp.state} />
        <span className="mono small">{shortId(exp.id, 12)}</span>
        {exp.grader_id ? (
          <Link to={`/graders/${exp.grader_id}`} className="small">
            grader {shortId(exp.grader_id)}
          </Link>
        ) : null}
        <span className="muted small">{formatDate(exp.created_at)}</span>
      </div>
      {!done && exp.job_id ? <JobProgress jobId={exp.job_id} title="Export job" onFinished={onFinished} compact /> : null}
      {exp.error ? <ErrorBox error={new Error(exp.error)} prefix="Export failed" /> : null}
      {done ? <DownloadButton exp={exp} /> : null}
      {extraLinks.length > 0 ? (
        <ul className="kv-list small">
          {extraLinks.map(([k, v]) => (
            <li key={k}>
              <span className="kv-key">{k}</span>
              <span className="kv-val mono">{String(v)}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}

export function ExportsPage() {
  const { projectId = '' } = useParams()
  const graders = useAsync(() => api.listGraders(projectId), [projectId])
  const listed = useAsync(() => api.listExports(projectId), [projectId])
  const [local, setLocal] = useState<ExportOut[]>([])
  const [kind, setKind] = useState<'FULL' | 'GRADER'>('FULL')
  const [graderId, setGraderId] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const { key, reset } = useIdempotencyKey()
  const listUnavailable = isNotAvailable(listed.error)

  // When the list endpoint is missing, fall back to exports remembered in this browser.
  useEffect(() => {
    if (!listUnavailable) return
    let cancelled = false
    Promise.all(loadKnownIds(projectId).map((id) => api.getExport(id).catch(() => null))).then((rows) => {
      if (!cancelled) setLocal(rows.filter((r): r is ExportOut => r !== null))
    })
    return () => {
      cancelled = true
    }
  }, [projectId, listUnavailable])

  const create = async () => {
    setBusy(true)
    setError(null)
    try {
      const exp = await api.createExport(projectId, { kind, grader_id: kind === 'GRADER' ? graderId : undefined, idempotency_key: key() })
      reset()
      rememberId(projectId, exp.id)
      setLocal((rows) => [exp, ...rows.filter((r) => r.id !== exp.id)])
      listed.reload()
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  const refreshOne = async (id: string) => {
    try {
      const fresh = await api.getExport(id)
      setLocal((rows) => rows.map((r) => (r.id === id ? fresh : r)))
      listed.reload()
    } catch {
      listed.reload()
    }
  }

  const rows: ExportOut[] = listUnavailable ? local : [...local.filter((l) => !(listed.data ?? []).some((r) => r.id === l.id)), ...(listed.data ?? [])]
  const featureUnavailable = isNotAvailable(error)

  return (
    <div>
      <ProjectNav projectId={projectId} />
      <h1>Exports</h1>
      <p className="muted small">
        A FULL export contains JSONL traces, separate human judgments and MACHINE predictions, grader manifests, policy epoch,
        split/exposure provenance, dataset hashes, optimizer configuration and eligible audit reports. A GRADER export is one
        grader bundle the CLI can reconstruct. Secrets, sealed audit material and executable pickles are never included.
      </p>
      <section className="card">
        <h2>Create export</h2>
        {featureUnavailable ? (
          <NotAvailable feature="Exports" />
        ) : (
          <div className="row gap">
            <label className="field">
              <span>Kind</span>
              <select value={kind} onChange={(e) => setKind(e.target.value as 'FULL' | 'GRADER')}>
                <option value="FULL">FULL</option>
                <option value="GRADER">GRADER bundle</option>
              </select>
            </label>
            {kind === 'GRADER' ? (
              <label className="field">
                <span>Grader</span>
                <select value={graderId} onChange={(e) => setGraderId(e.target.value)}>
                  <option value="">Select a grader…</option>
                  {(graders.data ?? []).map((g) => (
                    <option key={g.id} value={g.id}>
                      {g.label} ({g.origin})
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            <div className="field">
              <span>&nbsp;</span>
              <button type="button" className="btn btn-primary" onClick={create} disabled={busy || (kind === 'GRADER' && !graderId)}>
                {busy ? 'Creating…' : 'Create export'}
              </button>
            </div>
          </div>
        )}
        {!featureUnavailable ? <ErrorBox error={error} /> : null}
      </section>
      <section>
        <h2>Exports</h2>
        {listed.loading && !listUnavailable ? (
          <Loading />
        ) : listed.error && !listUnavailable ? (
          <ErrorBox error={listed.error} onRetry={listed.reload} />
        ) : rows.length === 0 ? (
          <EmptyState title="No exports yet">
            {listUnavailable ? 'Exports created from this browser will appear here.' : 'Create a FULL or GRADER export above.'}
          </EmptyState>
        ) : (
          rows.map((exp) => <ExportRow key={exp.id} exp={exp} onFinished={() => void refreshOne(exp.id)} />)
        )}
      </section>
    </div>
  )
}
