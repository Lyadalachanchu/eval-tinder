import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, isNotAvailable } from '../api'
import { Badge, HumanBadge, MachineBadge, ProvisionalBadge, StateBadge } from '../components/Badge'
import { JobProgress } from '../components/JobProgress'
import { JsonBlock, TextBlock } from '../components/JsonBlock'
import { ProjectNav } from '../components/ProjectNav'
import { EmptyState, ErrorBox, Loading, NotAvailable } from '../components/Status'
import { truncate } from '../format'
import { useAsync } from '../hooks/useAsync'
import { useIdempotencyKey } from '../hooks/useIdempotencyKey'

const PAGE_SIZE = 25

export function TracesPage() {
  const { projectId = '' } = useParams()
  const [partition, setPartition] = useState<'' | 'TRAIN' | 'DEV'>('')
  const [offset, setOffset] = useState(0)
  const [expanded, setExpanded] = useState<string | null>(null)
  const page = useAsync(() => api.listTraces(projectId, { partition: partition || undefined, limit: PAGE_SIZE, offset }), [projectId, partition, offset])
  const dash = useAsync(() => api.getProject(projectId), [projectId])
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobError, setJobError] = useState<unknown>(null)
  const [busy, setBusy] = useState(false)
  const { key, reset } = useIdempotencyKey()

  const shadow = dash.data?.shadow_grader ?? null

  const runBulk = async () => {
    if (!shadow) return
    setBusy(true)
    setJobError(null)
    try {
      const job = await api.createGradingJob(projectId, { grader_id: shadow.id, partition: partition || undefined, idempotency_key: key() })
      reset()
      setJobId(job.id)
    } catch (e) {
      setJobError(e)
    } finally {
      setBusy(false)
    }
  }

  const total = page.data?.total ?? 0
  const items = page.data?.items ?? []

  return (
    <div>
      <ProjectNav projectId={projectId} name={dash.data?.project.name} />
      <h1>Traces</h1>
      <p className="muted small">
        TRAIN and DEV traces only; AUDIT_RESERVE material is sealed. Human labels (<HumanBadge />) are shown next to, never replaced
        by, provisional shadow predictions (<MachineBadge /> <ProvisionalBadge />).
      </p>

      <section className="card">
        <div className="row gap">
          <label className="field">
            <span>Partition</span>
            <select
              value={partition}
              onChange={(e) => {
                setPartition(e.target.value as '' | 'TRAIN' | 'DEV')
                setOffset(0)
              }}
            >
              <option value="">TRAIN + DEV</option>
              <option value="TRAIN">TRAIN</option>
              <option value="DEV">DEV</option>
            </select>
          </label>
          <div className="field">
            <span>Bulk grading</span>
            <button type="button" className="btn btn-primary" onClick={runBulk} disabled={!shadow || busy}>
              {busy ? 'Starting…' : 'Run bulk grading with shadow grader'}
            </button>
          </div>
          {shadow ? (
            <span className="small">
              shadow: <Link to={`/graders/${shadow.id}`}>{shadow.label}</Link> <ProvisionalBadge />
            </span>
          ) : (
            <span className="muted small">No shadow grader selected; choose one on an optimization run first.</span>
          )}
        </div>
        {isNotAvailable(jobError) ? <NotAvailable feature="Bulk grading" /> : <ErrorBox error={jobError} />}
        {jobId ? <JobProgress jobId={jobId} title="Bulk grading job" onFinished={() => page.reload()} /> : null}
      </section>

      <section className="card">
        {page.loading && !page.data ? (
          <Loading />
        ) : page.error ? (
          <ErrorBox error={page.error} onRetry={page.reload} />
        ) : items.length === 0 ? (
          <EmptyState title="No browsable traces">
            {total === 0 ? 'Import a JSONL file from the dashboard, or change the partition filter.' : 'This page is empty.'}
          </EmptyState>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>External id</th>
                  <th>Partition</th>
                  <th>Input</th>
                  <th>Output</th>
                  <th>Human label</th>
                  <th>Shadow prediction</th>
                </tr>
              </thead>
              <tbody>
                {items.map((row) => {
                  const t = row.trace
                  const open = expanded === t.id
                  return [
                    <tr key={t.id} className={`clickable${open ? ' selected' : ''}`} onClick={() => setExpanded(open ? null : t.id)}>
                      <td className="mono small">{t.external_id}</td>
                      <td>
                        <Badge kind="info">{row.partition}</Badge>
                      </td>
                      <td className="small">{truncate(t.input)}</td>
                      <td className="small">{truncate(t.output)}</td>
                      <td>
                        {row.human_judgment ? (
                          <span className="row gap">
                            <HumanBadge />
                            <StateBadge state={row.human_judgment.verdict} />
                          </span>
                        ) : (
                          <span className="muted">—</span>
                        )}
                      </td>
                      <td>
                        {row.shadow_prediction ? (
                          <span className="row gap">
                            <MachineBadge />
                            <ProvisionalBadge />
                            <StateBadge state={row.shadow_prediction.verdict} />
                            {row.shadow_prediction.status !== 'OK' ? <StateBadge state={row.shadow_prediction.status} /> : null}
                          </span>
                        ) : (
                          <span className="muted">—</span>
                        )}
                      </td>
                    </tr>,
                    open ? (
                      <tr key={`${t.id}-detail`}>
                        <td colSpan={6}>
                          <div className="grid-2">
                            <div>
                              <h3>User request</h3>
                              <TextBlock text={t.input} />
                              <h3>Target output</h3>
                              <TextBlock text={t.output} className="prominent" />
                              <details>
                                <summary>Full snapshot</summary>
                                <JsonBlock value={t} maxHeight={320} />
                              </details>
                            </div>
                            <div>
                              <h3>
                                Human judgment <HumanBadge />
                              </h3>
                              {row.human_judgment ? (
                                <p>
                                  <StateBadge state={row.human_judgment.verdict} />{' '}
                                  {row.human_judgment.cannot_judge_reason ? <Badge kind="warn">{row.human_judgment.cannot_judge_reason}</Badge> : null}{' '}
                                  <span className="small">{row.human_judgment.explanation || ''}</span>
                                </p>
                              ) : (
                                <p className="muted">No human label in this policy epoch.</p>
                              )}
                              <h3>
                                Shadow prediction <MachineBadge /> <ProvisionalBadge />
                              </h3>
                              {row.shadow_prediction ? (
                                <>
                                  <p>
                                    <StateBadge state={row.shadow_prediction.verdict} /> <StateBadge state={row.shadow_prediction.status} />{' '}
                                    <Link to={`/graders/${row.shadow_prediction.grader_id}`} className="small">
                                      grader
                                    </Link>
                                  </p>
                                  {row.shadow_prediction.explanation ? <TextBlock text={row.shadow_prediction.explanation} /> : null}
                                  {row.shadow_prediction.evidence ? (
                                    <details>
                                      <summary>Evidence</summary>
                                      <JsonBlock value={row.shadow_prediction.evidence} maxHeight={200} />
                                    </details>
                                  ) : null}
                                  <p className="muted small">Provisional: no confidence percentage is fabricated from votes or self-reports.</p>
                                </>
                              ) : (
                                <p className="muted">No shadow prediction stored for this trace.</p>
                              )}
                            </div>
                          </div>
                        </td>
                      </tr>
                    ) : null,
                  ]
                })}
              </tbody>
            </table>
          </div>
        )}
        <div className="pagination">
          <button type="button" className="btn btn-small" onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))} disabled={offset === 0}>
            Previous
          </button>
          <span className="muted small">
            {total === 0 ? '0' : `${offset + 1}–${Math.min(offset + PAGE_SIZE, total)}`} of {total}
          </span>
          <button type="button" className="btn btn-small" onClick={() => setOffset(offset + PAGE_SIZE)} disabled={offset + PAGE_SIZE >= total}>
            Next
          </button>
        </div>
      </section>
    </div>
  )
}
