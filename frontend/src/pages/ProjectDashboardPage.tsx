import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, isNotAvailable } from '../api'
import { Badge, ProvisionalBadge, StateBadge } from '../components/Badge'
import { JobProgress } from '../components/JobProgress'
import { JsonBlock } from '../components/JsonBlock'
import { ProjectNav } from '../components/ProjectNav'
import { ReasonForm } from '../components/ReasonForm'
import { StartRunForm } from '../components/StartRunForm'
import { EmptyState, ErrorBox, LoadFailure, Loading, NotAvailable, Notice } from '../components/Status'
import { formatDate, shortId } from '../format'
import { useAsync } from '../hooks/useAsync'
import { useIdempotencyKey } from '../hooks/useIdempotencyKey'
import { usePolling } from '../hooks/usePolling'
import type { ImportOut, ProjectDashboard, ReviewRequestOut, SelectionRoundOut } from '../types'

// ------------------------------------------------------------------ import

function ImportCard({ projectId, onChanged }: { projectId: string; onChanged: () => void }) {
  const imports = useAsync(() => api.listImports(projectId), [projectId])
  const [file, setFile] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const [latest, setLatest] = useState<ImportOut | null>(null)
  const { key, reset } = useIdempotencyKey()
  const [inputKey, setInputKey] = useState(0)

  const upload = async (event: FormEvent) => {
    event.preventDefault()
    if (!file) {
      setError(new Error('Choose a JSONL file first.'))
      return
    }
    setBusy(true)
    setError(null)
    try {
      const result = await api.createImport(projectId, file, key())
      reset()
      setLatest(result)
      setFile(null)
      setInputKey((k) => k + 1)
      imports.reload()
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="card">
      <h2>Import traces</h2>
      <form onSubmit={upload} className="stack">
        <p className="muted small">
          One JSON object per line with external_id, group_id, input, context, tool_calls, output, metadata. Related groups
          are assigned to TRAIN, DEV or AUDIT_RESERVE before any label is inspected.
        </p>
        <div className="row gap">
          <input key={inputKey} type="file" accept=".jsonl,.json,.txt,application/jsonl" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          <button type="submit" className="btn btn-primary" disabled={busy || !file}>
            {busy ? 'Uploading…' : 'Upload JSONL'}
          </button>
        </div>
        <ErrorBox error={error} />
      </form>
      {latest ? (
        <div className="stack">
          <p>
            Import <span className="mono">{shortId(latest.id)}</span> accepted <StateBadge state={latest.state} />
          </p>
          <JobProgress
            jobId={latest.job_id}
            title="Import job"
            onFinished={() => {
              imports.reload()
              onChanged()
            }}
          />
        </div>
      ) : null}
      <h3>Previous imports</h3>
      {imports.loading ? (
        <Loading />
      ) : imports.error ? (
        <ErrorBox error={imports.error} onRetry={imports.reload} />
      ) : !imports.data || imports.data.length === 0 ? (
        <EmptyState title="Nothing imported yet">Upload a JSONL file to populate the partitions.</EmptyState>
      ) : (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>File</th>
                <th>State</th>
                <th>Counts</th>
                <th>Line errors</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {imports.data.map((b) => (
                <tr key={b.id}>
                  <td>{b.filename}</td>
                  <td>
                    <StateBadge state={b.state} />
                  </td>
                  <td className="small mono">
                    {Object.entries(b.counts ?? {})
                      .map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : String(v)}`)
                      .join(' ') || '—'}
                  </td>
                  <td>
                    {b.line_errors?.length ? (
                      <details>
                        <summary>{b.line_errors.length}</summary>
                        <JsonBlock value={b.line_errors.slice(0, 20)} maxHeight={200} />
                      </details>
                    ) : (
                      '0'
                    )}
                  </td>
                  <td>{formatDate(b.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

// ------------------------------------------------------------------ data overview

function DataCard({ d }: { d: ProjectDashboard }) {
  const partitions = ['TRAIN', 'DEV', 'AUDIT_RESERVE'] as const
  const totalTraces = partitions.reduce((n, p) => n + (d.partitions[p] ?? 0), 0)
  const labelRows = Object.entries(d.labels ?? {})
  const reviewStates = Object.entries(d.review_states ?? {})
  return (
    <section className="card">
      <h2>Data</h2>
      {totalTraces === 0 ? (
        <EmptyState title="No traces yet">Partition counts, labels and review states appear after the first import.</EmptyState>
      ) : (
        <>
          <h3>Groups per partition</h3>
          <div className="stat-row">
            {partitions.map((p) => (
              <div className="stat" key={p}>
                <div className="stat-value">{d.partitions[p] ?? 0}</div>
                <div className="stat-label">{p}</div>
              </div>
            ))}
          </div>
          <p className="muted small">AUDIT_RESERVE groups are sealed: they are never browsable and only enter through a locked audit.</p>
          <h3>Resolved labels per partition</h3>
          {labelRows.length === 0 ? (
            <p className="muted">No labels yet.</p>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Partition</th>
                  <th className="num">PASS</th>
                  <th className="num">FAIL</th>
                  <th className="num">CANNOT_JUDGE</th>
                  <th className="num">Resolved (PASS+FAIL)</th>
                </tr>
              </thead>
              <tbody>
                {labelRows.map(([partition, counts]) => (
                  <tr key={partition}>
                    <td>{partition}</td>
                    <td className="num">{counts.PASS ?? 0}</td>
                    <td className="num">{counts.FAIL ?? 0}</td>
                    <td className="num">{counts.CANNOT_JUDGE ?? 0}</td>
                    <td className="num">{counts.resolved ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <h3>Review requests by state</h3>
          {reviewStates.length === 0 ? (
            <p className="muted">No review requests yet.</p>
          ) : (
            <ul className="kv-list">
              {reviewStates.map(([state, n]) => (
                <li key={state}>
                  <span className="kv-key">
                    <StateBadge state={state} />
                  </span>
                  <span className="kv-val">{n}</span>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  )
}

// ------------------------------------------------------------------ readiness

function ReadinessCard({ d }: { d: ProjectDashboard }) {
  const r = d.readiness
  return (
    <section className="card">
      <h2>Readiness</h2>
      <Notice kind="info">These are bootstrap counts only, not sample-size guarantees.</Notice>
      {r.ready_to_optimize_again ? (
        <Notice kind="good">
          <strong>Ready to optimize again.</strong> {r.new_train_labels_since_last_run} new resolved TRAIN labels since the last run.
        </Notice>
      ) : null}
      <ul className="kv-list">
        <li>
          <span className="kv-key">Resolved TRAIN labels</span>
          <span className="kv-val">
            {r.resolved_train} / {r.bootstrap_train_labels} bootstrap target
          </span>
        </li>
        <li>
          <span className="kv-key">Resolved DEV labels</span>
          <span className="kv-val">
            {r.resolved_dev} / {r.bootstrap_dev_labels} bootstrap target
          </span>
        </li>
        <li>
          <span className="kv-key">Bootstrap ready</span>
          <span className="kv-val">{r.bootstrap_ready ? 'yes' : 'not yet'}</span>
        </li>
        <li>
          <span className="kv-key">New TRAIN labels since last run</span>
          <span className="kv-val">{r.new_train_labels_since_last_run}</span>
        </li>
        <li>
          <span className="kv-key">DEV top-up target</span>
          <span className="kv-val">{r.dev_topup_target}</span>
        </li>
        <li>
          <span className="kv-key">Last run</span>
          <span className="kv-val">{r.last_run_id ? <Link to={`/optimization-runs/${r.last_run_id}`}>{shortId(r.last_run_id)}</Link> : '—'}</span>
        </li>
        <li>
          <span className="kv-key">Active run</span>
          <span className="kv-val">{r.active_run ? <Link to={`/optimization-runs/${r.active_run}`}>{shortId(r.active_run)}</Link> : 'none'}</span>
        </li>
        <li>
          <span className="kv-key">Automatic optimization</span>
          <span className="kv-val">{r.automatic_optimization ? 'opt-in enabled' : 'off (opt-in)'}</span>
        </li>
      </ul>
      {r.note ? <p className="muted small">{r.note}</p> : null}
    </section>
  )
}

// ------------------------------------------------------------------ review batches

function BatchForm({
  projectId,
  purpose,
  kind,
  defaultSize,
  title,
  description,
  onCreated,
}: {
  projectId: string
  purpose: 'TRAIN' | 'DEV'
  kind: 'SEED' | 'DEV_RANDOM'
  defaultSize: number
  title: string
  description: string
  onCreated: () => void
}) {
  const [size, setSize] = useState(String(defaultSize))
  const [seed, setSeed] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const [created, setCreated] = useState<ReviewRequestOut[] | null>(null)
  const { key, reset } = useIdempotencyKey()

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const n = Math.max(1, Math.trunc(Number(size) || defaultSize))
      const result = await api.createReviewBatch(projectId, {
        purpose,
        kind,
        size: n,
        seed: seed.trim() ? Math.trunc(Number(seed)) : undefined,
        idempotency_key: key(),
      })
      reset()
      setCreated(result)
      onCreated()
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={submit} className="stack">
      <h3>{title}</h3>
      <p className="muted small">{description}</p>
      <div className="row gap">
        <label className="field">
          <span>Size</span>
          <input type="number" min={1} max={200} value={size} onChange={(e) => setSize(e.target.value)} style={{ width: 90 }} />
        </label>
        <label className="field">
          <span>Seed (optional)</span>
          <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} style={{ width: 120 }} placeholder="random" />
        </label>
        <button type="submit" className="btn btn-primary" disabled={busy}>
          {busy ? 'Creating…' : title}
        </button>
      </div>
      <ErrorBox error={error} />
      {created ? (
        <p>
          Created {created.length} review request{created.length === 1 ? '' : 's'}.{' '}
          <Link to={`/projects/${projectId}/review?purpose=${purpose}`}>Start reviewing {purpose}</Link>
        </p>
      ) : null}
    </form>
  )
}

function ReviewActionsCard({ d, onChanged }: { d: ProjectDashboard; onChanged: () => void }) {
  const projectId = d.project.id
  const r = d.readiness
  const devDefault = Math.max(1, r.dev_topup_target - r.resolved_dev)
  const open = (d.review_states.OPEN ?? 0) + (d.review_states.LEASED ?? 0)
  return (
    <section className="card">
      <h2>Human review</h2>
      <p>
        {open > 0 ? (
          <>
            {open} request{open === 1 ? '' : 's'} waiting.{' '}
          </>
        ) : (
          'No open review requests. '
        )}
        <Link to={`/projects/${projectId}/review?purpose=TRAIN`}>Review TRAIN</Link> ·{' '}
        <Link to={`/projects/${projectId}/review?purpose=DEV`}>Review DEV</Link>
      </p>
      <BatchForm
        projectId={projectId}
        purpose="TRAIN"
        kind="SEED"
        defaultSize={r.bootstrap_train_labels}
        title="Create seed TRAIN batch"
        description="A varied sample of TRAIN groups for the first labels. Selection reasons are never shown before you judge."
        onCreated={onChanged}
      />
      <BatchForm
        projectId={projectId}
        purpose="DEV"
        kind="DEV_RANDOM"
        defaultSize={devDefault}
        title="Create DEV random batch"
        description={`Uniformly random DEV groups. Default size tops DEV up to the target (${r.dev_topup_target}); DEV review is always blind.`}
        onCreated={onChanged}
      />
    </section>
  )
}

// ------------------------------------------------------------------ selection rounds

function isRoundDone(round: SelectionRoundOut): boolean {
  return round.state === 'COMPLETE' || round.state === 'FAILED'
}

function SelectionRoundSummary({ round }: { round: SelectionRoundOut }) {
  const byCategory = new Map<string, number>()
  for (const s of round.selected_requests ?? []) byCategory.set(s.category, (byCategory.get(s.category) ?? 0) + 1)
  return (
    <div className="stack small">
      <div className="row gap">
        <StateBadge state={round.state} />
        <span className="mono">{shortId(round.id)}</span>
        {round.exhausted ? <Badge kind="warn">POOL EXHAUSTED</Badge> : null}
      </div>
      {round.error ? <ErrorBox error={new Error(round.error)} /> : null}
      <ul className="kv-list">
        <li>
          <span className="kv-key">Committee</span>
          <span className="kv-val">
            {round.committee_ids?.length ?? 0} member{(round.committee_ids?.length ?? 0) === 1 ? '' : 's'}
            {round.committee_report?.diversity_claimed === false ? ' · diversity not claimed' : ''}
            {round.committee_report?.reason ? ` · ${round.committee_report.reason}` : ''}
          </span>
        </li>
        <li>
          <span className="kv-key">Probe / pool size</span>
          <span className="kv-val">
            {round.probe_size ?? '—'} / {round.pool_size ?? '—'}
          </span>
        </li>
        <li>
          <span className="kv-key">Selected</span>
          <span className="kv-val">
            {round.selected_requests?.length ?? 0}
            {byCategory.size ? ` (${Array.from(byCategory, ([c, n]) => `${c}: ${n}`).join(', ')})` : ''}
          </span>
        </li>
      </ul>
      {round.committee_report ? (
        <details>
          <summary>Committee report</summary>
          <JsonBlock value={round.committee_report} maxHeight={260} />
        </details>
      ) : null}
      {round.context_repair ? (
        <details>
          <summary>Context repair</summary>
          <JsonBlock value={round.context_repair} maxHeight={200} />
        </details>
      ) : null}
    </div>
  )
}

function SelectionRoundCard({ projectId, onChanged }: { projectId: string; onChanged: () => void }) {
  const rounds = useAsync(() => api.listSelectionRounds(projectId), [projectId])
  const [seed, setSeed] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const [activeId, setActiveId] = useState<string | null>(null)
  const { key, reset } = useIdempotencyKey()
  const polled = usePolling(activeId, (id) => api.getSelectionRound(id), isRoundDone, 1500)
  const notified = useRef<string | null>(null)
  const reloadRounds = rounds.reload

  useEffect(() => {
    if (polled.done && polled.data && notified.current !== polled.data.id) {
      notified.current = polled.data.id
      reloadRounds()
      onChanged()
    }
  }, [polled.done, polled.data, reloadRounds, onChanged])

  const create = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const round = await api.createSelectionRound(projectId, {
        seed: seed.trim() ? Math.trunc(Number(seed)) : undefined,
        idempotency_key: key(),
      })
      reset()
      setActiveId(round.id)
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  const unavailable = isNotAvailable(rounds.error) || isNotAvailable(error)

  return (
    <section className="card">
      <h2>Selection round (active TRAIN batch)</h2>
      <p className="muted small">
        Candidate graders probe the TRAIN pool and select the next cases from disagreements, underrepresented cases and
        random exploration. Which grader said what stays hidden until you judge.
      </p>
      {unavailable ? (
        <NotAvailable feature="Selection rounds" />
      ) : (
        <>
          <form onSubmit={create} className="row gap">
            <label className="field">
              <span>Seed (optional)</span>
              <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} style={{ width: 120 }} placeholder="random" />
            </label>
            <button type="submit" className="btn btn-primary" disabled={busy}>
              {busy ? 'Creating…' : 'Create selection round'}
            </button>
          </form>
          <ErrorBox error={error} />
          {activeId ? (
            polled.error ? (
              <ErrorBox error={polled.error} />
            ) : polled.data ? (
              <SelectionRoundSummary round={polled.data} />
            ) : (
              <Loading label="Waiting for the selection round…" />
            )
          ) : null}
          <h3>Previous rounds</h3>
          {rounds.loading ? (
            <Loading />
          ) : rounds.error ? (
            <ErrorBox error={rounds.error} onRetry={rounds.reload} />
          ) : !rounds.data || rounds.data.length === 0 ? (
            <EmptyState title="No selection rounds yet">Run one after the first optimization run has produced candidates.</EmptyState>
          ) : (
            <div className="stack">
              {rounds.data.slice(0, 5).map((round) => (
                <SelectionRoundSummary key={round.id} round={round} />
              ))}
            </div>
          )}
        </>
      )}
    </section>
  )
}

// ------------------------------------------------------------------ shadow / automation

function ShadowCard({ d, onChanged }: { d: ProjectDashboard; onChanged: () => void }) {
  const shadow = d.shadow_grader
  return (
    <section className="card">
      <div className="card-header">
        <h2>Shadow grader</h2>
        <ProvisionalBadge />
      </div>
      {shadow ? (
        <>
          <p>
            <Link to={`/graders/${shadow.id}`}>{shadow.label}</Link> <span className="muted small mono">{shortId(shadow.manifest_hash, 12)}</span>
          </p>
          <p className="muted small">
            Shadow predictions are provisional. They are stored next to human labels, never over them, and never enable automation.
          </p>
          <ReasonForm
            actionLabel="Clear shadow grader"
            danger
            onSubmit={async (reason) => {
              await api.selectShadowGrader(d.project.id, { grader_id: null, reason })
              onChanged()
            }}
          />
        </>
      ) : (
        <EmptyState title="No shadow grader selected">
          Pick a candidate from an <Link to={`/projects/${d.project.id}/optimization`}>optimization run</Link>. A shadow choice is
          explicit and recorded with a reason.
        </EmptyState>
      )}
    </section>
  )
}

function AutomationCard({ d }: { d: ProjectDashboard }) {
  const a = d.automation
  const state = a?.state ?? 'DISABLED'
  return (
    <section className="card">
      <div className="card-header">
        <h2>Automation</h2>
        <StateBadge state={state} />
      </div>
      {a ? (
        <ul className="kv-list">
          <li>
            <span className="kv-key">Pipeline hash</span>
            <span className="kv-val mono">{shortId(a.pipeline_hash, 16)}</span>
          </li>
          <li>
            <span className="kv-key">Audit</span>
            <span className="kv-val">{a.audit_id ? <Link to={`/audits/${a.audit_id}`}>{shortId(a.audit_id)}</Link> : '—'}</span>
          </li>
        </ul>
      ) : (
        <p className="muted">Disabled by default. No acceptable error target is ever assumed.</p>
      )}
      <p className="muted small">
        Automation can only be enabled for an exact frozen pipeline by an independent audit whose predeclared gate passed.{' '}
        <Link to={`/projects/${d.project.id}/audits`}>Go to audits</Link>
      </p>
    </section>
  )
}

// ------------------------------------------------------------------ page

export function ProjectDashboardPage() {
  const { projectId = '' } = useParams()
  const dash = useAsync(() => api.getProject(projectId), [projectId])
  const graders = useAsync(() => api.listGraders(projectId), [projectId])

  if (dash.loading && !dash.data) return <Loading label="Loading project…" />
  if (dash.error) return <LoadFailure error={dash.error} feature="Project" onRetry={dash.reload} />
  const d = dash.data
  if (!d) return null

  return (
    <div>
      <ProjectNav projectId={projectId} name={d.project.name} />
      <div className="card-header">
        <h1>{d.project.name}</h1>
        <Badge kind="neutral">policy epoch {d.project.policy_epoch}</Badge>
        <span className="muted small">created {formatDate(d.project.created_at)}</span>
      </div>
      {d.project.description ? <p className="muted">{d.project.description}</p> : <p className="muted">No description. There is no rubric: rules come from labels.</p>}
      <p className="muted small">
        {d.graders} grader version{d.graders === 1 ? '' : 's'} · {d.runs} optimization run{d.runs === 1 ? '' : 's'} ·{' '}
        <Link to={`/projects/${projectId}/traces`}>Traces</Link> · <Link to={`/projects/${projectId}/exports`}>Exports</Link>
      </p>

      <div className="grid-2">
        <ImportCard projectId={projectId} onChanged={dash.reload} />
        <DataCard d={d} />
        <ReadinessCard d={d} />
        <ReviewActionsCard d={d} onChanged={dash.reload} />
        <SelectionRoundCard projectId={projectId} onChanged={dash.reload} />
        <section className="card">
          <h2>Optimization</h2>
          {!d.readiness.bootstrap_ready ? (
            <Notice kind="warn">
              Bootstrap labels are incomplete ({d.readiness.resolved_train}/{d.readiness.bootstrap_train_labels} TRAIN,{' '}
              {d.readiness.resolved_dev}/{d.readiness.bootstrap_dev_labels} DEV). The server may still reject a run.
            </Notice>
          ) : null}
          <StartRunForm projectId={projectId} graders={graders.data ?? []} />
          <p className="small">
            <Link to={`/projects/${projectId}/optimization`}>All runs</Link>
          </p>
        </section>
        <ShadowCard d={d} onChanged={dash.reload} />
        <AutomationCard d={d} />
      </div>
    </div>
  )
}
