import { useState, type FormEvent } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, isNotAvailable } from '../api'
import { AuditEvidenceBadge, StateBadge } from '../components/Badge'
import { ProjectNav } from '../components/ProjectNav'
import { EmptyState, ErrorBox, Loading, NotAvailable } from '../components/Status'
import { formatDate, shortId } from '../format'
import { useAsync } from '../hooks/useAsync'
import { useIdempotencyKey } from '../hooks/useIdempotencyKey'
import type { AuditCreate, GraderOut } from '../types'

function parseRatio(value: string, label: string, errors: string[]): number {
  if (!value.trim()) {
    errors.push(`${label} is required.`)
    return NaN
  }
  const n = Number(value)
  if (!Number.isFinite(n) || n < 0 || n > 1) errors.push(`${label} must be a number between 0 and 1.`)
  return n
}

function toIso(local: string): string | undefined {
  if (!local.trim()) return undefined
  const d = new Date(local)
  return Number.isNaN(d.getTime()) ? local : d.toISOString()
}

/** Lock form. Risk fields have no defaults on purpose: the user must declare them before seeing audit labels. */
function LockAuditForm({ projectId, graders }: { projectId: string; graders: GraderOut[] }) {
  const navigate = useNavigate()
  const { key, reset } = useIdempotencyKey()
  const [graderId, setGraderId] = useState('')
  const [plannedN, setPlannedN] = useState('')
  const [seed, setSeed] = useState('')
  const [windowStart, setWindowStart] = useState('')
  const [windowEnd, setWindowEnd] = useState('')
  const [taskTypes, setTaskTypes] = useState('')
  const [independenceDocumented, setIndependenceDocumented] = useState(false)
  const [independenceNote, setIndependenceNote] = useState('')
  const [permitPass, setPermitPass] = useState(false)
  const [permitFail, setPermitFail] = useState(false)
  const [maxErrorRate, setMaxErrorRate] = useState('')
  const [minCoverage, setMinCoverage] = useState('')
  const [confidence, setConfidence] = useState('')
  const [falsePassGate, setFalsePassGate] = useState('')
  const [unresolvedRule, setUnresolvedRule] = useState<'' | 'block' | 'count_as_error'>('')
  const [errors, setErrors] = useState<string[]>([])
  const [error, setError] = useState<unknown>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    const problems: string[] = []
    if (!graderId) problems.push('Choose exactly one grader to audit.')
    const n = Math.trunc(Number(plannedN))
    if (!plannedN.trim() || !Number.isFinite(n) || n < 1) problems.push('Planned sample size must be a positive integer.')
    if (!independenceDocumented) problems.push('Confirm that the independence assumption is documented.')
    if (!independenceNote.trim()) problems.push('Write the independence note.')
    const permitted: ('PASS' | 'FAIL')[] = []
    if (permitPass) permitted.push('PASS')
    if (permitFail) permitted.push('FAIL')
    if (permitted.length === 0) problems.push('Select at least one permitted automatic verdict.')
    const maxError = parseRatio(maxErrorRate, 'Maximum error rate', problems)
    const minCov = parseRatio(minCoverage, 'Minimum coverage', problems)
    const conf = parseRatio(confidence, 'Confidence level', problems)
    if (Number.isFinite(conf) && (conf <= 0 || conf >= 1)) problems.push('Confidence level must be strictly between 0 and 1.')
    let fpGate: number | undefined
    if (falsePassGate.trim()) {
      fpGate = parseRatio(falsePassGate, 'False-pass gate', problems)
    }
    if (!unresolvedRule) problems.push('Choose how unresolved automatic decisions are handled.')
    setErrors(problems)
    if (problems.length > 0) return

    const body: AuditCreate = {
      grader_id: graderId,
      planned_n: n,
      seed: seed.trim() ? Math.trunc(Number(seed)) : undefined,
      population: {
        source_type: 'PRODUCTION',
        partition: 'AUDIT_RESERVE',
        time_window: windowStart.trim() || windowEnd.trim() ? { start: toIso(windowStart), end: toIso(windowEnd) } : null,
        task_types: taskTypes.trim()
          ? taskTypes
              .split(',')
              .map((s) => s.trim())
              .filter(Boolean)
          : null,
      },
      sampling_plan: {
        unit: 'group',
        method: 'uniform_random',
        independence_assumption_documented: independenceDocumented,
        independence_note: independenceNote.trim(),
      },
      risk_targets: {
        permitted_verdicts: permitted,
        max_error_rate: maxError,
        min_coverage: minCov,
        confidence: conf,
        gate_false_pass_rate: fpGate ?? null,
        joint_allocation: 'bonferroni',
        unresolved_automatic_rule: unresolvedRule as 'block' | 'count_as_error',
      },
      idempotency_key: key(),
    }
    setBusy(true)
    setError(null)
    try {
      const audit = await api.createAudit(projectId, body)
      reset()
      navigate(`/audits/${audit.id}`)
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={submit} className="stack" noValidate>
      <p className="muted small">
        Locking freezes the policy epoch, grader manifest, rendering/parsing behaviour, scope, sample ids, planned size and risk targets
        before any audit label is seen. Audit one grader; do not audit several and pick the winner on the same test.
      </p>
      <fieldset>
        <legend>Grader and sample</legend>
        <div className="form-grid">
          <label className="field">
            <span>Grader (one, frozen)</span>
            <select value={graderId} onChange={(e) => setGraderId(e.target.value)} required>
              <option value="">Select a grader…</option>
              {graders.map((g) => (
                <option key={g.id} value={g.id}>
                  {g.label} ({g.origin}){g.is_active_shadow ? ' · active shadow' : ''}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Planned sample size (groups)</span>
            <input type="number" min={1} value={plannedN} onChange={(e) => setPlannedN(e.target.value)} required />
          </label>
          <label className="field">
            <span>Sampling seed (optional)</span>
            <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="random" />
          </label>
        </div>
      </fieldset>
      <fieldset>
        <legend>Population (declared scope)</legend>
        <div className="form-grid">
          <div className="field">
            <span>Source type</span>
            <span>PRODUCTION (synthetic cases are excluded)</span>
          </div>
          <div className="field">
            <span>Partition</span>
            <span>AUDIT_RESERVE (untouched groups only)</span>
          </div>
          <label className="field">
            <span>Time window start (optional)</span>
            <input type="datetime-local" value={windowStart} onChange={(e) => setWindowStart(e.target.value)} />
          </label>
          <label className="field">
            <span>Time window end (optional)</span>
            <input type="datetime-local" value={windowEnd} onChange={(e) => setWindowEnd(e.target.value)} />
          </label>
          <label className="field">
            <span>Task types (optional, comma separated)</span>
            <input value={taskTypes} onChange={(e) => setTaskTypes(e.target.value)} placeholder="e.g. cancellation, refund" />
          </label>
        </div>
      </fieldset>
      <fieldset>
        <legend>Sampling plan</legend>
        <p className="small">
          Unit: one designated target response per independent production group · Method: uniform random over untouched eligible
          groups. Results are group-weighted for the declared window, not output-weighted accuracy.
        </p>
        <label className="row gap">
          <input type="checkbox" checked={independenceDocumented} onChange={(e) => setIndependenceDocumented(e.target.checked)} />
          <span>I have documented why groups can be treated as independent trials (unique ids do not prove independence).</span>
        </label>
        <label className="field">
          <span>Independence note</span>
          <textarea
            value={independenceNote}
            onChange={(e) => setIndependenceNote(e.target.value)}
            placeholder="How groups were formed, why they do not share users/sessions, and what could violate independence."
          />
        </label>
      </fieldset>
      <fieldset>
        <legend>Risk targets (predeclared, no defaults)</legend>
        <div className="form-grid">
          <div className="field">
            <span>Permitted automatic verdicts</span>
            <span className="row gap">
              <label className="row gap">
                <input type="checkbox" checked={permitPass} onChange={(e) => setPermitPass(e.target.checked)} /> PASS
              </label>
              <label className="row gap">
                <input type="checkbox" checked={permitFail} onChange={(e) => setPermitFail(e.target.checked)} /> FAIL
              </label>
            </span>
          </div>
          <label className="field">
            <span>Maximum automatic error rate (0–1)</span>
            <input type="number" step="any" min={0} max={1} value={maxErrorRate} onChange={(e) => setMaxErrorRate(e.target.value)} required />
          </label>
          <label className="field">
            <span>Minimum automatic coverage (0–1)</span>
            <input type="number" step="any" min={0} max={1} value={minCoverage} onChange={(e) => setMinCoverage(e.target.value)} required />
          </label>
          <label className="field">
            <span>Confidence level (0–1, e.g. 0.95)</span>
            <input type="number" step="any" min={0} max={1} value={confidence} onChange={(e) => setConfidence(e.target.value)} required />
          </label>
          <label className="field">
            <span>False-pass rate gate (optional, 0–1)</span>
            <input type="number" step="any" min={0} max={1} value={falsePassGate} onChange={(e) => setFalsePassGate(e.target.value)} placeholder="none" />
          </label>
          <div className="field">
            <span>Joint confidence allocation</span>
            <span>Bonferroni across all gated bounds</span>
          </div>
          <label className="field">
            <span>Unresolved automatic decisions</span>
            <select value={unresolvedRule} onChange={(e) => setUnresolvedRule(e.target.value as '' | 'block' | 'count_as_error')} required>
              <option value="">Select a rule…</option>
              <option value="block">block enablement</option>
              <option value="count_as_error">count as errors</option>
            </select>
          </label>
        </div>
      </fieldset>
      {errors.length > 0 ? (
        <div className="error-box" role="alert">
          <ul style={{ margin: 0, paddingLeft: '1.2rem' }}>
            {errors.map((e) => (
              <li key={e}>{e}</li>
            ))}
          </ul>
        </div>
      ) : null}
      <ErrorBox error={error} />
      <div className="row gap">
        <button type="submit" className="btn btn-primary" disabled={busy}>
          {busy ? 'Locking…' : 'Lock audit'}
        </button>
      </div>
    </form>
  )
}

export function AuditsPage() {
  const { projectId = '' } = useParams()
  const audits = useAsync(() => api.listAudits(projectId), [projectId])
  const graders = useAsync(() => api.listGraders(projectId), [projectId])
  const unavailable = isNotAvailable(audits.error)

  return (
    <div>
      <ProjectNav projectId={projectId} />
      <div className="card-header">
        <h1>Audits</h1>
        <AuditEvidenceBadge />
      </div>
      <p className="muted small">
        An independent audit is the only source of automation evidence. It is locked before any label is seen, applies to its declared
        population and window only, and is marked SPENT once its results influence a revision.
      </p>
      {unavailable ? (
        <NotAvailable feature="Audits" />
      ) : (
        <div className="grid-2">
          <section className="card">
            <h2>Audits</h2>
            {audits.loading ? (
              <Loading />
            ) : audits.error ? (
              <ErrorBox error={audits.error} onRetry={audits.reload} />
            ) : !audits.data || audits.data.length === 0 ? (
              <EmptyState title="No audits yet">
                Lock one on the right after selecting a release candidate. Until then automation stays disabled.
              </EmptyState>
            ) : (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Audit</th>
                      <th>State</th>
                      <th>Grader</th>
                      <th className="num">Planned</th>
                      <th className="num">Judged</th>
                      <th>Gate</th>
                      <th>Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {audits.data.map((a) => (
                      <tr key={a.id}>
                        <td>
                          <Link to={`/audits/${a.id}`} className="mono">
                            {shortId(a.id)}
                          </Link>
                        </td>
                        <td>
                          <StateBadge state={a.state} />
                        </td>
                        <td>
                          <Link to={`/graders/${a.grader_id}`} className="mono">
                            {shortId(a.grader_id)}
                          </Link>
                        </td>
                        <td className="num">{a.planned_n}</td>
                        <td className="num">{a.judged_count}</td>
                        <td>{a.report?.gate ? (a.report.gate.passed ? 'passed' : 'not passed') : '—'}</td>
                        <td>{formatDate(a.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
          <section className="card">
            <h2>Lock a new audit</h2>
            {graders.error ? <ErrorBox error={graders.error} onRetry={graders.reload} /> : null}
            {graders.data && graders.data.length === 0 ? (
              <EmptyState title="No graders to audit">Run an optimization first; the seed grader appears once a run exists.</EmptyState>
            ) : (
              <LockAuditForm projectId={projectId} graders={graders.data ?? []} />
            )}
          </section>
        </div>
      )}
    </div>
  )
}
