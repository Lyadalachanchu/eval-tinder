import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, isNotAvailable } from '../api'
import { AuditEvidenceBadge, Badge, StateBadge } from '../components/Badge'
import { JobProgress } from '../components/JobProgress'
import { JsonBlock } from '../components/JsonBlock'
import { MetricValue, MetricWithCounts } from '../components/MetricValue'
import { ProjectNav } from '../components/ProjectNav'
import { ReasonForm } from '../components/ReasonForm'
import { EmptyState, ErrorBox, LoadFailure, Loading, Notice } from '../components/Status'
import { formatDate, shortId } from '../format'
import { useAsync } from '../hooks/useAsync'
import type { AuditReport, AutomationPolicyOut } from '../types'

const HUMAN_ROWS = ['PASS', 'FAIL'] as const
const MACHINE_COLS = ['PASS', 'FAIL', 'REVIEW'] as const

function ConfusionTable({ table }: { table: AuditReport['table'] }) {
  if (!table) return <p className="muted">No table.</p>
  return (
    <table className="table" aria-label="Human versus machine verdicts">
      <thead>
        <tr>
          <th>Human \ Machine</th>
          {MACHINE_COLS.map((c) => (
            <th key={c} className="num">
              {c}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {HUMAN_ROWS.map((h) => (
          <tr key={h}>
            <td>{h}</td>
            {MACHINE_COLS.map((m) => (
              <td key={m} className="num">
                {table[h]?.[m] ?? 0}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function UnresolvedList({ title, items }: { title: string; items: unknown[] | undefined }) {
  return (
    <div>
      <h4>
        {title} ({items?.length ?? 0})
      </h4>
      {items && items.length > 0 ? <JsonBlock value={items} maxHeight={220} /> : <p className="muted small">none</p>}
    </div>
  )
}

function ReportSection({ report }: { report: AuditReport | null }) {
  if (!report) {
    return (
      <EmptyState title="No report yet">
        The report is computed from audit judgments. Review the locked cases, then recompute.
      </EmptyState>
    )
  }
  const metrics = Object.entries(report.metrics ?? {})
  const intervals = report.intervals
  const gate = report.gate
  return (
    <div className="stack">
      <p className="row gap">
        <Badge kind={report.complete ? 'good' : 'warn'}>{report.complete ? 'COMPLETE' : 'INCOMPLETE'}</Badge>
        <span className="muted small">
          {report.complete
            ? 'Every planned case has a judgment or a documented unresolved outcome.'
            : 'Incomplete until every planned case has a judgment or a documented unresolved outcome.'}
        </span>
      </p>
      {report.counts ? (
        <ul className="kv-list small">
          {Object.entries(report.counts).map(([k, v]) => (
            <li key={k}>
              <span className="kv-key">{k}</span>
              <span className="kv-val">{typeof v === 'object' ? JSON.stringify(v) : String(v)}</span>
            </li>
          ))}
        </ul>
      ) : null}

      <h3>Human versus machine (human-determinate cases)</h3>
      <ConfusionTable table={report.table} />
      <p className="muted small">Operational failures are reported separately and count as REVIEW for coverage.</p>

      <h3>Metrics</h3>
      {metrics.length === 0 ? (
        <p className="muted">No metrics.</p>
      ) : (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Definition</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map(([name, entry]) => (
                <tr key={name}>
                  <td className="mono small">{name}</td>
                  <td>
                    <MetricWithCounts entry={entry} />
                  </td>
                  <td className="small muted">{entry?.definition ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="muted small">A zero denominator is shown as NOT_ESTIMABLE, never as 0% error or 100% reliability.</p>

      {report.baselines ? (
        <details>
          <summary>Baselines (always PASS / always FAIL on the same labels)</summary>
          <JsonBlock value={report.baselines} maxHeight={260} />
        </details>
      ) : null}

      <h3>Upper bounds</h3>
      {intervals ? (
        intervals.supported ? (
          <ul className="kv-list small">
            <li>
              <span className="kv-key">Design</span>
              <span className="kv-val">supported: one-sided exact binomial upper bounds</span>
            </li>
            <li>
              <span className="kv-key">Joint confidence</span>
              <span className="kv-val">
                {intervals.confidence ?? '—'} (per bound {intervals.per_bound_confidence ?? '—'}, Bonferroni)
              </span>
            </li>
            <li>
              <span className="kv-key">Automatic error rate upper bound</span>
              <span className="kv-val">
                <MetricValue value={intervals.automatic_error_rate_upper} digits={2} />
              </span>
            </li>
            <li>
              <span className="kv-key">False-pass rate upper bound</span>
              <span className="kv-val">
                <MetricValue value={intervals.false_pass_rate_upper} digits={2} />
              </span>
            </li>
          </ul>
        ) : (
          <Notice kind="warn">
            Interval estimates are not supported for this audit: {intervals.reason ?? 'the sampling design does not support them.'}{' '}
            Descriptive sample metrics above still apply; statistical enablement is disabled.
          </Notice>
        )
      ) : (
        <p className="muted">No interval information.</p>
      )}

      <h3>Gate</h3>
      {gate ? (
        <div>
          <p>
            <Badge kind={gate.passed ? 'good' : 'bad'}>{gate.passed ? 'GATE PASSED' : 'GATE NOT PASSED'}</Badge>
          </p>
          <ul className="check-list">
            {(gate.checks ?? []).map((c, i) => (
              <li key={`${c.name}-${i}`}>
                <span className={c.passed ? 'check-pass' : 'check-fail'}>{c.passed ? 'PASS' : 'FAIL'}</span> {c.name}
                {c.detail ? <span className="muted small"> — {c.detail}</span> : null}
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <p className="muted">No gate evaluated.</p>
      )}

      <div className="grid">
        <UnresolvedList title="Human unresolved (CANNOT_JUDGE)" items={report.human_unresolved} />
        <UnresolvedList title="Unresolved automatic decisions" items={report.unresolved_automatic_decisions} />
        <UnresolvedList title="Operational failures" items={report.operational_failures} />
      </div>

      {report.scope ? (
        <details>
          <summary>Scope</summary>
          {typeof report.scope === 'string' ? <p>{report.scope}</p> : <JsonBlock value={report.scope} maxHeight={220} />}
        </details>
      ) : null}
      {report.notes && report.notes.length > 0 ? (
        <div>
          <h4>Notes</h4>
          <ul>
            {report.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  )
}

export function AuditDetailPage() {
  const { auditId = '' } = useParams()
  const audit = useAsync(() => api.getAudit(auditId), [auditId])
  const projectId = audit.data?.project_id ?? ''
  const policy = useAsync(async () => (projectId ? api.getAutomationPolicy(projectId) : null), [projectId])
  const [actionError, setActionError] = useState<unknown>(null)
  const [policyResult, setPolicyResult] = useState<AutomationPolicyOut | null>(null)
  const [busy, setBusy] = useState(false)

  if (audit.loading && !audit.data) return <Loading label="Loading audit…" />
  if (audit.error) return <LoadFailure error={audit.error} feature="Audits" onRetry={audit.reload} />
  const a = audit.data
  if (!a) return null

  const gatePassed = a.report?.gate?.passed === true
  const currentPolicy = policyResult ?? (isNotAvailable(policy.error) ? null : policy.data) ?? null
  const remaining = Math.max(0, a.planned_n - a.judged_count - a.unresolved_count)
  const reviewable = a.state === 'LOCKED' || a.state === 'IN_REVIEW'

  const recompute = async () => {
    setBusy(true)
    setActionError(null)
    try {
      audit.setData(await api.recomputeAudit(a.id))
    } catch (e) {
      setActionError(e)
    } finally {
      setBusy(false)
    }
  }

  const setPolicy = async (enable: boolean, reason: string) => {
    if (!projectId) throw new Error('This audit does not report its project id; cannot change the automation policy.')
    const result = await api.setAutomationPolicy(projectId, { audit_id: a.id, enable, reason })
    setPolicyResult(result)
    audit.reload()
  }

  return (
    <div>
      {projectId ? <ProjectNav projectId={projectId} /> : null}
      <div className="card-header">
        <h1>Audit {shortId(a.id)}</h1>
        <StateBadge state={a.state} />
        <AuditEvidenceBadge />
      </div>
      <ErrorBox error={actionError} />

      <div className="grid-2">
        <section className="card">
          <h2>Lock</h2>
          <ul className="kv-list">
            <li>
              <span className="kv-key">Grader</span>
              <span className="kv-val">
                <Link to={`/graders/${a.grader_id}`} className="mono">
                  {shortId(a.grader_id)}
                </Link>
              </span>
            </li>
            <li>
              <span className="kv-key">Pipeline hash</span>
              <span className="kv-val mono">{a.pipeline_hash}</span>
            </li>
            <li>
              <span className="kv-key">Policy epoch</span>
              <span className="kv-val">{a.policy_epoch}</span>
            </li>
            <li>
              <span className="kv-key">Created</span>
              <span className="kv-val">{formatDate(a.created_at)}</span>
            </li>
            <li>
              <span className="kv-key">Completed</span>
              <span className="kv-val">{formatDate(a.completed_at)}</span>
            </li>
            <li>
              <span className="kv-key">Report version</span>
              <span className="kv-val">{a.report_version ?? '—'}</span>
            </li>
          </ul>
          <div className="stat-row">
            {[
              ['Planned', a.planned_n],
              ['Locked', a.locked_count],
              ['Judged', a.judged_count],
              ['Unresolved', a.unresolved_count],
            ].map(([label, value]) => (
              <div className="stat" key={String(label)}>
                <div className="stat-value">{value}</div>
                <div className="stat-label">{label}</div>
              </div>
            ))}
          </div>
          {reviewable ? (
            <p>
              <Link to={`/audits/${a.id}/review`} className="btn btn-primary">
                Review next audit case (blind)
              </Link>{' '}
              <span className="muted small">{remaining} case{remaining === 1 ? '' : 's'} still without an outcome</span>
            </p>
          ) : (
            <p className="muted small">This audit is {a.state}; no further review.</p>
          )}
          {a.grading_job_id ? <JobProgress jobId={a.grading_job_id} title="Audit grading job" onFinished={() => audit.reload()} compact /> : null}
          <details>
            <summary>Population, sampling plan and risk targets</summary>
            <JsonBlock value={{ population: a.population_definition, sampling_plan: a.sampling_plan, risk_targets: a.risk_targets }} maxHeight={360} />
          </details>
          {a.correction_history && a.correction_history.length > 0 ? (
            <details>
              <summary>Correction history ({a.correction_history.length})</summary>
              <JsonBlock value={a.correction_history} maxHeight={220} />
            </details>
          ) : null}
        </section>

        <section className="card">
          <h2>Actions</h2>
          <div className="stack">
            <div className="row gap">
              <button type="button" className="btn" onClick={recompute} disabled={busy}>
                Recompute report
              </button>
              <span className="muted small">Re-derives the report from the recorded judgments; the lock never changes.</span>
            </div>
            <div>
              <h3>Enable automation</h3>
              <p className="row gap small">
                Current policy: <StateBadge state={currentPolicy?.state ?? 'DISABLED'} />
                {currentPolicy?.audit_id ? <span className="muted">audit {shortId(currentPolicy.audit_id)}</span> : null}
              </p>
              <ReasonForm
                actionLabel="Enable automation"
                disabled={!gatePassed}
                disabledReason="Enablement requires this audit's predeclared gate to pass, every needed denominator to be estimable, a supported sampling design and handled unresolved cases."
                note="Applies only to the exact frozen pipeline hash above, for the declared population and window."
                onSubmit={(reason) => setPolicy(true, reason)}
              />
              {currentPolicy?.state === 'ENABLED' ? (
                <ReasonForm actionLabel="Disable automation" danger onSubmit={(reason) => setPolicy(false, reason)} />
              ) : null}
              {policyResult ? (
                <Notice kind={policyResult.state === 'ENABLED' ? 'good' : 'info'}>
                  Automation policy is now {policyResult.state}. {policyResult.reason ?? ''} {policyResult.note ?? ''}
                  {policyResult.gate_result ? <JsonBlock value={policyResult.gate_result} maxHeight={160} /> : null}
                </Notice>
              ) : null}
            </div>
            <div>
              <h3>Mark as spent</h3>
              <ReasonForm
                actionLabel="Spend audit"
                danger
                disabled={a.state === 'SPENT'}
                disabledReason="This audit is already spent."
                note="Once results influence a revision, the audit can no longer serve as an independent test. It remains a historical report."
                onSubmit={async (reason) => {
                  audit.setData(await api.spendAudit(a.id, reason))
                }}
              />
            </div>
          </div>
        </section>
      </div>

      <section className="card">
        <div className="card-header">
          <h2>Report</h2>
          <AuditEvidenceBadge />
          <span className="muted small">Evidence for the declared population and window only. Not a DEV agreement number.</span>
        </div>
        <ReportSection report={a.report} />
      </section>
    </div>
  )
}

