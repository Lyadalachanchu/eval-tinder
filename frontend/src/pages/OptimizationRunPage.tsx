import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api'
import { Badge, DevAgreementBadge, ProvisionalBadge, StateBadge } from '../components/Badge'
import { DiffView } from '../components/DiffView'
import { JobProgress } from '../components/JobProgress'
import { JsonBlock } from '../components/JsonBlock'
import { MetricValue, MetricWithCounts } from '../components/MetricValue'
import { ProjectNav } from '../components/ProjectNav'
import { ReasonForm } from '../components/ReasonForm'
import { EmptyState, ErrorBox, LoadFailure, Loading, Notice } from '../components/Status'
import { formatDate, formatNumber, shortId } from '../format'
import { useAsync } from '../hooks/useAsync'
import type { BaselineEntry, CandidateOut, OptimizationRunOut, ShadowSelectResponse } from '../types'

const ACTIVE_STATES = new Set(['QUEUED', 'RUNNING'])

function BaselineRows({ baselines }: { baselines: { always_pass?: BaselineEntry; always_fail?: BaselineEntry } | undefined }) {
  if (!baselines) return null
  const rows: [string, BaselineEntry | undefined][] = [
    ['always PASS', baselines.always_pass],
    ['always FAIL', baselines.always_fail],
  ]
  return (
    <>
      {rows.map(([name, entry]) =>
        entry ? (
          <tr key={name} className="muted">
            <td>—</td>
            <td title={entry.definition}>
              baseline: {name} <Badge>BASELINE</Badge>
            </td>
            <td>
              <MetricWithCounts entry={entry.agreement} />
            </td>
            <td className="num">{entry.false_pass_rate_among_accepted ? entry.false_pass_rate_among_accepted.numerator : '—'}</td>
            <td>—</td>
            <td>
              <MetricWithCounts entry={entry.failure_recall} />
            </td>
            <td>—</td>
          </tr>
        ) : null,
      )}
    </>
  )
}

function CandidateDetail({ run, candidate, onShadowSelected }: { run: OptimizationRunOut; candidate: CandidateOut; onShadowSelected: (r: ShadowSelectResponse) => void }) {
  const agg = candidate.evaluation?.aggregate_metrics
  return (
    <section className="card">
      <div className="card-header">
        <h2>
          Candidate {candidate.candidate_index ?? '—'} · {candidate.label}
        </h2>
        {candidate.is_seed ? <Badge kind="info">SEED</Badge> : null}
        {candidate.is_member ? <Badge kind="neutral">COMMITTEE MEMBER</Badge> : null}
        {run.result_summary.recommended_grader_id === candidate.grader_id ? <Badge kind="good">RECOMMENDED</Badge> : null}
        <Link to={`/graders/${candidate.grader_id}`} className="small">
          grader page
        </Link>
      </div>
      <p className="muted small mono">manifest {candidate.manifest_hash}</p>
      {agg ? (
        <div className="row gap small">
          <DevAgreementBadge />
          <span>
            agreement <MetricValue value={agg.agreement} />
          </span>
          <span>false passes {formatNumber(agg.false_passes)}</span>
          <span>
            coverage <MetricValue value={agg.coverage} />
          </span>
          <span>
            failure recall <MetricValue value={agg.failure_recall} />
          </span>
          <span className="muted">
            {agg.evaluated_cases ?? '—'}/{agg.dev_size ?? run.dev_size} DEV cases{agg.complete === false ? ' (incomplete)' : ''}
          </span>
        </div>
      ) : (
        <p className="muted small">No DEV evaluation for this candidate.</p>
      )}
      <h3>Instruction text</h3>
      <pre className="code-block instruction-text">{candidate.instruction_text}</pre>
      <h3>Diff from seed</h3>
      <DiffView diff={candidate.diff_from_seed} emptyLabel="Identical to the seed instruction." />
      <h3>Use as shadow grader</h3>
      <ReasonForm
        actionLabel="Use as shadow grader"
        note={
          <>
            <ProvisionalBadge /> Shadow predictions are provisional and never enable automation. The reason is recorded in the
            project history.
          </>
        }
        onSubmit={async (reason) => {
          onShadowSelected(await api.selectShadowGrader(run.project_id, { grader_id: candidate.grader_id, reason }))
        }}
      />
    </section>
  )
}

export function OptimizationRunPage() {
  const { runId = '' } = useParams()
  const run = useAsync(() => api.getOptimizationRun(runId), [runId])
  const [selected, setSelected] = useState<string | null>(null)
  const [shadowResult, setShadowResult] = useState<ShadowSelectResponse | null>(null)
  const [cancelError, setCancelError] = useState<unknown>(null)
  const isActive = run.data ? ACTIVE_STATES.has(run.data.state) : false
  const reload = run.reload

  useEffect(() => {
    if (!isActive) return
    const t = setInterval(reload, 3000)
    return () => clearInterval(t)
  }, [isActive, reload])

  if (run.loading && !run.data) return <Loading label="Loading run…" />
  if (run.error) return <LoadFailure error={run.error} feature="Optimization run" onRetry={run.reload} />
  const r = run.data
  if (!r) return null

  const summary = r.result_summary ?? {}
  const comparison = summary.comparison ?? null
  const candidates = r.candidates ?? []
  const selectedCandidate = candidates.find((c) => c.grader_id === selected) ?? null
  const evalWithBaselines = candidates.find((c) => c.evaluation?.aggregate_metrics?.baselines)?.evaluation?.aggregate_metrics
  const insufficient =
    Boolean(comparison?.insufficient_class_coverage) || candidates.some((c) => c.evaluation?.aggregate_metrics?.insufficient_class_coverage)
  const estimate = r.budgets?.estimate

  const cancel = async () => {
    setCancelError(null)
    try {
      await api.cancelOptimizationRun(r.id)
      run.reload()
    } catch (e) {
      setCancelError(e)
    }
  }

  return (
    <div>
      <ProjectNav projectId={r.project_id} />
      <div className="card-header">
        <h1>Run {shortId(r.id)}</h1>
        <StateBadge state={r.state} />
        <DevAgreementBadge long />
        {isActive ? (
          <button type="button" className="btn btn-small" onClick={cancel}>
            Cancel run
          </button>
        ) : null}
      </div>
      <ErrorBox error={cancelError} />
      {r.error ? <ErrorBox error={new Error(r.error)} prefix="Run error" /> : null}
      {summary.partial ? <Notice kind="warn">Partial result: {summary.partial_reason ?? 'the run did not complete its budget.'}</Notice> : null}

      <div className="grid-2">
        <section className="card">
          <h2>Setup</h2>
          <ul className="kv-list">
            <li>
              <span className="kv-key">Label</span>
              <span className="kv-val">{String(r.config?.label ?? '') || '—'}</span>
            </li>
            <li>
              <span className="kv-key">Seed grader</span>
              <span className="kv-val">
                <Link to={`/graders/${r.seed_grader_id}`}>{shortId(r.seed_grader_id)}</Link> <span className="muted">({r.seed_choice})</span>
              </span>
            </li>
            <li>
              <span className="kv-key">TRAIN / DEV size</span>
              <span className="kv-val">
                {r.train_size} / {r.dev_size} (frozen snapshots {shortId(r.train_snapshot_id)} / {shortId(r.dev_snapshot_id)})
              </span>
            </li>
            <li>
              <span className="kv-key">Policy epoch · metric</span>
              <span className="kv-val">
                {r.policy_epoch} · {r.metric_version}
              </span>
            </li>
            <li>
              <span className="kv-key">Created / finished</span>
              <span className="kv-val">
                {formatDate(r.created_at)} / {formatDate(r.finished_at)}
              </span>
            </li>
          </ul>
          <details>
            <summary>Optimizer configuration</summary>
            <JsonBlock value={r.config} maxHeight={260} />
          </details>
          {r.job_id ? <JobProgress jobId={r.job_id} title="Optimization job" onFinished={() => run.reload()} compact /> : null}
        </section>

        <section className="card">
          <h2>Budget and usage</h2>
          {estimate ? (
            <>
              <h3>
                Estimate <Badge kind="warn">ESTIMATE, NOT A GUARANTEE</Badge>
              </h3>
              <ul className="kv-list">
                <li>
                  <span className="kv-key">Grading calls</span>
                  <span className="kv-val">{formatNumber(estimate.grading_calls)}</span>
                </li>
                <li>
                  <span className="kv-key">Reflection calls</span>
                  <span className="kv-val">{formatNumber(estimate.reflection_calls)}</span>
                </li>
                <li>
                  <span className="kv-key">Cost (USD)</span>
                  <span className="kv-val">{estimate.cost_usd == null ? 'not configured' : `$${estimate.cost_usd.toFixed(4)}`}</span>
                </li>
              </ul>
              <p className="muted small">
                {estimate.note} {estimate.cost_note}
              </p>
            </>
          ) : null}
          <h3>Budgets</h3>
          <JsonBlock value={Object.fromEntries(Object.entries(r.budgets ?? {}).filter(([k]) => k !== 'estimate'))} maxHeight={200} />
          <h3>Measured usage</h3>
          {Object.keys(r.usage ?? {}).length === 0 ? <p className="muted">No usage recorded yet.</p> : <JsonBlock value={r.usage} maxHeight={220} />}
        </section>
      </div>

      <section className="card">
        <div className="card-header">
          <h2>Result</h2>
          <DevAgreementBadge />
        </div>
        {summary.note ? <Notice kind="info">{summary.note}</Notice> : null}
        {insufficient ? (
          <Notice kind="warn">
            <strong>Insufficient class coverage:</strong> the DEV snapshot has only one human class. Failure-detection capability cannot
            be inferred from it.
          </Notice>
        ) : null}
        <ul className="kv-list">
          <li>
            <span className="kv-key">Seed agreement</span>
            <span className="kv-val">
              <MetricValue value={summary.seed_agreement} />
            </span>
          </li>
          <li>
            <span className="kv-key">Best agreement</span>
            <span className="kv-val">
              <MetricValue value={summary.best_agreement} />
            </span>
          </li>
          <li>
            <span className="kv-key">Improved</span>
            <span className="kv-val">{summary.improved ? 'yes' : 'no'}</span>
          </li>
          <li>
            <span className="kv-key">Recommended grader</span>
            <span className="kv-val">
              {summary.recommended_grader_id ? (
                <Link to={`/graders/${summary.recommended_grader_id}`}>{shortId(summary.recommended_grader_id)}</Link>
              ) : (
                'none (incumbent retained)'
              )}
            </span>
          </li>
        </ul>
        {comparison ? (
          <details open>
            <summary>Comparison rule</summary>
            <p className="small">
              {comparison.rule ? <span className="muted">{comparison.rule}. </span> : null}
              <strong>{comparison.recommend ? 'Recommend' : 'Do not recommend'}</strong>
              {comparison.reason ? `: ${comparison.reason}` : ''}
            </p>
            {comparison.incumbent && comparison.candidate ? (
              <table className="table">
                <thead>
                  <tr>
                    <th></th>
                    <th>Agreement</th>
                    <th className="num">False passes</th>
                    <th>Coverage</th>
                    <th>Failure recall</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ['Incumbent', comparison.incumbent],
                    ['Candidate', comparison.candidate],
                  ].map(([name, side]) =>
                    typeof side === 'object' ? (
                      <tr key={String(name)}>
                        <td>
                          {String(name)} <span className="muted small mono">{shortId(side.grader_id)}</span>
                        </td>
                        <td>
                          <MetricValue value={side.agreement} />
                        </td>
                        <td className="num">{formatNumber(side.false_passes)}</td>
                        <td>
                          <MetricValue value={side.coverage} />
                        </td>
                        <td>
                          <MetricValue value={side.failure_recall} />
                        </td>
                      </tr>
                    ) : null,
                  )}
                </tbody>
              </table>
            ) : null}
          </details>
        ) : null}
      </section>

      <section className="card">
        <div className="card-header">
          <h2>Candidates</h2>
          <DevAgreementBadge long />
        </div>
        {candidates.length === 0 ? (
          <EmptyState title="No candidates yet">
            {isActive ? 'The run is still in progress; candidates appear when it finishes.' : 'This run produced no candidates.'}
          </EmptyState>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Label</th>
                  <th>DEV agreement</th>
                  <th className="num">False passes</th>
                  <th>Coverage</th>
                  <th>Failure recall</th>
                  <th>Flags</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map((c) => {
                  const agg = c.evaluation?.aggregate_metrics
                  return (
                    <tr
                      key={c.grader_id}
                      className={`clickable${selected === c.grader_id ? ' selected' : ''}`}
                      onClick={() => setSelected(c.grader_id)}
                    >
                      <td>{c.candidate_index ?? '—'}</td>
                      <td>{c.label}</td>
                      <td>{agg ? <MetricValue value={agg.agreement} /> : <span className="muted">not evaluated</span>}</td>
                      <td className="num">{agg ? formatNumber(agg.false_passes) : '—'}</td>
                      <td>{agg ? <MetricValue value={agg.coverage} /> : '—'}</td>
                      <td>{agg ? <MetricValue value={agg.failure_recall} /> : '—'}</td>
                      <td className="row gap">
                        {c.is_seed ? <Badge kind="info">SEED</Badge> : null}
                        {c.is_member ? <Badge>MEMBER</Badge> : null}
                        {summary.recommended_grader_id === c.grader_id ? <Badge kind="good">RECOMMENDED</Badge> : null}
                        {agg?.complete === false ? <Badge kind="warn">INCOMPLETE</Badge> : null}
                      </td>
                    </tr>
                  )
                })}
                <BaselineRows baselines={evalWithBaselines?.baselines} />
              </tbody>
            </table>
          </div>
        )}
        <p className="muted small">Click a candidate to see its instruction text and the diff from the seed.</p>
      </section>

      {shadowResult ? (
        <Notice kind="good">
          Shadow grader set to {shortId(shadowResult.active_shadow_grader_id)} <ProvisionalBadge /> {shadowResult.note}
        </Notice>
      ) : null}
      {selectedCandidate ? (
        <CandidateDetail run={r} candidate={selectedCandidate} onShadowSelected={setShadowResult} />
      ) : null}
    </div>
  )
}
