import { Link, useParams } from 'react-router-dom'
import { api } from '../api'
import { Badge, DevAgreementBadge, ProvisionalBadge, StateBadge } from '../components/Badge'
import { DiffView } from '../components/DiffView'
import { JsonBlock } from '../components/JsonBlock'
import { MetricValue } from '../components/MetricValue'
import { ProjectNav } from '../components/ProjectNav'
import { EmptyState, LoadFailure, Loading } from '../components/Status'
import { formatDate, formatNumber, shortId } from '../format'
import { useAsync } from '../hooks/useAsync'

export function GraderPage() {
  const { graderId = '' } = useParams()
  const grader = useAsync(() => api.getGrader(graderId), [graderId])

  if (grader.loading && !grader.data) return <Loading label="Loading grader…" />
  if (grader.error) return <LoadFailure error={grader.error} feature="Grader" onRetry={grader.reload} />
  const g = grader.data
  if (!g) return null

  return (
    <div>
      <ProjectNav projectId={g.project_id} />
      <div className="card-header">
        <h1>{g.label}</h1>
        <StateBadge state={g.origin} />
        {g.is_active_shadow ? (
          <>
            <Badge kind="info">ACTIVE SHADOW</Badge> <ProvisionalBadge />
          </>
        ) : null}
      </div>
      <div className="grid-2">
        <section className="card">
          <h2>Identity</h2>
          <ul className="kv-list">
            <li>
              <span className="kv-key">Grader id</span>
              <span className="kv-val mono">{g.id}</span>
            </li>
            <li>
              <span className="kv-key">Manifest hash</span>
              <span className="kv-val mono">{g.manifest_hash}</span>
            </li>
            <li>
              <span className="kv-key">Pipeline hash</span>
              <span className="kv-val mono">{g.pipeline_hash}</span>
            </li>
            <li>
              <span className="kv-key">Policy epoch</span>
              <span className="kv-val">{g.policy_epoch}</span>
            </li>
            <li>
              <span className="kv-key">Renderer / parser</span>
              <span className="kv-val">
                {g.renderer_version} / {g.parser_version}
              </span>
            </li>
            <li>
              <span className="kv-key">Parents</span>
              <span className="kv-val">
                {g.parent_ids.length === 0
                  ? 'none (seed)'
                  : g.parent_ids.map((p) => (
                      <Link key={p} to={`/graders/${p}`} className="mono" style={{ marginRight: 8 }}>
                        {shortId(p)}
                      </Link>
                    ))}
              </span>
            </li>
            <li>
              <span className="kv-key">Optimization run</span>
              <span className="kv-val">
                {g.optimization_run_id ? (
                  <Link to={`/optimization-runs/${g.optimization_run_id}`} className="mono">
                    {shortId(g.optimization_run_id)}
                  </Link>
                ) : (
                  '—'
                )}
                {g.candidate_index != null ? ` · candidate ${g.candidate_index}` : ''}
              </span>
            </li>
            <li>
              <span className="kv-key">Created</span>
              <span className="kv-val">{formatDate(g.created_at)}</span>
            </li>
          </ul>
          <h3>Model configuration</h3>
          <JsonBlock value={g.model_config} maxHeight={200} />
          <h3>Manifest</h3>
          <JsonBlock value={g.manifest} maxHeight={360} />
        </section>
        <section className="card">
          <h2>Instruction text</h2>
          <pre className="code-block instruction-text">{g.instruction_text}</pre>
          {g.immutable_policy_context ? (
            <>
              <h3>Immutable policy context</h3>
              <pre className="code-block">{g.immutable_policy_context}</pre>
            </>
          ) : null}
          <h3>Diff from parent</h3>
          <DiffView diff={g.diff_from_parent} emptyLabel={g.parent_ids.length === 0 ? 'No parent: this is a seed grader.' : 'Identical to its parent.'} />
        </section>
      </div>
      <section className="card">
        <div className="card-header">
          <h2>DEV evaluations</h2>
          <DevAgreementBadge long />
        </div>
        {g.evaluations.length === 0 ? (
          <EmptyState title="No DEV evaluations">This grader has not been scored on a frozen DEV snapshot.</EmptyState>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>DEV snapshot</th>
                  <th>Source</th>
                  <th>Agreement</th>
                  <th className="num">False passes</th>
                  <th>Coverage</th>
                  <th>Failure recall</th>
                  <th>Complete</th>
                </tr>
              </thead>
              <tbody>
                {g.evaluations.map((ev) => {
                  const agg = ev.aggregate_metrics
                  return (
                    <tr key={ev.id}>
                      <td className="mono small">{shortId(ev.dev_snapshot_id)}</td>
                      <td className="small">{ev.source}</td>
                      <td>
                        <DevAgreementBadge /> <MetricValue value={agg?.agreement} />
                      </td>
                      <td className="num">{formatNumber(agg?.false_passes)}</td>
                      <td>
                        <MetricValue value={agg?.coverage} />
                      </td>
                      <td>
                        <MetricValue value={agg?.failure_recall} />
                      </td>
                      <td>{ev.complete ? 'yes' : 'no'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
