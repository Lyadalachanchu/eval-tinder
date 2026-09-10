import { Link, useParams } from 'react-router-dom'
import { api } from '../api'
import { DevAgreementBadge, StateBadge } from '../components/Badge'
import { ProjectNav } from '../components/ProjectNav'
import { StartRunForm } from '../components/StartRunForm'
import { EmptyState, ErrorBox, Loading } from '../components/Status'
import { formatDate, shortId } from '../format'
import { useAsync } from '../hooks/useAsync'

export function OptimizationPage() {
  const { projectId = '' } = useParams()
  const runs = useAsync(() => api.listOptimizationRuns(projectId), [projectId])
  const graders = useAsync(() => api.listGraders(projectId), [projectId])

  return (
    <div>
      <ProjectNav projectId={projectId} />
      <div className="card-header">
        <h1>Optimization runs</h1>
        <DevAgreementBadge long />
      </div>
      <p className="muted small">
        Each run freezes a TRAIN and a DEV snapshot, evolves the grader instruction with GEPA, and compares candidates with the
        seed on the same frozen DEV snapshot. Agreement numbers here are development results, never production accuracy.
      </p>
      <div className="grid-2">
        <section className="card">
          <h2>Runs</h2>
          {runs.loading ? (
            <Loading />
          ) : runs.error ? (
            <ErrorBox error={runs.error} onRetry={runs.reload} />
          ) : !runs.data || runs.data.length === 0 ? (
            <EmptyState title="No optimization runs yet">
              Label the bootstrap TRAIN and DEV sets first, then start a budgeted run. Candidates and their prompt diffs appear on the
              run page.
            </EmptyState>
          ) : (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>State</th>
                    <th>Label</th>
                    <th>Seed choice</th>
                    <th className="num">TRAIN</th>
                    <th className="num">DEV</th>
                    <th>Improved</th>
                    <th>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.data.map((run) => (
                    <tr key={run.id}>
                      <td>
                        <Link to={`/optimization-runs/${run.id}`} className="mono">
                          {shortId(run.id)}
                        </Link>
                      </td>
                      <td>
                        <StateBadge state={run.state} />
                      </td>
                      <td>{String(run.config?.label ?? '') || '—'}</td>
                      <td className="small">{run.seed_choice}</td>
                      <td className="num">{run.train_size}</td>
                      <td className="num">{run.dev_size}</td>
                      <td>{run.result_summary?.improved === undefined ? '—' : run.result_summary.improved ? 'yes' : 'no'}</td>
                      <td>{formatDate(run.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
        <section className="card">
          <h2>Start a run</h2>
          {graders.error ? <ErrorBox error={graders.error} onRetry={graders.reload} /> : null}
          <StartRunForm projectId={projectId} graders={graders.data ?? []} />
        </section>
      </div>
    </div>
  )
}
