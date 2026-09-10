import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import { EmptyState, ErrorBox, Loading } from '../components/Status'
import { formatDate } from '../format'
import { useAsync } from '../hooks/useAsync'
import { useIdempotencyKey } from '../hooks/useIdempotencyKey'

export function ProjectsPage() {
  const projects = useAsync(() => api.listProjects(), [])
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const { key, reset } = useIdempotencyKey()

  const create = async (event: FormEvent) => {
    event.preventDefault()
    if (!name.trim()) {
      setError(new Error('A project name is required.'))
      return
    }
    setBusy(true)
    setError(null)
    try {
      const project = await api.createProject({ name: name.trim(), description: description.trim(), idempotency_key: key() })
      reset()
      setName('')
      setDescription('')
      navigate(`/projects/${project.id}`)
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <h1>Projects</h1>
      <div className="grid-2">
        <section className="card">
          <h2>Create a project</h2>
          <form onSubmit={create} className="stack">
            <label className="field">
              <span>Name</span>
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Support assistant, cancellations" required />
            </label>
            <label className="field">
              <span>Description of the production application (optional)</span>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="What the application does and who its users are. Display-only context."
              />
            </label>
            <p className="muted small">
              There is deliberately no rubric field. Grading rules are evolved from your PASS/FAIL labels; the evolved
              instruction text is always shown in full with its diff, never hidden behind a summary.
            </p>
            <div className="row gap">
              <button type="submit" className="btn btn-primary" disabled={busy}>
                {busy ? 'Creating…' : 'Create project'}
              </button>
            </div>
            <ErrorBox error={error} />
          </form>
        </section>

        <section className="card">
          <h2>Existing projects</h2>
          {projects.loading ? (
            <Loading />
          ) : projects.error ? (
            <ErrorBox error={projects.error} onRetry={projects.reload} />
          ) : !projects.data || projects.data.length === 0 ? (
            <EmptyState title="No projects yet">Create one on the left, then import JSONL traces from its dashboard.</EmptyState>
          ) : (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Description</th>
                    <th>Policy epoch</th>
                    <th>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {projects.data.map((p) => (
                    <tr key={p.id}>
                      <td>
                        <Link to={`/projects/${p.id}`}>{p.name}</Link>
                      </td>
                      <td className="muted">{p.description || '—'}</td>
                      <td className="num">{p.policy_epoch}</td>
                      <td>{formatDate(p.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
