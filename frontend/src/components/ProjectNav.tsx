import { NavLink } from 'react-router-dom'

export function ProjectNav({ projectId, name }: { projectId: string; name?: string }) {
  const cls = ({ isActive }: { isActive: boolean }) => (isActive ? 'subnav-link active' : 'subnav-link')
  return (
    <nav className="subnav" aria-label="Project sections">
      {name ? <span className="subnav-title">{name}</span> : null}
      <NavLink to={`/projects/${projectId}`} end className={cls}>
        Dashboard
      </NavLink>
      <NavLink to={`/projects/${projectId}/review?purpose=TRAIN`} className={cls}>
        Review TRAIN
      </NavLink>
      <NavLink to={`/projects/${projectId}/review?purpose=DEV`} className={cls}>
        Review DEV
      </NavLink>
      <NavLink to={`/projects/${projectId}/optimization`} className={cls}>
        Optimization
      </NavLink>
      <NavLink to={`/projects/${projectId}/traces`} className={cls}>
        Traces
      </NavLink>
      <NavLink to={`/projects/${projectId}/audits`} className={cls}>
        Audits
      </NavLink>
      <NavLink to={`/projects/${projectId}/exports`} className={cls}>
        Exports
      </NavLink>
    </nav>
  )
}
