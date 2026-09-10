import { Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { AuditDetailPage } from './pages/AuditDetailPage'
import { AuditsPage } from './pages/AuditsPage'
import { ExportsPage } from './pages/ExportsPage'
import { GraderPage } from './pages/GraderPage'
import { NotFoundPage } from './pages/NotFoundPage'
import { OptimizationPage } from './pages/OptimizationPage'
import { OptimizationRunPage } from './pages/OptimizationRunPage'
import { ProjectDashboardPage } from './pages/ProjectDashboardPage'
import { ProjectsPage } from './pages/ProjectsPage'
import { ReviewPage } from './pages/ReviewPage'
import { TracesPage } from './pages/TracesPage'

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<ProjectsPage />} />
        <Route path="/projects/:projectId" element={<ProjectDashboardPage />} />
        <Route path="/projects/:projectId/review" element={<ReviewPage />} />
        <Route path="/projects/:projectId/optimization" element={<OptimizationPage />} />
        <Route path="/projects/:projectId/traces" element={<TracesPage />} />
        <Route path="/projects/:projectId/audits" element={<AuditsPage />} />
        <Route path="/projects/:projectId/exports" element={<ExportsPage />} />
        <Route path="/optimization-runs/:runId" element={<OptimizationRunPage />} />
        <Route path="/audits/:auditId" element={<AuditDetailPage />} />
        <Route path="/audits/:auditId/review" element={<ReviewPage />} />
        <Route path="/graders/:graderId" element={<GraderPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  )
}
