/**
 * Typed API client. Every call goes through `request`, which attaches the
 * optional bearer token, parses FastAPI `{detail}` errors into `ApiError`, and
 * flags 404/405/501 as "not available yet" so screens can degrade gracefully
 * for endpoints that are still being implemented.
 */
import { getApiBase, getToken } from './settings'
import type {
  AuditCreate,
  AuditOut,
  AutomationPolicyOut,
  ExportOut,
  GraderOut,
  Health,
  ImportOut,
  JobOut,
  JudgmentCreate,
  JudgmentOut,
  OptimizationRunCreate,
  OptimizationRunOut,
  PredictionPage,
  ProjectDashboard,
  ProjectOut,
  ReviewBatchCreate,
  ReviewCase,
  ReviewRequestOut,
  SelectionRoundOut,
  ShadowSelectResponse,
  TracePage,
} from './types'

export class ApiError extends Error {
  readonly status: number
  readonly detail: string
  readonly path: string

  constructor(status: number, detail: string, path: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
    this.path = path
  }

  /** True when the route itself appears to be missing (feature not implemented yet). */
  get notAvailable(): boolean {
    return this.status === 404 || this.status === 405 || this.status === 501
  }
}

export function isNotAvailable(error: unknown): boolean {
  return error instanceof ApiError && error.notAvailable
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.detail
  if (error instanceof Error) return error.message
  return String(error)
}

type Query = Record<string, string | number | boolean | null | undefined>

interface RequestOptions {
  json?: unknown
  form?: FormData
  query?: Query
  signal?: AbortSignal
}

function formatDetail(payload: unknown, status: number): string {
  if (payload && typeof payload === 'object' && 'detail' in payload) {
    const detail = (payload as { detail: unknown }).detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          if (item && typeof item === 'object') {
            const loc = Array.isArray((item as { loc?: unknown }).loc)
              ? ((item as { loc: unknown[] }).loc as unknown[]).filter((p) => p !== 'body').join('.')
              : ''
            const msg = (item as { msg?: unknown }).msg
            return loc ? `${loc}: ${String(msg ?? '')}` : String(msg ?? JSON.stringify(item))
          }
          return String(item)
        })
        .join('; ')
    }
    if (detail != null) return JSON.stringify(detail)
  }
  if (typeof payload === 'string' && payload.trim()) return payload.slice(0, 500)
  return `HTTP ${status}`
}

export function buildUrl(path: string, query?: Query): string {
  const base = getApiBase()
  let url = `${base}${path.startsWith('/') ? path : `/${path}`}`
  if (query) {
    const params = new URLSearchParams()
    for (const [key, value] of Object.entries(query)) {
      if (value === undefined || value === null || value === '') continue
      params.set(key, String(value))
    }
    const qs = params.toString()
    if (qs) url += `${url.includes('?') ? '&' : '?'}${qs}`
  }
  return url
}

export function authHeaders(): Record<string, string> {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export async function request<T>(method: string, path: string, options: RequestOptions = {}): Promise<T> {
  const url = buildUrl(path, options.query)
  const headers: Record<string, string> = { Accept: 'application/json', ...authHeaders() }
  let body: BodyInit | undefined
  if (options.form) {
    body = options.form
  } else if (options.json !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(options.json)
  }

  let response: Response
  try {
    // Resolved at call time so tests can stub globalThis.fetch.
    response = await globalThis.fetch(url, { method, headers, body, signal: options.signal })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError(0, `Network error reaching ${url}: ${errorMessage(error)}`, path)
  }

  const text = await response.text()
  let payload: unknown = undefined
  if (text) {
    try {
      payload = JSON.parse(text)
    } catch {
      payload = text
    }
  }

  if (!response.ok) {
    throw new ApiError(response.status, formatDetail(payload, response.status), path)
  }
  return payload as T
}

/** Download a binary endpoint with auth headers (plain links cannot carry the bearer token). */
export async function fetchBlob(path: string): Promise<Blob> {
  const url = buildUrl(path)
  let response: Response
  try {
    response = await globalThis.fetch(url, { headers: authHeaders() })
  } catch (error) {
    throw new ApiError(0, `Network error reaching ${url}: ${errorMessage(error)}`, path)
  }
  if (!response.ok) {
    const text = await response.text()
    let payload: unknown = text
    try {
      payload = JSON.parse(text)
    } catch {
      // keep text
    }
    throw new ApiError(response.status, formatDetail(payload, response.status), path)
  }
  return response.blob()
}

export const api = {
  // ---- health
  health: () => request<Health>('GET', '/health'),

  // ---- projects
  listProjects: () => request<ProjectOut[]>('GET', '/projects'),
  createProject: (body: { name: string; description: string; idempotency_key: string }) =>
    request<ProjectOut>('POST', '/projects', { json: body }),
  getProject: (projectId: string) => request<ProjectDashboard>('GET', `/projects/${projectId}`),

  // ---- imports / jobs
  createImport: (projectId: string, file: File, idempotencyKey: string) => {
    const form = new FormData()
    form.append('file', file)
    form.append('idempotency_key', idempotencyKey)
    return request<ImportOut>('POST', `/projects/${projectId}/imports`, { form })
  },
  listImports: (projectId: string) => request<ImportOut[]>('GET', `/projects/${projectId}/imports`),
  getImport: (importId: string) => request<ImportOut>('GET', `/imports/${importId}`),
  getJob: (jobId: string) => request<JobOut>('GET', `/jobs/${jobId}`),
  cancelJob: (jobId: string) => request<JobOut>('POST', `/jobs/${jobId}/cancel`),
  listJobs: (projectId: string) => request<JobOut[]>('GET', `/projects/${projectId}/jobs`),

  // ---- review
  createReviewBatch: (projectId: string, body: ReviewBatchCreate) =>
    request<ReviewRequestOut[]>('POST', `/projects/${projectId}/review-batches`, { json: body }),
  listReviewRequests: (projectId: string, query?: { state?: string; purpose?: string }) =>
    request<ReviewRequestOut[]>('GET', `/projects/${projectId}/review-requests`, { query }),
  nextReview: (projectId: string, purpose: 'TRAIN' | 'DEV') =>
    request<ReviewCase | null>('GET', `/projects/${projectId}/next-review`, { query: { purpose } }),
  getReviewRequest: (requestId: string) => request<ReviewCase>('GET', `/review-requests/${requestId}`),
  claimReviewRequest: (requestId: string) => request<ReviewRequestOut>('POST', `/review-requests/${requestId}/claim`, { json: {} }),
  releaseReviewRequest: (requestId: string) => request<ReviewRequestOut>('POST', `/review-requests/${requestId}/release`),
  skipReviewRequest: (requestId: string) => request<ReviewRequestOut>('POST', `/review-requests/${requestId}/skip`),
  submitJudgment: (requestId: string, body: JudgmentCreate) =>
    request<JudgmentOut>('POST', `/review-requests/${requestId}/judgments`, { json: body }),
  correctJudgment: (
    judgmentId: string,
    body: { verdict: string; explanation: string; cannot_judge_reason: string | null; idempotency_key: string },
  ) => request<JudgmentOut>('POST', `/judgments/${judgmentId}/corrections`, { json: body }),
  listJudgments: (projectId: string, partition?: 'TRAIN' | 'DEV') =>
    request<JudgmentOut[]>('GET', `/projects/${projectId}/judgments`, { query: { partition } }),

  // ---- optimization
  createOptimizationRun: (projectId: string, body: OptimizationRunCreate) =>
    request<OptimizationRunOut>('POST', `/projects/${projectId}/optimization-runs`, { json: body }),
  listOptimizationRuns: (projectId: string) =>
    request<OptimizationRunOut[]>('GET', `/projects/${projectId}/optimization-runs`),
  getOptimizationRun: (runId: string) => request<OptimizationRunOut>('GET', `/optimization-runs/${runId}`),
  cancelOptimizationRun: (runId: string) => request<OptimizationRunOut>('POST', `/optimization-runs/${runId}/cancel`),

  // ---- graders / shadow
  getGrader: (graderId: string) => request<GraderOut>('GET', `/graders/${graderId}`),
  listGraders: (projectId: string) => request<GraderOut[]>('GET', `/projects/${projectId}/graders`),
  selectShadowGrader: (projectId: string, body: { grader_id: string | null; reason: string }) =>
    request<ShadowSelectResponse>('POST', `/projects/${projectId}/shadow-grader`, { json: body }),

  // ---- traces
  listTraces: (projectId: string, query: { partition?: string; limit: number; offset: number }) =>
    request<TracePage>('GET', `/projects/${projectId}/traces`, { query }),

  // ---- selection rounds (may not be available yet)
  createSelectionRound: (projectId: string, body: { seed?: number; idempotency_key: string }) =>
    request<SelectionRoundOut>('POST', `/projects/${projectId}/selection-rounds`, { json: body }),
  getSelectionRound: (roundId: string) => request<SelectionRoundOut>('GET', `/selection-rounds/${roundId}`),
  listSelectionRounds: (projectId: string) => request<SelectionRoundOut[]>('GET', `/projects/${projectId}/selection-rounds`),

  // ---- grading jobs / predictions (may not be available yet)
  createGradingJob: (
    projectId: string,
    body: { grader_id: string; partition?: string; trace_ids?: string[]; idempotency_key: string },
  ) => request<JobOut>('POST', `/projects/${projectId}/grading-jobs`, { json: body }),
  listPredictions: (projectId: string, graderId: string) =>
    request<PredictionPage>('GET', `/projects/${projectId}/predictions`, { query: { grader_id: graderId } }),

  // ---- audits (may not be available yet)
  createAudit: (projectId: string, body: AuditCreate) => request<AuditOut>('POST', `/projects/${projectId}/audits`, { json: body }),
  listAudits: (projectId: string) => request<AuditOut[]>('GET', `/projects/${projectId}/audits`),
  getAudit: (auditId: string) => request<AuditOut>('GET', `/audits/${auditId}`),
  auditNextReview: (auditId: string) => request<ReviewCase | null>('GET', `/audits/${auditId}/next-review`),
  recomputeAudit: (auditId: string) => request<AuditOut>('POST', `/audits/${auditId}/recompute`),
  spendAudit: (auditId: string, reason: string) => request<AuditOut>('POST', `/audits/${auditId}/spend`, { json: { reason } }),

  // ---- automation policy (may not be available yet)
  setAutomationPolicy: (projectId: string, body: { audit_id: string; enable: boolean; reason: string }) =>
    request<AutomationPolicyOut>('POST', `/projects/${projectId}/automation-policy`, { json: body }),
  getAutomationPolicy: (projectId: string) => request<AutomationPolicyOut>('GET', `/projects/${projectId}/automation-policy`),

  // ---- exports (may not be available yet)
  createExport: (projectId: string, body: { kind: 'FULL' | 'GRADER'; grader_id?: string; idempotency_key: string }) =>
    request<ExportOut>('POST', `/projects/${projectId}/exports`, { json: body }),
  listExports: (projectId: string) => request<ExportOut[]>('GET', `/projects/${projectId}/exports`),
  getExport: (exportId: string) => request<ExportOut>('GET', `/exports/${exportId}`),
  exportDownloadPath: (exportId: string) => `/exports/${exportId}/download`,
}

export type Api = typeof api
