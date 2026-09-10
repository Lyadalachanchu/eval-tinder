import { vi } from 'vitest'

export interface MockRoute {
  method?: string
  /** Matched against pathname (string) or pathname+search (RegExp). */
  match: string | RegExp
  status?: number
  body?: unknown | ((url: URL, init?: RequestInit) => unknown)
}

export interface RecordedCall {
  method: string
  url: string
  path: string
  body: unknown
  headers: Record<string, string>
}

const HEALTH = { status: 'ok', llm_provider: 'fake', simulated: true, grader_model: null }

/**
 * Stub globalThis.fetch with a small router. Unmatched requests answer 404
 * with a FastAPI-style {detail}. Every call is recorded for assertions.
 */
export function installFetch(routes: MockRoute[]): { calls: RecordedCall[] } {
  const calls: RecordedCall[] = []
  const all: MockRoute[] = [{ match: '/api/health', body: HEALTH }, ...routes]
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
    const method = (init?.method ?? 'GET').toUpperCase()
    const u = new URL(url, 'http://localhost')
    let parsed: unknown = undefined
    if (typeof init?.body === 'string') {
      try {
        parsed = JSON.parse(init.body)
      } catch {
        parsed = init.body
      }
    } else if (init?.body instanceof FormData) {
      parsed = Object.fromEntries(init.body.entries())
    }
    const headers: Record<string, string> = {}
    if (init?.headers && !(init.headers instanceof Headers) && !Array.isArray(init.headers)) Object.assign(headers, init.headers)
    calls.push({ method, url, path: u.pathname, body: parsed, headers })
    for (const route of all) {
      if ((route.method ?? 'GET').toUpperCase() !== method) continue
      const hit = typeof route.match === 'string' ? u.pathname === route.match : route.match.test(u.pathname + u.search)
      if (!hit) continue
      const body = typeof route.body === 'function' ? (route.body as (url: URL, init?: RequestInit) => unknown)(u, init) : route.body
      const status = route.status ?? 200
      return new Response(body === undefined ? '' : JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
    }
    return new Response(JSON.stringify({ detail: `no mock for ${method} ${u.pathname}` }), {
      status: 404,
      headers: { 'Content-Type': 'application/json' },
    })
  })
  vi.stubGlobal('fetch', fn)
  return { calls }
}
