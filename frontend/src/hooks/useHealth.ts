import { useEffect, useState } from 'react'
import { api } from '../api'
import type { Health } from '../types'

/** GET /health once per mount; failures are reported as `error` without blocking the UI. */
export function useHealth(): { health: Health | null; error: unknown } {
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState<unknown>(null)
  useEffect(() => {
    let cancelled = false
    api.health().then(
      (h) => {
        if (!cancelled) setHealth(h)
      },
      (e: unknown) => {
        if (!cancelled) setError(e)
      },
    )
    return () => {
      cancelled = true
    }
  }, [])
  return { health, error }
}
