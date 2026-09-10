import { useCallback, useRef } from 'react'
import { newIdempotencyKey } from '../settings'

/**
 * A lazily created idempotency key that stays the same across retries of one
 * submission. Call `reset()` after the server accepted the request so the next
 * submission gets a fresh key.
 */
export function useIdempotencyKey(): { key: () => string; reset: () => void } {
  const ref = useRef<string | null>(null)
  const key = useCallback(() => {
    if (ref.current === null) ref.current = newIdempotencyKey()
    return ref.current
  }, [])
  const reset = useCallback(() => {
    ref.current = null
  }, [])
  return { key, reset }
}
