import { useCallback, useEffect, useRef } from 'react'

function now(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function' ? performance.now() : Date.now()
}

function isVisible(): boolean {
  return typeof document === 'undefined' || document.visibilityState !== 'hidden'
}

/**
 * Measures active review time in milliseconds for the current case, excluding
 * time while the tab is hidden (document.visibilitychange). The counter resets
 * whenever `caseKey` changes.
 */
export function useActiveReviewTimer(caseKey: string | null | undefined): { read: () => number; reset: () => void } {
  const accumulated = useRef(0)
  const visibleSince = useRef<number | null>(null)

  const reset = useCallback(() => {
    accumulated.current = 0
    visibleSince.current = isVisible() ? now() : null
  }, [])

  useEffect(() => {
    reset()
    const onVisibility = () => {
      if (isVisible()) {
        if (visibleSince.current === null) visibleSince.current = now()
      } else if (visibleSince.current !== null) {
        accumulated.current += now() - visibleSince.current
        visibleSince.current = null
      }
    }
    document.addEventListener('visibilitychange', onVisibility)
    return () => document.removeEventListener('visibilitychange', onVisibility)
  }, [caseKey, reset])

  const read = useCallback(() => {
    const live = visibleSince.current === null ? 0 : now() - visibleSince.current
    return Math.max(0, Math.round(accumulated.current + live))
  }, [])

  return { read, reset }
}
