import { useCallback, useEffect, useRef, useState } from 'react'

export interface AsyncState<T> {
  data: T | undefined
  error: unknown
  loading: boolean
  reload: () => void
  /** Replace the cached value locally (e.g. after a mutation returned the new object). */
  setData: (updater: T | ((previous: T | undefined) => T | undefined)) => void
}

/**
 * Run an async loader whenever `deps` change and expose loading/error/data.
 * Stale responses are ignored when a newer load has started.
 */
export function useAsync<T>(loader: () => Promise<T>, deps: unknown[]): AsyncState<T> {
  const [data, setData] = useState<T | undefined>(undefined)
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(true)
  const [version, setVersion] = useState(0)
  const latest = useRef(0)

  useEffect(() => {
    const ticket = ++latest.current
    setLoading(true)
    setError(null)
    loader().then(
      (value) => {
        if (ticket !== latest.current) return
        setData(value)
        setLoading(false)
      },
      (err: unknown) => {
        if (ticket !== latest.current) return
        setError(err)
        setLoading(false)
      },
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, ...deps])

  const reload = useCallback(() => setVersion((v) => v + 1), [])
  const set = useCallback((updater: T | ((previous: T | undefined) => T | undefined)) => {
    setData((previous) => (typeof updater === 'function' ? (updater as (p: T | undefined) => T | undefined)(previous) : updater))
  }, [])

  return { data, error, loading, reload, setData: set }
}
