import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { TERMINAL_JOB_STATES, type JobOut } from '../types'

export interface PollingState<T> {
  data: T | undefined
  error: unknown
  done: boolean
}

/**
 * Poll `loader` every `intervalMs` while `key` is set and `isDone(data)` is
 * false. Errors stop polling and are surfaced; a new key restarts it.
 */
export function usePolling<T>(
  key: string | null | undefined,
  loader: (key: string) => Promise<T>,
  isDone: (value: T) => boolean,
  intervalMs = 1500,
): PollingState<T> {
  const [data, setData] = useState<T | undefined>(undefined)
  const [error, setError] = useState<unknown>(null)
  const [done, setDone] = useState(false)
  const loaderRef = useRef(loader)
  const isDoneRef = useRef(isDone)
  loaderRef.current = loader
  isDoneRef.current = isDone

  useEffect(() => {
    setData(undefined)
    setError(null)
    setDone(false)
    if (!key) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    const tick = async () => {
      try {
        const value = await loaderRef.current(key)
        if (cancelled) return
        setData(value)
        if (isDoneRef.current(value)) {
          setDone(true)
          return
        }
      } catch (err) {
        if (cancelled) return
        setError(err)
        setDone(true)
        return
      }
      timer = setTimeout(tick, intervalMs)
    }
    void tick()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [key, intervalMs])

  return { data, error, done }
}

export function isTerminalJob(job: JobOut): boolean {
  return TERMINAL_JOB_STATES.includes(job.state)
}

/** Poll GET /jobs/{id} every 1.5s until the job reaches a terminal state. */
export function useJob(jobId: string | null | undefined, intervalMs = 1500): PollingState<JobOut> {
  return usePolling(jobId, (id) => api.getJob(id), isTerminalJob, intervalMs)
}
