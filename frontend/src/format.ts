import { NOT_ESTIMABLE, type MetricScalar } from './types'

export function formatPercent(value: MetricScalar, digits = 1): string {
  if (value === NOT_ESTIMABLE) return NOT_ESTIMABLE
  if (value === null || value === undefined) return '—'
  if (typeof value !== 'number' || Number.isNaN(value)) return String(value)
  return `${(value * 100).toFixed(digits)}%`
}

export function formatNumber(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(3)
  return String(value)
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return '—'
  const d = new Date(value)
  return Number.isNaN(d.getTime()) ? value : d.toLocaleString()
}

export function formatMs(ms: number): string {
  if (ms < 1000) return `${ms} ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(1)} s`
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`
}

export function shortId(id: string | null | undefined, n = 8): string {
  if (!id) return '—'
  return id.length > n ? `${id.slice(0, n)}…` : id
}

export function truncate(text: string | null | undefined, n = 90): string {
  if (!text) return ''
  return text.length > n ? `${text.slice(0, n)}…` : text
}

export function pretty(value: unknown): string {
  if (value === undefined) return 'undefined'
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}
