/**
 * Client-side settings kept in localStorage: an optional bearer token and an
 * optional API base override. Every access is wrapped because storage can be
 * unavailable (private windows, blocked site data, jsdom quirks).
 */

const TOKEN_KEY = 'eval-tinder.api-token'
const BASE_KEY = 'eval-tinder.api-base'

function read(key: string): string {
  try {
    return window.localStorage.getItem(key) ?? ''
  } catch {
    return ''
  }
}

function write(key: string, value: string): void {
  try {
    if (value) window.localStorage.setItem(key, value)
    else window.localStorage.removeItem(key)
  } catch {
    // storage unavailable; the value simply does not persist
  }
}

export function getToken(): string {
  return read(TOKEN_KEY)
}

export function setToken(token: string): void {
  write(TOKEN_KEY, token.trim())
}

export function getApiBaseOverride(): string {
  return read(BASE_KEY)
}

export function setApiBaseOverride(base: string): void {
  write(BASE_KEY, base.trim().replace(/\/+$/, ''))
}

/** Compile-time default: VITE_API_BASE for production builds, `/api` (the dev proxy) otherwise. */
export function defaultApiBase(): string {
  const env = (import.meta.env.VITE_API_BASE as string | undefined)?.trim()
  if (env) return env.replace(/\/+$/, '')
  return '/api'
}

export function getApiBase(): string {
  return getApiBaseOverride() || defaultApiBase()
}

/** A fresh UUID for idempotency keys. Callers keep it stable across retries of one submission. */
export function newIdempotencyKey(): string {
  const c = globalThis.crypto
  if (c && typeof c.randomUUID === 'function') return c.randomUUID()
  if (c && typeof c.getRandomValues === 'function') {
    const bytes = new Uint8Array(16)
    c.getRandomValues(bytes)
    bytes[6] = (bytes[6] & 0x0f) | 0x40
    bytes[8] = (bytes[8] & 0x3f) | 0x80
    const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
  }
  return `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}-${Math.random().toString(16).slice(2)}`
}
