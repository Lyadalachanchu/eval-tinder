import type { ReactNode } from 'react'
import { errorMessage, isNotAvailable } from '../api'

export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <p className="muted" role="status">
      {label}
    </p>
  )
}

export function ErrorBox({ error, onRetry, prefix }: { error: unknown; onRetry?: () => void; prefix?: string }) {
  if (!error) return null
  return (
    <div className="error-box" role="alert">
      <span>
        {prefix ? `${prefix}: ` : ''}
        {errorMessage(error)}
      </span>
      {onRetry ? (
        <button type="button" className="btn btn-small" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty-state">
      <p className="empty-title">{title}</p>
      {children ? <div className="empty-body">{children}</div> : null}
    </div>
  )
}

/** Friendly fallback for endpoints that the backend does not serve yet. */
export function NotAvailable({ feature }: { feature: string }) {
  return (
    <div className="empty-state not-available">
      <p className="empty-title">{feature} is not available yet</p>
      <p className="empty-body">The backend does not serve this endpoint on this deployment. Nothing else is affected.</p>
    </div>
  )
}

/** Renders NotAvailable for missing-route errors, ErrorBox otherwise. */
export function LoadFailure({ error, feature, onRetry }: { error: unknown; feature: string; onRetry?: () => void }) {
  if (isNotAvailable(error)) return <NotAvailable feature={feature} />
  return <ErrorBox error={error} onRetry={onRetry} />
}

export function Notice({ kind = 'info', children }: { kind?: 'info' | 'warn' | 'good'; children: ReactNode }) {
  return <div className={`notice notice-${kind}`}>{children}</div>
}
