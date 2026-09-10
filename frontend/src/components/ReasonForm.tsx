import { useState, type FormEvent, type ReactNode } from 'react'
import { ErrorBox } from './Status'

/**
 * A small inline form for policy-changing actions that must record a reason
 * (shadow selection, spending an audit, enabling automation).
 */
export function ReasonForm({
  actionLabel,
  onSubmit,
  disabled = false,
  disabledReason,
  note,
  placeholder = 'Reason (recorded in history)',
  danger = false,
}: {
  actionLabel: string
  onSubmit: (reason: string) => Promise<void>
  disabled?: boolean
  disabledReason?: ReactNode
  note?: ReactNode
  placeholder?: string
  danger?: boolean
}) {
  const [reason, setReason] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (!reason.trim()) {
      setError(new Error('A reason is required.'))
      return
    }
    setBusy(true)
    setError(null)
    try {
      await onSubmit(reason.trim())
      setReason('')
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  return (
    <form className="reason-form" onSubmit={submit}>
      {note ? <p className="muted small">{note}</p> : null}
      <div className="row gap">
        <input
          type="text"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          placeholder={placeholder}
          aria-label={`${actionLabel} reason`}
          disabled={disabled || busy}
        />
        <button type="submit" className={`btn ${danger ? 'btn-danger' : 'btn-primary'}`} disabled={disabled || busy || !reason.trim()}>
          {busy ? 'Working…' : actionLabel}
        </button>
      </div>
      {disabled && disabledReason ? <p className="muted small">{disabledReason}</p> : null}
      <ErrorBox error={error} />
    </form>
  )
}
