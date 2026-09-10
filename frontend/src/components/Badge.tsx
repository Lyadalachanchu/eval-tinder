import type { ReactNode } from 'react'

export type BadgeKind =
  | 'human'
  | 'machine'
  | 'provisional'
  | 'dev-agreement'
  | 'audit-evidence'
  | 'simulated'
  | 'neutral'
  | 'good'
  | 'bad'
  | 'warn'
  | 'info'

export function Badge({ kind = 'neutral', children, title }: { kind?: BadgeKind; children: ReactNode; title?: string }) {
  return (
    <span className={`badge badge-${kind}`} title={title}>
      {children}
    </span>
  )
}

/** A judgment made by a person. */
export function HumanBadge() {
  return <Badge kind="human" title="A verdict entered by a human reviewer">HUMAN</Badge>
}

/** A verdict produced by a grader. Always paired with PROVISIONAL outside an audit report. */
export function MachineBadge() {
  return <Badge kind="machine" title="A verdict produced by a grader prompt">MACHINE</Badge>
}

export function ProvisionalBadge() {
  return (
    <Badge kind="provisional" title="Shadow predictions are provisional; they never enable automation">
      PROVISIONAL
    </Badge>
  )
}

/** Marks DEV-snapshot agreement numbers: a development result, never production accuracy. */
export function DevAgreementBadge({ long = false }: { long?: boolean }) {
  return (
    <Badge kind="dev-agreement" title="Agreement with human labels on a frozen DEV snapshot. Not production accuracy.">
      {long ? 'DEVELOPMENT AGREEMENT (frozen DEV snapshot), not production accuracy' : 'DEVELOPMENT AGREEMENT'}
    </Badge>
  )
}

/** Marks numbers that come from a locked, independent audit. */
export function AuditEvidenceBadge() {
  return (
    <Badge kind="audit-evidence" title="Evidence from a locked independent audit of the declared population and window">
      AUDIT EVIDENCE
    </Badge>
  )
}

const STATE_KIND: Record<string, BadgeKind> = {
  SUCCEEDED: 'good',
  COMPLETE: 'good',
  ENABLED: 'good',
  JUDGED: 'good',
  OK: 'good',
  PASS: 'good',
  FAIL: 'bad',
  FAILED: 'bad',
  INVALIDATED: 'bad',
  CANCELLED: 'warn',
  BUDGET_EXHAUSTED: 'warn',
  NO_IMPROVEMENT: 'warn',
  SPENT: 'warn',
  CANNOT_JUDGE: 'warn',
  REVIEW: 'warn',
  RUNNING: 'info',
  QUEUED: 'info',
  IN_REVIEW: 'info',
  LOCKED: 'info',
  LEASED: 'info',
  OPEN: 'neutral',
  DISABLED: 'neutral',
  SKIPPED: 'neutral',
}

export function StateBadge({ state }: { state: string | null | undefined }) {
  if (!state) return <Badge>—</Badge>
  return <Badge kind={STATE_KIND[state] ?? 'neutral'}>{state}</Badge>
}
