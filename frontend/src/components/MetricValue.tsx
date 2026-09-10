import { formatPercent } from '../format'
import { NOT_ESTIMABLE, type MetricEntry, type MetricScalar } from '../types'

/** Shows a ratio as a percentage, or the literal NOT_ESTIMABLE sentinel. */
export function MetricValue({ value, digits = 1 }: { value: MetricScalar; digits?: number }) {
  if (value === NOT_ESTIMABLE) {
    return (
      <span className="not-estimable" title="Zero denominator: this rate cannot be estimated from the data">
        {NOT_ESTIMABLE}
      </span>
    )
  }
  return <span>{formatPercent(value, digits)}</span>
}

export function MetricWithCounts({ entry }: { entry: MetricEntry | undefined | null }) {
  if (!entry) return <span className="muted">—</span>
  return (
    <span title={entry.definition}>
      <MetricValue value={entry.value} />{' '}
      <span className="muted small">
        ({entry.numerator}/{entry.denominator})
      </span>
    </span>
  )
}
