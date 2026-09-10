import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import { Badge, HumanBadge, MachineBadge, ProvisionalBadge, StateBadge } from '../components/Badge'
import { JsonBlock, TextBlock } from '../components/JsonBlock'
import { ProjectNav } from '../components/ProjectNav'
import { EmptyState, ErrorBox, LoadFailure, Loading } from '../components/Status'
import { formatDate, formatMs } from '../format'
import { useActiveReviewTimer } from '../hooks/useActiveReviewTimer'
import { useIdempotencyKey } from '../hooks/useIdempotencyKey'
import { CANNOT_JUDGE_REASONS, type CannotJudgeReason, type HumanVerdict, type JudgmentOut, type ReviewCase } from '../types'

type ReviewMode = 'TRAIN' | 'DEV' | 'AUDIT'
type Phase = 'loading' | 'error' | 'empty' | 'case' | 'submitting' | 'judged'

const MODE_HINTS: Record<ReviewMode, string> = {
  TRAIN: 'Create a seed TRAIN batch or a selection round on the project dashboard.',
  DEV: 'Create a DEV random batch on the project dashboard.',
  AUDIT: 'Every locked audit case has been judged, or is leased to another reviewer.',
}

/** Case content only. Nothing here can carry predictions, prompt identities or selection reasons. */
function TracePanel({ current }: { current: ReviewCase }) {
  const { trace, request } = current
  const hasToolCalls = Array.isArray(trace.tool_calls) ? trace.tool_calls.length > 0 : trace.tool_calls != null
  return (
    <div className="card">
      <div className="card-header">
        <h2>Case</h2>
        <span className="mono">{trace.external_id}</span>
        <Badge kind="info">{request.purpose}</Badge>
        {trace.source_type ? <Badge>{trace.source_type}</Badge> : null}
      </div>
      <ul className="kv-list small">
        <li>
          <span className="kv-key">Group</span>
          <span className="kv-val mono">
            {trace.group_id} · revision {trace.revision}
          </span>
        </li>
        <li>
          <span className="kv-key">Timestamp</span>
          <span className="kv-val">{formatDate(trace.timestamp)}</span>
        </li>
        <li>
          <span className="kv-key">Expected reading length</span>
          <span className="kv-val">{request.expected_reading_length} chars</span>
        </li>
      </ul>

      <section className="review-section">
        <h3>User request</h3>
        <TextBlock text={trace.input} />
      </section>
      <section className="review-section">
        <h3>Context</h3>
        {typeof trace.context === 'string' ? <TextBlock text={trace.context} /> : <JsonBlock value={trace.context} maxHeight={320} />}
      </section>
      <section className="review-section">
        <h3>Tool calls</h3>
        {hasToolCalls ? <JsonBlock value={trace.tool_calls} maxHeight={320} /> : <p className="muted">No tool calls recorded.</p>}
      </section>
      <section className="review-section">
        <h3>Target output</h3>
        <TextBlock text={trace.output} className="prominent" />
      </section>
      {Object.keys(trace.metadata ?? {}).length > 0 ? (
        <details>
          <summary>Metadata</summary>
          <JsonBlock value={trace.metadata} maxHeight={200} />
        </details>
      ) : null}
      <details>
        <summary>Show full snapshot</summary>
        <JsonBlock value={trace} />
      </details>
      <p className="muted small">
        Snapshot hash <span className="mono">{current.shown_context_hash}</span> is sent with your judgment so the exact displayed
        evidence is recorded.
      </p>
    </div>
  )
}

function RevealedPanel({ requestId, revealed, error }: { requestId: string; revealed: ReviewCase | null; error: unknown }) {
  if (error) return <ErrorBox error={error} prefix="Could not load the revealed panel" />
  if (!revealed) return <Loading label="Loading what was hidden…" />
  const reason = revealed.request.selection_reason
  const hasReason = reason && Object.keys(reason).length > 0
  const predictions = revealed.predictions ?? []
  return (
    <section className="stack">
      <h3>Revealed after your judgment</h3>
      <p>
        Selection category: <Badge kind="info">{revealed.request.selection_category}</Badge>
        <span className="muted small mono"> request {requestId}</span>
      </p>
      {hasReason ? <JsonBlock value={reason} maxHeight={220} /> : <p className="muted">No selection reason was recorded for this case.</p>}
      <h4>
        Machine predictions <MachineBadge /> <ProvisionalBadge />
      </h4>
      {predictions.length === 0 ? (
        <p className="muted">No machine predictions were recorded for this case.</p>
      ) : (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Grader</th>
                <th>Verdict</th>
                <th>Status</th>
                <th>Explanation</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((p, i) => (
                <tr key={`${p.grader_id ?? 'grader'}-${i}`}>
                  <td>{p.grader_id ? <Link to={`/graders/${p.grader_id}`}>{p.grader_id.slice(0, 8)}…</Link> : '—'}</td>
                  <td>
                    <StateBadge state={p.verdict} />
                  </td>
                  <td>
                    <StateBadge state={p.status} />
                  </td>
                  <td className="small">{p.explanation || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

export function ReviewPage() {
  const params = useParams<{ projectId?: string; auditId?: string }>()
  const [searchParams] = useSearchParams()
  const projectId = params.projectId ?? ''
  const auditId = params.auditId ?? ''
  const mode: ReviewMode = auditId ? 'AUDIT' : searchParams.get('purpose') === 'DEV' ? 'DEV' : 'TRAIN'

  const [phase, setPhase] = useState<Phase>('loading')
  const [current, setCurrent] = useState<ReviewCase | null>(null)
  const [loadError, setLoadError] = useState<unknown>(null)
  const [submitError, setSubmitError] = useState<unknown>(null)
  const [judgment, setJudgment] = useState<JudgmentOut | null>(null)
  const [revealed, setRevealed] = useState<ReviewCase | null>(null)
  const [revealError, setRevealError] = useState<unknown>(null)
  const [explanation, setExplanation] = useState('')
  const [cannotJudgeOpen, setCannotJudgeOpen] = useState(false)
  const [cannotJudgeReason, setCannotJudgeReason] = useState<CannotJudgeReason | ''>('')
  const [validation, setValidation] = useState<string | null>(null)
  const { key: idempotencyKey, reset: resetIdempotencyKey } = useIdempotencyKey()
  const timer = useActiveReviewTimer(current?.request.id ?? null)
  const loadTicket = useRef(0)

  const loadNext = useCallback(async () => {
    const ticket = ++loadTicket.current
    setPhase('loading')
    setLoadError(null)
    setSubmitError(null)
    setJudgment(null)
    setRevealed(null)
    setRevealError(null)
    setExplanation('')
    setCannotJudgeOpen(false)
    setCannotJudgeReason('')
    setValidation(null)
    resetIdempotencyKey()
    try {
      const next = mode === 'AUDIT' ? await api.auditNextReview(auditId) : await api.nextReview(projectId, mode)
      if (ticket !== loadTicket.current) return
      setCurrent(next)
      setPhase(next ? 'case' : 'empty')
    } catch (e) {
      if (ticket !== loadTicket.current) return
      setCurrent(null)
      setLoadError(e)
      setPhase('error')
    }
  }, [mode, auditId, projectId, resetIdempotencyKey])

  useEffect(() => {
    void loadNext()
  }, [loadNext])

  const submit = async (verdict: HumanVerdict) => {
    if (!current) return
    if (verdict === 'CANNOT_JUDGE' && !cannotJudgeReason) {
      setValidation('Select a category: CANNOT_JUDGE requires one.')
      return
    }
    setValidation(null)
    setSubmitError(null)
    setPhase('submitting')
    try {
      const result = await api.submitJudgment(current.request.id, {
        verdict,
        explanation: explanation.trim(),
        cannot_judge_reason: verdict === 'CANNOT_JUDGE' ? (cannotJudgeReason as CannotJudgeReason) : null,
        shown_context_hash: current.shown_context_hash,
        active_review_ms: timer.read(),
        idempotency_key: idempotencyKey(),
      })
      resetIdempotencyKey()
      setJudgment(result)
      setPhase('judged')
      if (mode === 'TRAIN') {
        try {
          setRevealed(await api.getReviewRequest(current.request.id))
        } catch (e) {
          setRevealError(e)
        }
      }
    } catch (e) {
      setSubmitError(e)
      setPhase('case')
    }
  }

  const skip = async () => {
    if (!current) return
    setSubmitError(null)
    setPhase('submitting')
    try {
      await api.skipReviewRequest(current.request.id)
      await loadNext()
    } catch (e) {
      setSubmitError(e)
      setPhase('case')
    }
  }

  const busy = phase === 'submitting'
  const title = mode === 'AUDIT' ? 'Blind audit review' : `Blind review · ${mode}`

  return (
    <div>
      {projectId ? <ProjectNav projectId={projectId} /> : null}
      <div className="card-header">
        <h1>{title}</h1>
        {auditId ? (
          <Link to={`/audits/${auditId}`} className="small">
            audit {auditId.slice(0, 8)}…
          </Link>
        ) : null}
      </div>
      <p className="blind-note">
        <HumanBadge /> Candidate predictions, prompt identities and selection reasons are hidden until you submit.
        {mode === 'TRAIN'
          ? ' TRAIN cases reveal them after your judgment.'
          : ` ${mode === 'DEV' ? 'DEV' : 'Audit'} reviews stay blind: nothing is revealed afterwards.`}
      </p>

      {phase === 'loading' ? (
        <Loading label="Fetching the next case…" />
      ) : phase === 'error' ? (
        <LoadFailure error={loadError} feature={mode === 'AUDIT' ? 'Audit review' : 'Review'} onRetry={() => void loadNext()} />
      ) : phase === 'empty' || !current ? (
        <EmptyState title="No open review requests">
          {MODE_HINTS[mode]}{' '}
          <button type="button" className="btn btn-small" onClick={() => void loadNext()}>
            Check again
          </button>
        </EmptyState>
      ) : (
        <div className="review-layout">
          <TracePanel current={current} />
          {phase === 'judged' && judgment ? (
            <div className="card">
              <h2>Judgment recorded</h2>
              <p className="row gap">
                <HumanBadge />
                <StateBadge state={judgment.verdict} />
                {judgment.cannot_judge_reason ? <Badge kind="warn">{judgment.cannot_judge_reason}</Badge> : null}
                <span className="muted small">active review {formatMs(judgment.active_review_ms)}</span>
              </p>
              {mode === 'TRAIN' ? (
                <RevealedPanel requestId={current.request.id} revealed={revealed} error={revealError} />
              ) : (
                <p className="blind-note">
                  {mode === 'DEV' ? 'DEV' : 'Audit'} reviews stay blind: no predictions, prompt identities or selection reasons are
                  shown, before or after judging.
                </p>
              )}
              <div className="row gap" style={{ marginTop: '0.75rem' }}>
                <button type="button" className="btn btn-primary" onClick={() => void loadNext()}>
                  Next case
                </button>
              </div>
            </div>
          ) : (
            <div className="card">
              <h2>Your judgment</h2>
              <label className="field">
                <span>Explanation (optional)</span>
                <textarea
                  value={explanation}
                  onChange={(e) => setExplanation(e.target.value)}
                  placeholder="Why this output passes or fails, in your own words."
                  disabled={busy}
                />
              </label>
              <div className="verdict-buttons" style={{ marginTop: '0.75rem' }}>
                <button type="button" className="btn btn-good btn-large" onClick={() => void submit('PASS')} disabled={busy}>
                  PASS
                </button>
                <button type="button" className="btn btn-bad btn-large" onClick={() => void submit('FAIL')} disabled={busy}>
                  FAIL
                </button>
                <button
                  type="button"
                  className="btn btn-warn btn-large"
                  onClick={() => setCannotJudgeOpen(true)}
                  disabled={busy}
                  aria-expanded={cannotJudgeOpen}
                >
                  CANNOT_JUDGE
                </button>
                <button type="button" className="btn btn-large" onClick={() => void skip()} disabled={busy}>
                  SKIP
                </button>
              </div>
              {cannotJudgeOpen ? (
                <div className="stack" style={{ marginTop: '0.75rem' }}>
                  <label className="field">
                    <span>Category (required for CANNOT_JUDGE)</span>
                    <select
                      value={cannotJudgeReason}
                      onChange={(e) => {
                        setCannotJudgeReason(e.target.value as CannotJudgeReason | '')
                        setValidation(null)
                      }}
                      disabled={busy}
                      aria-label="CANNOT_JUDGE category"
                    >
                      <option value="">Select a category…</option>
                      {CANNOT_JUDGE_REASONS.map((r) => (
                        <option key={r} value={r}>
                          {r}
                        </option>
                      ))}
                    </select>
                  </label>
                  <div className="row gap">
                    <button
                      type="button"
                      className="btn btn-warn"
                      onClick={() => void submit('CANNOT_JUDGE')}
                      disabled={busy || !cannotJudgeReason}
                    >
                      Submit CANNOT_JUDGE
                    </button>
                    <button
                      type="button"
                      className="btn btn-small"
                      onClick={() => {
                        setCannotJudgeOpen(false)
                        setCannotJudgeReason('')
                        setValidation(null)
                      }}
                      disabled={busy}
                    >
                      Cancel
                    </button>
                  </div>
                  <p className="muted small">
                    CANNOT_JUDGE is not FAIL. It is excluded from agreement optimization and routed to context or policy
                    resolution.
                  </p>
                </div>
              ) : null}
              {validation ? (
                <p className="error-box" role="alert">
                  {validation}
                </p>
              ) : null}
              <ErrorBox error={submitError} />
              {submitError ? (
                <p className="small">
                  Retrying keeps the same idempotency key. If the lease or snapshot changed,{' '}
                  <button type="button" className="btn btn-small" onClick={() => void loadNext()}>
                    reload the case
                  </button>
                </p>
              ) : null}
              <p className="muted small" style={{ marginTop: '0.75rem' }}>
                Active review time excludes time while this tab is hidden. Submissions are idempotent: a retry reuses the same key.
                SKIP records no judgment.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
