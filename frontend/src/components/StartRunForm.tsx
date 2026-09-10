import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useIdempotencyKey } from '../hooks/useIdempotencyKey'
import type { GraderOut } from '../types'
import { ErrorBox } from './Status'

function optionalInt(value: string): number | undefined {
  const trimmed = value.trim()
  if (!trimmed) return undefined
  const n = Number(trimmed)
  return Number.isFinite(n) ? Math.trunc(n) : undefined
}

/** Budgeted optimization run form. Navigates to the run detail page on success. */
export function StartRunForm({ projectId, graders }: { projectId: string; graders: GraderOut[] }) {
  const navigate = useNavigate()
  const { key, reset } = useIdempotencyKey()
  const [label, setLabel] = useState('')
  const [maxMetricCalls, setMaxMetricCalls] = useState('')
  const [minibatch, setMinibatch] = useState('')
  const [threads, setThreads] = useState('')
  const [seed, setSeed] = useState('0')
  const [seedGrader, setSeedGrader] = useState('')
  const [maxProviderCalls, setMaxProviderCalls] = useState('')
  const [maxTokens, setMaxTokens] = useState('')
  const [evaluateAll, setEvaluateAll] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const run = await api.createOptimizationRun(projectId, {
        label: label.trim(),
        seed: optionalInt(seed) ?? 0,
        max_metric_calls: optionalInt(maxMetricCalls),
        reflection_minibatch_size: optionalInt(minibatch),
        num_threads: optionalInt(threads),
        seed_grader_id: seedGrader || null,
        max_provider_calls: optionalInt(maxProviderCalls) ?? null,
        max_total_tokens: optionalInt(maxTokens) ?? null,
        evaluate_all_candidates: evaluateAll,
        idempotency_key: key(),
      })
      reset()
      navigate(`/optimization-runs/${run.id}`)
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={submit} className="stack">
      <div className="form-grid">
        <label className="field">
          <span>Label</span>
          <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="optional" />
        </label>
        <label className="field">
          <span>Seed grader</span>
          <select value={seedGrader} onChange={(e) => setSeedGrader(e.target.value)}>
            <option value="">Automatic (shadow → previous best → generic seed)</option>
            {graders.map((g) => (
              <option key={g.id} value={g.id}>
                {g.label} ({g.origin})
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Max metric calls (GEPA budget)</span>
          <input type="number" min={1} value={maxMetricCalls} onChange={(e) => setMaxMetricCalls(e.target.value)} placeholder="server default" />
        </label>
        <label className="field">
          <span>Reflection minibatch size</span>
          <input type="number" min={1} max={50} value={minibatch} onChange={(e) => setMinibatch(e.target.value)} placeholder="server default" />
        </label>
        <label className="field">
          <span>Threads</span>
          <input type="number" min={1} max={32} value={threads} onChange={(e) => setThreads(e.target.value)} placeholder="server default" />
        </label>
        <label className="field">
          <span>Random seed</span>
          <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} />
        </label>
        <label className="field">
          <span>Max provider calls (job budget)</span>
          <input type="number" min={1} value={maxProviderCalls} onChange={(e) => setMaxProviderCalls(e.target.value)} placeholder="server default" />
        </label>
        <label className="field">
          <span>Max total tokens (job budget)</span>
          <input type="number" min={1} value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} placeholder="server default" />
        </label>
      </div>
      <label className="row gap">
        <input type="checkbox" checked={evaluateAll} onChange={(e) => setEvaluateAll(e.target.checked)} />
        <span>Evaluate all captured candidates on DEV (more grading calls)</span>
      </label>
      <p className="muted small">
        A metric-call budget is not a token budget. Cost figures shown on the run are estimates; actual usage is measured separately.
      </p>
      <div className="row gap">
        <button type="submit" className="btn btn-primary" disabled={busy}>
          {busy ? 'Starting…' : 'Start optimization run'}
        </button>
      </div>
      <ErrorBox error={error} />
    </form>
  )
}
