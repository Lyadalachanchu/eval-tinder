import type { AuditOut, JudgmentOut, OptimizationRunOut, ReviewCase } from '../types'

export const LEAKED_EXPLANATION = 'LEAKED_MACHINE_EXPLANATION_should_not_render'
export const LEAKED_REASON = 'SECRET_SELECTION_REASON_should_not_render'
export const LEAKED_GRADER = 'grader-leak-0001'

/** A case as a leaky backend might send it: predictions and reasons present even though the UI must not show them. */
export function makeCase(overrides: Partial<ReviewCase['trace']> = {}, purpose: 'TRAIN' | 'DEV' | 'AUDIT' = 'TRAIN'): ReviewCase {
  return {
    request: {
      id: 'req-1',
      project_id: 'p1',
      trace_id: 't1',
      purpose,
      state: 'LEASED',
      selection_category: 'DISAGREEMENT',
      selection_reason: { why: LEAKED_REASON, votes: { [LEAKED_GRADER]: 'FAIL' } },
      expected_reading_length: 240,
      lease_owner: 'local-expert',
      lease_expiry: null,
      judgment_id: null,
      batch_id: null,
      created_at: '2026-09-01T00:00:00Z',
    },
    trace: {
      id: 't1',
      external_id: 'chat-101-v1',
      group_id: 'chat-101',
      revision: 1,
      timestamp: '2026-02-03T09:05:00Z',
      input: 'I want to cancel my membership, effective today.',
      context: { subscription_id: 's-2061', plan: 'premium' },
      tool_calls: [{ name: 'cancel_subscription', arguments: { subscription_id: 's-2061' }, result: { status: 'completed' } }],
      output: 'The cancellation will be applied once billing picks it up.',
      metadata: { task_type: 'cancellation' },
      source_type: 'SYNTHETIC',
      content_hash: 'hash-abc',
      partition: purpose === 'DEV' ? 'DEV' : purpose === 'AUDIT' ? 'AUDIT_RESERVE' : 'TRAIN',
      ...overrides,
    },
    shown_context_hash: 'hash-abc',
    predictions: [
      { grader_id: LEAKED_GRADER, verdict: 'FAIL', status: 'OK', explanation: LEAKED_EXPLANATION, kind: 'MACHINE', provisional: true },
    ],
  }
}

export function makeJudgment(verdict: JudgmentOut['verdict'], reason: JudgmentOut['cannot_judge_reason'] = null): JudgmentOut {
  return {
    id: 'j1',
    trace_id: 't1',
    review_request_id: 'req-1',
    purpose: 'TRAIN',
    policy_epoch: 1,
    verdict,
    explanation: '',
    cannot_judge_reason: reason,
    reviewer_id: 'local-expert',
    active_review_ms: 1234,
    supersedes_id: null,
    superseded_by_id: null,
    created_at: '2026-09-01T00:00:00Z',
  }
}

const metric = (value: number | 'NOT_ESTIMABLE', numerator: number, denominator: number, definition = 'def') => ({
  value,
  numerator,
  denominator,
  definition,
})

export function makeRun(): OptimizationRunOut {
  const baselines = {
    always_pass: {
      agreement: metric(0.75, 6, 8),
      failure_recall: metric(0, 0, 2),
      false_pass_rate_among_accepted: metric(0.25, 2, 8),
      definition: 'always PASS',
    },
    always_fail: {
      agreement: metric(0.25, 2, 8),
      failure_recall: metric(1, 2, 2),
      false_pass_rate_among_accepted: metric('NOT_ESTIMABLE', 0, 0),
      definition: 'always FAIL',
    },
  }
  return {
    id: 'run-1',
    project_id: 'p1',
    state: 'SUCCEEDED',
    seed_grader_id: 'g-seed',
    seed_choice: 'generic_seed',
    train_snapshot_id: 'snap-train',
    dev_snapshot_id: 'snap-dev',
    train_size: 12,
    dev_size: 8,
    policy_epoch: 1,
    metric_version: 'v1',
    config: { label: 'first run', max_metric_calls: 40 },
    budgets: { max_metric_calls: 40, estimate: { grading_calls: 104, reflection_calls: 5, cost_usd: null, note: 'Estimate only.', cost_note: 'Configure PRICING_TABLE.' } },
    usage: { grading: { calls: 60 } },
    result_summary: {
      seed_agreement: 0.625,
      best_agreement: 0.875,
      recommended_grader_id: 'g-cand-2',
      improved: true,
      partial: false,
      note: 'DEV agreement is a development result on a frozen snapshot, not evidence of production accuracy.',
      comparison: {
        recommend: true,
        reason: 'meets the conservative rule',
        rule: 'recommend only if agreement improves',
        incumbent: { grader_id: 'g-seed', agreement: 0.625, false_passes: 2, coverage: 1, failure_recall: 0.5 },
        candidate: { grader_id: 'g-cand-2', agreement: 0.875, false_passes: 1, coverage: 1, failure_recall: 0.75 },
        insufficient_class_coverage: false,
      },
    },
    job_id: null,
    error: null,
    created_at: '2026-09-01T00:00:00Z',
    finished_at: '2026-09-01T00:10:00Z',
    candidates: [
      {
        grader_id: 'g-seed',
        candidate_index: 0,
        label: 'seed',
        parent_ids: [],
        instruction_text: 'Judge the response.\nBe strict.\n',
        manifest_hash: 'mh-seed',
        evaluation: {
          id: 'ev-0',
          dev_snapshot_id: 'snap-dev',
          complete: true,
          source: 'app',
          kind: 'DEVELOPMENT_AGREEMENT',
          aggregate_metrics: { agreement: 0.625, false_passes: 2, coverage: 1, failure_recall: 0.5, baselines, complete: true, evaluated_cases: 8, dev_size: 8 },
        },
        is_seed: true,
        is_member: true,
        diff_from_seed: '',
      },
      {
        grader_id: 'g-cand-2',
        candidate_index: 2,
        label: 'candidate 2',
        parent_ids: ['g-seed'],
        instruction_text: 'Judge the response.\nBe strict about cancellations.\n',
        manifest_hash: 'mh-2',
        evaluation: {
          id: 'ev-2',
          dev_snapshot_id: 'snap-dev',
          complete: true,
          source: 'app',
          kind: 'DEVELOPMENT_AGREEMENT',
          aggregate_metrics: { agreement: 0.875, false_passes: 1, coverage: 1, failure_recall: 0.75, baselines, complete: true, evaluated_cases: 8, dev_size: 8 },
        },
        is_seed: false,
        is_member: true,
        diff_from_seed: '--- seed\n+++ candidate\n@@ -1,2 +1,2 @@\n Judge the response.\n-Be strict.\n+Be strict about cancellations.\n',
      },
    ],
  }
}

export function makeAudit(gatePassed: boolean): AuditOut {
  return {
    id: 'audit-1',
    project_id: 'p1',
    grader_id: 'g-cand-2',
    pipeline_hash: 'pipe-hash-1',
    policy_epoch: 1,
    state: 'COMPLETE',
    population_definition: { source_type: 'PRODUCTION', partition: 'AUDIT_RESERVE' },
    sampling_plan: { unit: 'group', method: 'uniform_random', independence_assumption_documented: true, independence_note: 'n' },
    risk_targets: { permitted_verdicts: ['PASS', 'FAIL'], max_error_rate: 0.05, min_coverage: 0.8, confidence: 0.95 },
    planned_n: 10,
    locked_count: 10,
    judged_count: 10,
    unresolved_count: 0,
    report_version: 1,
    report: {
      kind: 'AUDIT_EVIDENCE',
      complete: true,
      counts: { human_pass: 10, human_fail: 0, human_unresolved: 0, operational_failures: 0, total_cases: 10 },
      table: { PASS: { PASS: 10, FAIL: 0, REVIEW: 0 }, FAIL: { PASS: 0, FAIL: 0, REVIEW: 0 } },
      metrics: {
        agreement: metric(1, 10, 10, 'agreement def'),
        failure_recall: metric('NOT_ESTIMABLE', 0, 0, 'human FAIL cases the machine also marked FAIL'),
        false_pass_rate_among_accepted: metric(0, 0, 10, 'fp def'),
      },
      baselines: {},
      human_unresolved: [],
      unresolved_automatic_decisions: [],
      operational_failures: [],
      intervals: { supported: true, confidence: 0.95, per_bound_confidence: 0.975, automatic_error_rate_upper: 0.31, false_pass_rate_upper: 0.31 },
      gate: {
        passed: gatePassed,
        checks: [
          { name: 'sampling design supported', passed: true },
          { name: 'failure_recall estimable', passed: gatePassed, detail: gatePassed ? '' : 'no human FAIL cases in the sample' },
        ],
      },
      scope: { window: 'all', weighting: 'group' },
      notes: ['Evidence applies to the declared population only.'],
    },
    correction_history: [],
    grading_job_id: null,
    created_at: '2026-09-02T00:00:00Z',
  }
}
