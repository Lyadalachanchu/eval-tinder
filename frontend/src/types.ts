/**
 * API types mirrored from backend/eval_tinder/api/schemas.py and the router
 * modules. Loose dictionaries from the backend stay `Record<string, unknown>`
 * so the UI never invents structure the server did not promise.
 */

export type Partition = 'TRAIN' | 'DEV' | 'AUDIT_RESERVE'
export type ReviewPurpose = 'TRAIN' | 'DEV' | 'AUDIT'
export type HumanVerdict = 'PASS' | 'FAIL' | 'CANNOT_JUDGE'
export type CannotJudgeReason = 'MISSING_CONTEXT' | 'AMBIGUOUS_POLICY' | 'OUT_OF_SCOPE' | 'OTHER'
export type MachineVerdict = 'PASS' | 'FAIL' | 'REVIEW'
export type JobState = 'QUEUED' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'CANCELLED' | 'BUDGET_EXHAUSTED'
export type OptimizationState =
  | 'QUEUED'
  | 'RUNNING'
  | 'SUCCEEDED'
  | 'NO_IMPROVEMENT'
  | 'FAILED'
  | 'CANCELLED'
  | 'BUDGET_EXHAUSTED'
export type AuditState = 'LOCKED' | 'IN_REVIEW' | 'COMPLETE' | 'INVALIDATED' | 'SPENT'
export type AutomationState = 'DISABLED' | 'ENABLED' | 'INVALIDATED'
export type SelectionRoundState = 'QUEUED' | 'RUNNING' | 'COMPLETE' | 'FAILED'

export const CANNOT_JUDGE_REASONS: CannotJudgeReason[] = [
  'MISSING_CONTEXT',
  'AMBIGUOUS_POLICY',
  'OUT_OF_SCOPE',
  'OTHER',
]

export const TERMINAL_JOB_STATES: JobState[] = ['SUCCEEDED', 'FAILED', 'CANCELLED', 'BUDGET_EXHAUSTED']

/** The backend's sentinel for a ratio with a zero denominator. Always shown literally. */
export const NOT_ESTIMABLE = 'NOT_ESTIMABLE'
export type MetricScalar = number | typeof NOT_ESTIMABLE | null | undefined

export type Dict = Record<string, unknown>

// ---------------------------------------------------------------- health

export interface Health {
  status: string
  llm_provider: string
  simulated: boolean
  grader_model: string | null
}

// ---------------------------------------------------------------- projects

export interface ProjectOut {
  id: string
  name: string
  description: string
  policy_epoch: number
  policy_notes: string
  configuration: Dict
  active_shadow_grader_id: string | null
  automation_policy_id: string | null
  created_at: string
}

export interface LabelCounts {
  PASS: number
  FAIL: number
  CANNOT_JUDGE: number
  resolved: number
}

export interface Readiness {
  resolved_train: number
  resolved_dev: number
  bootstrap_train_labels: number
  bootstrap_dev_labels: number
  bootstrap_ready: boolean
  new_train_labels_since_last_run: number
  ready_to_optimize_again: boolean
  dev_topup_target: number
  last_run_id: string | null
  active_run: string | null
  automatic_optimization?: boolean
  note: string
}

export interface ProjectDashboard {
  project: ProjectOut
  partitions: Record<string, number>
  labels: Record<string, Partial<LabelCounts>>
  review_states: Record<string, number>
  readiness: Readiness
  graders: number
  runs: number
  shadow_grader: { id: string; label: string; manifest_hash: string; status: string } | null
  automation: { id: string; state: AutomationState; pipeline_hash: string; audit_id: string | null } | null
}

// ---------------------------------------------------------------- imports / jobs

export interface ImportOut {
  id: string
  project_id: string
  filename: string
  state: string
  counts: Dict
  line_errors: Dict[]
  job_id: string | null
  created_at: string
}

export interface JobOut {
  id: string
  project_id: string | null
  kind: string
  state: JobState
  progress: Dict
  result: Dict
  attempts: number
  error: string | null
  cancel_requested: boolean
  created_at: string
  started_at: string | null
  finished_at: string | null
}

// ---------------------------------------------------------------- review

export interface ReviewRequestOut {
  id: string
  project_id: string
  trace_id: string
  purpose: ReviewPurpose
  state: string
  selection_category: string
  selection_reason: Dict | null
  expected_reading_length: number
  lease_owner: string | null
  lease_expiry: string | null
  judgment_id: string | null
  batch_id: string | null
  created_at: string
}

export interface TraceView {
  id: string
  external_id: string
  group_id: string
  revision: number
  timestamp: string | null
  input: string
  context: unknown
  tool_calls: unknown
  output: string
  metadata: Dict
  source_type: string
  content_hash: string
  partition: string | null
}

export interface MachinePrediction {
  grader_id?: string
  verdict: MachineVerdict | string
  status: string
  explanation?: string
  evidence?: unknown
  kind: 'MACHINE'
  provisional: true
}

export interface ReviewCase {
  request: ReviewRequestOut
  trace: TraceView
  shown_context_hash: string
  predictions: MachinePrediction[] | null
}

export interface JudgmentCreate {
  verdict: HumanVerdict
  explanation: string
  cannot_judge_reason: CannotJudgeReason | null
  shown_context_hash: string
  active_review_ms: number
  idempotency_key: string
}

export interface JudgmentOut {
  id: string
  trace_id: string
  review_request_id: string | null
  purpose: ReviewPurpose
  policy_epoch: number
  verdict: HumanVerdict
  explanation: string
  cannot_judge_reason: CannotJudgeReason | null
  reviewer_id: string
  active_review_ms: number
  supersedes_id: string | null
  superseded_by_id: string | null
  created_at: string
}

export interface ReviewBatchCreate {
  purpose: 'TRAIN' | 'DEV'
  kind: 'SEED' | 'DEV_RANDOM' | 'RANDOM'
  size?: number
  seed?: number
  idempotency_key: string
}

// ---------------------------------------------------------------- optimization

export interface MetricEntry {
  value: MetricScalar
  numerator: number
  denominator: number
  definition: string
}

export interface BaselineEntry {
  agreement: MetricEntry
  failure_recall: MetricEntry
  false_pass_rate_among_accepted: MetricEntry
  definition: string
}

export interface AggregateMetrics {
  metrics?: Record<string, MetricEntry | Dict>
  baselines?: { always_pass?: BaselineEntry; always_fail?: BaselineEntry }
  agreement?: MetricScalar
  false_passes?: number
  coverage?: MetricScalar
  failure_recall?: MetricScalar
  human_classes?: { PASS: number; FAIL: number }
  insufficient_class_coverage?: boolean
  evaluated_cases?: number
  dev_size?: number
  complete?: boolean
  partial_reason?: string
}

export interface CandidateEvaluation {
  id: string
  dev_snapshot_id: string
  complete: boolean
  source: string
  aggregate_metrics: AggregateMetrics | null
  per_case_scores?: Dict
  verdicts?: Dict
  kind: 'DEVELOPMENT_AGREEMENT'
}

export interface CandidateOut {
  grader_id: string
  candidate_index: number | null
  label: string
  parent_ids: string[]
  instruction_text: string
  manifest_hash: string
  evaluation: CandidateEvaluation | null
  is_seed: boolean
  is_member: boolean
  diff_from_seed: string
}

export interface ComparisonSide {
  grader_id: string
  agreement: MetricScalar
  false_passes: number | null
  coverage: MetricScalar
  failure_recall: MetricScalar
}

export interface Comparison {
  dev_snapshot_id?: string
  incumbent?: ComparisonSide
  candidate?: ComparisonSide
  insufficient_class_coverage?: boolean
  rule?: string
  recommend?: boolean
  reason?: string
}

export interface ResultSummary {
  seed_agreement?: MetricScalar
  best_agreement?: MetricScalar
  recommended_grader_id?: string | null
  comparison?: Comparison | null
  improved?: boolean
  partial?: boolean
  partial_reason?: string | null
  note?: string
  member_indices?: number[]
  candidate_grader_ids?: Record<string, string>
  [key: string]: unknown
}

export interface BudgetEstimate {
  grading_calls?: number
  reflection_calls?: number
  cost_usd?: number | null
  cost_note?: string
  note?: string
}

export interface OptimizationRunOut {
  id: string
  project_id: string
  state: OptimizationState
  seed_grader_id: string
  seed_choice: string
  train_snapshot_id: string
  dev_snapshot_id: string
  train_size: number
  dev_size: number
  policy_epoch: number
  metric_version: string
  config: Dict
  budgets: Dict & { estimate?: BudgetEstimate }
  usage: Dict
  result_summary: ResultSummary
  job_id: string | null
  error: string | null
  created_at: string
  finished_at: string | null
  candidates: CandidateOut[]
}

export interface OptimizationRunCreate {
  max_metric_calls?: number
  reflection_minibatch_size?: number
  num_threads?: number
  seed: number
  seed_grader_id?: string | null
  label: string
  max_provider_calls?: number | null
  max_total_tokens?: number | null
  evaluate_all_candidates?: boolean
  idempotency_key: string
}

// ---------------------------------------------------------------- graders

export interface GraderOut {
  id: string
  project_id: string
  label: string
  origin: string
  parent_ids: string[]
  optimization_run_id: string | null
  candidate_index: number | null
  instruction_text: string
  immutable_policy_context: string
  model_config: Dict
  renderer_version: string
  parser_version: string
  policy_epoch: number
  manifest: Dict
  manifest_hash: string
  pipeline_hash: string
  created_at: string
  diff_from_parent: string | null
  evaluations: CandidateEvaluation[]
  is_active_shadow: boolean
}

export interface ShadowSelectResponse {
  project_id: string
  active_shadow_grader_id: string | null
  status: 'PROVISIONAL'
  note: string
  history: Dict[]
}

// ---------------------------------------------------------------- traces

export interface TraceRow {
  trace: TraceView
  partition: Partition
  human_judgment: {
    verdict: HumanVerdict
    kind: 'HUMAN'
    cannot_judge_reason?: CannotJudgeReason | null
    explanation?: string
    judgment_id?: string
  } | null
  shadow_prediction: (MachinePrediction & { grader_id: string }) | null
}

export interface TracePage {
  items: TraceRow[]
  total: number
  limit: number
  offset: number
}

// ---------------------------------------------------------------- selection rounds (in progress on the backend)

export interface SelectionRoundOut {
  id: string
  state: SelectionRoundState
  job_id?: string | null
  committee_ids?: string[]
  committee_report?: { members?: unknown[]; log?: unknown[]; diversity_claimed?: boolean; reason?: string } | null
  probe_size?: number
  pool_size?: number
  selected_requests?: { request_id: string; trace_id: string; category: string }[]
  exhausted?: boolean
  context_repair?: unknown
  batch_id?: string | null
  error?: string | null
  created_at?: string
}

// ---------------------------------------------------------------- predictions / grading jobs (in progress on the backend)

export interface PredictionRow {
  trace_id: string
  external_id: string
  verdict: string
  status: string
  kind: 'MACHINE'
  provisional: boolean
  audit_status?: string
  automation_status?: string
  explanation?: string
  evidence?: unknown
}

export interface PredictionPage {
  items: PredictionRow[]
  total: number
}

// ---------------------------------------------------------------- audits (in progress on the backend)

export interface AuditPopulation {
  source_type: 'PRODUCTION'
  partition: 'AUDIT_RESERVE'
  time_window?: { start?: string; end?: string } | null
  task_types?: string[] | null
}

export interface AuditSamplingPlan {
  unit: 'group'
  method: 'uniform_random'
  independence_assumption_documented: boolean
  independence_note: string
}

export interface AuditRiskTargets {
  permitted_verdicts: ('PASS' | 'FAIL')[]
  max_error_rate: number
  min_coverage: number
  confidence: number
  gate_false_pass_rate?: number | null
  joint_allocation: 'bonferroni'
  unresolved_automatic_rule: 'block' | 'count_as_error'
}

export interface AuditCreate {
  grader_id: string
  planned_n: number
  seed?: number
  population: AuditPopulation
  sampling_plan: AuditSamplingPlan
  risk_targets: AuditRiskTargets
  idempotency_key: string
}

export interface AuditGateCheck {
  name: string
  passed: boolean
  detail?: string
}

export interface AuditReport {
  kind: 'AUDIT_EVIDENCE'
  complete: boolean
  counts?: Dict
  table?: Record<string, Record<string, number>>
  metrics?: Record<string, MetricEntry>
  baselines?: Dict
  human_unresolved?: unknown[]
  unresolved_automatic_decisions?: unknown[]
  operational_failures?: unknown[]
  intervals?: {
    supported: boolean
    reason?: string
    confidence?: number
    per_bound_confidence?: number
    automatic_error_rate_upper?: MetricScalar
    false_pass_rate_upper?: MetricScalar
  }
  gate?: { passed: boolean; checks?: AuditGateCheck[] }
  scope?: Dict | string
  notes?: string[]
}

export interface AuditOut {
  id: string
  project_id?: string
  grader_id: string
  pipeline_hash: string
  policy_epoch: number
  state: AuditState
  population_definition: Dict
  sampling_plan: Dict
  risk_targets: Dict
  planned_n: number
  locked_count: number
  judged_count: number
  unresolved_count: number
  report: AuditReport | null
  report_version?: number
  correction_history?: unknown[]
  grading_job_id?: string | null
  created_at?: string
}

export interface AutomationPolicyOut {
  state: AutomationState
  pipeline_hash?: string | null
  audit_id?: string | null
  gate_result?: Dict | null
  reason?: string
  history?: Dict[]
}

// ---------------------------------------------------------------- exports (in progress on the backend)

export interface ExportOut {
  id: string
  project_id?: string
  kind: 'FULL' | 'GRADER'
  grader_id?: string | null
  state: string
  job_id: string | null
  download_url?: string | null
  created_at?: string
  [key: string]: unknown
}
