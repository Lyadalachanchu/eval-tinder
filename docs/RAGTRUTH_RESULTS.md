# RAGTruth experiment: measured results

Date: 2026-09-10. Model: `openai/gpt-5.6-luna` for grading and reflection.
Raw state (every round, score, and audit): `docs/ragtruth_experiment_state.json`.
Code: `backend/eval_tinder/benchmarks/` (converter, scorer, driver), launcher
`backend/scripts/run_ragtruth_experiment.py`.

## Setup

- Data: RAGTruth (Niu et al., ACL 2024), human span-level hallucination labels on
  RAG answers from six models. A seeded, task-stratified sample of the train split
  (300 source documents, 1,806 answers, 44% with a hallucination) was imported as
  PRODUCTION traces; all six answers to one document share a group. Label policy:
  any annotated span = FAIL, none = PASS, truncated = CANNOT_JUDGE.
- The official test split (450 documents, 2,700 answers) never entered the app. A
  612-answer stratified slice scored graders between rounds; the final graders and
  the generic seed were scored on all 2,699 determinate answers.
- Partitions (seeded): 223 TRAIN / 40 DEV / 37 AUDIT_RESERVE documents.
- Two strategies, each on its own fresh project: **random** batches versus
  **committee**-selected batches. Same bootstrap (12 TRAIN + 8 DEV), same label
  schedule (TRAIN 12, 32, 72, 152; DEV topped up to 8, 8, 18, 38), a 300
  metric-call GEPA round after each stage, 8 threads. Human labels were replayed
  through the ordinary review path (leases, blind requests, idempotent judgments).
- Cost: 8,431 model calls and 20.4 million tokens across both strategies (GEPA
  rounds, DEV re-evaluations, selection probes/pools, audits, benchmark scoring),
  4.0 hours wall clock.

## 1. Learning curve

DEV agreement is the app's own re-evaluation of the seed and best candidate on
the frozen DEV snapshot; the promotion rule requires better agreement, no added
false passes, and no coverage loss.

| Strategy | Round | TRAIN / DEV labels | Seed on DEV | Best candidate on DEV | Outcome |
|---|---|---|---|---|---|
| random | 0 | 12 / 8 | 0.750 | 0.750 | no improvement |
| random | 1 | 32 / 8 | 0.750 | 0.750 | no improvement |
| random | 2 | 72 / 18 | 0.722 | 0.722 | no improvement |
| random | 3 | 152 / 38 | 0.763 | 0.789 | **promoted** |
| committee | 0 | 12 / 8 | 0.750 | 0.750 | no improvement |
| committee | 1 | 32 / 8 | 0.750 | 0.750 | no improvement |
| committee | 2 | 72 / 18 | 0.667 | 0.722 | two candidates **refused** (added a false pass; one also lost coverage) |
| committee | 3 | 152 / 38 | 0.658 | 0.658 | no improvement |

Scores on the full test split (2,699 unseen answers). Precision, recall, and F1
are for detecting a hallucination (FAIL); coverage is the share answered
PASS/FAIL rather than REVIEW.

| Grader | Agreement | Precision | Recall | F1 | Coverage | False-pass rate |
|---|---|---|---|---|---|---|
| generic seed (both strategies' starting point; committee's final grader) | 68.0% | 0.567 | 88.5% | 0.691 | 93.3% | 4.3% |
| random strategy, promoted after 152 labels | 69.5% | 0.566 | 92.5% | 0.702 | 95.6% | 3.6% |

Per task, the promoted grader's gains came from summarization (agreement 59.1% to
67.6%, recall 80.9% to 91.2%); data-to-text was unchanged (F1 0.835 to 0.836) and
question answering traded agreement (71.1% to 66.4%) for recall (83.8% to 89.4%).

## 2. Did the rule reject bad revisions?

In committee round 2 the rule refused a candidate that beat the seed on DEV
(0.722 vs 0.667) because it added one false pass and lost coverage. Scored on the
612-answer test slice:

| | seed (kept) | refused candidate |
|---|---|---|
| agreement | 66.7% | 78.3% |
| precision | 0.546 | 0.730 |
| recall | 88.6% | 85.6% |
| F1 | 0.675 | 0.788 |
| false-pass rate among accepted answers | 3.8% | 7.6% |

The refusal was consistent with the declared priority (never add false passes):
the candidate really does let twice as many hallucinations through as PASS. It
also cost 12 points of agreement and 0.11 F1. Whether that trade is right depends
on the deployment; the rule made it visible and did not decide it silently.

## 3. Committee versus random

Committee selection did not beat random here: random ended with a promoted
grader (F1 0.702 on the full split) and committee ended with the seed (0.691).
The mechanism is in the log: through rounds 1 and 2 only one grader cleared the
committee quality floor, so the "committee" batches were coverage/random picks.
On an 8-answer DEV set one case is 0.125 of agreement, more than the 0.10 floor,
so every GEPA candidate one case below the seed was excluded. Committees of two
to four members formed only in round 3 (DEV = 18 and 38), where all four batches
were mostly disagreement-driven (12/4/4, with one batch reporting 6 exhausted
disagreement slots). The design assumption that candidates worth disagreeing
with exist early did not hold on this task with this model.

## 4. Audits on the untouched reserve

Both strategies audited the same 37 reserve answers (same partition seed, same
audit seed), one designated answer per document, with a predeclared target of
15% automatic error at 95% confidence.

| Audited grader | Measured error | 95% upper bound | Gate | Error on the 2,699-answer test split |
|---|---|---|---|---|
| random's promoted grader | 32.4% (12 of 37) | 47.2% | failed | 27.4% |
| generic seed (committee's final) | 14.3% (5 of 35; 2 REVIEW) | 27.7% | failed | 27.1% |

Both gates correctly refused automation. Both bounds contained the error measured
on the large test split; the seed's bound (27.7% versus a true 27.1%) shows how
little margin a 35-case audit gives, and the two audits of the same 37 answers
landing at 32% and 14% shows the sampling noise at that size. Automation stayed
disabled everywhere.

## 5. Comparison with the paper's baselines (Table 5, response level, test split)

| Method | QA F1 | Data-to-text F1 | Summary F1 | Overall precision / recall / F1 |
|---|---|---|---|---|
| Prompt, gpt-4-turbo (paper) | 45.6 | 78.3 | 47.6 | 46.9 / 97.9 / 63.4 |
| Finetuned Llama-2-13B (paper) | 68.2 | 88.1 | 59.1 | 76.9 / 80.7 / 78.7 |
| generic seed, gpt-5.6-luna (this run) | 56.3 | 83.5 | 50.3 | 56.7 / 88.5 / 69.1 |
| promoted grader, gpt-5.6-luna (this run) | 52.2 | 83.6 | 58.4 | 56.6 / 92.5 / 70.2 |
| refused candidate (612-answer slice only) | | | | 73.0 / 85.6 / 78.8 |

The prompted graders here sit between the paper's prompted GPT-4-turbo and its
fine-tuned detector: recall is high, precision is the weakness. The refused
candidate's precision-recall balance matches the fine-tuned detector on the slice
it was scored on, which is exactly the balance the promotion rule refused to trade
for.

## What this does and does not show

- The pipeline holds up on real human labels: leakage rules, hidden categories,
  frozen snapshots, honest no-improvement rounds, refusals that are explainable on
  held-out data, audits whose bounds contained the true error, and automation off.
- Learning from labels was modest with this model: one promotion in eight rounds,
  +0.011 F1 and +4 points recall on the full split. The generic prompt already
  performs above the paper's prompted GPT-4 baseline, leaving less headroom for
  prompt evolution than the synthetic demo suggested.
- The committee mechanism needs a DEV set large enough that the quality floor
  means something; a floor expressed in cases rather than a fixed 0.10 would let
  committees form earlier. That is a configuration finding, not a result.
- One run per strategy, one seed; differences of a point or two between the
  strategies are within noise.
