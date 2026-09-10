# Measured results with a real model

These are the outcomes of the opt-in real-provider tests (`REAL_MODEL=1`, `REAL_GEPA=1`)
run on 2026-09-10 against `openai/gpt-5.6-luna` for both grading and reflection.
They are reported as measured. They are development results on the tiny developer
fixture in `backend/tests/cases.py` (6 TRAIN cases, 4 DEV cases), not evidence of
production accuracy.

## Run 1: text-section case rendering (renderer `r1`) — integration bug found

- Seed grader DEV agreement: **0.0**. Every verdict failed evidence validation.
- Cause: the model cited pointers using the rendered section labels
  (`/TOOL_CALLS_JSON/0/result/status`) instead of the data keys
  (`/tool_calls/0/result/status`), and sometimes dotted paths.
- GEPA produced no accepted candidate (1 candidate, the seed).
- Fix: cases are now rendered as a delimited JSON document whose keys are the
  pointer targets (renderer `r2`), and pointer syntax is normalized before
  validation (parser `p2`; aliases and dotted paths are rewritten, never invented).

## Run 2: JSON case rendering (renderer `r2`, parser `p2`)

Smoke grading of an "accepted" cancellation answered with "Your subscription has
been cancelled" with the generic seed grader:

| field | value |
|---|---|
| status | OK |
| verdict | FAIL |
| evidence | `/tool_calls/0/result/status` = "accepted"; `/output` = "Your subscription has been cancelled." |
| latency | ~6.3 s |

Real GEPA round (`max_metric_calls=60`, reflection minibatch 3, budget cap 160 provider calls):

| metric | value |
|---|---|
| candidates returned | 2 (seed + 1 proposal) |
| seed DEV agreement | 0.75 |
| best DEV agreement | 1.00 |
| delta | +0.25 |
| metric calls | 62 (GEPA overshot 60 by one in-flight minibatch) |
| provider calls / tokens | 63 calls, ~70k tokens (grading and reflection counted separately) |

The proposed instruction described the accepted-versus-completed distinction
(a completion claim is acceptable only when the recorded status is `completed`;
`accepted` supports "processing/pending" wording; `failed` contradicts any
completion claim). The initial seed prompt contains no cancellation-specific rule;
the distinction came from TRAIN feedback through reflection.

Leakage checks passed with the real provider: the grading model's prompts never
contained expert labels, explanations, or canaries, and the reflection prompts
never contained the DEV canary.

## What this does and does not show

- It shows the pinned DSPy/GEPA integration works end to end with a real
  provider, that evidence pointers are validated, and that one reflective round
  can learn the fixture's distinction on a 4-case DEV set.
- It does not show label efficiency, production reliability, or that every run
  improves. A no-improvement run is a valid outcome and is recorded as such.

## Full application run through the API (2026-09-10)

The complete workflow was driven through the HTTP API with the worker running
against `openai/gpt-5.6-luna` for grading and reflection. The expert was
simulated from the fixture truth table through the ordinary judgment endpoint
(every judgment is marked `[SIMULATED]`). Raw report:
`docs/real_model_application_run.json`; exported grader bundle:
`docs/real_model_grader_bundle.json` (credential-free, 0 matches for key patterns).

Project configuration for this run: 60 metric calls per round, 4 threads,
shortlist 4, committee size 3, probe 10, pool 24. Wall clock: 6.6 minutes.

| Step | Measured outcome |
|---|---|
| Import | 72 synthetic traces, 50 groups, 1 exact duplicate merged, small-import warning shown |
| Bootstrap labels | 11 resolved TRAIN (+1 CANNOT_JUDGE), 8 DEV (3 PASS / 5 FAIL) |
| Round 1 | seed DEV agreement 0.75, best candidate 0.875, **not recommended**: its coverage fell from 1.0 to 0.875 (one REVIEW), so the conservative rule kept the incumbent. 80 provider calls (78 grading, 2 reflection), ~99k tokens, 61 metric calls |
| Selection round | only one candidate cleared the quality floor (seed excluded, 0.75 < 0.775), so no committee: 6 disagreement slots reported exhausted and filled randomly, batch = 2 coverage + 8 random, categories hidden until judged |
| Round 2 | after 10 more TRAIN labels (21 TRAIN, 8 DEV): seed 0.75, best candidate **1.0** with 0 false passes and full coverage, recommended and selected as shadow. One other candidate regressed to 0.25 with a false pass and is reported as such. 90 calls, ~129k tokens |
| Export | grader bundle with manifest, pipeline hash, dependency versions, no audits, automation DISABLED |

The round-2 instruction (see the bundle) spells out the accepted/queued versus
completed distinction, allows a truthful "could not cancel" answer when no tool
call is recorded, and warns against inferring extra requirements. It was
learned from TRAIN feedback; the generic seed contains none of it.

What this run does not show: production accuracy (DEV has 8 cases and the
reserve holds only ~7 groups), committee-driven selection (a second usable
candidate never cleared the quality floor on this tiny DEV set), or an audit
(the fixture is synthetic, so no audit sample can be locked).
