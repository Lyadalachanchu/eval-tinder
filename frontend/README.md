# eval-tinder frontend

React 18 + TypeScript + Vite single-page app for the eval-tinder backend (FastAPI at `http://localhost:8000`).

Experts label production traces PASS / FAIL / CANNOT_JUDGE in a **blind review** screen, GEPA evolves grader prompts, candidate
graders select the next traces to review, and only an **independent audit** can gate automation. The UI keeps three kinds of
numbers visually distinct:

| Badge | Meaning |
|---|---|
| `DEVELOPMENT AGREEMENT` | Agreement with human labels on a frozen DEV snapshot. Never production accuracy. |
| `AUDIT EVIDENCE` | Metrics from a locked, independent audit of a declared population and window. |
| `MACHINE` + `PROVISIONAL` | A shadow-grader prediction. Stored next to human labels, never over them; never enables automation. |
| `HUMAN` | A verdict entered by a person. |

## Requirements

- Node 22 and pnpm (`corepack enable` or `/opt/node22/bin/pnpm`)
- The backend running on `http://localhost:8000` (see the repository root `docker-compose.yml` / `backend/README.md`)

## Development

```sh
cd frontend
pnpm install
pnpm dev            # http://localhost:5173
```

In development every request goes to `/api/*` and the Vite dev server proxies it to `http://localhost:8000` with the `/api`
prefix stripped (`vite.config.ts`). The backend's loopback-only default auth therefore works without a token. Set
`API_PROXY_TARGET` to proxy somewhere else.

## Production build

```sh
VITE_API_BASE=https://api.example.com pnpm build   # writes dist/
pnpm preview                                         # serves dist/ on 0.0.0.0:5173
```

`VITE_API_BASE` is baked in at build time and defaults to `/api` when unset (so a reverse proxy can forward `/api` to the
backend). The in-app **Settings** panel (top right) can override the API base at runtime and stores an optional bearer token;
when present it is sent as `Authorization: Bearer <token>` on every request. Both live only in that browser's localStorage.
Note that the backend does not currently add CORS headers, so a separate-origin deployment needs a reverse proxy or a
backend CORS configuration.

## Docker

```sh
docker build -t eval-tinder-frontend --build-arg VITE_API_BASE=http://localhost:8000 frontend/
docker run -p 5173:5173 eval-tinder-frontend
```

The root `docker-compose.yml` builds this image as the `frontend` service.

## Quality checks

```sh
pnpm lint     # eslint (typescript-eslint + react-hooks + react-refresh)
pnpm build    # tsc -b && vite build
pnpm test     # vitest + @testing-library/react (jsdom)
```

The tests cover the product's safety rules: the review screen never renders predictions or selection reasons before a
judgment (and never for DEV/audit), CANNOT_JUDGE requires a category, HTML in trace text is rendered as text, the audit
report shows `NOT_ESTIMABLE` literally and keeps "Enable automation" disabled unless the predeclared gate passed, the
optimization screen labels agreement as development agreement, and active review time excludes hidden-tab time.

## Screens

| Route | Purpose |
|---|---|
| `/` | Projects list and create form (name, optional description; no rubric field by design) |
| `/projects/:id` | Dashboard: JSONL import with job polling, partition/label/review-state counts, readiness, seed TRAIN / DEV random batches, selection rounds, optimization run form, shadow grader, automation status |
| `/projects/:id/review?purpose=TRAIN\|DEV` | Blind review; TRAIN reveals predictions and the selection reason only after judging |
| `/audits/:auditId/review` | Blind audit review; nothing is ever revealed |
| `/projects/:id/optimization`, `/optimization-runs/:runId` | Run list; run detail with budget estimate, usage, candidate table, baselines, prompt diff, "Use as shadow grader" |
| `/projects/:id/traces` | Paginated TRAIN/DEV traces with human labels vs provisional shadow predictions; bulk grading job |
| `/projects/:id/audits`, `/audits/:auditId` | Lock an audit (all risk targets required, no defaults); evidence report, gate checklist, recompute / spend / enable automation |
| `/projects/:id/exports` | FULL or GRADER exports with job polling and authenticated download |
| `/graders/:id` | Manifest, instruction text, diff from parent, DEV evaluations, pipeline hash |

Endpoints the backend does not serve yet (selection rounds, grading jobs, audits, automation policy, exports) degrade to a
"not available yet" panel instead of failing the page.

## Conventions

- `src/api.ts` is the only place that talks to the network. Errors become `ApiError` with the FastAPI `{detail}` text.
- Every creating POST sends a fresh `crypto.randomUUID()` idempotency key that stays stable across retries of the same
  submission (`useIdempotencyKey`).
- Judgments carry `shown_context_hash` from the case and `active_review_ms` measured with `document.visibilitychange`.
- Trace text is rendered through React's default escaping only; `dangerouslySetInnerHTML` is not used anywhere.
