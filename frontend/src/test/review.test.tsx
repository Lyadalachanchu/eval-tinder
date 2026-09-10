import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { LEAKED_EXPLANATION, LEAKED_GRADER, LEAKED_REASON, makeCase, makeJudgment } from './fixtures'
import { installFetch } from './mockFetch'
import { renderApp } from './render'

describe('blind review screen', () => {
  it('never renders predictions, prompt identities or selection reasons before a judgment', async () => {
    installFetch([{ match: /^\/api\/projects\/p1\/next-review/, body: makeCase() }])
    renderApp('/projects/p1/review?purpose=TRAIN')

    expect(await screen.findByText('I want to cancel my membership, effective today.')).toBeInTheDocument()
    expect(screen.getByText('The cancellation will be applied once billing picks it up.')).toBeInTheDocument()

    expect(screen.queryByText(LEAKED_EXPLANATION)).not.toBeInTheDocument()
    expect(screen.queryByText(new RegExp(LEAKED_REASON))).not.toBeInTheDocument()
    expect(screen.queryByText(new RegExp(LEAKED_GRADER))).not.toBeInTheDocument()
    expect(screen.queryByText('DISAGREEMENT')).not.toBeInTheDocument()
    expect(screen.queryByText('MACHINE')).not.toBeInTheDocument()
    expect(screen.queryByText('PROVISIONAL')).not.toBeInTheDocument()
    expect(document.body.textContent).not.toContain(LEAKED_EXPLANATION)
    expect(document.body.textContent).not.toContain(LEAKED_REASON)
  })

  it('reveals machine predictions with MACHINE / PROVISIONAL badges only after a TRAIN judgment', async () => {
    const { calls } = installFetch([
      { match: /^\/api\/projects\/p1\/next-review/, body: makeCase() },
      { method: 'POST', match: '/api/review-requests/req-1/judgments', status: 201, body: makeJudgment('PASS') },
      { match: '/api/review-requests/req-1', body: makeCase() },
    ])
    renderApp('/projects/p1/review?purpose=TRAIN')
    const user = userEvent.setup()

    await user.click(await screen.findByRole('button', { name: 'PASS' }))

    expect(await screen.findByText('Judgment recorded')).toBeInTheDocument()
    expect(await screen.findByText(LEAKED_EXPLANATION)).toBeInTheDocument()
    expect(screen.getByText('MACHINE')).toBeInTheDocument()
    expect(screen.getByText('PROVISIONAL')).toBeInTheDocument()
    expect(screen.getByText('DISAGREEMENT')).toBeInTheDocument()

    const post = calls.find((c) => c.method === 'POST' && c.path === '/api/review-requests/req-1/judgments')
    expect(post).toBeDefined()
    const body = post!.body as Record<string, unknown>
    expect(body.verdict).toBe('PASS')
    expect(body.cannot_judge_reason).toBeNull()
    expect(body.shown_context_hash).toBe('hash-abc')
    expect(typeof body.active_review_ms).toBe('number')
    expect(typeof body.idempotency_key).toBe('string')
    expect((body.idempotency_key as string).length).toBeGreaterThan(8)
  })

  it('keeps DEV reviews blind after judging and never re-fetches the request', async () => {
    const { calls } = installFetch([
      { match: /^\/api\/projects\/p1\/next-review/, body: makeCase({}, 'DEV') },
      { method: 'POST', match: '/api/review-requests/req-1/judgments', status: 201, body: makeJudgment('FAIL') },
      { match: '/api/review-requests/req-1', body: makeCase({}, 'DEV') },
    ])
    renderApp('/projects/p1/review?purpose=DEV')
    const user = userEvent.setup()

    await user.click(await screen.findByRole('button', { name: 'FAIL' }))
    expect(await screen.findByText('Judgment recorded')).toBeInTheDocument()

    expect(screen.queryByText('MACHINE')).not.toBeInTheDocument()
    expect(screen.queryByText(LEAKED_EXPLANATION)).not.toBeInTheDocument()
    expect(screen.queryByText(new RegExp(LEAKED_REASON))).not.toBeInTheDocument()
    expect(calls.some((c) => c.method === 'GET' && c.path === '/api/review-requests/req-1')).toBe(false)
  })

  it('requires a category before CANNOT_JUDGE can be submitted', async () => {
    const { calls } = installFetch([
      { match: /^\/api\/projects\/p1\/next-review/, body: makeCase() },
      { method: 'POST', match: '/api/review-requests/req-1/judgments', status: 201, body: makeJudgment('CANNOT_JUDGE', 'MISSING_CONTEXT') },
      { match: '/api/review-requests/req-1', body: makeCase() },
    ])
    renderApp('/projects/p1/review?purpose=TRAIN')
    const user = userEvent.setup()

    await user.click(await screen.findByRole('button', { name: 'CANNOT_JUDGE' }))
    const submit = screen.getByRole('button', { name: 'Submit CANNOT_JUDGE' })
    expect(submit).toBeDisabled()
    await user.click(submit)
    expect(calls.filter((c) => c.method === 'POST')).toHaveLength(0)

    await user.selectOptions(screen.getByLabelText('CANNOT_JUDGE category'), 'MISSING_CONTEXT')
    expect(submit).toBeEnabled()
    await user.click(submit)

    await waitFor(() => expect(calls.filter((c) => c.method === 'POST')).toHaveLength(1))
    const body = calls.find((c) => c.method === 'POST')!.body as Record<string, unknown>
    expect(body.verdict).toBe('CANNOT_JUDGE')
    expect(body.cannot_judge_reason).toBe('MISSING_CONTEXT')
  })

  it('renders HTML in trace text as text, never as elements', async () => {
    installFetch([
      {
        match: /^\/api\/projects\/p1\/next-review/,
        body: makeCase({
          input: '<script>alert("xss")</script> please help',
          output: '<img src=x onerror="alert(1)"> done <b>bold</b>',
          context: { note: '<iframe src="https://evil.example"></iframe>' },
        }),
      },
    ])
    const { container } = renderApp('/projects/p1/review?purpose=TRAIN')

    expect(await screen.findByText(/<script>alert\("xss"\)<\/script> please help/)).toBeInTheDocument()
    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('iframe')).toBeNull()
    expect(container.querySelector('b')).toBeNull()
    expect(document.body.textContent).toContain('<script>')
    expect(document.body.textContent).toContain('<img src=x onerror="alert(1)">')
  })

  it('shows an empty state when no request is open', async () => {
    installFetch([{ match: /^\/api\/projects\/p1\/next-review/, body: null }])
    renderApp('/projects/p1/review?purpose=TRAIN')
    expect(await screen.findByText('No open review requests')).toBeInTheDocument()
  })
})
