import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { makeAudit } from './fixtures'
import { installFetch } from './mockFetch'
import { renderApp } from './render'

describe('audit report', () => {
  it('shows NOT_ESTIMABLE literally and disables enabling automation when the gate did not pass', async () => {
    installFetch([
      { match: '/api/audits/audit-1', body: makeAudit(false) },
      { match: '/api/projects/p1/automation-policy', body: { state: 'DISABLED', audit_id: null } },
    ])
    renderApp('/audits/audit-1')

    expect(await screen.findByText('GATE NOT PASSED')).toBeInTheDocument()
    expect(screen.getAllByText('NOT_ESTIMABLE').length).toBeGreaterThan(0)
    // The zero-denominator metric must read NOT_ESTIMABLE, never 0% or 100%.
    const recallRow = screen.getByText('failure_recall').closest('tr')
    expect(recallRow).not.toBeNull()
    expect(recallRow!.textContent).toContain('NOT_ESTIMABLE')
    expect(recallRow!.textContent).not.toMatch(/\d+\.\d%/)
    expect(screen.getAllByText('AUDIT EVIDENCE').length).toBeGreaterThan(0)

    const enable = screen.getByRole('button', { name: 'Enable automation' })
    expect(enable).toBeDisabled()
    expect(screen.getByLabelText('Enable automation reason')).toBeDisabled()
  })

  it('allows enabling automation with a reason once the gate passed', async () => {
    const { calls } = installFetch([
      { match: '/api/audits/audit-1', body: makeAudit(true) },
      { match: '/api/projects/p1/automation-policy', body: { state: 'DISABLED', audit_id: null } },
      {
        method: 'POST',
        match: '/api/projects/p1/automation-policy',
        body: { state: 'ENABLED', audit_id: 'audit-1', pipeline_hash: 'pipe-hash-1', reason: 'ok' },
      },
    ])
    renderApp('/audits/audit-1')
    const user = userEvent.setup()

    expect(await screen.findByText('GATE PASSED')).toBeInTheDocument()
    const enable = screen.getByRole('button', { name: 'Enable automation' })
    expect(enable).toBeDisabled()
    await user.type(screen.getByLabelText('Enable automation reason'), 'gate passed for the declared window')
    expect(enable).toBeEnabled()
    await user.click(enable)

    expect(await screen.findByText(/Automation policy is now ENABLED/)).toBeInTheDocument()
    const post = calls.find((c) => c.method === 'POST' && c.path === '/api/projects/p1/automation-policy')
    expect(post?.body).toMatchObject({ audit_id: 'audit-1', enable: true, reason: 'gate passed for the declared window' })
  })
})
