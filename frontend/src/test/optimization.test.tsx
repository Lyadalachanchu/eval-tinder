import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { makeRun } from './fixtures'
import { installFetch } from './mockFetch'
import { renderApp } from './render'

describe('optimization run screen', () => {
  it('labels agreement as development agreement, not production accuracy', async () => {
    installFetch([{ match: '/api/optimization-runs/run-1', body: makeRun() }])
    renderApp('/optimization-runs/run-1')

    expect(await screen.findByText('candidate 2')).toBeInTheDocument()
    expect(screen.getAllByText(/DEVELOPMENT AGREEMENT/).length).toBeGreaterThan(0)
    expect(screen.getAllByText('DEVELOPMENT AGREEMENT (frozen DEV snapshot), not production accuracy').length).toBeGreaterThan(0)
    expect(screen.getByText(/not evidence of production accuracy/)).toBeInTheDocument()
    expect(screen.getByText('ESTIMATE, NOT A GUARANTEE')).toBeInTheDocument()
    expect(screen.getAllByText('87.5%').length).toBeGreaterThan(0)
    expect(screen.getAllByText('RECOMMENDED').length).toBeGreaterThan(0)
    expect(screen.getByText(/baseline: always PASS/)).toBeInTheDocument()
  })

  it('shows the instruction text and colored diff for a selected candidate', async () => {
    installFetch([{ match: '/api/optimization-runs/run-1', body: makeRun() }])
    const { container } = renderApp('/optimization-runs/run-1')
    const user = userEvent.setup()

    await user.click(await screen.findByText('candidate 2'))
    expect(await screen.findByText('Diff from seed')).toBeInTheDocument()
    expect(container.querySelector('.diff-line.diff-add')?.textContent).toBe('+Be strict about cancellations.')
    expect(container.querySelector('.diff-line.diff-del')?.textContent).toBe('-Be strict.')
    expect(screen.getByRole('button', { name: 'Use as shadow grader' })).toBeDisabled()
    expect(screen.getByText(/never enable automation/)).toBeInTheDocument()
  })

  it('warns about insufficient class coverage', async () => {
    const run = makeRun()
    run.result_summary.comparison!.insufficient_class_coverage = true
    installFetch([{ match: '/api/optimization-runs/run-1', body: run }])
    renderApp('/optimization-runs/run-1')
    expect(await screen.findByText(/Insufficient class coverage/)).toBeInTheDocument()
  })
})
