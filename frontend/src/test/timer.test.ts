import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useActiveReviewTimer } from '../hooks/useActiveReviewTimer'

describe('useActiveReviewTimer', () => {
  let clock = 0
  let visibility: DocumentVisibilityState = 'visible'

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('excludes time while the document is hidden', () => {
    vi.spyOn(performance, 'now').mockImplementation(() => clock)
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => visibility })

    const { result } = renderHook(() => useActiveReviewTimer('case-1'))
    clock = 1000
    expect(result.current.read()).toBe(1000)

    act(() => {
      visibility = 'hidden'
      document.dispatchEvent(new Event('visibilitychange'))
    })
    clock = 6000
    expect(result.current.read()).toBe(1000)

    act(() => {
      visibility = 'visible'
      document.dispatchEvent(new Event('visibilitychange'))
    })
    clock = 6500
    expect(result.current.read()).toBe(1500)
  })
})
