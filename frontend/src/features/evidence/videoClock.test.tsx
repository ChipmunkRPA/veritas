// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { buildPtsIndex } from '../../lib/pts'
import { playbackSeekTime, useFramePlayer } from '../video/FramePlayer'

afterEach(cleanup)
const index = buildPtsIndex({ pts: [900000, 903000, 906000, 909000], time_base: '1/90000' })
const element = () => {
  const video = document.createElement('video')
  Object.defineProperty(video, 'readyState', { value: 1 })
  video.play = vi.fn().mockResolvedValue(undefined)
  video.pause = vi.fn()
  return video
}
describe('source clock to browser playback', () => {
  it('subtracts the exact recorded origin on seeks and adds it to browser clock updates', () => {
    const video = element()
    const { result } = renderHook(() => useFramePlayer(index, 1, '/source', 10))
    act(() => result.current.ref(video))
    expect(video.currentTime).toBeCloseTo(1 / 30 + 1 / 120, 8)
    act(() => { video.currentTime = 0.07; video.dispatchEvent(new Event('timeupdate')) })
    expect(result.current.frame).toBe(2)
    act(() => result.current.seekFrame(3))
    expect(video.currentTime).toBeCloseTo(0.1 + 1 / 120, 8)
  })
  it('maps requestVideoFrameCallback mediaTime and preserves the selected frame on token renewal', () => {
    const video = element() as HTMLVideoElement & { requestVideoFrameCallback: (cb: (now: number, meta: { mediaTime: number }) => void) => number }
    let callback: ((now: number, meta: { mediaTime: number }) => void) | undefined
    video.requestVideoFrameCallback = vi.fn((cb) => { callback = cb; return 1 })
    video.cancelVideoFrameCallback = vi.fn()
    const { result, rerender } = renderHook(({ source }) => useFramePlayer(index, null, source, 10), { initialProps: { source: '/source?token=first' } })
    act(() => result.current.ref(video))
    act(() => callback?.(0, { mediaTime: 0.08 }))
    expect(result.current.frame).toBe(2)
    rerender({ source: '/source?token=renewed' })
    expect(video.currentTime).toBeCloseTo(2 / 30 + 1 / 120, 8)
  })
  it('keeps unknown historical mappings unavailable instead of assuming a zero origin', () => {
    const video = element()
    const { result, rerender } = renderHook(({ origin }) => useFramePlayer(index, null, '/source', origin), { initialProps: { origin: null as number | null } })
    act(() => result.current.ref(video))
    act(() => { result.current.seekFrame(2); result.current.toggle() })
    expect(result.current.clockMapped).toBe(false)
    expect(video.currentTime).toBe(0)
    expect(video.play).not.toHaveBeenCalled()
    expect(result.current.frame).toBe(2)
    rerender({ origin: 10 })
    expect(video.currentTime).toBeCloseTo(2 / 30 + 1 / 120, 8)
    expect(playbackSeekTime(index, 0, null)).toBeNull()
    expect(playbackSeekTime(index, 0, 20)).toBeNull()
  })
  it('retains variable frame durations and negative source timestamps', () => {
    const variable = buildPtsIndex({ pts: [-100, -60, 10], time_base: '1/1000' })
    const video = element()
    const { result } = renderHook(() => useFramePlayer(variable, 1, '/source', -0.2))
    act(() => result.current.ref(video))
    expect(video.currentTime).toBeCloseTo(0.1575, 8)
    act(() => { video.currentTime = 0.22; video.dispatchEvent(new Event('timeupdate')) })
    expect(result.current.frame).toBe(2)
  })
})
