// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import type { ExhibitMeta } from '../../lib/api/queries'
import ImageDisplayEvidence from './ImageDisplayEvidence'

afterEach(cleanup)

function exhibit(id: string, kind: 'image' | 'table', title: string, caption: string): ExhibitMeta {
  return { id, analysis_id: 'example', kind, title, caption, inline: true, binaries: kind === 'image' ? ['payload'] : [] }
}

it('shows the same-source crop pair only when both crops and media access are available', () => {
  const table = exhibit('source', 'table', 'Color stored under PNG transparency', 'Checked 98,304 pixels.')
  const visible = exhibit('visible', 'image', 'Crop as displayed on white', 'Source x 256, y 128, 128 × 128 px.')
  const hidden = exhibit('hidden', 'image', 'Stored colors revealed in the same crop', visible.caption)
  const { rerender } = render(<ImageDisplayEvidence exhibits={[table, visible]} token="access" mediaReady />)
  expect(screen.getByRole('heading')).toHaveTextContent(table.title)
  expect(screen.queryAllByRole('img')).toHaveLength(0)
  rerender(<ImageDisplayEvidence exhibits={[table, visible, hidden]} token="access" mediaReady={false} />)
  expect(screen.queryAllByRole('img')).toHaveLength(0)
  rerender(<ImageDisplayEvidence exhibits={[table, visible, hidden]} token="access" mediaReady />)
  expect(screen.getByRole('img', { name: visible.title })).toHaveAttribute('src', expect.stringContaining('/exhibits/visible/payload'))
  expect(screen.getByRole('img', { name: hidden.title })).toHaveAttribute('src', expect.stringContaining('/exhibits/hidden/payload'))
  expect(screen.getAllByText(/Source x 256, y 128/)).toHaveLength(2)
})

it('shows the static fallback and animation opening with their changed source area', () => {
  const table = exhibit('comparison', 'table', 'Animated PNG opening comparison', '294 displayed pixels differ at source x 16, y 14.')
  const fallback = exhibit('fallback', 'image', 'Static fallback image', '')
  const opening = exhibit('opening', 'image', 'First animation frame', '')
  render(<ImageDisplayEvidence exhibits={[table, fallback, opening]} token="access" mediaReady />)
  expect(screen.getByRole('heading')).toHaveTextContent(table.title)
  expect(screen.getByText(/294 displayed pixels differ/)).toBeTruthy()
  expect(screen.getByRole('img', { name: opening.title })).toHaveAttribute('src', expect.stringContaining('/exhibits/opening/payload'))
  expect(screen.getByRole('img', { name: fallback.title })).toHaveAttribute('src', expect.stringContaining('/exhibits/fallback/payload'))
})

it('pairs a Motion Photo region with its time and the complete set of images', () => {
  const table = exhibit('motion', 'table', 'Motion Photo still and video', 'The displayed frame is at 1.169484 s.')
  const pictures = [
    exhibit('location', 'image', 'Highlighted area in the still', ''),
    exhibit('photo', 'image', 'Unmarked crop from the still', ''),
    exhibit('video', 'image', 'Corresponding crop from the matched video frame', ''),
    exhibit('mask', 'image', 'Measured difference (white)', ''),
  ]
  const { rerender } = render(<ImageDisplayEvidence exhibits={[table, ...pictures.slice(0, 3)]} token="access" mediaReady />)
  expect(screen.queryAllByRole('img')).toHaveLength(0)
  rerender(<ImageDisplayEvidence exhibits={[table, ...pictures]} token="access" mediaReady />)
  expect(screen.getByText('The displayed frame is at 1.169484 s.')).toBeTruthy()
  for (const picture of pictures) {
    expect(screen.getByRole('img', { name: picture.title })).toHaveAttribute('src', expect.stringContaining('/exhibits/' + picture.id + '/payload'))
  }
})
