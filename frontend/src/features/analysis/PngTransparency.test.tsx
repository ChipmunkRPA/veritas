// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest'
import { render, screen } from '@testing-library/react'
import { expect, it } from 'vitest'
import type { ExhibitMeta } from '../../lib/api/queries'
import PngTransparency from './PngTransparency'

function exhibit(id: string, kind: 'image' | 'table', title: string, caption: string): ExhibitMeta {
  return { id, analysis_id: 'example', kind, title, caption, inline: true, binaries: kind === 'image' ? ['payload'] : [] }
}

it('shows the same-source crop pair only when both crops and media access are available', () => {
  const table = exhibit('source', 'table', 'Color stored under PNG transparency', 'Checked 98,304 pixels.')
  const visible = exhibit('visible', 'image', 'Crop as displayed on white', 'Source x 256, y 128, 128 × 128 px.')
  const hidden = exhibit('hidden', 'image', 'Stored colors revealed in the same crop', visible.caption)
  const { rerender } = render(<PngTransparency exhibits={[table, visible]} token="access" mediaReady />)
  expect(screen.getByRole('heading')).toHaveTextContent(table.title)
  expect(screen.queryAllByRole('img')).toHaveLength(0)
  rerender(<PngTransparency exhibits={[table, visible, hidden]} token="access" mediaReady={false} />)
  expect(screen.queryAllByRole('img')).toHaveLength(0)
  rerender(<PngTransparency exhibits={[table, visible, hidden]} token="access" mediaReady />)
  expect(screen.getByRole('img', { name: visible.title })).toHaveAttribute('src', expect.stringContaining('/exhibits/visible/payload'))
  expect(screen.getByRole('img', { name: hidden.title })).toHaveAttribute('src', expect.stringContaining('/exhibits/hidden/payload'))
  expect(screen.getAllByText(/Source x 256, y 128/)).toHaveLength(2)
})
