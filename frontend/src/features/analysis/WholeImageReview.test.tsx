// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import type { ExhibitMeta } from '../../lib/api/queries'
import WholeImageReview from './WholeImageReview'

afterEach(cleanup)

const table = (caption: string) => ({ kind: 'table', id: 'review', title: 'Supplementary whole-image check', caption } as ExhibitMeta)

it('keeps the measured view linked to the review and its displayed source coordinates', () => {
  const view = { kind: 'image', id: 'crop-1', title: 'Inspected image view 1', caption: 'x 5, y 8, 700 × 700 px' } as ExhibitMeta
  render(<WholeImageReview exhibits={[table('The inspected views support a whole-image review.'), view]} token="safe token" mediaReady />)
  expect(screen.getByRole('region', { name: 'Supplementary whole-image check' })).toBeTruthy()
  expect(screen.getByText(/x 5, y 8, 700 × 700 px/)).toBeTruthy()
  expect(screen.getByRole('img', { name: view.title }).getAttribute('src')).toContain('/exhibits/crop-1/payload?token=safe%20token')
})

it('shows a recorded skip even when there are no crops', () => {
  render(<WholeImageReview exhibits={[table('The check could not assess this file.')]} token="" mediaReady={false} />)
  expect(screen.getByText('The check could not assess this file.')).toBeTruthy()
  expect(screen.queryByRole('img')).toBeNull()
})
