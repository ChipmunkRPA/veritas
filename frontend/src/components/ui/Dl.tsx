import type { CSSProperties, ReactNode } from 'react'
import { cx } from '../../lib/cx'

export interface DlRow { key: ReactNode; value: ReactNode; title?: string }

/**
 * Two-column definition list (Notion property rows). Keys are muted; values inherit.
 * `keyWidth` in px; `size` selects 13/20 (small) or 15/24 (body) values; `eyebrowKeys` sets keys in eyebrow style.
 * Below 640 px a list whose key column is 128 px or wider stacks each key above its value (L-7): the value keeps the full width.
 */
export function Dl({ rows, keyWidth = 120, size = 'small', eyebrowKeys = false, className, gap = 8 }: {
  rows: DlRow[]; keyWidth?: number; size?: 'small' | 'body' | 'caption'; eyebrowKeys?: boolean; className?: string; gap?: number
}) {
  const valueCls = size === 'body' ? 'text-body' : size === 'caption' ? 'text-caption' : 'text-small'
  const keyCls = eyebrowKeys ? 'eyebrow pt-[3px]' : cx(valueCls, 'text-text-muted')
  return (
    <dl className={cx('dl grid m-0', keyWidth >= 128 && 'dl-stack', className)} style={{ '--dl-key': `${keyWidth}px`, '--dl-gap': `${gap}px` } as CSSProperties}>
      {rows.map((r, i) => (
        <div key={i} className="contents" title={r.title}>
          <dt className={cx(keyCls, 'min-w-0')}>{r.key}</dt>
          <dd className={cx(valueCls, 'text-text m-0 min-w-0 break-words')}>{r.value}</dd>
        </div>
      ))}
    </dl>
  )
}
