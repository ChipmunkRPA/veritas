import type { HTMLAttributes, ReactNode } from 'react'
import { cx } from '../../lib/cx'

/** Section label: 12/16, letterspaced uppercase, muted. Renders as <p> unless `as` is given. */
export function Eyebrow({ as: Tag = 'p', className, children, ...rest }: HTMLAttributes<HTMLElement> & { as?: 'p' | 'span' | 'h2' | 'h3' | 'h4' | 'div' | 'dt'; children: ReactNode }) {
  return <Tag className={cx('eyebrow', className)} {...rest}>{children}</Tag>
}
