import type { ReactNode } from 'react'
import { cx } from '../lib/cx'

/**
 * A Markdown record of the repository shown as a document (N-10, D-71): headings, paragraphs, lists, tables, fenced code,
 * quotes and rules, with the inline marks for code, strong, emphasis and links.
 *
 * Every element is built from the text; nothing of the record is interpreted as HTML, so markup inside it prints as text.
 * A link is followed only to an http(s) address: a repository path is no address of this application and prints as its text.
 * What the reader does not know stays a paragraph, word for word.
 */

export type Block =
  | { kind: 'heading'; level: number; text: string }
  | { kind: 'paragraph'; text: string }
  | { kind: 'list'; ordered: boolean; items: { text: string; depth: number }[] }
  | { kind: 'table'; head: string[]; rows: string[][] }
  | { kind: 'code'; text: string }
  | { kind: 'quote'; text: string }
  | { kind: 'rule' }

const FENCE = /^\s*(```+|~~~+)/
const HEADING = /^(#{1,6})\s+(.*?)(?:\s+#+)?\s*$/
const RULE = /^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/
const ITEM = /^(\s*)(?:[-*+]|(\d+)[.)])\s+(.*)$/
const QUOTE = /^\s*>\s?(.*)$/
const DIVIDER = /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$/

/** The cells of a table row: split at each `|` that no backslash escapes, without the outer pipes. */
export function cells(line: string): string[] {
  const out: string[] = []
  let cell = ''
  const row = line.trim().replace(/^\|/, '')
  for (let i = 0; i < row.length; i++) {
    if (row[i] === '\\' && row[i + 1] === '|') { cell += '\\|'; i++ } else if (row[i] === '|') { out.push(cell.trim()); cell = '' } else cell += row[i]
  }
  if (cell.trim() !== '') out.push(cell.trim())
  return out
}

const isTableStart = (lines: string[], i: number): boolean => lines[i].includes('|') && i + 1 < lines.length && DIVIDER.test(lines[i + 1]) && lines[i + 1].includes('-')
const startsBlock = (lines: string[], i: number): boolean =>
  FENCE.test(lines[i]) || HEADING.test(lines[i]) || RULE.test(lines[i]) || ITEM.test(lines[i]) || QUOTE.test(lines[i]) || isTableStart(lines, i)

export function parseBlocks(source: string): Block[] {
  const lines = source.replace(/\r\n?/g, '\n').split('\n')
  const blocks: Block[] = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (line.trim() === '') { i++; continue }
    const fence = FENCE.exec(line)
    if (fence) {
      const body: string[] = []
      i++
      while (i < lines.length && !lines[i].trim().startsWith(fence[1])) body.push(lines[i++])
      i++  // the closing fence, or the end of the record
      blocks.push({ kind: 'code', text: body.join('\n') })
      continue
    }
    const heading = HEADING.exec(line)
    if (heading) { blocks.push({ kind: 'heading', level: heading[1].length, text: heading[2] }); i++; continue }
    if (isTableStart(lines, i)) {
      const head = cells(line)
      const rows: string[][] = []
      i += 2
      while (i < lines.length && lines[i].trim() !== '' && lines[i].includes('|')) rows.push(cells(lines[i++]))
      blocks.push({ kind: 'table', head, rows })
      continue
    }
    if (RULE.test(line)) { blocks.push({ kind: 'rule' }); i++; continue }
    const first = ITEM.exec(line)
    if (first) {
      const items: { text: string; depth: number }[] = []
      const ordered = first[2] !== undefined
      while (i < lines.length) {
        const item = ITEM.exec(lines[i])
        if (item && (item[2] !== undefined) !== ordered && item[1].length < 2) break  // a list of the other kind begins
        if (item) { items.push({ text: item[3], depth: Math.min(3, Math.floor(item[1].length / 2)) }); i++; continue }
        // an indented line continues the item above it; a blank line ends the list unless an item of its kind follows it
        if (lines[i].trim() !== '' && /^\s{2,}/.test(lines[i]) && !FENCE.test(lines[i])) { items[items.length - 1].text += ` ${lines[i].trim()}`; i++; continue }
        const next = lines[i].trim() === '' && i + 1 < lines.length ? ITEM.exec(lines[i + 1]) : null
        if (next && (next[2] !== undefined) === ordered) { i++; continue }
        break
      }
      blocks.push({ kind: 'list', ordered, items })
      continue
    }
    if (QUOTE.test(line)) {
      const body: string[] = []
      while (i < lines.length && QUOTE.test(lines[i])) body.push((QUOTE.exec(lines[i++]) as RegExpExecArray)[1])
      blocks.push({ kind: 'quote', text: body.join(' ') })
      continue
    }
    const body = [line.trim()]
    i++
    while (i < lines.length && lines[i].trim() !== '' && !startsBlock(lines, i)) body.push(lines[i++].trim())
    blocks.push({ kind: 'paragraph', text: body.join(' ') })
  }
  return blocks
}

// an escaped character | a code span | strong | emphasis | a link. Inside strong and emphasis an escaped character is one unit, so
// `\*` neither opens nor closes them; the address of a link may hold one pair of parentheses.
const UNIT = String.raw`(?:[^*\\]|\\.)`
const EDGE = String.raw`(?:[^\s*\\]|\\.)`
const INLINE = new RegExp([
  String.raw`\\([\\\`*_{}[\]()#+\-.!|~<>])`,
  String.raw`(\`+)([^\`]|[^\`][\s\S]*?[^\`])\2(?!\`)`,
  String.raw`\*\*(${EDGE}(?:${UNIT}|\*(?!\*))*?)\*\*`,
  String.raw`\*(${EDGE}(?:${UNIT}*${EDGE})?)\*`,
  String.raw`\[([^\]\n]+)\]\(((?:[^()\s]|\([^()\s]*\))+)\)`,
].join('|'), 'g')

export function inline(text: string, key = 'i'): ReactNode[] {
  const out: ReactNode[] = []
  let last = 0
  let n = 0
  for (const m of text.matchAll(INLINE)) {
    const at = m.index ?? 0
    if (at > last) out.push(text.slice(last, at))
    const k = `${key}-${n++}`
    if (m[1] !== undefined) out.push(m[1])
    else if (m[3] !== undefined) out.push(<code key={k} className="font-mono text-[0.92em] break-words">{m[3].trim()}</code>)
    else if (m[4] !== undefined) out.push(<strong key={k} className="font-medium text-text">{inline(m[4], k)}</strong>)
    else if (m[5] !== undefined) out.push(<em key={k}>{inline(m[5], k)}</em>)
    else if (/^https?:\/\//i.test(m[7])) out.push(<a key={k} href={m[7]} target="_blank" rel="noreferrer" className="text-accent underline underline-offset-[3px] break-all">{inline(m[6], k)}</a>)
    else out.push(...inline(m[6], k))
    last = at + m[0].length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

const INDENT = ['', 'ml-4', 'ml-8', 'ml-12']

function BlockView({ block, id }: { block: Block; id: string }) {
  switch (block.kind) {
    case 'heading':
      if (block.level === 1) return <h3 className="m-0 text-body font-medium text-text max-w-[80ch]">{inline(block.text, id)}</h3>
      if (block.level === 2) return <h4 className="m-0 mt-3 eyebrow">{inline(block.text, id)}</h4>
      return <h5 className="m-0 mt-1 text-small font-medium text-text">{inline(block.text, id)}</h5>
    case 'paragraph':
      return <p className="m-0 max-w-[80ch] break-words">{inline(block.text, id)}</p>
    case 'list': {
      const Tag = block.ordered ? 'ol' : 'ul'
      return (
        <Tag className={cx('m-0 pl-5 space-y-1 max-w-[80ch]', block.ordered ? 'list-decimal' : 'list-disc')}>
          {block.items.map((item, i) => <li key={i} className={cx('break-words', INDENT[item.depth])}>{inline(item.text, `${id}-${i}`)}</li>)}
        </Tag>
      )
    }
    case 'table':
      return (
        <div className="overflow-x-auto max-w-full">
          <table className="w-full min-w-[640px] border-collapse text-small">
            <thead>
              <tr className="border-b border-border-strong">
                {block.head.map((h, i) => <th key={i} scope="col" className="eyebrow text-left font-medium py-1.5 pr-4 align-bottom">{inline(h, `${id}-h${i}`)}</th>)}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, r) => (
                <tr key={r} className="border-b border-border">
                  {block.head.map((_, c) => <td key={c} className="py-1.5 pr-4 align-top break-words">{inline(row[c] ?? '', `${id}-${r}-${c}`)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
    case 'code':
      return <pre className="m-0 max-w-full overflow-x-auto bg-surface-2 border border-border rounded-sm p-3 font-mono text-caption leading-[18px] text-text whitespace-pre">{block.text}</pre>
    case 'quote':
      return <blockquote className="m-0 pl-3 border-l border-border-strong text-text-muted max-w-[80ch]">{inline(block.text, id)}</blockquote>
    case 'rule':
      return <hr className="m-0 border-0 border-t border-border" />
  }
}

export function Markdown({ source, className, testId }: { source: string; className?: string; testId?: string }) {
  return (
    <div data-testid={testId} className={cx('flex flex-col gap-3 min-w-0 text-small leading-[20px] text-text', className)}>
      {parseBlocks(source).map((block, i) => <BlockView key={i} block={block} id={`b${i}`} />)}
    </div>
  )
}
