import type { Kind } from '../lib/api'
import type { AnalysisProfile } from '../lib/api/intake'

const MB = 1024 * 1024

/** Analysis profiles (ARCHITECTURE §2.7; sampling in `engine/profiles.py`). '' leaves the choice to the instance default. */
export const PROFILES: { id: AnalysisProfile | ''; label: string; detail: string }[] = [
  { id: '', label: 'Instance default', detail: 'Uses this instance’s configured choice. Select a profile to see which checks are available.' },
  { id: 'quick', label: 'Quick', detail: 'An initial review with reduced sampling and the checks enabled for quick. Portions of a long recording may not be scored.' },
  { id: 'full', label: 'Full', detail: 'Broader coverage with the checks enabled for full, more video samples and available face, voice and lip-sync analysis. Takes longer than quick.' },
  { id: 'deep', label: 'Deep', detail: 'Additional checks and a larger analysis budget where supported. The slowest profile; more checks do not guarantee a conclusive result.' },
]

interface TabSpec {
  kind: Kind; label: string; noun: string; accept: string; exts: string[]; mimePrefix?: string; maxBytes: number; formats: string
}

export const TABS: TabSpec[] = [
  {
    kind: 'image', label: 'Image', noun: 'an image', accept: 'image/*,.apng,.heic,.heif', exts: ['jpg', 'jpeg', 'png', 'apng', 'webp', 'heic', 'heif', 'bmp', 'tif', 'tiff', 'gif'], mimePrefix: 'image/', maxBytes: 60 * MB,
    formats: 'JPEG · PNG · APNG · WebP · HEIC · up to 60 MB',
  },
  {
    kind: 'video', label: 'Video', noun: 'a video', accept: 'video/*,.mkv', exts: ['mp4', 'mov', 'webm', 'mkv', 'm4v', 'avi'], mimePrefix: 'video/', maxBytes: 500 * MB,
    formats: 'MP4 · MOV · WebM · MKV · up to 500 MB',
  },
  {
    kind: 'audio', label: 'Audio', noun: 'a recording', accept: 'audio/*,.m4a,.opus,.flac', exts: ['wav', 'mp3', 'm4a', 'ogg', 'opus', 'flac'], mimePrefix: 'audio/', maxBytes: 150 * MB,
    formats: 'WAV · MP3 · M4A · OGG · OPUS · FLAC · up to 150 MB',
  },
  {
    kind: 'document', label: 'Document', noun: 'a document', accept: '.pdf,.docx,.odt,.pptx,.odp,.xlsx,.ods,.csv,.tsv,.epub,.rtf,.txt,.md,.markdown,.html,.htm,.xhtml,.eml',
    exts: ['pdf', 'docx', 'odt', 'pptx', 'odp', 'xlsx', 'ods', 'csv', 'tsv', 'epub', 'rtf', 'txt', 'md', 'markdown', 'html', 'htm', 'xhtml', 'eml'], maxBytes: 25 * MB,
    formats: 'Word (DOCX) · PDF · ODT · PowerPoint (PPTX) · LibreOffice Impress (ODP) · Excel (XLSX) · LibreOffice Calc (ODS) · CSV/TSV · EPUB · RTF · TXT · Markdown · HTML/XHTML · Email (EML) · up to 25 MB',
  },
]

const sheetExport = 'Save the sheets containing writing as macro-free XLSX, CSV or PDF, then upload them.'
const delimitedCoverage = 'Writing is checked within each text field. Reports show the row and column, counting headers and blank records.'
const htmlCoverage = 'Saved web pages are checked without navigation and text explicitly marked hidden. Stylesheets and scripts are outside this check.'

const documentCoverage = new Map(Object.entries({
  docx: 'Word checks body paragraphs, supported tables, filled form controls, and simple text boxes. It also checks supported footnotes and endnotes. Export to PDF to include headers, footers, or text in complex layouts.',
  odt: 'OpenDocument checks the saved body text. Hidden and conditionally displayed sections are outside this check.',
  pdf: 'PDF checks selectable text and can try to recognize text on scans. Some pages or areas may remain unchecked.',
  pptx: 'PowerPoint checks editable text and tables on visible slides. Hidden groups are excluded. Export to PDF to include writing that appears only in images.',
  odp: 'Impress checks saved text in supported shapes and tables. Hidden slides, hidden layers and speaker notes are outside this check.',
  xlsx: 'Excel checks saved writing on visible sheets, rows and columns. Cells directly formatted to hide their values are excluded. Formulas, calculated results, comments and images are outside this check.',
  ods: 'OpenDocument spreadsheets check saved writing on visible sheets, rows and columns. Linked sheets, formulas, comments and images are outside this check.',
  csv: delimitedCoverage,
  tsv: delimitedCoverage,
  epub: 'EPUB checks the main reading text. Navigation, optional chapters, and text in images are outside this check.',
  rtf: 'Rich Text checks saved body text and leaves out directly hidden text. Layout and style-based visibility are outside this check.',
  html: htmlCoverage,
  htm: htmlCoverage,
  xhtml: htmlCoverage,
  eml: 'Email checks the message body. If the plain-text version is empty, it can read the matching HTML version. Upload attachments separately.',
}))

const fileExtension = (filename: string) => /\.([^./]+)$/.exec(filename)?.[1]?.toLowerCase() ?? ''

export function documentCoverageNotice(filename: string): string | undefined {
  return documentCoverage.get(fileExtension(filename))
}

const documentExport = new Map(Object.entries({
  gdoc: 'Download the Google Docs document as Word (DOCX) or PDF, then upload the downloaded file.',
  gsheet: 'Download the Google Sheets spreadsheet as Excel (XLSX) or PDF, then upload the downloaded file.',
  gslides: 'Download the Google Slides presentation as PowerPoint (PPTX) or PDF, then upload the downloaded file.',
  doc: 'Save the older Word file as DOCX or PDF, then upload it.',
  docm: 'Save the Word file as a macro-free DOCX or PDF, then upload it.',
  pages: 'Export the Apple Pages file as Word (DOCX) or PDF, then upload it.',
  ppt: 'Save the older PowerPoint file as PPTX or PDF, then upload it.',
  pptm: 'Save the presentation as a macro-free PPTX or PDF, then upload it.',
  key: 'Export the Keynote file as PowerPoint (PPTX) or PDF, then upload it.',
  msg: 'Export the Outlook message as EML or PDF, then upload it.',
  xls: sheetExport,
  xlsm: sheetExport,
  numbers: 'Export the Apple Numbers spreadsheet as Excel (XLSX), CSV or PDF, then upload the exported file.',
}))

export function isDocumentFilename(filename: string): boolean {
  const extension = fileExtension(filename)
  return TABS.some((tab) => tab.kind === 'document' && tab.exts.includes(extension)) || documentExport.has(extension)
}

export function validate(f: File, t: TabSpec): string | null {
  const ext = fileExtension(f.name)
  const mimeOk = t.mimePrefix ? f.type.startsWith(t.mimePrefix) : false
  if (!mimeOk && !t.exts.includes(ext)) {
    const exportHelp = t.kind === 'document' ? documentExport.get(ext) : undefined
    return exportHelp ?? `Unsupported type. Accepted: ${t.formats.split(' · ').slice(0, -1).join(', ')}`
  }
  if (f.size > t.maxBytes) return `File exceeds ${Math.round(t.maxBytes / MB)} MB`
  if (f.size === 0) return 'File is empty'
  return null
}

export const intakeKind = (value: string | null): Kind => TABS.find((tab) => tab.kind === value)?.kind ?? 'image'

/** Why "Begin analysis" is switched off, in one sentence (N-19a; design review, P2 "screens" 10: the button was disabled
 *  without a reason before a file was chosen). Null when the form can be sent. */
export function beginBlocked(s: { kind: Kind; hasFile: boolean; words: number; minWords: number; batch: number; maxBatch: number; problem: string | null }): string | null {
  if (s.batch > 1) return s.batch > s.maxBatch ? `A batch holds at most ${s.maxBatch} files: remove ${s.batch - s.maxBatch} to submit it.` : null
  if (!s.hasFile) {
    if (s.kind !== 'document') return `Choose ${TABS.find((t) => t.kind === s.kind)!.noun} to begin.`
    if (s.words === 0) return `Choose a document or paste at least ${s.minWords} words to begin.`
    if (s.words < s.minWords) return `The pasted text has ${s.words} ${s.words === 1 ? 'word' : 'words'}: ${s.minWords} are needed to begin.`
  }
  return s.problem ? 'Correct the case and custody fields to begin.' : null
}
