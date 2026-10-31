/** Batch submission and progress (`POST /api/v2/batches`, `GET /api/v2/batches/{id}`; PRD API-01; ARCHITECTURE §7.2 `/batches/:id`).
 *  The submission is an XHR, because fetch reports no upload progress. The server detects each file's kind from its contents. */
import { useQuery } from '@tanstack/react-query'
import { apiClient, authHeaders, unwrap } from './client'
import { intakeError, type AnalysisProfile } from './intake'
import type { components } from './schema'

export type Batch = components['schemas']['BatchOut']
export type BatchItem = components['schemas']['BatchItemOut']
export type BatchItemStatus = BatchItem['status']

/** Most files one batch holds (VERITAS_BATCH_MAX_ITEMS default). */
export const MAX_BATCH_FILES = 1000

export const batchKey = (id: string) => ['batch', id] as const

export function batchForm(files: readonly File[], caseId?: string, profile?: AnalysisProfile): FormData {
  const fd = new FormData()
  for (const f of files) fd.append('files', f, f.name)
  if (caseId) fd.append('case_id', caseId)
  if (profile) fd.append('profile', profile)
  return fd
}

export function submitBatch(files: readonly File[], caseId: string | undefined, onProgress?: (fraction: number) => void,
  profile?: AnalysisProfile): Promise<Batch> {
  return new Promise<Batch>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/v2/batches')
    for (const [k, v] of Object.entries(authHeaders('POST'))) xhr.setRequestHeader(k, v)
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress?.(e.loaded / e.total) }
    xhr.onerror = () => reject(new Error('Network error during upload'))
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText) as Batch) } catch { reject(new Error('Malformed response')) }
      } else {
        reject(new Error(intakeError(xhr.status, xhr.statusText, xhr.responseText)))
      }
    }
    xhr.send(batchForm(files, caseId, profile))
  })
}

/** Files of a drop, folders walked recursively; names keep their folder path (`set/a.png`). Dot-files are left out. */
export async function droppedFiles(dt: DataTransfer): Promise<File[]> {
  // Entries must be taken while the drop event is being dispatched, before the first await.
  const entries = Array.from(dt.items ?? []).map((item) => item.webkitGetAsEntry?.() ?? null).filter((e): e is FileSystemEntry => e !== null)
  if (entries.length === 0) return usable(Array.from(dt.files ?? []))
  const out: File[] = []
  const walk = async (entry: FileSystemEntry, prefix: string): Promise<void> => {
    if (entry.isFile) {
      const file = await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject))
      out.push(prefix ? new File([file], `${prefix}${file.name}`, { type: file.type, lastModified: file.lastModified }) : file)
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader()
      for (;;) {
        const chunk = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject))
        if (chunk.length === 0) break
        for (const child of chunk) await walk(child, `${prefix}${entry.name}/`)
      }
    }
  }
  for (const entry of entries) await walk(entry, '')
  return usable(out)
}

/** A folder picked with `webkitdirectory` keeps each file's relative path as its name. */
export function withRelativePaths(files: readonly File[]): File[] {
  return usable(files.map((f) => (f.webkitRelativePath && f.webkitRelativePath !== f.name
    ? new File([f], f.webkitRelativePath, { type: f.type, lastModified: f.lastModified }) : f)))
}

export function usable(files: readonly File[]): File[] {
  return files.filter((f) => !f.name.split('/').some((part) => part.startsWith('.')))
}

export const isBatchTerminal = (status: Batch['status'] | undefined): boolean => status === 'done' || status === 'failed'

const getBatch = (id: string) => unwrap(apiClient.GET('/api/v2/batches/{batch_id}', { params: { path: { batch_id: id } } }))

/** The batch, polled every 1.5 s while items are expanding or running. */
export function useBatch(id: string) {
  return useQuery({
    queryKey: batchKey(id), queryFn: () => getBatch(id), enabled: id !== '',
    refetchInterval: (query) => (isBatchTerminal(query.state.data?.status) ? false : 1500),
  })
}

/** Download a summary with the API credentials (a plain link would carry none). */
export async function downloadSummary(id: string, format: 'csv' | 'json'): Promise<void> {
  const response = await fetch(`/api/v2/batches/${encodeURIComponent(id)}/summary.${format}`, { headers: authHeaders() })
  if (!response.ok) throw new Error(intakeError(response.status, response.statusText, await response.text()))
  const url = URL.createObjectURL(await response.blob())
  const a = document.createElement('a')
  a.href = url
  a.download = `veritas-batch-${id.slice(0, 8)}.${format}`
  a.click()
  URL.revokeObjectURL(url)
}
