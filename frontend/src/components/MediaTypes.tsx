import { AudioLines, FileText, Image, Video, type LucideIcon } from 'lucide-react'
import type { Kind } from '../lib/api'

/** Navigation describes supported inputs, not a promise that a detector will produce a conclusion. */
export const MEDIA_TYPES: { kind: Kind; label: string; description: string; icon: LucideIcon }[] = [
  { kind: 'image', label: 'Image', description: 'Photographs & generated images', icon: Image },
  { kind: 'video', label: 'Video', description: 'Frames, faces & speech', icon: Video },
  { kind: 'audio', label: 'Audio', description: 'Recordings & synthetic voice', icon: AudioLines },
  { kind: 'document', label: 'Document', description: 'Documents & pasted text', icon: FileText },
]

export function MediaIcon({ kind, size = 20, className }: { kind: string; size?: number; className?: string }) {
  const Icon = MEDIA_TYPES.find((item) => item.kind === kind)?.icon ?? FileText
  return <Icon size={size} strokeWidth={1.5} aria-hidden className={className} />
}
