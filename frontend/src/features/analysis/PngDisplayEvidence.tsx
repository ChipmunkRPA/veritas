import type { ExhibitMeta } from '../../lib/api/queries'
import { exhibitUrl } from '../viewer/useBase'

const checks = [
  { id: 'png-transparency', title: 'Color stored under PNG transparency',
    pictures: ['Crop as displayed on white', 'Stored colors revealed in the same crop'] },
  { id: 'png-animation', title: 'Animated PNG opening comparison',
    pictures: ['Static fallback image', 'First animation frame'] },
]

export default function PngDisplayEvidence({ exhibits, token, mediaReady }: {
  exhibits: readonly ExhibitMeta[]
  token: string
  mediaReady: boolean
}) {
  return <>{checks.map(({ id, title, pictures }) => {
    const table = exhibits.find(item => item.kind === 'table' && item.title === title)
    if (!table) return null
    const crops = pictures.flatMap(name => {
      const crop = exhibits.find(item => item.kind === 'image' && item.title === name)
      return crop ? [crop] : []
    })

    return <section key={id} id={id} aria-labelledby={id + '-heading'}
      className="space-y-4 border-t border-border pt-8 scroll-mt-14">
      <h2 className="m-0 eyebrow" id={id + '-heading'}>{title}</h2>
      <p className="m-0 max-w-[76ch] text-body">{table.caption}</p>
      {crops.length === 2 && <div className="grid max-w-2xl grid-cols-1 gap-4 sm:grid-cols-2">
        {crops.map(crop => <figure key={crop.id} className="m-0">
          {mediaReady && <img src={exhibitUrl(crop.id, 'payload', token)} alt={crop.title} loading="lazy"
            className="max-h-56 max-w-full rounded-sm border border-border object-contain" />}
          <figcaption className="mt-2 text-caption text-text-muted">{crop.title}{crop.caption && '. ' + crop.caption}</figcaption>
        </figure>)}
      </div>}
    </section>
  })}</>
}
