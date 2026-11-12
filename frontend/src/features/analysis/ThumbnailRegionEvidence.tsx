import type { ExhibitMeta } from '../../lib/api/queries'
import { exhibitUrl } from '../viewer/useBase'

const tableTitle = 'Embedded preview difference locations'
const mainTitle = 'Main image crop for preview difference'
const previewTitle = 'Embedded preview crop for difference'

export default function ThumbnailRegionEvidence({ exhibits, token, mediaReady }: {
  exhibits: readonly ExhibitMeta[]
  token: string
  mediaReady: boolean
}) {
  const table = exhibits.find(item => item.kind === 'table' && item.title === tableTitle)
  if (!table) return null

  const byTitle = new Map(exhibits.filter(item => item.kind === 'image').map(item => [item.title, item]))
  const pairs = [1, 2, 3].flatMap(region => {
    const preview = byTitle.get(`${previewTitle} ${region}`)
    const main = byTitle.get(`${mainTitle} ${region}`)
    return preview && main ? [{ region, preview, main }] : []
  })
  if (pairs.length === 0) return null

  return <section aria-labelledby="embedded-preview-difference-heading" className="space-y-4 border-t border-border pt-8">
    <h2 className="m-0 eyebrow" id="embedded-preview-difference-heading">Embedded preview differences</h2>
    <p className="m-0 max-w-[76ch] text-body text-text">{table.caption}</p>
    {pairs.map(({ region, preview, main }) => <article key={region} className="space-y-2">
      <h3 className="m-0 text-small font-medium">Region {region}</h3>
      <div className="grid gap-4 sm:grid-cols-2">
        {[{ item: preview, label: 'Embedded preview crop' }, { item: main, label: 'Main image crop' }].map(({ item, label }) => <figure key={item.id} className="m-0">
          {mediaReady && <img src={exhibitUrl(item.id, 'payload', token)} alt={`${label} for region ${region}`} loading="lazy" className="max-h-56 max-w-full rounded-sm object-contain" />}
          <figcaption className="mt-2 text-caption text-text-muted">{label} · {item.caption}</figcaption>
        </figure>)}
      </div>
    </article>)}
  </section>
}
