import type { ExhibitMeta } from '../../lib/api/queries'
import { exhibitUrl } from '../viewer/useBase'

const title = 'Supplementary whole-image check'

export default function WholeImageReview({ exhibits, token, mediaReady }: {
  exhibits: readonly ExhibitMeta[]
  token: string
  mediaReady: boolean
}) {
  const review = exhibits.find(item => item.kind === 'table' && item.title === title)
  if (!review) return null

  const crops = [1, 2, 3].map(number => exhibits.find(item =>
    item.kind === 'image' && item.title === `Inspected image view ${number}`))

  return <section id="whole-image-review" aria-labelledby="whole-image-review-heading"
    className="space-y-4 border-t border-border pt-8 scroll-mt-14">
    <h2 className="m-0 eyebrow" id="whole-image-review-heading">{title}</h2>
    <p className="m-0 max-w-[76ch] text-body">{review.caption}</p>
    {crops.some(Boolean) && <div className="grid gap-4 sm:grid-cols-3">
      {crops.map(item => item && <figure key={item.id} className="m-0">
        {mediaReady && <img src={exhibitUrl(item.id, 'payload', token)} alt={item.title} loading="lazy"
          width={528} height={528} className="h-auto w-full rounded-sm object-contain" />}
        <figcaption className="mt-2 text-caption text-text-muted">{item.title} · {item.caption}</figcaption>
      </figure>)}
    </div>}
  </section>
}
