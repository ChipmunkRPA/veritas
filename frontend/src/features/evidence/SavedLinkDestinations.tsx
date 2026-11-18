import type { DocumentSavedLinkHosts, DocumentSavedLinkLocation } from './multimediaPackets'

function sourceLabel(location: DocumentSavedLinkLocation) {
  switch (location.kind) {
    case 'word_paragraph':
      return `Word body path ${location.body_path.join(' / ')} · Original Word paragraph characters`
    case 'decoded_email_html':
      return `Email MIME part ${location.mime_part} · Decoded email HTML source characters`
    case 'decoded_html':
      return 'Original decoded HTML source characters'
  }
}

export default function SavedLinkDestinations({ review }: { review: DocumentSavedLinkHosts }) {
  if (!review.findings.length) return null
  return <section id="document-saved-link-hosts" aria-labelledby="document-saved-link-hosts-heading" className="mt-4 scroll-mt-20 rounded-md border border-border p-4">
    <h3 id="document-saved-link-hosts-heading" className="m-0 text-body font-medium">Saved link destinations</h3>
    <p className="mt-2 mb-0 text-small">Each link displays one hostname and stores a different destination hostname. Ordinary redirects can explain a difference.</p>
    <ul className="mt-3 mb-0 list-none space-y-3 p-0">
      {review.findings.map(({ label_host, saved_host, location }) => {
        const source = sourceLabel(location)
        const ranges = location.ranges.map(([start, end]) => `[${start}, ${end})`).join(', ')
        return <li key={`${source} ${ranges}`} className="border-t border-border pt-3">
          <dl className="m-0 grid gap-3 text-small sm:grid-cols-2">
            <div className="min-w-0">
              <dt className="font-medium">Displayed hostname</dt>
              <dd className="m-0"><bdi dir="ltr" className="break-all font-mono">{label_host}</bdi></dd>
            </div>
            <div className="min-w-0">
              <dt className="font-medium">Saved destination hostname</dt>
              <dd className="m-0"><bdi dir="ltr" className="break-all font-mono">{saved_host}</bdi></dd>
            </div>
          </dl>
          <p className="mt-2 mb-0 text-caption text-text-muted">{source} <span className="font-mono">{ranges}</span></p>
        </li>
      })}
    </ul>
    <p className="mt-2 mb-0 text-caption text-text-muted">Positions count Unicode code points starting at zero and exclude the end offset.</p>
    {review.omitted > 0 && <p className="mt-2 mb-0 text-caption text-text-muted">
      {review.omitted} additional {review.omitted === 1 ? 'difference exceeds' : 'differences exceed'} the display limit.
    </p>}
    <p className="mt-2 mb-0 text-small text-text-muted">The check compares normalized hostnames from the saved file without opening the links. The final destination remains unverified. This observation adds no weight to the AI authorship estimate.</p>
  </section>
}
