import type { Analysis } from '../../lib/api'

type Stats = NonNullable<NonNullable<Analysis['result']>['stats']>

/** Text-length-weighted model classifications (AI-like, mixed, human-like), shown under the v2 answer module and the v1 verdict. */
export default function DocumentShare({ stats, estimate }: { stats: Stats; estimate: number | null }) {
  if (estimate == null || !Number.isFinite(estimate) || estimate < 0 || estimate > 1) return null
  const scored = stats.scored_sentences ?? (stats.ai_sentences + stats.mixed_sentences + stats.human_sentences)
  if (!Number.isFinite(scored) || scored <= 0) return null
  const human = Math.max(0, 1 - stats.ai_share - stats.mixed_share)
  return (
    <div className="mt-6" data-tour="share">
      <p className="mt-0 mb-2 text-caption text-text-muted">Share of scored text, weighted by length · {scored} scored {scored === 1 ? 'sentence' : 'sentences'}</p>
      <div className="flex h-2 gap-px overflow-hidden" style={{ borderRadius: 1 }} role="img" aria-label={`Share of scored text: AI-like ${Math.round(stats.ai_share * 100)} percent, mixed ${Math.round(stats.mixed_share * 100)} percent, human-like ${Math.round(human * 100)} percent`}>
        <span className="bg-hl-ai-rule" style={{ width: `${stats.ai_share * 100}%` }} />
        <span className="bg-hl-mixed-rule" style={{ width: `${stats.mixed_share * 100}%` }} />
        <span className="bg-border flex-1" />
      </div>
      <p className="mt-2 text-small text-text-muted num">
        <span className="text-text">AI-like {Math.round(stats.ai_share * 100)}%</span> · {stats.ai_sentences} sentences
        <span className="mx-3 text-text-faint" aria-hidden>|</span>
        <span className="text-text">mixed {Math.round(stats.mixed_share * 100)}%</span> · {stats.mixed_sentences}
        <span className="mx-3 text-text-faint" aria-hidden>|</span>
        <span className="text-text">human-like {Math.round(human * 100)}%</span> · {stats.human_sentences}
      </p>
    </div>
  )
}
