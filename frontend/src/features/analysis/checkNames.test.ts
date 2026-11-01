import { describe, expect, it } from 'vitest'
import type { ModelCard, SignalV2 } from '../../lib/api/queries'
import { anonymizeDetectorDescription, anonymizeFindingText, anonymizeModelNames } from './checkNames'

describe('anonymizeModelNames', () => {
  it('removes full and shortened detector names in saved findings without changing scores', () => {
    expect(anonymizeModelNames('DFDC EfficientNet-B7 (selimsef, two-checkpoint ensemble) returned a score of 0.53.'))
      .toBe('A detection check returned a score of 0.53.')
    expect(anonymizeModelNames('DDA returned a score of -0.64; dima806 deepfake_vs_real_image_detection returned 0.94.'))
      .toBe('A detection check returned a score of -0.64; a detection check returned 0.94.')
  })

  it('handles an unknown future model from its card and signal', () => {
    const card = { id: 'img.newmodel', name: 'NewModel (rev 2)', models: ['newmodel'] } as ModelCard
    const signal = { detector_id: 'img.newmodel', category: 'model', name: 'NewModel (rev 2)' } as SignalV2
    expect(anonymizeModelNames('NewModel returned 0.7; img.newmodel was used.', [card], [signal]))
      .toBe('A detection check returned 0.7; a detection check was used.')
  })

  it('preserves source generator names and model-free evidence', () => {
    const evidence = 'The file mentions Stable Diffusion and Midjourney. C2PA credentials were not present. The radar was visible.'
    expect(anonymizeModelNames(evidence)).toBe(evidence)
  })

  it('does not rewrite model-like words quoted from source metadata', () => {
    const text = 'The file’s metadata mentions “RADAR” and “DDA”.'
    expect(anonymizeFindingText({ family: 'F2_self_declared_metadata' }, text)).toBe(text)
    expect(anonymizeFindingText({ family: 'F1_provenance' }, text)).toBe(text)
  })

  it('hides architecture details in detector references while keeping their meaning', () => {
    const reference = 'DINOv2 ViT-L/14 with LoRA trained on photographs (Dual Data Alignment); decision boundary at logit 0'
    expect(anonymizeDetectorDescription(reference)).toBe('A detection model trained on photographs; decision boundary at logit 0')
    expect(anonymizeFindingText({ family: 'F2_self_declared_metadata' }, reference)).toBe(reference)
  })
})
