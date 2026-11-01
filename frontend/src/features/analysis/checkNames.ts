import type { Finding, ModelCard, SignalV2 } from '../../lib/api/queries'

/** Pinned model-dependent detector identities. This fallback keeps saved findings readable even while the catalogue loads.
 * Source-media generator names are deliberately not included: they can be evidence about the file itself. */
const PINNED_NAMES: readonly string[] = [
  "Lip-sync lane: audio-visual offset and confidence (LatentSync StableSyncNet) per speaking face track",
  "Synthesis lane: Community Forensics (ViT-S/16) on native 384 px crops of clip frames",
  "Synthesis lane: DDA (DINOv2 ViT-L/14 + LoRA) on native 336 px crops of clip frames",
  "Synthesis lane: D3 second-order temporal statistic (X-CLIP B/16) per shot clip",
  "Face tracks: YuNet on every 2nd low-resolution frame, DINOv2-small appearance",
  "DFDC EfficientNet-B7 on face frames (selimsef, two-checkpoint ensemble)",
  "Binoculars zero-shot (Qwen2.5-1.5B observer / 1.5B-Instruct performer)",
  "RADAR (DINOv2 ViT-B/14 + Depth Anything V2 Base, diffusion inpainting)",
  "dima806 deepfake_vs_real_image_detection on face frames (ViT-B/16)",
  "haywoodsloan ai-image-detector-deploy per sampled frame (Swin-v2)",
  "Ateeqq ai-vs-human-image-detector per sampled frame (SigLIP)",
  "YuNet face detector on sampled frames (OpenCV Zoo 2023mar)",
  "DFDC EfficientNet-B7 (selimsef, two-checkpoint ensemble)",
  "Wvolf ViT_Deepfake_Detection on face frames (ViT-B/16)",
  "UnivFD per sampled frame (CLIP ViT-L/14 linear probe)",
  "mo-thecreator wav2vec2-base deepfake audio classifier",
  "Community Forensics (ViT-S/16, native 384 px crops)",
  "Face-swap lane: GenD (CLIP ViT-L/14) per face track",
  "desklib-ft-v2 (Veritas fine-tune, DeBERTa-v3-large)",
  "dima806 deepfake_vs_real_image_detection (ViT-B/16)",
  "Mesorch (ConvNeXt-T + SegFormer MiT-B3, DCT bands)",
  "DDA (DINOv2 ViT-L/14 + LoRA, native 336 px crops)",
  "desklib ai-text-detector v1.01 (DeBERTa-v3-large)",
  "Gustking wav2vec2-XLSR deepfake audio classifier",
  "SPAI (ViT-B/16 MFM, spectral, native resolution)",
  "haywoodsloan ai-image-detector-deploy (Swin-v2)",
  "Ateeqq ai-vs-human-image-detector (SigLIP)",
  "GenD (CLIP ViT-L/14, aligned face crops)",
  "YuNet face detector (OpenCV Zoo 2023mar)",
  "Hello-SimpleAI chatgpt-detector-roberta",
  "Wvolf ViT_Deepfake_Detection (ViT-B/16)",
  "UnivFD (CLIP ViT-L/14 linear probe)",
  "vid.synth.frames.commfor384",
  "vid.frames.hf.haywoodsloan",
  "vid.lipsync.stable_syncnet",
  "vid.tracks.yunet_dinov2",
  "aud.mo_thecreator_w2v2",
  "vid.swap.gend_clip_l14",
  "txt.hf.hello_simpleai",
  "vid.frames.hf.dima806",
  "vid.frames.hf.ateeqq",
  "vid.synth.frames.dda",
  "img.hf.haywoodsloan",
  "vid.frames.hf.wvolf",
  "face.gend_clip_l14",
  "vid.frames.dfdc_b7",
  "vid.synth.d3_xclip",
  "aud.gustking_xlsr",
  "txt.desklib_ft_v2",
  "vid.frames.univfd",
  "img.faces.yunet",
  "vid.faces.yunet",
  "img.commfor384",
  "img.hf.dima806",
  "txt.binoculars",
  "img.hf.ateeqq",
  "img.hf.wvolf",
  "img.dfdc_b7",
  "loc.mesorch",
  "txt.desklib",
  "img.univfd",
  "loc.radar",
  "img.spai",
  "img.dda"
]

// Abbreviations and model-specific variants used in older persisted finding sentences.
const SHORT_NAMES = [
  'DFDC EfficientNet-B7', 'Community Forensics', 'Ateeqq ai-vs-human-image-detector',
  'dima806 deepfake_vs_real_image_detection', 'haywoodsloan ai-image-detector-deploy',
  'Wvolf ViT_Deepfake_Detection', 'desklib-ft-v2', 'desklib ai-text-detector v1.01',
  'Hello-SimpleAI chatgpt-detector-roberta', 'Gustking wav2vec2-XLSR deepfake audio classifier',
  'mo-thecreator wav2vec2-base deepfake audio classifier', 'YuNet face detector',
  'LatentSync StableSyncNet', 'DINOv2-small', 'Binoculars', 'Mesorch', 'UnivFD', 'GenD', 'SPAI', 'RADAR', 'DDA',
]

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** Hide the identities of the detector models, not names declared by the source or other evidence.
 * Replacement is display-only: stored findings and evidence identifiers are never modified. */
export function anonymizeModelNames(text: string, cards: readonly ModelCard[] = [], signals: readonly SignalV2[] = []): string {
  const learned = cards.filter((card) => (card.model_entries?.length ?? 0) > 0 || (card.models?.length ?? 0) > 0)
  const ids = new Set(learned.map((card) => card.id))
  const signalNames = signals.filter((signal) => ids.has(signal.detector_id) || signal.category === 'model').map((signal) => signal.name)
  const names = [...PINNED_NAMES, ...SHORT_NAMES, ...learned.flatMap((card) => [card.name, card.id]), ...signalNames]
    .flatMap((name) => {
      const trimmed = name.trim()
      const withoutArchitecture = trimmed.replace(/\s*\([^()]*\)$/, '').trim()
      return [trimmed, withoutArchitecture]
    }).filter((name) => name.length >= 3)
  // Longest first so a short acronym cannot leave a model's longer name behind.
  const unique = [...new Set(names)].sort((a, b) => b.length - a.length)
  return unique.reduce((value, name) => value.replace(
    new RegExp(`(?<![\\p{L}\\p{N}_])${escapeRegExp(name)}(?![\\p{L}\\p{N}_])`, 'gu'),
    (_match, offset: number, whole: string) => {
      const before = whole.slice(0, offset)
      if (/\b(?:the|a|an)\s+$/i.test(before)) return 'detection check'
      return offset === 0 || /(?:^|[.!?]\s+)$/.test(before) ? 'A detection check' : 'a detection check'
    },
  ), text)
}

/** Provenance and metadata findings may quote source material; keep its names intact. */
export function anonymizeFindingText(finding: Pick<Finding, 'family'>, text: string, cards: readonly ModelCard[] = [], signals: readonly SignalV2[] = []): string {
  if (['F1_provenance', 'F2_self_declared_metadata', 'F8_metadata_consistency'].includes(finding.family)) return text
  return anonymizeModelNames(text, cards, signals)
}

/** For detector-authored reference descriptions only, never for source observations or quoted metadata. */
export function anonymizeDetectorDescription(text: string, cards: readonly ModelCard[] = [], signals: readonly SignalV2[] = []): string {
  const withoutArchitecture = text.replace(/DINOv2 ViT-L\/14 with LoRA/g, 'A detection model').replace(/\s*\(Dual Data Alignment\)/g, '')
  return anonymizeModelNames(withoutArchitecture, cards, signals)
}
