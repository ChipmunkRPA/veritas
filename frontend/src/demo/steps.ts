import type { DemoAnalysis, TourStep } from './types'

const A = (id: string) => `/analyses/${id}`
const EVIDENCE = (id: string) => `/analyses/${id}/evidence`
const T = (name: string) => `[data-tour="${name}"]`
const S = (section: string) => `#section-${section}`
const ID = (testId: string) => `[data-testid="${testId}"]`

/** L-2: the camera-photograph step describes the decision the screen shows. The served image model can call this photograph AI:
 *  that is a false positive, and the step says so with the measured reasons (`docs/limitations.md`, "Operating point"). */
export function cameraPhotoBody(code: string | null | undefined): string {
  if (code === 'ai_supported' || code === 'ai_supported_strong') {
    return 'This is a camera original, and the model calls it AI-generated: a false positive. The decision threshold was set on calibration images and does not transfer exactly: it flags 7.5 % of the benchmark’s non-AI test images against the 5 % it targets. In the field test 5 of 52 camera originals received an AI decision (9.6 %, Wilson interval 4.2–20.6 %), against 3.6 % of the benchmark’s camera originals; with so few photographs the two do not differ significantly (Fisher exact test, p = 0.06). Treat a flag as a reason to check the source, never as proof.'
  }
  if (code === 'inconclusive' || code === 'not_assessable') {
    return 'The same pipeline on a camera original: the checks do not reach a firm assessment either way. That is not a finding about the photograph; the answer says why, and the next step says what would help.'
  }
  if (code === 'no_ai_indicators' || code === 'no_ai_indicators_strong') {
    return 'The same pipeline on a camera original: the checks find no AI indicators. That does not confirm the photograph is unedited; detectors miss some AI content.'
  }
  return 'The same pipeline on a camera original. Read the answer: it states what the checks found, and a flag on a real photograph is a known kind of error.'
}

/** N-17 (browser QA P1-5): the face-swap step says what the served model does with the demo clip, as the photograph's step does.
 *  The clip is a face swap; an answer that does not flag it is a miss, and the step says so. The rate is the benchmark's
 *  (`docs/limitations.md`, "Video fuser"). */
export function faceSwapVideoBody(code: string | null | undefined): string {
  if (code === 'ai_supported' || code === 'ai_supported_strong') {
    return 'This clip is a face swap, and the model flags it. Faces are tracked through the clip and the sampled frames are scored; the key findings name the checks that moved the estimate. A flag is a reason to examine the clip, never proof.'
  }
  if (code === 'not_assessable') {
    return 'This clip is a face swap, and the model could not assess it: no estimate was formed, so the swap goes unflagged. The answer lists the reasons, and the timeline lanes show what each check recorded, for inspection only.'
  }
  if (code === 'inconclusive') {
    return 'This clip is a face swap, and the model does not flag it: the answer is inconclusive, which for this clip is a miss. That is common for video: 50.2 % of the benchmark’s test videos (149 of 297) come out inconclusive under the served policy. The answer gives the reason for this clip, and the timeline lanes show what each check found on it.'
  }
  if (code === 'no_ai_indicators' || code === 'no_ai_indicators_strong') {
    return 'This clip is a face swap, and the model finds no indications of it: a false negative. Detectors miss some manipulated video, so a clean answer never confirms that a clip is unedited. The timeline lanes show what each check found on it.'
  }
  return 'A face-swap clip through the video pipeline: faces are tracked and the sampled frames are scored. Read the answer: it states what the checks found, and a miss on a manipulated clip is a known kind of error.'
}

/**
 * Builds the walkthrough of the v2 workspace (E-8) from the samples returned by `/api/demo/start`, in the order the backend
 * supplies them: AI portrait, camera photograph, face-swap video, AI-written answer, human-written answer. A sample whose
 * analysis failed contributes no steps (the provider re-builds when statuses change). Steps on sections a sample may not have
 * (the viewer, lanes, methods, review) are optional: the overlay skips them when the anchor never appears.
 */
export function buildSteps(items: DemoAnalysis[]): TourStep[] {
  const steps: TourStep[] = []
  const byKind = (kind: string) => items.map((it, i) => ({ it, i })).filter((x) => x.it.kind === kind && x.it.status !== 'failed')
  const [imgAi, imgReal] = byKind('image')
  const [video] = byKind('video')
  const [docAi, docHuman] = byKind('document')

  if (imgAi) {
    const r = A(imgAi.it.id), item = imgAi.i
    steps.push(
      { key: 'img1-answer', route: r, item, intro: true, selector: T('verdict'), title: 'The answer', body: 'What the checks found, in plain words: the headline, then the number this result can state (a calibrated likelihood with its range, or a score that says it is not calibrated), what it means and what it does not, and the next step. It is an estimate, never proof.' },
      { key: 'img1-scale', route: r, item, selector: T('scale'), optional: true, title: 'Where it falls', body: 'The scale runs from no indications to strong indications. The marker is this exhibit’s estimate. Under a calibrated result the headline follows the calibrated levels for inputs like this one, not the segment the marker is in; a headline of one answer also needs the marker in that answer’s part of the scale. Under it the page says how often a result like this was wrong in testing, or that the rate is not established.' },
      { key: 'img1-findings', route: r, item, selector: S('findings'), title: 'Key findings', body: 'The observations that count for this result, each with its direction and strength, a plain reason and the evidence ids it rests on. One that had little effect says so.' },
      { key: 'img1-exhibit', route: r, item, selector: S('exhibit'), optional: true, title: 'The exhibit', body: 'The image with the lenses the findings cite. Open the evidence viewer to inspect them at full resolution.' },
      { key: 'img1-viewer', route: EVIDENCE(imgAi.it.id), item, selector: ID('lens-rail'), optional: true, ms: 10000, title: 'Evidence viewer', body: 'Heat maps with their reliability mask, regions, faces and notes over the original pixels. Zoom, probe a value, compare with the source; the view is kept in the link.' },
      { key: 'img1-reached', route: r, item, selector: S('reached'), optional: true, title: 'How it was reached', body: 'The bars start from the model’s estimate with every check at its neutral point, which is a property of the model and not evidence about the file; each check then adds or removes weight from its own neutral point. The bars add up to the answer, or the section explains why they cannot.' },
      { key: 'img1-evidence', route: r, item, selector: T('evidence'), action: { type: 'expand-first-row' }, ms: 10000, title: 'Every signal', body: 'Each detector with what it observed against its reference, its probability, or its raw score where the check is not calibrated, its contribution and its reliability. Detectors that did not run are listed with the reason. The first row is open.' },
      { key: 'img1-methods', route: r, item, selector: S('methods'), optional: true, title: 'Methods and validation', body: 'Whether this media falls inside the validated envelope, the measured error rates with their intervals, and "not established" wherever no measurement exists.' },
      { key: 'img1-review', route: r, item, selector: S('review'), optional: true, title: 'Examiner review', body: 'The examiner layer: dispositions on each finding, numbered notes, a conclusion and sign-off, which locks the analysis. Until then every surface says it has not been reviewed.' },
    )
  }
  if (imgReal) {
    const r = A(imgReal.it.id), item = imgReal.i
    steps.push(
      { key: 'img2-answer', route: r, item, intro: true, selector: T('verdict'), title: 'A camera photograph', body: cameraPhotoBody(imgReal.it.decision_code) },
      { key: 'img2-provenance', route: r, item, selector: S('provenance'), optional: true, title: 'Capture metadata', body: 'What the file says about itself: every metadata field as recorded (a camera file can hold hundreds of maker-note rows), content credentials when present, and any anomalies. Metadata is easy to strip or edit, so it is shown for the examiner and does not enter the fused estimate.' },
    )
  }
  if (video) {
    const r = A(video.it.id), item = video.i
    steps.push(
      { key: 'vid-answer', route: r, item, intro: true, selector: T('verdict'), title: 'A face-swap video', body: faceSwapVideoBody(video.it.decision_code) },
      { key: 'vid-exhibit', route: r, item, selector: S('exhibit'), optional: true, title: 'The video exhibit', body: 'A preview of the timeline lanes. A stretch that a lane flagged is listed below them as a segment; when no lane reached the threshold, the preview says so. Open the video workspace to step through the clip frame by frame.' },
      { key: 'vid-lanes', route: EVIDENCE(video.it.id), item, selector: ID('timeline'), optional: true, ms: 10000, title: 'Timeline lanes', body: 'One lane per face track, then lip-sync, audio, synthesis, codec and provenance. Click a segment to seek to its frame; shift-drag to note an interval.' },
      { key: 'vid-segments', route: EVIDENCE(video.it.id), item, selector: T('segments'), optional: true, title: 'Flagged segments', body: 'The segments that a lane flagged, ranked by uncalibrated score, each with its time range and the finding it supports. When no lane reached the segment threshold, as can happen with this clip, the list says so.' },
    )
  }
  if (docAi) {
    const r = A(docAi.it.id), item = docAi.i
    steps.push(
      { key: 'doc1-answer', route: r, item, intro: true, selector: T('share'), fallback: T('verdict'), title: 'Patterns across scored text', body: 'The bar groups scored text into AI-like, mixed and human-like patterns, weighted by text length. Review these classifications alongside the sentence evidence.' },
      { key: 'doc1-manuscript', route: r, item, selector: T('manuscript'), action: { type: 'select-first-ai-sentence' }, ms: 10000, title: 'Each sentence is scored', body: 'Highlighted by uncalibrated score band: tint and rule for AI-like, dotted for mixed. Click any sentence to see its evidence; one is selected for you.' },
      { key: 'doc1-rail', route: r, item, selector: T('rail'), optional: true, title: 'The sentence rail', body: 'Band, uncalibrated score, the reasons behind the call and the raw detector scores for the selected sentence.' },
    )
  }
  if (docHuman) {
    const r = A(docHuman.it.id), item = docHuman.i
    steps.push(
      { key: 'doc2-answer', route: r, item, intro: true, selector: T('verdict'), title: 'By contrast', body: 'A human-written answer through the same detectors. Compare its headline and its reasons with the document before it; the answer says whether its number is a calibrated likelihood or a score.' },
      { key: 'doc2-report', route: r, item, selector: T('report'), title: 'Take it with you', body: 'Every analysis exports a PDF report and the raw JSON: the same answer, findings and signals you see here.' },
    )
  }
  steps.push({ key: 'finish', route: '/', selector: ID('cases'), final: true, title: 'Cases keep the record', body: 'A quick analysis opens an unfiled case with its own custody log, members and reports; a named case files several exhibits under one matter. Try your own file.' })
  return steps
}
