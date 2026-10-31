/** Where the tour's card goes, and what stays lit (BUILD_SPEC N-19d; browser QA "Demo tour": the card covered part of what
 *  it highlighted on steps 1, 3, 6 to 13, 17 and 19). The card is put beside the part of the target that is in the window:
 *  below, above, right, left. A target that leaves no room on any side (a section taller than the window) has the card in
 *  the bottom-left corner, and the lit area ends above it, so the card always stands on the dimmed part of the page. */
export const GAP = 12
export const PAD = 8
const CORNER = 24

export type Box = { top: number; left: number; width: number; height: number }
export type Placement = { top: number; left: number; anchored: boolean }
export type Cut = { top: number; left: number; right: number; bottom: number }

export function place(t: Box, w: number, h: number, vw: number, vh: number): Placement {
  const clampX = (x: number) => Math.min(Math.max(GAP, x), vw - w - GAP)
  const clampY = (y: number) => Math.min(Math.max(GAP, y), vh - h - GAP)
  // the part of the target in the window, with the padding of the lit area
  const top = Math.max(0, t.top - PAD), bottom = Math.min(vh, t.top + t.height + PAD)
  const left = Math.max(0, t.left - PAD), right = Math.min(vw, t.left + t.width + PAD)
  // a target outside the window (before the page has scrolled to it) has the card in the corner
  if (bottom <= 0 || top >= vh || right <= 0 || left >= vw) return { top: vh - h - CORNER, left: CORNER, anchored: false }
  if (bottom + GAP + h <= vh) return { top: bottom + GAP, left: clampX(t.left), anchored: true }
  if (top - GAP - h >= 0) return { top: top - GAP - h, left: clampX(t.left), anchored: true }
  if (right + GAP + w <= vw) return { top: clampY(top), left: right + GAP, anchored: true }
  if (left - GAP - w >= 0) return { top: clampY(top), left: left - GAP - w, anchored: true }
  return { top: vh - h - CORNER, left: CORNER, anchored: false }
}

/** The lit area: the target with its padding, within the window; under a card that found no room beside the target it ends
 *  above the card. */
export function cutOut(t: Box, pos: Placement | null, vw: number, vh: number): Cut {
  const cut = { top: Math.max(0, t.top - PAD), left: Math.max(0, t.left - PAD), right: Math.min(vw, t.left + t.width + PAD), bottom: Math.min(vh, t.top + t.height + PAD) }
  if (pos && !pos.anchored) cut.bottom = Math.max(cut.top, Math.min(cut.bottom, pos.top - GAP))
  return cut
}

/** Whether a card of `w` × `h` at `pos` stands on the lit area. */
export function covers(pos: Placement, w: number, h: number, cut: Cut): boolean {
  return pos.left < cut.right && pos.left + w > cut.left && pos.top < cut.bottom && pos.top + h > cut.top
}
