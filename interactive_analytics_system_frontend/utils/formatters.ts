/** Formatting utilities for homography reprojection error display. */

/** Format a reprojection error (px) as a labelled string with pass/warn/fail symbol. */
export function reprErrorLabel(val: number | undefined): string {
  if (val === undefined) return '—'
  if (val < 10) return `${val}px ✓`
  if (val < 20) return `${val}px ⚠`
  return `${val}px ✗`
}

/** Return a CSS colour string for a reprojection error value. */
export function reprErrorColor(val: number | undefined): string {
  if (val === undefined) return ''
  if (val < 10) return '#2d7a2d'
  if (val < 20) return '#b8860b'
  return '#cc2222'
}

/** Format an overall homography quality string as a labelled badge. */
export function qualityBadge(q: string): string {
  if (q === 'good') return '✅ good'
  if (q === 'warning') return '⚠️ warning'
  return '❌ bad'
}

/** Return a CSS colour string for an overall quality value. */
export function qualityColor(q: string): string {
  if (q === 'good') return '#2d7a2d'
  if (q === 'warning') return '#b8860b'
  return '#cc2222'
}

/** Format a per-keypoint verdict ('good' | 'high' | 'outlier') as a symbol. */
export function verdictBadge(v: string): string {
  if (v === 'good') return '✓'
  if (v === 'high') return '⚠'
  return '✗'
}

/** Return a CSS colour string for a keypoint impact value. */
export function impactColor(impact: string): string {
  if (impact === 'helpful') return '#2d7a2d'
  if (impact === 'marginal') return '#b8860b'
  return '#cc2222'
}
