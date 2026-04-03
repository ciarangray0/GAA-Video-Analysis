/** Formatting utilities for homography reprojection error display. */

export function reprErrorLabel(val: number | undefined): string {
  if (val === undefined) return '—'
  if (val < 10) return `${val}px ✓`
  if (val < 20) return `${val}px ⚠`
  return `${val}px ✗`
}

export function reprErrorColor(val: number | undefined): string {
  if (val === undefined) return ''
  if (val < 10) return '#2d7a2d'
  if (val < 20) return '#b8860b'
  return '#cc2222'
}

export function qualityBadge(q: string): string {
  if (q === 'good') return '✅ good'
  if (q === 'warning') return '⚠️ warning'
  return '❌ bad'
}

export function qualityColor(q: string): string {
  if (q === 'good') return '#2d7a2d'
  if (q === 'warning') return '#b8860b'
  return '#cc2222'
}

export function verdictBadge(v: string): string {
  if (v === 'good') return '✓'
  if (v === 'high') return '⚠'
  return '✗'
}

export function impactColor(impact: string): string {
  if (impact === 'helpful') return '#2d7a2d'
  if (impact === 'marginal') return '#b8860b'
  return '#cc2222'
}