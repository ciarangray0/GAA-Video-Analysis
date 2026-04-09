/**
 * Draw a precision crosshair marker (2px circle + crosshair arms) at
 * canvas coords (cx, cy).
 */
export function drawCrosshair(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  color: string,
  label?: string,
): void {
  const r = 2
  const arm = 7

  // Dark shadow for contrast on any background
  ctx.save()
  ctx.strokeStyle = 'rgba(0,0,0,0.65)'
  ctx.lineWidth = 2.5
  ctx.beginPath()
  ctx.moveTo(cx - r - arm, cy); ctx.lineTo(cx - r, cy)
  ctx.moveTo(cx + r, cy);       ctx.lineTo(cx + r + arm, cy)
  ctx.moveTo(cx, cy - r - arm); ctx.lineTo(cx, cy - r)
  ctx.moveTo(cx, cy + r);       ctx.lineTo(cx, cy + r + arm)
  ctx.stroke()

  // Coloured arms
  ctx.strokeStyle = color
  ctx.lineWidth = 1.2
  ctx.beginPath()
  ctx.moveTo(cx - r - arm, cy); ctx.lineTo(cx - r, cy)
  ctx.moveTo(cx + r, cy);       ctx.lineTo(cx + r + arm, cy)
  ctx.moveTo(cx, cy - r - arm); ctx.lineTo(cx, cy - r)
  ctx.moveTo(cx, cy + r);       ctx.lineTo(cx, cy + r + arm)
  ctx.stroke()

  // Centre circle
  ctx.fillStyle = color
  ctx.strokeStyle = 'rgba(0,0,0,0.65)'
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.arc(cx, cy, r, 0, 2 * Math.PI)
  ctx.fill()
  ctx.stroke()

  if (label) {
    const pad = 2
    ctx.font = '7px monospace'
    const tw = ctx.measureText(label).width
    const lx = cx + r + arm + 3
    const ly = cy + 2
    ctx.fillStyle = 'rgba(0,0,0,0.30)'
    ctx.fillRect(lx - pad, ly - 8, tw + pad * 2, 10)
    ctx.fillStyle = 'rgba(255,255,255,0.70)'
    ctx.textAlign = 'left'
    ctx.fillText(label, lx, ly)
  }
  ctx.restore()
}