import type { AnchorFrame, PlayerPosition, TeamClassifications } from '../types'
import {
  PITCH_DISPLAY_WIDTH,
  PITCH_DISPLAY_HEIGHT,
  GAA_PITCH_WIDTH,
  GAA_PITCH_LENGTH,
  GAA_PITCH_VERTICES,
  PITCH_LINE_SEGMENTS,
  PITCH_CANVAS_W,
  PITCH_CANVAS_H,
  DISPLAY_SCALE,
} from './constants'

// Symmetric horizontal line pairs (top_y, bottom_y) in meters
const PITCH_HORIZONTAL_LINES = [
  [13, 127],
  [20, 120],
  [45,  95],
  [65,  75],
] as const

export function pitchToCanvas(pitchX: number, pitchY: number): { x: number; y: number } {
  return {
    x: (pitchX / GAA_PITCH_WIDTH) * PITCH_DISPLAY_WIDTH,
    y: (pitchY / GAA_PITCH_LENGTH) * PITCH_DISPLAY_HEIGHT,
  }
}

function drawHorizontalLinePair(
  ctx: CanvasRenderingContext2D,
  y1: number,
  y2: number,
  width: number,
): void {
  ctx.beginPath()
  ctx.moveTo(0, y1)
  ctx.lineTo(width, y1)
  ctx.moveTo(0, y2)
  ctx.lineTo(width, y2)
  ctx.stroke()
}

export function drawPitchDiagram(
  canvas: HTMLCanvasElement,
  anchorFrames: AnchorFrame[],
  currentAnchorIdx: number,
  pendingFrameClick: { x: number; y: number } | null,
  _pendingLinePoint1: { x: number; y: number } | null,
): void {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  canvas.width = PITCH_DISPLAY_WIDTH
  canvas.height = PITCH_DISPLAY_HEIGHT

  ctx.fillStyle = '#2d5016'
  ctx.fillRect(0, 0, PITCH_DISPLAY_WIDTH, PITCH_DISPLAY_HEIGHT)

  ctx.strokeStyle = '#ffffff'
  ctx.lineWidth = 2
  ctx.strokeRect(0, 0, PITCH_DISPLAY_WIDTH, PITCH_DISPLAY_HEIGHT)

  // Center line
  const centerY = pitchToCanvas(0, 70).y
  ctx.beginPath()
  ctx.moveTo(0, centerY)
  ctx.lineTo(PITCH_DISPLAY_WIDTH, centerY)
  ctx.stroke()

  // Horizontal line pairs (13m, 20m, 45m, 65m)
  for (const [top, bottom] of PITCH_HORIZONTAL_LINES) {
    drawHorizontalLinePair(
      ctx,
      pitchToCanvas(0, top).y,
      pitchToCanvas(0, bottom).y,
      PITCH_DISPLAY_WIDTH,
    )
  }

  // 13m box vertical lines
  const box13Left = pitchToCanvas(33, 0).x
  const box13Right = pitchToCanvas(52, 0).x
  const line13Top = pitchToCanvas(0, 13).y
  const line13Bottom = pitchToCanvas(0, 127).y
  ctx.beginPath()
  ctx.moveTo(box13Left, 0)
  ctx.lineTo(box13Left, line13Top)
  ctx.moveTo(box13Right, 0)
  ctx.lineTo(box13Right, line13Top)
  ctx.moveTo(box13Left, line13Bottom)
  ctx.lineTo(box13Left, PITCH_DISPLAY_HEIGHT)
  ctx.moveTo(box13Right, line13Bottom)
  ctx.lineTo(box13Right, PITCH_DISPLAY_HEIGHT)
  ctx.stroke()

  // Goalie box
  const goalieLeft = pitchToCanvas(35.5, 0).x
  const goalieRight = pitchToCanvas(49.5, 0).x
  const goalieTop = pitchToCanvas(0, 4.5).y
  const goalieBottom = pitchToCanvas(0, 135.5).y
  ctx.beginPath()
  ctx.moveTo(goalieLeft, 0)
  ctx.lineTo(goalieLeft, goalieTop)
  ctx.lineTo(goalieRight, goalieTop)
  ctx.lineTo(goalieRight, 0)
  ctx.moveTo(goalieLeft, PITCH_DISPLAY_HEIGHT)
  ctx.lineTo(goalieLeft, goalieBottom)
  ctx.lineTo(goalieRight, goalieBottom)
  ctx.lineTo(goalieRight, PITCH_DISPLAY_HEIGHT)
  ctx.stroke()

  const currentAnchor = anchorFrames[currentAnchorIdx]
  const annotatedIds = currentAnchor ? currentAnchor.points.map(p => p.pitch_id) : []

  // Highlight selectable line segments when awaiting a pitch point click
  if (pendingFrameClick) {
    ctx.strokeStyle = 'rgba(255, 255, 100, 0.4)'
    ctx.lineWidth = 6
    for (const seg of PITCH_LINE_SEGMENTS) {
      const p1 = pitchToCanvas(seg.x1, seg.y1)
      const p2 = pitchToCanvas(seg.x2, seg.y2)
      ctx.beginPath()
      ctx.moveTo(p1.x, p1.y)
      ctx.lineTo(p2.x, p2.y)
      ctx.stroke()
    }
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 2
  }

  // Draw all vertex points
  for (const [id, coords] of Object.entries(GAA_PITCH_VERTICES)) {
    const pos = pitchToCanvas(coords[0], coords[1])
    const isAnnotated = annotatedIds.includes(id)
    ctx.beginPath()
    ctx.arc(pos.x, pos.y, 6, 0, 2 * Math.PI)
    ctx.fillStyle = isAnnotated ? '#00ff00' : pendingFrameClick ? '#ffff00' : '#ff6600'
    ctx.fill()
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 2
    ctx.stroke()
  }

  if (pendingFrameClick) {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)'
    ctx.fillRect(10, PITCH_DISPLAY_HEIGHT - 44, PITCH_DISPLAY_WIDTH - 20, 34)
    ctx.fillStyle = '#ffffff'
    ctx.font = '10px Arial'
    ctx.textAlign = 'center'
    ctx.fillText('Click a vertex (●) or anywhere on a line', PITCH_DISPLAY_WIDTH / 2, PITCH_DISPLAY_HEIGHT - 28)
    ctx.fillStyle = '#aaaaaa'
    ctx.fillText(`Frame: (${pendingFrameClick.x}, ${pendingFrameClick.y})`, PITCH_DISPLAY_WIDTH / 2, PITCH_DISPLAY_HEIGHT - 14)
    ctx.textAlign = 'left'
  }
}

/** Convert OpenCV HSV (H 0-179, S 0-255, V 0-255) to a CSS rgb() string. */
function hsvToCss(h_cv: number, s_cv: number, v_cv: number): string {
  const h = (h_cv * 2) / 360
  const s = s_cv / 255
  const v = v_cv / 255
  const i = Math.floor(h * 6)
  const f = h * 6 - i
  const p = v * (1 - s)
  const q = v * (1 - f * s)
  const t = v * (1 - (1 - f) * s)
  let r: number, g: number, b: number
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break
    case 1: r = q; g = v; b = p; break
    case 2: r = p; g = v; b = t; break
    case 3: r = p; g = q; b = v; break
    case 4: r = t; g = p; b = v; break
    default: r = v; g = p; b = q; break
  }
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`
}

export { hsvToCss }

export function drawPitch(
  canvas: HTMLCanvasElement,
  positions: PlayerPosition[],
  frame: number,
  teamClassifications?: TeamClassifications,
  showTrails = false,
): void {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const W = PITCH_DISPLAY_WIDTH
  const H = PITCH_DISPLAY_HEIGHT

  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W
    canvas.height = H
  }

  ctx.fillStyle = '#2d5016'
  ctx.fillRect(0, 0, W, H)

  ctx.strokeStyle = '#ffffff'
  ctx.lineWidth = 2
  ctx.strokeRect(2, 2, W - 4, H - 4)

  // All pitch markings at uniform opacity
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)'
  ctx.lineWidth = 1

  // Center line
  ctx.beginPath()
  ctx.moveTo(0, H / 2)
  ctx.lineTo(W, H / 2)
  ctx.stroke()

  // 20m semicircles (radius 13m, curving into pitch)
  const arcR = (13 / GAA_PITCH_LENGTH) * H
  const arcCx = W / 2
  ctx.beginPath()
  ctx.arc(arcCx, (20 / GAA_PITCH_LENGTH) * H, arcR, 0, Math.PI, false)
  ctx.stroke()
  ctx.beginPath()
  ctx.arc(arcCx, (120 / GAA_PITCH_LENGTH) * H, arcR, 0, Math.PI, true)
  ctx.stroke()

  // Symmetric horizontal line pairs (13m, 20m, 45m)
  for (const y_m of [13, 20, 45, 65] as const) {
    drawHorizontalLinePair(
      ctx,
      (y_m / GAA_PITCH_LENGTH) * H,
      ((GAA_PITCH_LENGTH - y_m) / GAA_PITCH_LENGTH) * H,
      W,
    )
  }

  // 13m box vertical lines (x=33m, x=52m, from endline to 13m line)
  const box13L = (33 / GAA_PITCH_WIDTH) * W
  const box13R = (52 / GAA_PITCH_WIDTH) * W
  const y13top = (13 / GAA_PITCH_LENGTH) * H
  const y13bot = (127 / GAA_PITCH_LENGTH) * H
  ctx.beginPath()
  ctx.moveTo(box13L, 0);    ctx.lineTo(box13L, y13top)
  ctx.moveTo(box13R, 0);    ctx.lineTo(box13R, y13top)
  ctx.moveTo(box13L, y13bot); ctx.lineTo(box13L, H)
  ctx.moveTo(box13R, y13bot); ctx.lineTo(box13R, H)
  ctx.stroke()

  // Small (goalie) box (x=35.5m–49.5m, depth=4.5m from endline)
  const boxSL = (35.5 / GAA_PITCH_WIDTH) * W
  const boxSR = (49.5 / GAA_PITCH_WIDTH) * W
  const ySTop = (4.5 / GAA_PITCH_LENGTH) * H
  const ySBot = (135.5 / GAA_PITCH_LENGTH) * H
  ctx.beginPath()
  ctx.moveTo(boxSL, 0);  ctx.lineTo(boxSL, ySTop); ctx.lineTo(boxSR, ySTop); ctx.lineTo(boxSR, 0)
  ctx.moveTo(boxSL, H);  ctx.lineTo(boxSL, ySBot); ctx.lineTo(boxSR, ySBot); ctx.lineTo(boxSR, H)
  ctx.stroke()

  const validTrackIds = new Set(positions.map(p => p.track_id))

  const framePositions = positions.filter(p => p.frame_idx === frame && validTrackIds.has(p.track_id))
  const activeTrackIds = new Set(framePositions.map(p => p.track_id))

  // Returns the dot colour for a track, or null if the track should be hidden.
  const getPlayerColor = (trackId: number): string | null => {
    if (teamClassifications) {
      const cls = teamClassifications[trackId.toString()]
      if (cls?.team === 'referee' || cls?.team === 'ignore') return null
      if (cls?.team === 'ellistown') return '#FFD700'
      if (cls?.team === 'opposition') return '#4488FF'
    }
    return `hsl(${(trackId * 137.508) % 360}, 70%, 50%)`
  }

  // Draw trails — full trajectories as faded lines, coloured by team
  if (showTrails) {
    // Group all positions by track
    const byTrack = new Map<number, PlayerPosition[]>()
    for (const p of positions) {
      if (!byTrack.has(p.track_id)) byTrack.set(p.track_id, [])
      byTrack.get(p.track_id)!.push(p)
    }

    byTrack.forEach((pts, trackId) => {
      const color = getPlayerColor(trackId)
      if (color === null) return  // referee/ignore

      // Sort by frame so the line goes in temporal order
      pts.sort((a, b) => a.frame_idx - b.frame_idx)

      // Parse team colour to draw the trail in a matching shade
      // Use the same colour string at low alpha for the trail
      ctx.save()
      ctx.globalAlpha = 0.35
      ctx.strokeStyle = color
      ctx.lineWidth = 1.5
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      let started = false
      for (const p of pts) {
        const x = Math.max(0, Math.min(W, (p.x_pitch / PITCH_CANVAS_W) * W))
        const y = Math.max(0, Math.min(H, (p.y_pitch / PITCH_CANVAS_H) * H))
        if (!started) { ctx.moveTo(x, y); started = true }
        else ctx.lineTo(x, y)
      }
      ctx.stroke()
      ctx.setLineDash([])
      ctx.restore()
    })
  }

  const outOfBounds = framePositions.filter(
    p => p.x_pitch < 0 || p.x_pitch > PITCH_CANVAS_W || p.y_pitch < 0 || p.y_pitch > PITCH_CANVAS_H,
  )
  if (outOfBounds.length > 0) {
    console.warn(`Frame ${frame}: ${outOfBounds.length} out-of-bounds positions:`, outOfBounds)
  }

  // For each track that has been seen at some point up to this frame but is
  // not active now, find its most recent position and draw a grey ghost dot.
  const pastPositions = positions.filter(p => p.frame_idx < frame && validTrackIds.has(p.track_id))
  const lastKnownByTrack = new Map<number, PlayerPosition>()
  for (const p of pastPositions) {
    const existing = lastKnownByTrack.get(p.track_id)
    if (!existing || p.frame_idx > existing.frame_idx) {
      lastKnownByTrack.set(p.track_id, p)
    }
  }

  const padding = 8

  // Draw ghost dots first (underneath active dots)
  lastKnownByTrack.forEach((pos, trackId) => {
    if (activeTrackIds.has(trackId)) return  // active — drawn below
    if (getPlayerColor(trackId) === null) return  // referee/ignore — hidden
    const x = (pos.x_pitch / PITCH_CANVAS_W) * W
    const y = (pos.y_pitch / PITCH_CANVAS_H) * H
    const clampedX = Math.max(padding, Math.min(W - padding, x))
    const clampedY = Math.max(padding, Math.min(H - padding, y))

    ctx.fillStyle = 'rgba(160, 160, 160, 0.45)'
    ctx.beginPath()
    ctx.arc(clampedX, clampedY, 7, 0, 2 * Math.PI)
    ctx.fill()

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
    ctx.lineWidth = 1
    ctx.stroke()

    ctx.fillStyle = 'rgba(200, 200, 200, 0.6)'
    ctx.font = 'bold 9px Arial'
    ctx.textAlign = 'center'
    ctx.fillText(trackId.toString(), clampedX, clampedY + 3)
  })

  // Draw active players on top
  framePositions.forEach((pos) => {
    const color = getPlayerColor(pos.track_id)
    if (color === null) return  // referee/ignore — hidden

    const x = (pos.x_pitch / PITCH_CANVAS_W) * W
    const y = (pos.y_pitch / PITCH_CANVAS_H) * H
    const isOutOfBounds =
      pos.x_pitch < 0 || pos.x_pitch > PITCH_CANVAS_W ||
      pos.y_pitch < 0 || pos.y_pitch > PITCH_CANVAS_H
    const clampedX = Math.max(padding, Math.min(W - padding, x))
    const clampedY = Math.max(padding, Math.min(H - padding, y))

    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(clampedX, clampedY, 8, 0, 2 * Math.PI)
    ctx.fill()

    ctx.strokeStyle = isOutOfBounds ? '#ff0000' : '#ffffff'
    ctx.lineWidth = isOutOfBounds ? 3 : 2
    ctx.stroke()

    ctx.fillStyle = '#ffffff'
    ctx.font = 'bold 10px Arial'
    ctx.textAlign = 'center'
    ctx.fillText(pos.track_id.toString(), clampedX, clampedY + 3)
  })

  ctx.fillStyle = 'rgba(0, 0, 0, 0.6)'
  ctx.fillRect(5, 5, 120, 50)
  ctx.fillStyle = '#ffffff'
  ctx.font = '14px Arial'
  ctx.textAlign = 'left'
  ctx.fillText(`Frame: ${frame}`, 10, 25)
  ctx.fillText(`Players: ${framePositions.length}`, 10, 45)
}
