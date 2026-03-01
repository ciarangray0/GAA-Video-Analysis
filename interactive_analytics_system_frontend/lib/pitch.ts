import type { AnchorFrame, PlayerPosition } from '../types'
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

export function pitchToCanvas(pitchX: number, pitchY: number): { x: number; y: number } {
  const x = (pitchX / GAA_PITCH_WIDTH) * PITCH_DISPLAY_WIDTH
  const y = (pitchY / GAA_PITCH_LENGTH) * PITCH_DISPLAY_HEIGHT
  return { x, y }
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

  // 13m lines
  const line13Top = pitchToCanvas(0, 13).y
  const line13Bottom = pitchToCanvas(0, 127).y
  ctx.beginPath()
  ctx.moveTo(0, line13Top)
  ctx.lineTo(PITCH_DISPLAY_WIDTH, line13Top)
  ctx.moveTo(0, line13Bottom)
  ctx.lineTo(PITCH_DISPLAY_WIDTH, line13Bottom)
  ctx.stroke()

  // 20m lines
  const line20Top = pitchToCanvas(0, 20).y
  const line20Bottom = pitchToCanvas(0, 120).y
  ctx.beginPath()
  ctx.moveTo(0, line20Top)
  ctx.lineTo(PITCH_DISPLAY_WIDTH, line20Top)
  ctx.moveTo(0, line20Bottom)
  ctx.lineTo(PITCH_DISPLAY_WIDTH, line20Bottom)
  ctx.stroke()

  // 45m lines
  const line45Top = pitchToCanvas(0, 45).y
  const line45Bottom = pitchToCanvas(0, 95).y
  ctx.beginPath()
  ctx.moveTo(0, line45Top)
  ctx.lineTo(PITCH_DISPLAY_WIDTH, line45Top)
  ctx.moveTo(0, line45Bottom)
  ctx.lineTo(PITCH_DISPLAY_WIDTH, line45Bottom)
  ctx.stroke()

  // 65m lines
  const line65Top = pitchToCanvas(0, 65).y
  const line65Bottom = pitchToCanvas(0, 75).y
  ctx.beginPath()
  ctx.moveTo(0, line65Top)
  ctx.lineTo(PITCH_DISPLAY_WIDTH, line65Top)
  ctx.moveTo(0, line65Bottom)
  ctx.lineTo(PITCH_DISPLAY_WIDTH, line65Bottom)
  ctx.stroke()

  // 13m box vertical lines
  const box13Left = pitchToCanvas(33, 0).x
  const box13Right = pitchToCanvas(52, 0).x
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

  // Highlight line segments when a pending click is waiting for a pitch point
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

    if (isAnnotated) {
      ctx.fillStyle = '#00ff00'
    } else if (pendingFrameClick) {
      ctx.fillStyle = '#ffff00'
    } else {
      ctx.fillStyle = '#ff6600'
    }
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

export function drawPitch(
  canvas: HTMLCanvasElement,
  positions: PlayerPosition[],
  frame: number,
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

  // Center line
  ctx.beginPath()
  ctx.moveTo(0, H / 2)
  ctx.lineTo(W, H / 2)
  ctx.stroke()

  // Center circle
  ctx.beginPath()
  ctx.arc(W / 2, H / 2, 40 * DISPLAY_SCALE, 0, 2 * Math.PI)
  ctx.stroke()

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)'
  ctx.lineWidth = 1

  // 13m lines
  const line13mTop = (13 / GAA_PITCH_LENGTH) * H
  const line13mBottom = ((GAA_PITCH_LENGTH - 13) / GAA_PITCH_LENGTH) * H
  ctx.beginPath()
  ctx.moveTo(0, line13mTop)
  ctx.lineTo(W, line13mTop)
  ctx.moveTo(0, line13mBottom)
  ctx.lineTo(W, line13mBottom)
  ctx.stroke()

  // 20m lines
  const line20mTop = (20 / GAA_PITCH_LENGTH) * H
  const line20mBottom = ((GAA_PITCH_LENGTH - 20) / GAA_PITCH_LENGTH) * H
  ctx.beginPath()
  ctx.moveTo(0, line20mTop)
  ctx.lineTo(W, line20mTop)
  ctx.moveTo(0, line20mBottom)
  ctx.lineTo(W, line20mBottom)
  ctx.stroke()

  // 45m lines
  const line45mTop = (45 / GAA_PITCH_LENGTH) * H
  const line45mBottom = ((GAA_PITCH_LENGTH - 45) / GAA_PITCH_LENGTH) * H
  ctx.beginPath()
  ctx.moveTo(0, line45mTop)
  ctx.lineTo(W, line45mTop)
  ctx.moveTo(0, line45mBottom)
  ctx.lineTo(W, line45mBottom)
  ctx.stroke()

  // Filter positions for this frame
  const framePositions = positions.filter(p => p.frame_idx === frame)

  const outOfBounds = framePositions.filter(p =>
    p.x_pitch < 0 || p.x_pitch > PITCH_CANVAS_W ||
    p.y_pitch < 0 || p.y_pitch > PITCH_CANVAS_H
  )
  if (outOfBounds.length > 0) {
    console.warn(`Frame ${frame}: ${outOfBounds.length} out-of-bounds positions:`, outOfBounds)
  }

  const getPlayerColor = (trackId: number): string => {
    const hue = (trackId * 137.508) % 360
    return `hsl(${hue}, 70%, 50%)`
  }

  const padding = 8
  framePositions.forEach((pos) => {
    const x = (pos.x_pitch / PITCH_CANVAS_W) * W
    const y = (pos.y_pitch / PITCH_CANVAS_H) * H
    const isOutOfBounds =
      pos.x_pitch < 0 || pos.x_pitch > PITCH_CANVAS_W ||
      pos.y_pitch < 0 || pos.y_pitch > PITCH_CANVAS_H
    const clampedX = Math.max(padding, Math.min(W - padding, x))
    const clampedY = Math.max(padding, Math.min(H - padding, y))

    ctx.fillStyle = getPlayerColor(pos.track_id)
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
