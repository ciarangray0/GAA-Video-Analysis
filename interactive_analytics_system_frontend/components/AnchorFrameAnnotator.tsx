import { useState, useRef, useEffect, useCallback } from 'react'
import type { VideoMetadata, AnchorFrame, LineAnnotation, PitchPoint } from '../types'
import { drawPitchDiagram, pitchToCanvas } from '../lib/pitch'
import { AVAILABLE_LINES, GAA_PITCH_VERTICES, PITCH_LINE_SEGMENTS, GAA_PITCH_WIDTH, GAA_PITCH_LENGTH } from '../lib/pitchConfig'
import { API_URL } from '../lib/api'
import { drawCrosshair } from '../utils/canvasUtils'

interface AnchorFrameAnnotatorProps {
  videoMetadata: VideoMetadata
  videoFilename?: string
  anchorFrames: AnchorFrame[]
  currentAnchorIdx: number
  onAnchorFramesChange: (frames: AnchorFrame[]) => void
  onCurrentIdxChange: (idx: number) => void
}

export default function AnchorFrameAnnotator({
  videoMetadata,
  videoFilename,
  anchorFrames,
  currentAnchorIdx,
  onAnchorFramesChange,
  onCurrentIdxChange,
}: AnchorFrameAnnotatorProps) {
  const [loadingFrame, setLoadingFrame] = useState(false)
  const [annotationMode, setAnnotationMode] = useState<'point' | 'line'>('point')
  const [selectedLineId, setSelectedLineId] = useState<string>('20m_top')
  const [pendingLinePoint1, setPendingLinePoint1] = useState<{ x: number; y: number } | null>(null)
  const [pendingFrameClick, setPendingFrameClick] = useState<{ x: number; y: number } | null>(null)
  const [zoom, setZoom] = useState(1)
  const [canvasDims, setCanvasDims] = useState({ w: 0, h: 0 })
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number } | null>(null)

  const frameCanvasRef = useRef<HTMLCanvasElement>(null)
  const frameImageRef = useRef<HTMLImageElement | null>(null)
  const [pitchDiagramEl, setPitchDiagramEl] = useState<HTMLCanvasElement | null>(null)
  const importAnnotationsRef = useRef<HTMLInputElement>(null)
  // Tracks which frame index the most recent loadFrameImage call is for,
  // preventing a slower earlier load from overwriting a newer one.
  const loadingFrameIdxRef = useRef<number | null>(null)

  const currentAnchor = anchorFrames[currentAnchorIdx]

  // Auto-save to localStorage
  useEffect(() => {
    if (anchorFrames.length > 0 && videoFilename) {
      localStorage.setItem(`gaa_annotations_${videoFilename}`, JSON.stringify(anchorFrames))
    }
  }, [anchorFrames, videoFilename])

  const loadFrameImage = useCallback((frameIdx: number) => {
    loadingFrameIdxRef.current = frameIdx
    setLoadingFrame(true)
    const url = `${API_URL}/videos/${videoMetadata.video_id}/frame/${frameIdx}`
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      // Discard stale loads so the wrong image never shows with wrong annotations
      if (loadingFrameIdxRef.current === frameIdx) {
        frameImageRef.current = img
        setLoadingFrame(false)
      }
    }
    img.onerror = () => {
      if (loadingFrameIdxRef.current === frameIdx) setLoadingFrame(false)
    }
    img.src = `${url}?t=${Date.now()}`
  }, [videoMetadata.video_id])

  const hasLoadedRef = useRef(false)
  useEffect(() => {
    if (!hasLoadedRef.current && anchorFrames.length > 0) {
      hasLoadedRef.current = true
      loadFrameImage(anchorFrames[currentAnchorIdx].frame_idx)
    }
  }, [anchorFrames, currentAnchorIdx, loadFrameImage])

  const drawFrameWithPoints = useCallback(() => {
    const canvas = frameCanvasRef.current
    const img = frameImageRef.current
    if (!canvas || !img || anchorFrames.length === 0) return
    if (!img.complete || img.naturalWidth === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Use up to 1600px buffer for better precision (was 1000)
    const scale = Math.min(1, 1600 / img.naturalWidth)
    const newW = Math.round(img.naturalWidth * scale)
    const newH = Math.round(img.naturalHeight * scale)
    if (canvas.width !== newW || canvas.height !== newH) {
      canvas.width = newW
      canvas.height = newH
      setCanvasDims({ w: newW, h: newH })
    }

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    const anchor = anchorFrames[currentAnchorIdx]
    // Separate X and Y scales to correctly handle frames whose height rounds differently from width
    const imgScaleX = canvas.width / img.naturalWidth
    const imgScaleY = canvas.height / img.naturalHeight

    // Draw line annotations
    if (anchor?.lines) {
      for (const line of anchor.lines) {
        const x1 = line.u1 * imgScaleX
        const y1 = line.v1 * imgScaleY
        const x2 = line.u2 * imgScaleX
        const y2 = line.v2 * imgScaleY

        const isVertical = AVAILABLE_LINES[line.line_id]?.orientation === 'vertical'
        const lineColor = isVertical ? 'rgba(255, 165, 0, 0.7)' : 'rgba(0, 220, 255, 0.7)'
        const crosshairColor = isVertical ? '#ffa500' : '#00dcff'

        ctx.save()
        ctx.strokeStyle = lineColor
        ctx.lineWidth = 1
        ctx.setLineDash([5, 4])
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.stroke()
        ctx.setLineDash([])
        ctx.restore()

        const labelText = AVAILABLE_LINES[line.line_id]?.label || line.line_id
        drawCrosshair(ctx, x1, y1, crosshairColor)
        drawCrosshair(ctx, x2, y2, crosshairColor, labelText)
      }
    }

    // Pending line first-point indicator
    if (pendingLinePoint1) {
      const px = pendingLinePoint1.x * imgScaleX
      const py = pendingLinePoint1.y * imgScaleY
      drawCrosshair(ctx, px, py, '#ffff00', '←2nd point')
    }

    // Annotated keypoints
    if (anchor?.points) {
      for (const point of anchor.points) {
        const px = point.x_img * imgScaleX
        const py = point.y_img * imgScaleY
        drawCrosshair(ctx, px, py, '#00ff80', point.pitch_id)
      }
    }
  }, [anchorFrames, currentAnchorIdx, pendingLinePoint1])

  useEffect(() => {
    if (!loadingFrame && frameImageRef.current && anchorFrames.length > 0) {
      drawFrameWithPoints()
    }
  }, [anchorFrames, currentAnchorIdx, drawFrameWithPoints, loadingFrame, pendingLinePoint1])

  useEffect(() => {
    if (anchorFrames.length > 0 && pitchDiagramEl) {
      drawPitchDiagram(pitchDiagramEl, anchorFrames, currentAnchorIdx, pendingFrameClick, pendingLinePoint1)
    }
  }, [pendingFrameClick, anchorFrames, currentAnchorIdx, pendingLinePoint1, pitchDiagramEl])

  /** Map a mouse event on the frame canvas to original image pixel coordinates.
   *
   * Key: we use outline (not border) on .frame-canvas so getBoundingClientRect()
   * returns the exact content area. rect.width already accounts for any CSS zoom
   * applied via style.width, so the formula is simply:
   *   x_image = (clientX - rect.left) * img.naturalWidth / rect.width
   */
  const canvasEventToImageCoords = (
    e: React.MouseEvent<HTMLCanvasElement>,
  ): { x: number; y: number } | null => {
    const canvas = frameCanvasRef.current
    const img = frameImageRef.current
    if (!canvas || !img || !img.naturalWidth) return null
    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left) * img.naturalWidth / rect.width
    const y = (e.clientY - rect.top) * img.naturalHeight / rect.height
    return { x: Math.round(x), y: Math.round(y) }
  }

  const handleFrameClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (anchorFrames.length === 0) return
    const coords = canvasEventToImageCoords(e)
    if (!coords) return
    const { x, y } = coords

    if (annotationMode === 'line') {
      if (!pendingLinePoint1) {
        setPendingLinePoint1({ x, y })
      } else {
        const newLine: LineAnnotation = {
          line_id: selectedLineId,
          u1: pendingLinePoint1.x,
          v1: pendingLinePoint1.y,
          u2: x,
          v2: y,
        }
        const updated = [...anchorFrames]
        updated[currentAnchorIdx] = {
          ...updated[currentAnchorIdx],
          lines: [
            ...updated[currentAnchorIdx].lines.filter(l => l.line_id !== selectedLineId),
            newLine,
          ],
        }
        onAnchorFramesChange(updated)
        setPendingLinePoint1(null)
      }
    } else {
      setPendingFrameClick({ x, y })
    }
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const coords = canvasEventToImageCoords(e)
    if (coords) setHoverPos(coords)
  }

  const handleMouseLeave = () => setHoverPos(null)

  const handlePitchDiagramClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!pendingFrameClick) return
    const canvas = pitchDiagramEl
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const clickX = (e.clientX - rect.left) * canvas.width / rect.width
    const clickY = (e.clientY - rect.top) * canvas.height / rect.height

    // Find closest pitch vertex within 20px
    let closestId: string | null = null
    let closestDist = Infinity
    for (const [id, coords] of Object.entries(GAA_PITCH_VERTICES)) {
      const pos = pitchToCanvas(coords[0], coords[1])
      const dist = Math.hypot(pos.x - clickX, pos.y - clickY)
      if (dist < closestDist && dist < 20) {
        closestDist = dist
        closestId = id
      }
    }

    if (closestId) {
      addKeypoint(closestId)
      return
    }

    // Try nearest pitch line segment (within 15px)
    let nearestLine: typeof PITCH_LINE_SEGMENTS[0] | null = null
    let nearestLineDist = Infinity
    let lineT = 0
    for (const seg of PITCH_LINE_SEGMENTS) {
      const p1 = pitchToCanvas(seg.x1, seg.y1)
      const p2 = pitchToCanvas(seg.x2, seg.y2)
      const dx = p2.x - p1.x
      const dy = p2.y - p1.y
      const lenSq = dx * dx + dy * dy
      const t = lenSq > 0 ? Math.max(0, Math.min(1, ((clickX - p1.x) * dx + (clickY - p1.y) * dy) / lenSq)) : 0
      const projX = p1.x + t * dx
      const projY = p1.y + t * dy
      const dist = Math.hypot(projX - clickX, projY - clickY)
      if (dist < nearestLineDist && dist < 15) {
        nearestLineDist = dist
        nearestLine = seg
        lineT = t
      }
    }

    if (nearestLine) {
      const pitchX = nearestLine.x1 + lineT * (nearestLine.x2 - nearestLine.x1)
      const pitchY = nearestLine.y1 + lineT * (nearestLine.y2 - nearestLine.y1)
      addKeypoint(`line_${nearestLine.name}_x${pitchX.toFixed(1)}_y${pitchY.toFixed(1)}`)
    }
  }

  const addKeypoint = (pitchId: string) => {
    if (!pendingFrameClick) return
    const newPoint: PitchPoint = {
      pitch_id: pitchId,
      x_img: pendingFrameClick.x,
      y_img: pendingFrameClick.y,
    }
    const updated = [...anchorFrames]
    updated[currentAnchorIdx] = {
      ...updated[currentAnchorIdx],
      points: [
        ...updated[currentAnchorIdx].points.filter(p => p.pitch_id !== pitchId),
        newPoint,
      ],
    }
    onAnchorFramesChange(updated)
    setPendingFrameClick(null)
  }

  const goToAnchorFrame = (idx: number) => {
    if (idx >= 0 && idx < anchorFrames.length) {
      onCurrentIdxChange(idx)
      loadFrameImage(anchorFrames[idx].frame_idx)
    }
  }

  const toggleSkipFrame = () => {
    const updated = [...anchorFrames]
    updated[currentAnchorIdx] = {
      ...updated[currentAnchorIdx],
      isSkipped: !updated[currentAnchorIdx].isSkipped,
    }
    onAnchorFramesChange(updated)
  }

  const removePoint = (pointIdx: number) => {
    const updated = [...anchorFrames]
    updated[currentAnchorIdx] = {
      ...updated[currentAnchorIdx],
      points: updated[currentAnchorIdx].points.filter((_, i) => i !== pointIdx),
    }
    onAnchorFramesChange(updated)
  }

  const removeLine = (lineIdx: number) => {
    const updated = [...anchorFrames]
    updated[currentAnchorIdx] = {
      ...updated[currentAnchorIdx],
      lines: updated[currentAnchorIdx].lines.filter((_, i) => i !== lineIdx),
    }
    onAnchorFramesChange(updated)
  }

  const exportAnnotations = () => {
    const filename = videoFilename || 'unknown'
    const data = {
      videoFilename: filename,
      anchorFrames: anchorFrames.map(af => ({
        frame_idx: af.frame_idx,
        isSkipped: af.isSkipped,
        points: af.points,
        lines: af.lines || [],
      })),
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `annotations_${filename}_${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const importAnnotations = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      try {
        const parsed = JSON.parse(ev.target?.result as string)
        if (!parsed.anchorFrames || !Array.isArray(parsed.anchorFrames)) {
          alert('Invalid annotation file format.')
          return
        }
        const imported: AnchorFrame[] = parsed.anchorFrames.map((af: any) => ({
          frame_idx: af.frame_idx,
          isSkipped: !!af.isSkipped,
          points: Array.isArray(af.points) ? af.points : [],
          lines: Array.isArray(af.lines) ? af.lines : [],
        }))
        if (anchorFrames.length > 0) {
          const frameIndices = new Set(anchorFrames.map(f => f.frame_idx))
          const hasMismatch = imported.some(f => !frameIndices.has(f.frame_idx))
          if (hasMismatch) {
            if (!confirm('Some imported frame indices do not match current anchor frames. Import anyway?')) return
          }
          const merged = anchorFrames.map(f => {
            const match = imported.find(p => p.frame_idx === f.frame_idx)
            return match ? { ...f, isSkipped: match.isSkipped, points: match.points, lines: match.lines } : f
          })
          onAnchorFramesChange(merged)
          if (merged.length > 0) loadFrameImage(merged[0].frame_idx)
        } else {
          onAnchorFramesChange(imported)
          if (imported.length > 0) loadFrameImage(imported[0].frame_idx)
        }
        onCurrentIdxChange(0)
      } catch (err: any) {
        alert(`Failed to parse annotation file: ${err?.message || 'Invalid format'}`)
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  const getPointLabel = (id: string): string =>
    id.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())

  // Coverage quality metrics
  const pointCount = currentAnchor ? currentAnchor.points.length : 0
  const coverageColor = pointCount < 4 ? '#ff4444' : pointCount < 7 ? '#ffaa00' : '#44ff44'
  let coveragePercent = 0
  let clustered = false
  if (currentAnchor && currentAnchor.points.length >= 2) {
    const pitchCoords = currentAnchor.points.map(p => {
      const vert = GAA_PITCH_VERTICES[p.pitch_id]
      if (vert) return { x: vert[0], y: vert[1] }
      const match = p.pitch_id.match(/x([-\d.]+)_y([-\d.]+)$/)
      if (match) return { x: parseFloat(match[1]), y: parseFloat(match[2]) }
      return null
    }).filter(Boolean) as Array<{ x: number; y: number }>
    if (pitchCoords.length >= 2) {
      const xs = pitchCoords.map(c => c.x)
      const ys = pitchCoords.map(c => c.y)
      const bboxW = Math.max(...xs) - Math.min(...xs)
      const bboxH = Math.max(...ys) - Math.min(...ys)
      coveragePercent = Math.round(bboxW * bboxH / (GAA_PITCH_WIDTH * GAA_PITCH_LENGTH) * 100)
      clustered = coveragePercent < 10
    }
  }

  // Canvas CSS size driven by zoom (no border on canvas — use outline so
  // getBoundingClientRect returns the exact content box)
  const canvasStyle: React.CSSProperties = canvasDims.w > 0 ? {
    width: `${canvasDims.w * zoom}px`,
    height: `${canvasDims.h * zoom}px`,
    imageRendering: zoom >= 2 ? 'pixelated' : 'auto',
  } : {}

  return (
    <div className="annotation-section">
      <h2>3. Annotate Anchor Frames</h2>

      {/* Anchor frame navigation tabs */}
      <div className="anchor-nav">
        <div className="anchor-tabs">
          {anchorFrames.map((af, idx) => (
            <button
              key={idx}
              className={`anchor-tab ${idx === currentAnchorIdx ? 'active' : ''} ${af.isSkipped ? 'skipped' : ''} ${af.points.length >= 4 ? 'complete' : ''}`}
              onClick={() => goToAnchorFrame(idx)}
              title={`Frame ${af.frame_idx}${af.isSkipped ? ' (skipped)' : ''}`}
            >
              {idx + 1}
              {af.points.length >= 4 && !af.isSkipped && ' ✓'}
              {af.isSkipped && ' ✗'}
            </button>
          ))}
        </div>
      </div>

      {currentAnchor && (
        <div className="current-anchor-info">
          <span>
            Anchor {currentAnchorIdx + 1} of {anchorFrames.length} |
            Frame {currentAnchor.frame_idx} |
            Points: {currentAnchor.points.length}/4+ | Lines: {currentAnchor.lines?.length || 0}
          </span>
          <span style={{ marginLeft: 12, fontSize: 12 }}>
            <span style={{ color: coverageColor }}>●</span>{' '}
            {pointCount} points
            {currentAnchor.points.length >= 2 && ` | Coverage: ${coveragePercent}%`}
            {clustered && ' | ⚠ Clustered'}
          </span>
          <div className="anchor-actions">
            <button onClick={toggleSkipFrame} className={currentAnchor.isSkipped ? 'warning' : ''}>
              {currentAnchor.isSkipped ? 'Unskip Frame' : 'Skip Frame'}
            </button>
          </div>
        </div>
      )}

      {!currentAnchor?.isSkipped && (
        <>
          {/* Annotation Mode Toggle */}
          <div className="annotation-mode-toggle">
            <label className="mode-label">Annotation Mode:</label>
            <div className="mode-buttons">
              <button
                className={`mode-btn ${annotationMode === 'point' ? 'active' : ''}`}
                onClick={() => { setAnnotationMode('point'); setPendingLinePoint1(null) }}
              >
                 Point Mode
              </button>
              <button
                className={`mode-btn ${annotationMode === 'line' ? 'active' : ''}`}
                onClick={() => { setAnnotationMode('line'); setPendingFrameClick(null) }}
              >
                  Line Mode
              </button>
            </div>

            {annotationMode === 'line' && (
              <div className="line-selector">
                <label>Select Line:</label>
                <select value={selectedLineId} onChange={(e) => setSelectedLineId(e.target.value)}>
                  {Object.entries(AVAILABLE_LINES).map(([id, info]) => (
                    <option key={id} value={id}>
                      {info.label} {info.orientation === 'vertical' ? `(X=${info.x_meters}m)` : `(Y=${info.y_meters}m)`}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          {/* Instructions */}
          <div className="annotation-instructions">
            {annotationMode === 'line' ? (
              pendingLinePoint1 ? (
                <p className="pending-instruction line-mode">
                  ✓ First point: ({pendingLinePoint1.x}, {pendingLinePoint1.y}).
                  <strong> Click the second point on the {AVAILABLE_LINES[selectedLineId]?.label || selectedLineId}.</strong>
                  <button onClick={() => setPendingLinePoint1(null)} className="cancel-btn">Cancel</button>
                </p>
              ) : (
                <p className="line-mode-instruction">
                  <strong>Line Mode:</strong> Click two points on the <em>{AVAILABLE_LINES[selectedLineId]?.label || selectedLineId}</em> in the video frame.
                  <br />
                  <small>Line constraints improve homography accuracy in midfield regions where point intersections aren't visible.</small>
                </p>
              )
            ) : pendingFrameClick ? (
              <p className="pending-instruction">
                ✓ Frame point: ({pendingFrameClick.x}, {pendingFrameClick.y}).
                <strong> Now click the corresponding point on the pitch diagram →</strong>
                <button onClick={() => setPendingFrameClick(null)} className="cancel-btn">Cancel</button>
              </p>
            ) : (
              <p> <strong>Point Mode:</strong> Click a feature in the video frame, then select the matching location on the pitch diagram.</p>
            )}
          </div>

          {/* Side-by-side workspace */}
          <div className="annotation-workspace">
            <div className="frame-panel">
              <div className="frame-panel-header">
                <h4>Video Frame</h4>
                {/* Zoom controls */}
                <div className="zoom-controls">
                  {[1, 1.5, 2, 3, 4].map(z => (
                    <button
                      key={z}
                      className={`zoom-btn ${zoom === z ? 'active' : ''}`}
                      onClick={() => setZoom(z)}
                    >
                      {z}×
                    </button>
                  ))}
                  <span className="pixel-readout">
                    {hoverPos ? `px (${hoverPos.x}, ${hoverPos.y})` : 'hover to inspect'}
                  </span>
                </div>
              </div>

              {loadingFrame ? (
                <div className="loading">
                  <div className="spinner"></div>
                  <p>Loading frame...</p>
                </div>
              ) : (
                <div className="frame-canvas-scroller">
                  <canvas
                    ref={frameCanvasRef}
                    onClick={handleFrameClick}
                    onMouseMove={handleMouseMove}
                    onMouseLeave={handleMouseLeave}
                    style={canvasStyle}
                    className={`frame-canvas ${annotationMode === 'line' ? 'line-mode' : ''} ${pendingFrameClick || pendingLinePoint1 ? 'has-pending' : ''}`}
                  />
                </div>
              )}
            </div>

            <div className="pitch-panel">
              <h4>Pitch Diagram</h4>
              <canvas
                ref={setPitchDiagramEl}
                onClick={handlePitchDiagramClick}
                className={`pitch-diagram ${pendingFrameClick ? 'awaiting-click' : ''}`}
              />
              <div className="pitch-legend">
                <span className="legend-item"><span className="dot orange"></span> Available</span>
                <span className="legend-item"><span className="dot green"></span> Annotated</span>
                <span className="legend-item"><span className="dot yellow"></span> Select now</span>
              </div>
            </div>
          </div>

          {/* Points list */}
          {currentAnchor && currentAnchor.points.length > 0 && (
            <div className="points-list">
              <h4>Annotated Points ({currentAnchor.points.length}):</h4>
              <div className="points-grid">
                {currentAnchor.points.map((point, idx) => {
                  const isLinePoint = point.pitch_id.startsWith('line_')
                  return (
                    <div key={idx} className="point-item">
                      <span>
                        <strong style={isLinePoint ? { color: '#88ccff' } : undefined}>
                          {getPointLabel(point.pitch_id)}
                          {isLinePoint && <em style={{ fontWeight: 'normal', fontSize: '0.85em' }}> (line)</em>}
                        </strong>
                        <br />
                        <small>({point.x_img}, {point.y_img})</small>
                      </span>
                      <button onClick={() => removePoint(idx)} className="remove-btn">×</button>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Lines list */}
          {currentAnchor && currentAnchor.lines && currentAnchor.lines.length > 0 && (
            <div className="lines-list">
              <h4> Annotated Lines ({currentAnchor.lines.length}):</h4>
              <div className="lines-grid">
                {currentAnchor.lines.map((line, idx) => (
                  <div key={idx} className="line-item">
                    <span>
                      <strong>{AVAILABLE_LINES[line.line_id]?.label || line.line_id}</strong>
                      <br />
                      <small>
                        ({Math.round(line.u1)}, {Math.round(line.v1)}) → ({Math.round(line.u2)}, {Math.round(line.v2)})
                      </small>
                    </span>
                    <button onClick={() => removeLine(idx)} className="remove-btn">×</button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* Navigation */}
      <div className="nav-buttons">
        <button onClick={() => goToAnchorFrame(currentAnchorIdx - 1)} disabled={currentAnchorIdx === 0}>
          ← Previous Frame
        </button>
        <button onClick={() => goToAnchorFrame(currentAnchorIdx + 1)} disabled={currentAnchorIdx === anchorFrames.length - 1}>
          Next Frame →
        </button>
      </div>

      {/* Summary + Import/Export */}
      <div className="process-section">
        <div className="annotation-summary">
          <p>
            <strong>Ready to process:</strong>{' '}
            {anchorFrames.filter(af => !af.isSkipped && af.points.length >= 4).length} frames annotated |{' '}
            {anchorFrames.filter(af => af.isSkipped).length} frames skipped |{' '}
            {anchorFrames.filter(af => !af.isSkipped && af.points.length < 4).length} frames incomplete
          </p>
        </div>
        <div className="annotation-io-buttons">
          <button onClick={exportAnnotations} className="secondary-btn">⬇ Export Annotations</button>
          <button onClick={() => importAnnotationsRef.current?.click()} className="secondary-btn">⬆ Import Annotations</button>
          <input
            ref={importAnnotationsRef}
            type="file"
            accept=".json"
            style={{ display: 'none' }}
            onChange={importAnnotations}
          />
        </div>
      </div>
    </div>
  )
}
