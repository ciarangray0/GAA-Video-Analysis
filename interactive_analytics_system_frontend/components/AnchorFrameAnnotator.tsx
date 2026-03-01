import { useState, useRef, useEffect, useCallback } from 'react'
import type { VideoMetadata, AnchorFrame, LineAnnotation, PitchPoint } from '../types'
import { drawPitchDiagram, pitchToCanvas } from '../lib/pitch'
import { AVAILABLE_LINES, GAA_PITCH_VERTICES, PITCH_LINE_SEGMENTS, GAA_PITCH_WIDTH, GAA_PITCH_LENGTH } from '../lib/constants'
import { API_URL } from '../lib/api'

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
  const [copyStatus, setCopyStatus] = useState('')

  const frameCanvasRef = useRef<HTMLCanvasElement>(null)
  const frameImageRef = useRef<HTMLImageElement | null>(null)
  const pitchDiagramRef = useRef<HTMLCanvasElement>(null)
  const importAnnotationsRef = useRef<HTMLInputElement>(null)

  const currentAnchor = anchorFrames[currentAnchorIdx]

  // Auto-save to localStorage when anchorFrames change
  useEffect(() => {
    if (anchorFrames.length > 0 && videoFilename) {
      localStorage.setItem(`gaa_annotations_${videoFilename}`, JSON.stringify(anchorFrames))
    }
  }, [anchorFrames, videoFilename])

  const loadFrameImage = useCallback(async (frameIdx: number) => {
    setLoadingFrame(true)
    try {
      const url = `${API_URL}/videos/${videoMetadata.video_id}/frame/${frameIdx}`
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.onload = () => {
        frameImageRef.current = img
        setLoadingFrame(false)
      }
      img.onerror = () => {
        setLoadingFrame(false)
      }
      img.src = `${url}?t=${Date.now()}`
    } catch {
      setLoadingFrame(false)
    }
  }, [videoMetadata.video_id])

  const hasLoadedRef = useRef(false)
  // Load first frame once when the annotator first receives anchor frames
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

    const maxWidth = 1000
    const scale = Math.min(1, maxWidth / img.naturalWidth)
    canvas.width = img.naturalWidth * scale
    canvas.height = img.naturalHeight * scale
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    const anchor = anchorFrames[currentAnchorIdx]
    const imgScale = canvas.width / img.naturalWidth

    // Draw line annotations
    if (anchor && anchor.lines) {
      anchor.lines.forEach((line) => {
        const x1 = line.u1 * imgScale
        const y1 = line.v1 * imgScale
        const x2 = line.u2 * imgScale
        const y2 = line.v2 * imgScale

        ctx.strokeStyle = 'rgba(0, 255, 255, 0.5)'
        ctx.lineWidth = 1.5
        ctx.setLineDash([6, 4])
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.stroke()
        ctx.setLineDash([])

        ctx.fillStyle = 'rgba(0, 255, 255, 0.6)'
        ctx.beginPath()
        ctx.arc(x1, y1, 4, 0, 2 * Math.PI)
        ctx.fill()
        ctx.beginPath()
        ctx.arc(x2, y2, 4, 0, 2 * Math.PI)
        ctx.fill()

        const midX = (x1 + x2) / 2
        const midY = (y1 + y2) / 2
        ctx.font = '9px Arial'
        const labelText = AVAILABLE_LINES[line.line_id]?.label || line.line_id
        const textW = ctx.measureText(labelText).width
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)'
        ctx.fillRect(midX - textW / 2 - 3, midY - 8, textW + 6, 14)
        ctx.fillStyle = '#00ffff'
        ctx.textAlign = 'center'
        ctx.fillText(labelText, midX, midY + 3)
        ctx.textAlign = 'left'
      })
    }

    // Draw pending line first point
    if (pendingLinePoint1) {
      const x = pendingLinePoint1.x * imgScale
      const y = pendingLinePoint1.y * imgScale
      ctx.fillStyle = 'rgba(255, 255, 0, 0.7)'
      ctx.beginPath()
      ctx.arc(x, y, 6, 0, 2 * Math.PI)
      ctx.fill()
      ctx.strokeStyle = '#000000'
      ctx.lineWidth = 1
      ctx.stroke()
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)'
      ctx.fillRect(x - 75, y + 12, 150, 20)
      ctx.fillStyle = '#ffffff'
      ctx.font = '10px Arial'
      ctx.textAlign = 'center'
      ctx.fillText('Click second point on line', x, y + 26)
      ctx.textAlign = 'left'
    }

    // Draw annotation points
    if (anchor && anchor.points) {
      anchor.points.forEach((point) => {
        const x = point.x_img * imgScale
        const y = point.y_img * imgScale
        ctx.fillStyle = 'rgba(0, 255, 0, 0.6)'
        ctx.beginPath()
        ctx.arc(x, y, 5, 0, 2 * Math.PI)
        ctx.fill()
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 1
        ctx.stroke()
        ctx.font = '10px Arial'
        const labelText = point.pitch_id
        const textW = ctx.measureText(labelText).width
        const lx = x + 14
        const ly = y + 4
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)'
        ctx.fillRect(lx - 2, ly - 10, textW + 4, 13)
        ctx.fillStyle = '#ffffff'
        ctx.fillText(labelText, lx, ly)
      })
    }
  }, [anchorFrames, currentAnchorIdx, pendingLinePoint1])

  // Redraw when annotations or image loads
  useEffect(() => {
    if (!loadingFrame && frameImageRef.current && anchorFrames.length > 0) {
      drawFrameWithPoints()
    }
  }, [anchorFrames, currentAnchorIdx, drawFrameWithPoints, loadingFrame, pendingLinePoint1])

  // Redraw pitch diagram
  useEffect(() => {
    if (anchorFrames.length > 0 && pitchDiagramRef.current) {
      drawPitchDiagram(pitchDiagramRef.current, anchorFrames, currentAnchorIdx, pendingFrameClick, pendingLinePoint1)
    }
  }, [pendingFrameClick, anchorFrames, currentAnchorIdx, pendingLinePoint1])

  const handleFrameClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = frameCanvasRef.current
    const img = frameImageRef.current
    if (!canvas || !img || anchorFrames.length === 0) return
    if (!img.naturalWidth || !img.naturalHeight) return

    const rect = canvas.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const clickY = e.clientY - rect.top
    const cssToCanvasX = canvas.width / rect.width
    const cssToCanvasY = canvas.height / rect.height
    const canvasToImageX = img.naturalWidth / canvas.width
    const canvasToImageY = img.naturalHeight / canvas.height
    const x = clickX * cssToCanvasX * canvasToImageX
    const y = clickY * cssToCanvasY * canvasToImageY

    if (annotationMode === 'line') {
      if (!pendingLinePoint1) {
        setPendingLinePoint1({ x: Math.round(x), y: Math.round(y) })
      } else {
        const newLine: LineAnnotation = {
          line_id: selectedLineId,
          u1: pendingLinePoint1.x,
          v1: pendingLinePoint1.y,
          u2: Math.round(x),
          v2: Math.round(y),
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
      setPendingFrameClick({ x: Math.round(x), y: Math.round(y) })
    }
  }

  const handlePitchDiagramClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!pendingFrameClick) return
    const canvas = pitchDiagramRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const cssToCanvasX = canvas.width / rect.width
    const cssToCanvasY = canvas.height / rect.height
    const clickX = (e.clientX - rect.left) * cssToCanvasX
    const clickY = (e.clientY - rect.top) * cssToCanvasY

    // Find closest pitch vertex within 20px
    let closestId: string | null = null
    let closestDist = Infinity
    for (const [id, coords] of Object.entries(GAA_PITCH_VERTICES)) {
      const pos = pitchToCanvas(coords[0], coords[1])
      const dist = Math.sqrt(Math.pow(pos.x - clickX, 2) + Math.pow(pos.y - clickY, 2))
      if (dist < closestDist && dist < 20) {
        closestDist = dist
        closestId = id
      }
    }

    if (closestId) {
      const newPoint: PitchPoint = {
        pitch_id: closestId,
        x_img: pendingFrameClick.x,
        y_img: pendingFrameClick.y,
      }
      const updated = [...anchorFrames]
      updated[currentAnchorIdx] = {
        ...updated[currentAnchorIdx],
        points: [
          ...updated[currentAnchorIdx].points.filter(p => p.pitch_id !== closestId),
          newPoint,
        ],
      }
      onAnchorFramesChange(updated)
      setPendingFrameClick(null)
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
      const dist = Math.sqrt(Math.pow(projX - clickX, 2) + Math.pow(projY - clickY, 2))
      if (dist < nearestLineDist && dist < 15) {
        nearestLineDist = dist
        nearestLine = seg
        lineT = t
      }
    }

    if (nearestLine) {
      const pitchX = nearestLine.x1 + lineT * (nearestLine.x2 - nearestLine.x1)
      const pitchY = nearestLine.y1 + lineT * (nearestLine.y2 - nearestLine.y1)
      const pitchId = `line_${nearestLine.name}_x${pitchX.toFixed(1)}_y${pitchY.toFixed(1)}`
      const newPoint: PitchPoint = {
        pitch_id: pitchId,
        x_img: pendingFrameClick.x,
        y_img: pendingFrameClick.y,
      }
      const updated = [...anchorFrames]
      updated[currentAnchorIdx] = {
        ...updated[currentAnchorIdx],
        points: [...updated[currentAnchorIdx].points, newPoint],
      }
      onAnchorFramesChange(updated)
      setPendingFrameClick(null)
    }
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

  const swapAnchorFrame = (newFrameIdx: number) => {
    if (newFrameIdx < 0 || newFrameIdx >= videoMetadata.num_frames) return
    const updated = [...anchorFrames]
    updated[currentAnchorIdx] = { frame_idx: newFrameIdx, isSkipped: false, points: [], lines: [] }
    onAnchorFramesChange(updated)
    loadFrameImage(newFrameIdx)
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

  const copyFromPrevious = () => {
    let srcIdx = currentAnchorIdx - 1
    while (srcIdx >= 0 && anchorFrames[srcIdx].isSkipped) srcIdx--
    if (srcIdx < 0 || anchorFrames[srcIdx].points.length === 0) return
    const updated = [...anchorFrames]
    updated[currentAnchorIdx] = {
      ...updated[currentAnchorIdx],
      points: [...anchorFrames[srcIdx].points],
      lines: [...(anchorFrames[srcIdx].lines || [])],
    }
    onAnchorFramesChange(updated)
    setCopyStatus('Copied from previous')
    setTimeout(() => setCopyStatus(''), 2000)
  }

  const copyFromNext = () => {
    let srcIdx = currentAnchorIdx + 1
    while (srcIdx < anchorFrames.length && anchorFrames[srcIdx].isSkipped) srcIdx++
    if (srcIdx >= anchorFrames.length || anchorFrames[srcIdx].points.length === 0) return
    const updated = [...anchorFrames]
    updated[currentAnchorIdx] = {
      ...updated[currentAnchorIdx],
      points: [...anchorFrames[srcIdx].points],
      lines: [...(anchorFrames[srcIdx].lines || [])],
    }
    onAnchorFramesChange(updated)
    setCopyStatus('Copied from next')
    setTimeout(() => setCopyStatus(''), 2000)
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
            <button onClick={() => {
              const newFrame = prompt('Enter new frame number:', String(currentAnchor.frame_idx))
              if (newFrame) swapAnchorFrame(parseInt(newFrame))
            }}>
              Swap Frame
            </button>
            <button
              onClick={copyFromPrevious}
              disabled={currentAnchorIdx === 0 || !anchorFrames.slice(0, currentAnchorIdx).some(f => !f.isSkipped && f.points.length > 0)}
              title="Copy annotations from the previous non-skipped frame"
            >
              ← Copy Previous
            </button>
            <button
              onClick={copyFromNext}
              disabled={currentAnchorIdx === anchorFrames.length - 1 || !anchorFrames.slice(currentAnchorIdx + 1).some(f => !f.isSkipped && f.points.length > 0)}
              title="Copy annotations from the next non-skipped frame"
            >
              Copy Next →
            </button>
          </div>
          {copyStatus && (
            <span style={{ marginLeft: 8, fontSize: 12, color: '#44ff44' }}>{copyStatus}</span>
          )}
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
                📍 Point Mode
              </button>
              <button
                className={`mode-btn ${annotationMode === 'line' ? 'active' : ''}`}
                onClick={() => { setAnnotationMode('line'); setPendingFrameClick(null) }}
              >
                📏 Line Mode
              </button>
            </div>

            {annotationMode === 'line' && (
              <div className="line-selector">
                <label>Select Line:</label>
                <select value={selectedLineId} onChange={(e) => setSelectedLineId(e.target.value)}>
                  {Object.entries(AVAILABLE_LINES).map(([id, info]) => (
                    <option key={id} value={id}>
                      {info.label} (Y={info.y_meters}m)
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
                  ✓ First point selected at ({pendingLinePoint1.x}, {pendingLinePoint1.y}).
                  <strong> Click the second point on the {AVAILABLE_LINES[selectedLineId]?.label || selectedLineId}.</strong>
                  <button onClick={() => setPendingLinePoint1(null)} className="cancel-btn">Cancel</button>
                </p>
              ) : (
                <p className="line-mode-instruction">
                  📏 <strong>Line Mode:</strong> Click two points on the <em>{AVAILABLE_LINES[selectedLineId]?.label || selectedLineId}</em> in the video frame.
                  <br />
                  <small>Line constraints improve homography accuracy in midfield regions where point intersections aren't visible.</small>
                </p>
              )
            ) : pendingFrameClick ? (
              <p className="pending-instruction">
                ✓ Frame point selected at ({pendingFrameClick.x}, {pendingFrameClick.y}).
                <strong> Now click the corresponding point on the pitch diagram →</strong>
                <button onClick={() => setPendingFrameClick(null)} className="cancel-btn">Cancel</button>
              </p>
            ) : (
              <p>📍 <strong>Point Mode:</strong> Click a point on the video frame, then select the corresponding pitch location on the diagram.</p>
            )}
          </div>

          {/* Side-by-side frame and pitch diagram */}
          <div className="annotation-workspace">
            <div className="frame-panel">
              <h4>Video Frame</h4>
              {loadingFrame ? (
                <div className="loading">
                  <div className="spinner"></div>
                  <p>Loading frame...</p>
                </div>
              ) : (
                <canvas
                  ref={frameCanvasRef}
                  onClick={handleFrameClick}
                  className={`frame-canvas ${annotationMode === 'line' ? 'line-mode' : ''} ${pendingFrameClick || pendingLinePoint1 ? 'has-pending' : ''}`}
                />
              )}
            </div>

            <div className="pitch-panel">
              <h4>Pitch Diagram</h4>
              <canvas
                ref={pitchDiagramRef}
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
                        <small>Frame: ({point.x_img}, {point.y_img})</small>
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
              <h4>📏 Annotated Lines ({currentAnchor.lines.length}):</h4>
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

      {/* Navigation buttons */}
      <div className="nav-buttons">
        <button onClick={() => goToAnchorFrame(currentAnchorIdx - 1)} disabled={currentAnchorIdx === 0}>
          ← Previous Frame
        </button>
        <button onClick={() => goToAnchorFrame(currentAnchorIdx + 1)} disabled={currentAnchorIdx === anchorFrames.length - 1}>
          Next Frame →
        </button>
      </div>

      {/* Annotation summary and import/export */}
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
