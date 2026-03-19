import { useCallback, useMemo, useState } from 'react'
import type { VideoMetadata, AnchorFrame, PlayerPosition, AnchorFrameAnnotation } from '../types'
import { API_URL, getDetections, mapPlayers, interpolateTrajectories, getPlayerPositions } from '../lib/api'

interface StepBResult {
  frames: number[]
  per_frame_count: number
  info: Record<string, any>
}


interface AnchorQualityPoint {
  pitch_id: string
  x_img: number
  y_img: number
  error_px: number
  verdict: 'good' | 'high' | 'outlier'
  impact: 'helpful' | 'marginal' | 'harmful'
}

interface AnchorQualityFrame {
  frame_idx: number
  n_keypoints: number
  n_lines: number
  mean_error_px: number
  max_error_px: number
  n_outlier_points: number
  n_helpful_points: number
  overall_quality: 'good' | 'warning' | 'bad'
  keypoints: AnchorQualityPoint[]
  recommendation: string
}

interface AnchorQualityData {
  anchors: AnchorQualityFrame[]
}

interface PipelineStepsProps {
  videoMetadata: VideoMetadata | null
  anchorFrames: AnchorFrame[]
  stepAResult: { frames_processed: number; tracks: number; num_detections: number } | null
  stepBResult: StepBResult | null
  stepCResult: { positions: PlayerPosition[]; total: number } | null
  stepDResult: { frames_generated: number; method: string } | null
  staleSteps: Set<string>
  runningSteps: Set<string>
  onStepAComplete: (result: { frames_processed: number; tracks: number; num_detections: number }) => void
  onStepBComplete: (result: StepBResult, homographyFrames: number[]) => void
  onStepCComplete: (result: { positions: PlayerPosition[]; total: number }) => void
  onStepDComplete: (result: { frames_generated: number; method: string }, allPositions: PlayerPosition[], startFrame: number, endFrame: number, fps: number) => void
  onStepsMarkedStale: (steps: string[]) => void
  onStepsClearedStale: (steps: string[]) => void
  onRunningStepChange: (step: string, action: 'add' | 'remove') => void
  onError: (msg: string) => void
  onStatusChange: (msg: string) => void
  logApiCall: (entry: string) => void
}

function reprErrorLabel(val: number | undefined): string {
  if (val === undefined) return '—'
  if (val < 10) return `${val}px ✓`
  if (val < 20) return `${val}px ⚠`
  return `${val}px ✗`
}

function reprErrorColor(val: number | undefined): string {
  if (val === undefined) return ''
  if (val < 10) return '#2d7a2d'
  if (val < 20) return '#b8860b'
  return '#cc2222'
}

function qualityBadge(q: string): string {
  if (q === 'good') return '✅ good'
  if (q === 'warning') return '⚠️ warning'
  return '❌ bad'
}

function qualityColor(q: string): string {
  if (q === 'good') return '#2d7a2d'
  if (q === 'warning') return '#b8860b'
  return '#cc2222'
}

function verdictBadge(v: string): string {
  if (v === 'good') return '✓'
  if (v === 'high') return '⚠'
  return '✗'
}

function impactColor(impact: string): string {
  if (impact === 'helpful') return '#2d7a2d'
  if (impact === 'marginal') return '#b8860b'
  return '#cc2222'
}

export default function PipelineSteps({
  videoMetadata,
  anchorFrames,
  stepAResult,
  stepBResult,
  stepCResult,
  stepDResult,
  staleSteps,
  runningSteps,
  onStepAComplete,
  onStepBComplete,
  onStepCComplete,
  onStepDComplete,
  onStepsMarkedStale,
  onStepsClearedStale,
  onRunningStepChange,
  onError,
  onStatusChange,
  logApiCall,
}: PipelineStepsProps) {
  const [anchorQuality, setAnchorQuality] = useState<AnchorQualityData | null>(null)
  const [anchorQualityError, setAnchorQualityError] = useState<string | null>(null)
  // Incremented each time step B completes, used to bust the browser image cache
  const [stepBVersion, setStepBVersion] = useState(0)
  // Step D smoothing params
  const [sgLongWindow, setSgLongWindow] = useState(15)
  const [sgMidWindow, setSgMidWindow] = useState(11)
  const [maxVelPx, setMaxVelPx] = useState(4.0)
  // Step B distortion mode: 0=none, 1=after H, 2=integrated

  const apiFetch = useCallback(async (url: string, options?: RequestInit): Promise<Response> => {
    const method = options?.method || 'GET'
    logApiCall(`→ ${method} ${url}`)
    const start = Date.now()
    try {
      const res = await fetch(url, options)
      logApiCall(`← ${res.status} (${Date.now() - start}ms)`)
      return res
    } catch (err) {
      logApiCall(`✗ ${String(err)}`)
      throw err
    }
  }, [logApiCall])

  const runStepA = useCallback(async () => {
    if (!videoMetadata) { onError('Please upload a video first'); return }
    onRunningStepChange('A', 'add')
    onError('')
    try {
      const res = await apiFetch(`${API_URL}/videos/${videoMetadata.video_id}/track`, { method: 'POST' })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Tracking failed')
      }
      const data = await res.json()
      const detections = await getDetections(videoMetadata.video_id)
      onStepAComplete({ frames_processed: data.frames_processed, tracks: data.tracks, num_detections: detections.length })
      onStepsMarkedStale(['B', 'C', 'D'])
      onStepsClearedStale(['A'])
    } catch (err: any) {
      onError(err.message || 'Tracking failed')
    } finally {
      onRunningStepChange('A', 'remove')
    }
  }, [videoMetadata, apiFetch, onStepAComplete, onStepsMarkedStale, onStepsClearedStale, onRunningStepChange, onError])

  const runStepB = useCallback(async () => {
    if (!videoMetadata) { onError('Please upload a video first'); return }

    // A frame is usable if it has any manual points OR any line annotations —
    // line intersections will derive the required keypoints automatically.
    const validAnnotations: AnchorFrameAnnotation[] = anchorFrames
      .filter(af => !af.isSkipped && (af.points.length > 0 || (af.lines || []).length > 0))
      .map(af => ({ frame_idx: af.frame_idx, points: af.points, lines: af.lines || [] }))

    if (validAnnotations.length === 0) {
      onError('Please annotate at least one anchor frame with points or line annotations')
      return
    }

    onRunningStepChange('B', 'add')
    onError('')
    setAnchorQuality(null)
    setAnchorQualityError(null)
    try {
      const endpoint = `${API_URL}/videos/${videoMetadata.video_id}/homographies/v3`

      const res = await apiFetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(validAnnotations),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Homography computation failed')
      }
      const data = await res.json()
      const result: StepBResult = {
        frames: data.frames || [],
        per_frame_count: data.per_frame_count ?? (data.frames || []).length,
        info: data.info || {},
      }
      onStepBComplete(result, data.frames || [])
      setStepBVersion(v => v + 1)
      onStepsMarkedStale(['C', 'D'])
      onStepsClearedStale(['B'])

      // Immediately fetch anchor quality so the user sees per-point errors
      try {
        const qRes = await apiFetch(`${API_URL}/videos/${videoMetadata.video_id}/homographies/anchor-quality`)
        if (qRes.ok) {
          const qData: AnchorQualityData = await qRes.json()
          setAnchorQuality(qData)
        } else {
          const qErr = await qRes.json()
          setAnchorQualityError(qErr.detail || 'Anchor quality fetch failed')
        }
      } catch (qe: any) {
        setAnchorQualityError(qe.message || 'Anchor quality fetch failed')
      }
    } catch (err: any) {
      onError(err.message || 'Homography computation failed')
    } finally {
      onRunningStepChange('B', 'remove')
    }
  }, [videoMetadata, anchorFrames, apiFetch, onStepBComplete, onStepsMarkedStale, onStepsClearedStale, onRunningStepChange, onError])

  const runStepC = useCallback(async () => {
    if (!videoMetadata) { onError('Please upload a video first'); return }
    if (!stepBResult) { onError('Please compute homographies first (Step B)'); return }

    onRunningStepChange('C', 'add')
    onError('')
    try {
      const positions = await mapPlayers(videoMetadata.video_id)
      onStepCComplete({ positions, total: positions.length })
      onStepsMarkedStale(['D'])
      onStepsClearedStale(['C'])
    } catch (err: any) {
      onError(err.message || 'Player mapping failed')
    } finally {
      onRunningStepChange('C', 'remove')
    }
  }, [videoMetadata, stepBResult, onStepCComplete, onStepsMarkedStale, onStepsClearedStale, onRunningStepChange, onError])

  const runStepD = useCallback(async () => {
    if (!videoMetadata) { onError('Please upload a video first'); return }
    if (!stepCResult) { onError('Please map players first (Step C)'); return }

    const startFrame = 0
    const endFrame = videoMetadata.num_frames - 1

    onRunningStepChange('D', 'add')
    onError('')
    try {
      const data = await interpolateTrajectories(videoMetadata.video_id, startFrame, endFrame, {
        sgLongWindow, sgMidWindow, maxVelPx,
      })
      onStepsClearedStale(['D'])
      const allPositions = await getPlayerPositions(videoMetadata.video_id)
      onStepDComplete(data, allPositions, startFrame, endFrame, videoMetadata.fps)
      onStatusChange('Pipeline complete!')
    } catch (err: any) {
      onError(err.message || 'Interpolation failed')
    } finally {
      onRunningStepChange('D', 'remove')
    }
  }, [videoMetadata, stepCResult, sgLongWindow, sgMidWindow, maxVelPx, onStepDComplete, onStepsClearedStale, onRunningStepChange, onError, onStatusChange])


  // A frame is valid if it has any annotations (points or lines)
  const validAnnotationCount = useMemo(
    () => anchorFrames.filter(af => !af.isSkipped && (af.points.length > 0 || (af.lines || []).length > 0)).length,
    [anchorFrames]
  )

  if (!videoMetadata) return null

  return (
    <div className="process-section">
      {/* Step A */}
      <div className="pipeline-step">
        <div className="step-header">
          <h4>Step A: Upload &amp; Run Tracking</h4>
          {staleSteps.has('A') && <span className="stale-badge">STALE</span>}
        </div>
        <button onClick={runStepA} disabled={runningSteps.has('A')} className="process-btn">
          {runningSteps.has('A') ? 'Running...' : 'Upload & Run Tracking'}
        </button>
        {stepAResult && (
          <div className="step-result">
            <p>✅ Tracking complete</p>
            <p><strong>video_id:</strong> {videoMetadata.video_id}</p>
            <p><strong>fps:</strong> {videoMetadata.fps} | <strong>frames:</strong> {videoMetadata.num_frames}</p>
            <p><strong>Detections:</strong> {stepAResult.num_detections} | <strong>Unique tracks:</strong> {stepAResult.tracks}</p>
          </div>
        )}
      </div>

      {/* Step B */}
      <div className="pipeline-step">
        <div className="step-header">
          <h4>Step B: Compute Homographies</h4>
          {staleSteps.has('B') && <span className="stale-badge">STALE</span>}
        </div>
        <button onClick={runStepB} disabled={runningSteps.has('B') || validAnnotationCount === 0} className="process-btn">
          {runningSteps.has('B') ? 'Computing...' : 'Compute Homographies'}
        </button>
        {stepBResult && (
          <div className="step-result">
            <p>✅ Homographies computed for {stepBResult.frames.length} anchor frames</p>
            <p><strong>Anchor frames:</strong> {stepBResult.frames.join(', ')}</p>
            <p><strong>Per-frame Hs propagated:</strong> {stepBResult.per_frame_count}</p>

            {/* Failed frames */}
            {(() => {
              const failed = Object.entries(stepBResult.info)
                .filter(([, v]) => v.quality === 'bad')
                .sort(([a], [b]) => Number(a) - Number(b))
              return failed.length > 0 ? (
                <p style={{ color: '#cc2222', margin: '4px 0' }}>
                  ⚠️ {failed.length} frame{failed.length > 1 ? 's' : ''} rejected:{' '}
                  {failed.map(([f, v]) => `${f} (${v.error ?? 'unknown'})`).join(', ')}
                </p>
              ) : null
            })()}

            {/* Per-anchor summary table */}
            {Object.keys(stepBResult.info).length > 0 && (
              <table className="debug-table">
                <thead>
                  <tr>
                    <th>Frame</th>
                    <th>Keypoints</th>
                    <th>Lines</th>
                    <th>Converged</th>
                    <th>Repr Error (mean)</th>
                    <th>Repr Error (max)</th>
                    <th>Warnings</th>
                  </tr>
                </thead>
                <tbody>
                  {stepBResult.frames.map(f => {
                    const info = stepBResult.info[String(f)] || {}
                    return (
                      <tr key={f}>
                        <td>{f}</td>
                        <td>{info.num_keypoints ?? '—'}</td>
                        <td>{info.valid_lines ?? 0}</td>
                        <td>{info.converged !== undefined ? (info.converged ? '✅' : '❌') : '—'}</td>
                        <td style={{ color: reprErrorColor(info.repr_mean) }}>
                          {reprErrorLabel(info.repr_mean)}
                        </td>
                        <td style={{ color: reprErrorColor(info.repr_max) }}>
                          {reprErrorLabel(info.repr_max)}
                        </td>
                        <td style={{ fontSize: '0.8em' }}>{info.line_warnings?.length ? info.line_warnings.join('; ') : '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            )}

            {/* Anchor quality — per-point reprojection breakdown */}
            {anchorQualityError && (
              <p style={{ color: '#cc2222', marginTop: '8px' }}>Anchor quality: {anchorQualityError}</p>
            )}
            {anchorQuality && anchorQuality.anchors.length > 0 && (
              <details className="step-details" style={{ marginTop: '10px' }} open>
                <summary><strong>Annotation Quality — per-keypoint reprojection errors</strong></summary>
                <div style={{ marginTop: '8px' }}>
                  {anchorQuality.anchors.map(anchor => (
                    <div key={anchor.frame_idx} style={{ marginBottom: '16px' }}>
                      <p style={{ margin: '4px 0' }}>
                        <strong>Frame {anchor.frame_idx}</strong>
                        {' — '}
                        <span style={{ color: qualityColor(anchor.overall_quality), fontWeight: 600 }}>
                          {qualityBadge(anchor.overall_quality)}
                        </span>
                        {' · '}
                        mean: <span style={{ color: reprErrorColor(anchor.mean_error_px) }}>{anchor.mean_error_px}px</span>
                        {' · '}
                        max: <span style={{ color: reprErrorColor(anchor.max_error_px) }}>{anchor.max_error_px}px</span>
                        {anchor.n_outlier_points > 0 && (
                          <span style={{ color: '#cc2222', marginLeft: '8px' }}>
                            {anchor.n_outlier_points} outlier{anchor.n_outlier_points > 1 ? 's' : ''}
                          </span>
                        )}
                      </p>
                      <p style={{ fontSize: '0.85em', color: '#555', margin: '2px 0 6px 0' }}>
                        {anchor.recommendation}
                      </p>
                      <table className="debug-table">
                        <thead>
                          <tr>
                            <th>pitch_id</th>
                            <th>error (px)</th>
                            <th>verdict</th>
                            <th>impact</th>
                          </tr>
                        </thead>
                        <tbody>
                          {anchor.keypoints
                            .slice()
                            .sort((a, b) => b.error_px - a.error_px)
                            .map(kp => (
                              <tr key={kp.pitch_id} style={{ opacity: kp.impact === 'harmful' ? 1 : 0.85 }}>
                                <td style={{ fontFamily: 'monospace', fontSize: '0.85em' }}>{kp.pitch_id}</td>
                                <td style={{ color: reprErrorColor(kp.error_px), fontWeight: kp.verdict === 'outlier' ? 700 : 400 }}>
                                  {kp.error_px}px {verdictBadge(kp.verdict)}
                                </td>
                                <td style={{ color: reprErrorColor(kp.error_px) }}>{kp.verdict}</td>
                                <td style={{ color: impactColor(kp.impact) }}>{kp.impact}</td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  ))}
                </div>
              </details>
            )}

            {/* Warped frame previews */}
            <div className="warped-thumbs">
              {stepBResult.frames.map(f => (
                <div key={f} className="warped-thumb-item">
                  <p className="thumb-label">Frame {f}</p>
                  <div className="thumb-row">
                    <div>
                      <p className="thumb-sublabel">Original</p>
                      <img src={`${API_URL}/videos/${videoMetadata.video_id}/frame/${f}`} alt={`Original frame ${f}`} className="thumb-img" />
                    </div>
                    <div>
                      <p className="thumb-sublabel">Warped</p>
                      <img src={`${API_URL}/videos/${videoMetadata.video_id}/frames/${f}/warped?v=${stepBVersion}`} alt={`Warped frame ${f}`} className="thumb-img" />
                    </div>
                    {stepCResult && (
                      <div>
                        <p className="thumb-sublabel">With Players</p>
                        <img src={`${API_URL}/videos/${videoMetadata.video_id}/frames/${f}/warped?players=true&v=${stepBVersion}`} alt={`Warped with players frame ${f}`} className="thumb-img" />
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Step C */}
      <div className="pipeline-step">
        <div className="step-header">
          <h4>Step C: Map Players to Pitch</h4>
          {staleSteps.has('C') && <span className="stale-badge">STALE</span>}
        </div>
        <button onClick={runStepC} disabled={runningSteps.has('C') || !stepBResult || !stepAResult} className="process-btn">
          {runningSteps.has('C') ? 'Mapping...' : 'Map Players'}
        </button>
        {stepCResult && (
          <div className="step-result">
            <p>✅ Mapped {stepCResult.total} player positions</p>
            <details className="step-details">
              <summary>Sample positions (first 20)</summary>
              <table className="debug-table">
                <thead>
                  <tr><th>frame_idx</th><th>track_id</th><th>x_pitch</th><th>y_pitch</th><th>source</th></tr>
                </thead>
                <tbody>
                  {stepCResult.positions.slice(0, 20).map((p, i) => (
                    <tr key={i}>
                      <td>{p.frame_idx}</td>
                      <td>#{p.track_id}</td>
                      <td>{p.x_pitch.toFixed(1)}</td>
                      <td>{p.y_pitch.toFixed(1)}</td>
                      <td>{p.source}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          </div>
        )}
      </div>

      {/* Step D */}
      <div className="pipeline-step">
        <div className="step-header">
          <h4>Step D: Interpolate Trajectories</h4>
          {staleSteps.has('D') && <span className="stale-badge">STALE</span>}
        </div>
        <div className="config-row" style={{ gap: 16, marginBottom: 10 }}>
          <label style={{ fontSize: 13 }}>
            SG window (long tracks &gt;20f):&nbsp;
            <input
              type="number" min={3} max={51} step={2} value={sgLongWindow}
              onChange={e => setSgLongWindow(Number(e.target.value))}
              style={{ width: 52 }}
            />
          </label>
          <label style={{ fontSize: 13 }}>
            SG window (mid tracks 10–20f):&nbsp;
            <input
              type="number" min={3} max={31} step={2} value={sgMidWindow}
              onChange={e => setSgMidWindow(Number(e.target.value))}
              style={{ width: 52 }}
            />
          </label>
          <label style={{ fontSize: 13 }}>
            Max vel (px/frame):&nbsp;
            <input
              type="number" min={0} max={50} step={0.5} value={maxVelPx}
              onChange={e => setMaxVelPx(Number(e.target.value))}
              style={{ width: 52 }}
            />
          </label>
        </div>
        <button onClick={runStepD} disabled={runningSteps.has('D') || !stepCResult} className="process-btn">
          {runningSteps.has('D') ? 'Interpolating...' : 'Interpolate'}
        </button>
        {stepDResult && (
          <div className="step-result">
            <p>✅ Interpolated {stepDResult.frames_generated} frames (method: {stepDResult.method})</p>
            <p>Results playback is now active below ↓</p>
          </div>
        )}
      </div>
    </div>
  )
}
