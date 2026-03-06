import { useCallback, useMemo, useState } from 'react'
import type { VideoMetadata, AnchorFrame, PlayerPosition, AnchorFrameAnnotation } from '../types'
import { API_URL, getDetections, mapPlayers, interpolateTrajectories, getPlayerPositions } from '../lib/api'

interface StepBResult {
  frames: number[]
  per_frame_count: number
  info: Record<string, any>
}

interface PerFrameMappingFrame {
  frame: number
  y_pf: number
  y_na: number
  err_pf: number
  err_na: number
  improvement: number
}

interface PerFrameMappingSummary {
  median_pf: number
  mean_pf: number
  max_pf: number
  median_na: number
  mean_na: number
  max_na: number
  pct_better: number
}

interface PerFrameMappingData {
  frames: PerFrameMappingFrame[]
  summary: PerFrameMappingSummary | null
}

interface PipelineStepsProps {
  videoMetadata: VideoMetadata | null
  anchorFrames: AnchorFrame[]
  trimStartSeconds: number
  trimEndSeconds: number | null
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

export default function PipelineSteps({
  videoMetadata,
  anchorFrames,
  trimStartSeconds,
  trimEndSeconds,
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
  const [perFrameData, setPerFrameData] = useState<PerFrameMappingData | null>(null)
  const [loadingDiagnostic, setLoadingDiagnostic] = useState(false)
  const [diagnosticError, setDiagnosticError] = useState<string | null>(null)

  // Internal fetch wrapper that logs API calls
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
      const params = new URLSearchParams()
      params.set('trim_start', String(trimStartSeconds))
      if (trimEndSeconds !== null) params.set('trim_end', String(trimEndSeconds))
      const res = await apiFetch(`${API_URL}/videos/${videoMetadata.video_id}/track?${params}`, { method: 'POST' })
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
  }, [videoMetadata, trimStartSeconds, trimEndSeconds, apiFetch, onStepAComplete, onStepsMarkedStale, onStepsClearedStale, onRunningStepChange, onError])

  const runStepB = useCallback(async () => {
    if (!videoMetadata) { onError('Please upload a video first'); return }

    const validAnnotations: AnchorFrameAnnotation[] = anchorFrames
      .filter(af => !af.isSkipped && af.points.length >= 4)
      .map(af => ({ frame_idx: af.frame_idx, points: af.points, lines: af.lines || [] }))

    if (validAnnotations.length === 0) {
      onError('Please annotate at least one anchor frame with 4+ points')
      return
    }

    onRunningStepChange('B', 'add')
    onError('')
    try {
      // Always use v2 — it works without lines and propagates per-frame Hs
      const endpoint = `${API_URL}/videos/${videoMetadata.video_id}/homographies/v2`

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
      onStepsMarkedStale(['C', 'D'])
      onStepsClearedStale(['B'])
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

    const startFrame = Math.floor(trimStartSeconds * videoMetadata.fps)
    const endFrame = trimEndSeconds !== null
      ? Math.floor(trimEndSeconds * videoMetadata.fps)
      : videoMetadata.num_frames - 1

    onRunningStepChange('D', 'add')
    onError('')
    try {
      const data = await interpolateTrajectories(videoMetadata.video_id, startFrame, endFrame)
      onStepsClearedStale(['D'])
      const allPositions = await getPlayerPositions(videoMetadata.video_id)
      onStepDComplete(data, allPositions, startFrame, endFrame, videoMetadata.fps)
      onStatusChange('Pipeline complete!')
    } catch (err: any) {
      onError(err.message || 'Interpolation failed')
    } finally {
      onRunningStepChange('D', 'remove')
    }
  }, [videoMetadata, stepCResult, trimStartSeconds, trimEndSeconds, onStepDComplete, onStepsClearedStale, onRunningStepChange, onError, onStatusChange])

  const loadPerFrameDiagnostic = useCallback(async () => {
    if (!videoMetadata) return
    setLoadingDiagnostic(true)
    setDiagnosticError(null)
    try {
      const res = await apiFetch(`${API_URL}/videos/${videoMetadata.video_id}/diagnostics/per-frame-mapping`)
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Diagnostic failed')
      }
      const data: PerFrameMappingData = await res.json()
      setPerFrameData(data)
    } catch (err: any) {
      setDiagnosticError(err.message || 'Failed to load diagnostic')
    } finally {
      setLoadingDiagnostic(false)
    }
  }, [videoMetadata, apiFetch])

  const validAnnotationCount = useMemo(
    () => anchorFrames.filter(af => !af.isSkipped && af.points.length >= 4).length,
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
                        <td>{info.line_warnings ? info.line_warnings.join('; ') : '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            )}
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
                      <img src={`${API_URL}/videos/${videoMetadata.video_id}/frames/${f}/warped`} alt={`Warped frame ${f}`} className="thumb-img" />
                    </div>
                    {stepCResult && (
                      <div>
                        <p className="thumb-sublabel">With Players</p>
                        <img src={`${API_URL}/videos/${videoMetadata.video_id}/frames/${f}/warped_with_players`} alt={`Warped with players frame ${f}`} className="thumb-img" />
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

      {/* Per-frame mapping diagnostic (only shown after Step D) */}
      {stepDResult && (
        <div className="pipeline-step">
          <details className="step-details">
            <summary>Per-Frame Mapping Diagnostic</summary>
            <div style={{ marginTop: '8px' }}>
              <p style={{ fontSize: '0.85em', color: '#555' }}>
                Measures per-frame H vs nearest-anchor H accuracy using annotated 20m_top line points as simulated players.
                Requires at least one anchor frame annotated with a 20m_top line.
              </p>
              <button
                onClick={loadPerFrameDiagnostic}
                disabled={loadingDiagnostic}
                className="process-btn"
                style={{ marginBottom: '8px' }}
              >
                {loadingDiagnostic ? 'Loading…' : 'Load Diagnostic'}
              </button>
              {diagnosticError && <p style={{ color: 'red' }}>{diagnosticError}</p>}
              {perFrameData && perFrameData.frames.length === 0 && (
                <p>No frames available — ensure at least one anchor has a 20m_top line annotation.</p>
              )}
              {perFrameData && perFrameData.summary && (
                <>
                  <p><strong>Summary (20m_top line mapping error)</strong></p>
                  <table className="debug-table">
                    <thead>
                      <tr><th></th><th>Median (px)</th><th>Mean (px)</th><th>Max (px)</th></tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Per-frame H</td>
                        <td>{perFrameData.summary.median_pf}</td>
                        <td>{perFrameData.summary.mean_pf}</td>
                        <td>{perFrameData.summary.max_pf}</td>
                      </tr>
                      <tr>
                        <td>Nearest anchor H</td>
                        <td>{perFrameData.summary.median_na}</td>
                        <td>{perFrameData.summary.mean_na}</td>
                        <td>{perFrameData.summary.max_na}</td>
                      </tr>
                    </tbody>
                  </table>
                  <p>Per-frame H better on <strong>{perFrameData.summary.pct_better}%</strong> of frames (by &gt;2px)</p>
                </>
              )}
              {perFrameData && perFrameData.frames.length > 0 && (
                <details style={{ marginTop: '8px' }}>
                  <summary>Frame-by-frame data ({perFrameData.frames.length} frames)</summary>
                  <table className="debug-table">
                    <thead>
                      <tr>
                        <th>Frame</th>
                        <th>PF y (px)</th>
                        <th>NA y (px)</th>
                        <th>PF err (px)</th>
                        <th>NA err (px)</th>
                        <th>Improvement (px)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {perFrameData.frames.map(r => (
                        <tr key={r.frame}>
                          <td>{r.frame}</td>
                          <td>{r.y_pf}</td>
                          <td>{r.y_na}</td>
                          <td>{r.err_pf}</td>
                          <td>{r.err_na}</td>
                          <td style={{ color: r.improvement > 0 ? '#2d7a2d' : r.improvement < 0 ? '#cc2222' : undefined }}>
                            {r.improvement > 0 ? '+' : ''}{r.improvement}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </details>
              )}
            </div>
          </details>
        </div>
      )}
    </div>
  )
}
