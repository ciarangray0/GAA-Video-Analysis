import { useCallback } from 'react'
import type { VideoMetadata, AnchorFrame, PlayerPosition, AnchorFrameAnnotation } from '../types'
import { API_URL, getDetections, mapPlayers, interpolateTrajectories, getPlayerPositions } from '../lib/api'

interface PipelineStepsProps {
  videoMetadata: VideoMetadata | null
  anchorFrames: AnchorFrame[]
  trimStartSeconds: number
  trimEndSeconds: number | null
  stepAResult: { frames_processed: number; tracks: number; num_detections: number } | null
  stepBResult: { frames: number[]; info: Record<string, any> } | null
  stepCResult: { positions: PlayerPosition[]; total: number } | null
  stepDResult: { frames_generated: number; method: string } | null
  staleSteps: Set<string>
  runningStep: string | null
  onStepAComplete: (result: { frames_processed: number; tracks: number; num_detections: number }) => void
  onStepBComplete: (result: { frames: number[]; info: Record<string, any> }, homographyFrames: number[]) => void
  onStepCComplete: (result: { positions: PlayerPosition[]; total: number }) => void
  onStepDComplete: (result: { frames_generated: number; method: string }, allPositions: PlayerPosition[], startFrame: number, endFrame: number, fps: number) => void
  onStepsMarkedStale: (steps: string[]) => void
  onStepsClearedStale: (steps: string[]) => void
  onRunningStepChange: (step: string | null) => void
  onError: (msg: string) => void
  onStatusChange: (msg: string) => void
  logApiCall: (entry: string) => void
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
  runningStep,
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
    onRunningStepChange('A')
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
      onRunningStepChange(null)
    }
  }, [videoMetadata, apiFetch, onStepAComplete, onStepsMarkedStale, onStepsClearedStale, onRunningStepChange, onError])

  const runStepB = useCallback(async () => {
    if (!videoMetadata) { onError('Please upload a video first'); return }
    if (!stepAResult) { onError('Please run tracking first (Step A)'); return }

    const validAnnotations: AnchorFrameAnnotation[] = anchorFrames
      .filter(af => !af.isSkipped && af.points.length >= 4)
      .map(af => ({ frame_idx: af.frame_idx, points: af.points, lines: af.lines || [] }))

    if (validAnnotations.length === 0) {
      onError('Please annotate at least one anchor frame with 4+ points')
      return
    }

    onRunningStepChange('B')
    onError('')
    try {
      const hasLines = validAnnotations.some(a => a.lines.length > 0)
      const endpoint = hasLines
        ? `${API_URL}/videos/${videoMetadata.video_id}/homographies/v2`
        : `${API_URL}/videos/${videoMetadata.video_id}/homographies`

      const res = await apiFetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(hasLines
          ? validAnnotations
          : validAnnotations.map(a => ({ frame_idx: a.frame_idx, points: a.points }))
        ),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Homography computation failed')
      }
      const data = await res.json()
      const result = { frames: data.frames || [], info: data.info || {} }
      onStepBComplete(result, data.frames || [])
      onStepsMarkedStale(['C', 'D'])
      onStepsClearedStale(['B'])
    } catch (err: any) {
      onError(err.message || 'Homography computation failed')
    } finally {
      onRunningStepChange(null)
    }
  }, [videoMetadata, stepAResult, anchorFrames, apiFetch, onStepBComplete, onStepsMarkedStale, onStepsClearedStale, onRunningStepChange, onError])

  const runStepC = useCallback(async () => {
    if (!videoMetadata) { onError('Please upload a video first'); return }
    if (!stepBResult) { onError('Please compute homographies first (Step B)'); return }

    onRunningStepChange('C')
    onError('')
    try {
      const positions = await mapPlayers(videoMetadata.video_id)
      onStepCComplete({ positions, total: positions.length })
      onStepsMarkedStale(['D'])
      onStepsClearedStale(['C'])
    } catch (err: any) {
      onError(err.message || 'Player mapping failed')
    } finally {
      onRunningStepChange(null)
    }
  }, [videoMetadata, stepBResult, onStepCComplete, onStepsMarkedStale, onStepsClearedStale, onRunningStepChange, onError])

  const runStepD = useCallback(async () => {
    if (!videoMetadata) { onError('Please upload a video first'); return }
    if (!stepCResult) { onError('Please map players first (Step C)'); return }

    const startFrame = Math.floor(trimStartSeconds * videoMetadata.fps)
    const endFrame = trimEndSeconds !== null
      ? Math.floor(trimEndSeconds * videoMetadata.fps)
      : videoMetadata.num_frames - 1

    onRunningStepChange('D')
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
      onRunningStepChange(null)
    }
  }, [videoMetadata, stepCResult, trimStartSeconds, trimEndSeconds, onStepDComplete, onStepsClearedStale, onRunningStepChange, onError, onStatusChange])

  if (!videoMetadata) return null

  return (
    <div className="process-section">
      {/* Step A */}
      <div className="pipeline-step">
        <div className="step-header">
          <h4>Step A: Upload &amp; Run Tracking</h4>
          {staleSteps.has('A') && <span className="stale-badge">STALE</span>}
        </div>
        <button onClick={runStepA} disabled={runningStep !== null} className="process-btn">
          {runningStep === 'A' ? 'Running...' : 'Upload & Run Tracking'}
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
        <button onClick={runStepB} disabled={!stepAResult || runningStep !== null} className="process-btn">
          {runningStep === 'B' ? 'Computing...' : 'Compute Homographies'}
        </button>
        {stepBResult && (
          <div className="step-result">
            <p>✅ Homographies computed for {stepBResult.frames.length} frames</p>
            <p><strong>Anchor frames:</strong> {stepBResult.frames.join(', ')}</p>
            {Object.keys(stepBResult.info).length > 0 && (
              <details className="step-details">
                <summary>Computation Info</summary>
                <table className="debug-table">
                  <thead>
                    <tr><th>Frame</th><th>Keypoints</th><th>Lines</th><th>Converged</th><th>Warnings</th></tr>
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
                          <td>{info.warnings ? info.warnings.join('; ') : '—'}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </details>
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
        <button onClick={runStepC} disabled={!stepBResult || runningStep !== null} className="process-btn">
          {runningStep === 'C' ? 'Mapping...' : 'Map Players'}
        </button>
        {stepCResult && (
          <div className="step-result">
            <p>✅ Mapped {stepCResult.total} player positions</p>
            <details className="step-details">
              <summary>Sample positions (first 20)</summary>
              <table className="debug-table">
                <thead>
                  <tr><th>frame_idx</th><th>track_id</th><th>x_pitch</th><th>y_pitch</th></tr>
                </thead>
                <tbody>
                  {stepCResult.positions.slice(0, 20).map((p, i) => (
                    <tr key={i}>
                      <td>{p.frame_idx}</td>
                      <td>#{p.track_id}</td>
                      <td>{p.x_pitch.toFixed(1)}</td>
                      <td>{p.y_pitch.toFixed(1)}</td>
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
        <button onClick={runStepD} disabled={!stepCResult || runningStep !== null} className="process-btn">
          {runningStep === 'D' ? 'Interpolating...' : 'Interpolate'}
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
