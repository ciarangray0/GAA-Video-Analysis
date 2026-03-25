import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import type { VideoMetadata, PlayerPosition, TeamClassifications, TeamName, ClassifyTeamsSummary, KpiSummary } from '../types'
import { drawPitch, hsvToCss } from '../lib/pitch'
import { PITCH_DISPLAY_WIDTH, PITCH_DISPLAY_HEIGHT, PITCH_CANVAS_W, PITCH_CANVAS_H } from '../lib/constants'
import { API_URL, classifyTeams, getTeamClassifications, overrideTeamClassification, computeKpis } from '../lib/api'


interface ResultsViewerProps {
  videoMetadata: VideoMetadata
  videoFile: File
  playerPositions: PlayerPosition[]
  currentFrame: number
  onFrameChange: (frame: number) => void
  processedStartFrame: number
  processedEndFrame: number
  homographyFrameIndices: number[]
  processedFps: number
}

export default function ResultsViewer({
  videoMetadata,
  videoFile,
  playerPositions,
  currentFrame,
  onFrameChange,
  processedStartFrame,
  processedEndFrame,
  homographyFrameIndices,
  processedFps,
}: ResultsViewerProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(1)
  const [showBotSortOverlay, setShowBotSortOverlay] = useState(false)
  const [videoObjectUrl, setVideoObjectUrl] = useState<string | null>(null)

  // Team classification state
  const [teamClassifications, setTeamClassifications] = useState<TeamClassifications>({})
  const [classifySummary, setClassifySummary] = useState<ClassifyTeamsSummary | null>(null)
  const [isClassifying, setIsClassifying] = useState(false)
  const [classifyError, setClassifyError] = useState<string | null>(null)

  // Clip mode
  const [clipMode, setClipMode] = useState<'score' | 'defense'>('score')

  // Trails toggle
  const [showTrails, setShowTrails] = useState(false)

  // KPI state
  const [kpiSummary, setKpiSummary] = useState<KpiSummary | null>(null)
  const [isComputingKpis, setIsComputingKpis] = useState(false)
  const [kpiError, setKpiError] = useState<string | null>(null)

  // Debug panel state
  const [showMappingView, setShowMappingView] = useState(false)

  // Analysis trim — limits which positions are used for pitch display and KPI computation.
  // Video playback and annotations are unaffected.
  // trimEndFrame is the committed value (drives canvas redraws / KPI).
  // trimDragFrame is the live slider position — only updates the label while dragging.
  const [trimEndFrame, setTrimEndFrame] = useState(processedEndFrame)
  const [trimDragFrame, setTrimDragFrame] = useState(processedEndFrame)
  useEffect(() => { setTrimEndFrame(processedEndFrame); setTrimDragFrame(processedEndFrame) }, [processedEndFrame])

  const analysisPositions = useMemo(
    () => playerPositions.filter(p => p.frame_idx <= trimEndFrame),
    [playerPositions, trimEndFrame]
  )

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const videoPlayerRef = useRef<HTMLVideoElement>(null)
  const animFrameRef = useRef<number>(0)

  // Create object URL for the video file
  useEffect(() => {
    const url = URL.createObjectURL(videoFile)
    setVideoObjectUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [videoFile])

  // Load any previously computed classifications on mount
  useEffect(() => {
    getTeamClassifications(videoMetadata.video_id).then(cls => {
      if (Object.keys(cls).length > 0) setTeamClassifications(cls)
    })
  }, [videoMetadata.video_id])

  const handleClassifyTeams = useCallback(async () => {
    setIsClassifying(true)
    setClassifyError(null)
    try {
      const result = await classifyTeams(videoMetadata.video_id)
      setTeamClassifications(result.classifications)
      setClassifySummary(result.summary)
    } catch (e: any) {
      setClassifyError(e.message || 'Classification failed')
    } finally {
      setIsClassifying(false)
    }
  }, [videoMetadata.video_id])

  const handleComputeKpis = useCallback(async () => {
    setIsComputingKpis(true)
    setKpiError(null)
    try {
      const result = await computeKpis(videoMetadata.video_id, trimEndFrame)
      setKpiSummary(result)
    } catch (e: any) {
      setKpiError(e.message || 'KPI computation failed')
    } finally {
      setIsComputingKpis(false)
    }
  }, [videoMetadata.video_id, trimEndFrame])

  const handleOverrideTeam = useCallback(async (trackId: number, team: string) => {
    try {
      const updated = await overrideTeamClassification(videoMetadata.video_id, trackId, team)
      setTeamClassifications(updated)
    } catch (e: any) {
      console.error('Override failed:', e.message)
    }
  }, [videoMetadata.video_id])

  // Redraw pitch when frame, classifications, or trails toggle changes
  useEffect(() => {
    if (canvasRef.current && analysisPositions.length > 0) {
      drawPitch(canvasRef.current, analysisPositions, currentFrame, teamClassifications, showTrails)
    }
  }, [currentFrame, analysisPositions, teamClassifications, showTrails])

  const getFramesWithPositions = useCallback(() => {
    const frames = new Set(analysisPositions.map(p => p.frame_idx))
    return Array.from(frames).sort((a, b) => a - b)
  }, [analysisPositions])

  const goToFrame = useCallback((frameIdx: number) => {
    const frames = getFramesWithPositions()
    if (frames.length === 0) return
    let nearest = frames[0]
    let minDist = Math.abs(frameIdx - nearest)
    for (const f of frames) {
      const dist = Math.abs(frameIdx - f)
      if (dist < minDist) { minDist = dist; nearest = f }
    }
    onFrameChange(nearest)
    if (!isPlaying && videoPlayerRef.current) {
      videoPlayerRef.current.currentTime = nearest / videoMetadata.fps
    }
  }, [getFramesWithPositions, isPlaying, videoMetadata.fps, onFrameChange])

  const skipFrames = useCallback((delta: number) => {
    const frames = getFramesWithPositions()
    if (frames.length === 0) return
    const currentIdx = frames.indexOf(currentFrame)
    const newIdx = Math.max(0, Math.min(frames.length - 1, currentIdx + delta))
    onFrameChange(frames[newIdx])
  }, [currentFrame, getFramesWithPositions, onFrameChange])

  const stopPlayback = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current)
      animFrameRef.current = 0
    }
    if (videoPlayerRef.current) videoPlayerRef.current.pause()
    setIsPlaying(false)
  }, [])

  const onPlaybackFrame = useCallback(() => {
    const video = videoPlayerRef.current
    if (!video || video.paused) { setIsPlaying(false); return }
    const fps = processedFps || videoMetadata.fps || 25
    const frameIdx = Math.round(video.currentTime * fps)
    if (frameIdx > trimEndFrame) {
      video.pause()
      setIsPlaying(false)
      return
    }
    onFrameChange(frameIdx)
    animFrameRef.current = requestAnimationFrame(onPlaybackFrame)
  }, [processedFps, trimEndFrame, videoMetadata.fps, onFrameChange])

  const startPlayback = useCallback(() => {
    const video = videoPlayerRef.current
    if (!video || playerPositions.length === 0) return
    video.playbackRate = playbackSpeed
    const fps = processedFps || videoMetadata.fps || 25
    const startTime = processedStartFrame / fps
    // Reset to start if video has ended or is past the end; also honour startTime
    if (video.ended || video.currentTime >= trimEndFrame / fps) {
      video.currentTime = startTime
    } else if (video.currentTime < startTime) {
      video.currentTime = startTime
    }
    setIsPlaying(true)
    video.play()
      .then(() => {
        animFrameRef.current = requestAnimationFrame(onPlaybackFrame)
      })
      .catch(err => {
        console.warn('Playback blocked:', err)
        setIsPlaying(false)
      })
  }, [playbackSpeed, playerPositions.length, processedFps, processedStartFrame, trimEndFrame, videoMetadata.fps, onPlaybackFrame])

  const togglePlayback = useCallback(() => {
    isPlaying ? stopPlayback() : startPlayback()
  }, [isPlaying, startPlayback, stopPlayback])

  // Update playback rate when speed changes
  useEffect(() => {
    if (isPlaying && videoPlayerRef.current) {
      videoPlayerRef.current.playbackRate = playbackSpeed
    }
  }, [playbackSpeed, isPlaying])

  // Sync video to current frame when not playing
  useEffect(() => {
    if (!isPlaying && videoPlayerRef.current && playerPositions.length > 0) {
      const video = videoPlayerRef.current
      if (video.readyState >= 2) {
        const timeInSeconds = currentFrame / videoMetadata.fps
        if (Math.abs(video.currentTime - timeInSeconds) > 0.1) {
          video.currentTime = timeInSeconds
        }
      }
    }
  }, [currentFrame, isPlaying, videoMetadata.fps, playerPositions.length])

  // Cleanup animation frame on unmount
  useEffect(() => {
    return () => { if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current) }
  }, [])


  const framesWithPositions = getFramesWithPositions()

  return (
    <div className="results-section">
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 24, flexWrap: 'wrap', marginBottom: 8 }}>
        <h2 style={{ margin: 0 }}>4. Player Tracking Results</h2>
        <fieldset style={{ border: '1px solid #444', borderRadius: 6, padding: '4px 12px', fontSize: 13, display: 'flex', gap: 16, alignItems: 'center' }}>
          <legend style={{ fontSize: 11, color: '#aaa', padding: '0 4px' }}>Clip context</legend>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
            <input type="radio" name="clipMode" value="score" checked={clipMode === 'score'} onChange={() => setClipMode('score')} />
            Ellistown score
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
            <input type="radio" name="clipMode" value="defense" checked={clipMode === 'defense'} onChange={() => setClipMode('defense')} />
            Ellistown defense
          </label>
        </fieldset>
      </div>

      <div className="processing-info">
        <p>
          <strong>Processed frames:</strong> {processedStartFrame} - {processedEndFrame} |
          <strong> Total detections:</strong> {playerPositions.length} |
          <strong> Unique frames with players:</strong> {framesWithPositions.length} |
          <strong> Homography anchors:</strong> {homographyFrameIndices.length}
        </p>
      </div>

      {/* Playback controls */}
      <div className="playback-controls">
        <div className="playback-buttons">
          <button onClick={() => skipFrames(-10)} title="Back 10 frames">⏪</button>
          <button onClick={() => skipFrames(-1)} title="Previous frame">◀</button>
          <button onClick={togglePlayback} className="play-btn">{isPlaying ? '⏸ Pause' : '▶ Play'}</button>
          <button onClick={() => skipFrames(1)} title="Next frame">▶</button>
          <button onClick={() => skipFrames(10)} title="Forward 10 frames">⏩</button>
        </div>

        <div className="playback-options">
          <label>
            Speed:
            <select value={playbackSpeed} onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}>
              <option value={0.25}>0.25x</option>
              <option value={0.5}>0.5x</option>
              <option value={1}>1x</option>
              <option value={2}>2x</option>
              <option value={4}>4x</option>
            </select>
          </label>
          <button onClick={() => setShowBotSortOverlay(!showBotSortOverlay)} className={`sidebar-toggle ${showBotSortOverlay ? 'active' : ''}`}>
            🎯 BotSort Overlay
          </button>
          <button onClick={() => setShowTrails(t => !t)} className={`sidebar-toggle ${showTrails ? 'active' : ''}`}>
            〰 Show Trails
          </button>
          <button onClick={handleClassifyTeams} disabled={isClassifying} className="sidebar-toggle">
            {isClassifying ? '⏳ Classifying…' : '🎽 Classify Teams'}
          </button>
          <button onClick={handleComputeKpis} disabled={isComputingKpis} className="sidebar-toggle">
            {isComputingKpis ? '⏳ Computing…' : '📊 Compute KPIs'}
          </button>
        </div>
      </div>

      {/* Frame slider */}
      <div className="frame-slider">
        <input
          type="range"
          min={framesWithPositions[0] || 0}
          max={framesWithPositions[framesWithPositions.length - 1] || 100}
          value={currentFrame}
          onChange={(e) => goToFrame(parseInt(e.target.value))}
          className="slider"
        />
        <span className="frame-info">
          Frame {currentFrame} / {framesWithPositions[framesWithPositions.length - 1] || 0}
          {` (${(currentFrame / videoMetadata.fps).toFixed(2)}s)`}
        </span>
      </div>

      {/* Analysis trim slider */}
      <div className="frame-slider" style={{ gap: 10 }}>
        <span style={{ fontSize: 12, color: '#aaa', whiteSpace: 'nowrap' }}>Analysis trim end:</span>
        <input
          type="range"
          min={processedStartFrame}
          max={processedEndFrame}
          value={trimDragFrame}
          onChange={(e) => setTrimDragFrame(parseInt(e.target.value))}
          className="slider"
          style={{ accentColor: '#ff9900' }}
        />
        <span className="frame-info" style={{ color: '#8b949e', whiteSpace: 'nowrap', minWidth: 120 }}>
          {`frame ${trimDragFrame} / ${processedEndFrame} (${(trimDragFrame / videoMetadata.fps).toFixed(1)}s)`}
        </span>
        <button
          onClick={() => setTrimEndFrame(trimDragFrame)}
          style={{
            padding: '2px 8px', fontSize: 11, margin: 0,
            background: trimDragFrame === trimEndFrame ? '#1a7a3a' : '#b45a00',
            color: '#fff',
          }}
        >
          {trimDragFrame === trimEndFrame ? '✓ Trim applied' : 'Apply trim'}
        </button>
        <button
          onClick={() => { setTrimDragFrame(processedEndFrame); setTrimEndFrame(processedEndFrame) }}
          style={{ padding: '2px 8px', fontSize: 11, margin: 0, background: '#555', color: '#ccc' }}
        >
          Reset
        </button>
      </div>

      {/* Main content area */}
      <div className="results-content">
          {/* Side-by-side view */}
        <div className="results-main">
          <div className="video-frame-panel">
            <h4>Video Frame {currentFrame}</h4>
            {videoObjectUrl ? (
              <video
                ref={videoPlayerRef}
                src={videoObjectUrl}
                className="results-video"
                muted
                playsInline
                onTimeUpdate={() => {
                  if (videoPlayerRef.current && !isPlaying) {
                    const frameFromVideo = Math.round(videoPlayerRef.current.currentTime * videoMetadata.fps)
                    if (Math.abs(frameFromVideo - currentFrame) > 1) goToFrame(frameFromVideo)
                  }
                }}
              />
            ) : (
              <div className="video-placeholder"><p>Video not available</p></div>
            )}

            {/* Mapping view — warped frame for the current position, sits below the video */}
            <details className="debug-panel" style={{ marginTop: 8 }} onToggle={(e) => setShowMappingView((e.target as HTMLDetailsElement).open)}>
              <summary>🗺️ Mapping View</summary>
              {showMappingView && (
                <div className="debug-panel-body">
                  <p className="debug-info">
                    Frame {currentFrame} —{' '}
                    {homographyFrameIndices.includes(currentFrame)
                      ? <strong style={{ color: 'green' }}>anchor frame</strong>
                      : <span style={{ color: '#888' }}>propagated frame</span>}
                  </p>
                  <div style={{ overflow: 'hidden', border: '1px solid #ccc', borderRadius: 4 }}>
                    <img
                      key={currentFrame}
                      src={`${API_URL}/videos/${videoMetadata.video_id}/frames/${currentFrame}/warped`}
                      alt={`Warped frame ${currentFrame} with pitch lines`}
                      style={{ width: '100%', height: 'auto', display: 'block' }}
                      onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
                    />
                  </div>
                </div>
              )}
            </details>
          </div>

          <div className="pitch-view-panel">
            <h4>2D Pitch View</h4>
            <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
              <div>
                <canvas ref={canvasRef} width={PITCH_DISPLAY_WIDTH} height={PITCH_DISPLAY_HEIGHT} className="pitch-canvas" />
                <div className="pitch-legend">
                  {Object.keys(teamClassifications).length > 0 ? (
                    <>
                      <span style={{ color: '#FFD700' }}>● Ellistown</span>
                      <span style={{ color: '#4488FF', marginLeft: 12 }}>● Opposition</span>
                      <span style={{ color: '#888', marginLeft: 12 }}>● Unclassified</span>
                    </>
                  ) : (
                    <span>● Each player has a unique color based on their track ID</span>
                  )}
                </div>
              </div>

              {kpiSummary && (() => {
                const { spatial_summary, spatial_timeseries, zone_balance_timeseries } = kpiSummary

                // Average zone counts across all frames
                const totals: Record<string, Record<string, number>> = {}
                let frameCount = 0
                for (const frame of Object.values(zone_balance_timeseries)) {
                  frameCount++
                  for (const [team, zones] of Object.entries(frame)) {
                    if (!totals[team]) totals[team] = { defensive: 0, middle: 0, attacking: 0 }
                    totals[team].defensive += zones.defensive
                    totals[team].middle += zones.middle
                    totals[team].attacking += zones.attacking
                  }
                }

                // Auto-detect active zone from ALL players (including unclassified),
                // so zone detection works even before Classify Teams is run.
                const zoneActivity = { defensive: 0, middle: 0, attacking: 0 }
                for (const zones of Object.values(totals)) {
                  zoneActivity.defensive += zones.defensive
                  zoneActivity.middle    += zones.middle
                  zoneActivity.attacking += zones.attacking
                }
                const detectedZone = (Object.entries(zoneActivity) as [string, number][])
                  .sort(([, a], [, b]) => b - a)[0][0] as 'defensive' | 'middle' | 'attacking'
                const ZONE_RANGES = { defensive: '0–47m', middle: '47–93m', attacking: '93–140m' } as const
                const yRangeLabel = ZONE_RANGES[detectedZone]
                const clipZoneLabel = clipMode === 'score' ? 'attacking' : 'defensive'

                // Zone balance — one decimal, classified players only.
                // Falls back to total (unclassified) if Classify Teams hasn't been run.
                let zoneSentence = ''
                if (frameCount > 0) {
                  const ellNf = (totals['ellistown']?.[detectedZone] ?? 0) / frameCount
                  const oppNf = (totals['opposition']?.[detectedZone] ?? 0) / frameCount
                  const prefix = `In the ${clipZoneLabel} third (${yRangeLabel})`

                  if (ellNf === 0 && oppNf === 0) {
                    // No classified players — show total and prompt classification
                    const totalN = (zoneActivity[detectedZone] / frameCount).toFixed(1)
                    zoneSentence = `${prefix}: ${totalN} players avg (classify teams for breakdown)`
                  } else {
                    const ellN = ellNf.toFixed(1)
                    const oppN = oppNf.toFixed(1)
                    if (clipMode === 'score') {
                      if (oppNf > ellNf)
                        zoneSentence = `${prefix}: Opposition defended with a ${oppN}v${ellN} advantage`
                      else if (ellNf > oppNf)
                        zoneSentence = `${prefix}: Ellistown had a ${ellN}v${oppN} numerical overload`
                      else
                        zoneSentence = `${prefix}: Teams level at ${ellN}v${oppN}`
                    } else {
                      if (oppNf > ellNf)
                        zoneSentence = `${prefix}: Opposition had a ${oppN}v${ellN} numerical advantage`
                      else if (ellNf > oppNf)
                        zoneSentence = `${prefix}: Ellistown defended with a ${ellN}v${oppN} advantage`
                      else
                        zoneSentence = `${prefix}: Teams level at ${ellN}v${oppN}`
                    }
                  }
                }

                // Spread sentence
                const ellSpread = spatial_summary.per_team['ellistown']?.mean_spread_m2
                const oppSpread = spatial_summary.per_team['opposition']?.mean_spread_m2
                const spreadSentence = (ellSpread != null && oppSpread != null)
                  ? `Opposition spread: ${oppSpread} m²  /  Ellistown spread: ${ellSpread} m²`
                  : null

                // Depth: compare centroid y at the start and end of the clip rather than
                // the average — averages hide the movement that actually matters.
                // "goal-side" = closer to the contested goal:
                //   detectedZone='defensive' → goal at y=0, lower y = goal-side
                //   detectedZone='attacking' → goal at y=140, higher y = goal-side
                const tsKeys = Object.keys(spatial_timeseries).map(Number).sort((a, b) => a - b)
                const bothPresent = tsKeys.filter(k => {
                  const f = spatial_timeseries[k.toString()]
                  return f?.teams?.['ellistown']?.centroid_y_m != null
                    && f?.teams?.['opposition']?.centroid_y_m != null
                })
                let depthSentence: string | null = null
                if (bothPresent.length >= 2) {
                  const f0 = spatial_timeseries[bothPresent[0].toString()]
                  const f1 = spatial_timeseries[bothPresent[bothPresent.length - 1].toString()]
                  const eY0 = f0.teams['ellistown'].centroid_y_m
                  const oY0 = f0.teams['opposition'].centroid_y_m
                  const eY1 = f1.teams['ellistown'].centroid_y_m
                  const oY1 = f1.teams['opposition'].centroid_y_m
                  const oppGoalSide = (eY: number, oY: number) =>
                    detectedZone === 'attacking' ? oY > eY : oY < eY
                  const desc = (eY: number, oY: number): string => {
                    const gs = oppGoalSide(eY, oY)
                    const gap = Math.abs(eY - oY).toFixed(1)
                    if (clipMode === 'score') {
                      return gs
                        ? `Opposition ${gap}m goal-side`
                        : `Ellistown ${gap}m goal-side`
                    } else {
                      return gs
                        ? `Opposition ${gap}m goal-side`
                        : `Ellistown ${gap}m goal-side`
                    }
                  }
                  depthSentence = `Clip start: ${desc(eY0, oY0)} · Clip end: ${desc(eY1, oY1)}`
                }

                return (
                  <div style={{
                    background: '#1a2a1a',
                    border: '1px solid #3a5a3a',
                    borderRadius: 8,
                    padding: '14px 16px',
                    minWidth: 200,
                    maxWidth: 260,
                    fontSize: 13,
                    lineHeight: 1.6,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 12,
                  }}>
                    <div style={{ fontSize: 11, color: '#aaa', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      Clip summary · {clipMode === 'score' ? 'Ellistown score' : 'Ellistown defense'}
                    </div>

                    {zoneSentence && <div style={{ color: '#e0e0e0' }}>{zoneSentence}</div>}
                    {spreadSentence && <div style={{ color: '#e0e0e0' }}>{spreadSentence}</div>}
                    {depthSentence && <div style={{ color: '#e0e0e0' }}>{depthSentence}</div>}

                    {!zoneSentence && !spreadSentence && !depthSentence && (
                      <div style={{ color: '#888' }}>Run Compute KPIs to see summary</div>
                    )}
                  </div>
                )
              })()}

              {!kpiSummary && (
                <div style={{
                  background: '#1a2a1a',
                  border: '1px solid #3a5a3a',
                  borderRadius: 8,
                  padding: '14px 16px',
                  minWidth: 200,
                  maxWidth: 260,
                  fontSize: 13,
                  color: '#666',
                  lineHeight: 1.6,
                }}>
                  <div style={{ fontSize: 11, color: '#555', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
                    Clip summary
                  </div>
                  Press <strong style={{ color: '#888' }}>Compute KPIs</strong> to see zone balance, team spread, and centroid separation for this clip.
                </div>
              )}
            </div>
          </div>

        </div>
      </div>

      {showBotSortOverlay && (
        <div className="botsort-overlay-section">
          <h4>BotSort Detections — Frame {currentFrame}</h4>
          <img
            key={currentFrame}
            src={`${API_URL}/videos/${videoMetadata.video_id}/frames/${currentFrame}/detections_overlay`}
            alt={`BotSort detections frame ${currentFrame}`}
            className="botsort-overlay-img"
            onError={(e) => {
              const img = e.target as HTMLImageElement
              img.style.display = 'none'
              img.insertAdjacentHTML('afterend', '<p style="color:#888;padding:8px">No detections available for this frame</p>')
            }}
          />
        </div>
      )}

      {/* Player list for current frame */}
      <div className="current-frame-players">
        <h4>Players in Frame {currentFrame}</h4>
        <div className="player-list">
          {analysisPositions.filter(p => p.frame_idx === currentFrame).map((pos, idx) => (
            <span key={idx} className="player-badge">
              #{pos.track_id}: ({(pos.x_pitch / 10).toFixed(1)}m, {(pos.y_pitch / 10).toFixed(1)}m)
              <small style={{ color: '#888' }}> [{Math.round(pos.x_pitch)}, {Math.round(pos.y_pitch)}px]</small>
              <small>{pos.source}</small>
            </span>
          ))}
          {analysisPositions.filter(p => p.frame_idx === currentFrame).length === 0 && (
            <span className="no-players">No players detected in this frame</span>
          )}
        </div>
      </div>

      {/* Team Classification Panel */}
      {(Object.keys(teamClassifications).length > 0 || classifyError) && (
        <details className="debug-panel" open>
          <summary>🎽 Team Classifications</summary>
          <div className="debug-panel-body">
            {classifyError && <p style={{ color: '#f88' }}>{classifyError}</p>}

            {classifySummary && (
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 12, fontSize: 13 }}>
                <span>Ellistown: <strong>{classifySummary.num_ellistown}</strong></span>
                <span>Opposition: <strong>{classifySummary.num_opposition}</strong></span>
                <span>Avg confidence: <strong>{(classifySummary.mean_confidence * 100).toFixed(0)}%</strong></span>
                {classifySummary.hsv_cluster_separation !== null && (
                  <span>HSV separation: <strong>{classifySummary.hsv_cluster_separation}</strong></span>
                )}
                {classifySummary.low_confidence_tracks.length > 0 && (
                  <span style={{ color: '#f88' }}>
                    Low-confidence tracks: {classifySummary.low_confidence_tracks.join(', ')}
                  </span>
                )}
              </div>
            )}

            {(['ellistown', 'opposition', 'referee', 'ignore'] as TeamName[]).map(team => {
              const trackIds = Object.entries(teamClassifications)
                .filter(([, v]) => v.team === team)
                .map(([k]) => parseInt(k))
                .sort((a, b) => a - b)
              if (trackIds.length === 0) return null
              return (
                <div key={team} style={{ marginBottom: 12 }}>
                  <strong style={{ textTransform: 'capitalize' }}>{team}</strong>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 4 }}>
                    {trackIds.map(tid => {
                      const cls = teamClassifications[tid.toString()]
                      const [h, s, v] = cls.mean_hsv
                      const swatchColor = hsvToCss(h, s, v)
                      return (
                        <div key={tid} style={{
                          display: 'flex', alignItems: 'center', gap: 6,
                          background: '#1a1a2e', borderRadius: 6, padding: '4px 8px', fontSize: 12,
                        }}>
                          <div style={{
                            width: 14, height: 14, borderRadius: '50%',
                            background: swatchColor, border: '1px solid #555', flexShrink: 0,
                          }} title={`HSV: ${h.toFixed(0)}, ${s.toFixed(0)}, ${v.toFixed(0)}`} />
                          <span>#{tid}</span>
                          <div style={{
                            width: 40, height: 4, background: '#333', borderRadius: 2, overflow: 'hidden',
                          }}>
                            <div style={{
                              width: `${cls.confidence * 100}%`, height: '100%',
                              background: cls.confidence >= 0.6 ? '#4caf50' : '#f88',
                            }} />
                          </div>
                          <select
                            value={cls.team}
                            onChange={(e) => handleOverrideTeam(tid, e.target.value)}
                            style={{ fontSize: 11, background: '#222', color: '#fff', border: '1px solid #555', borderRadius: 3 }}
                          >
                            <option value="ellistown">Ellistown</option>
                            <option value="opposition">Opposition</option>
                            <option value="referee">Referee</option>
                            <option value="ignore">Ignore</option>
                          </select>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        </details>
      )}

      {/* KPI Panel */}
      {(kpiSummary || kpiError) && (
        <details className="debug-panel" open>
          <summary>📊 Spatial KPIs</summary>
          <div className="debug-panel-body">
            {kpiError && <p style={{ color: '#f88' }}>{kpiError}</p>}

            {kpiSummary && (() => {
              const { clip_meta: meta, per_player, spatial_summary, spatial_timeseries, zone_balance_timeseries } = kpiSummary

              const teamColor = (team: string) => {
                if (team === 'ellistown') return '#FFD700'
                if (team === 'opposition') return '#4488FF'
                return '#888'
              }

              // Zone balance: average across all frames
              const teamZoneTotals: Record<string, Record<string, number>> = {}
              const frameCount = Object.keys(zone_balance_timeseries).length
              for (const frame of Object.values(zone_balance_timeseries)) {
                for (const [team, zones] of Object.entries(frame)) {
                  if (!teamZoneTotals[team]) teamZoneTotals[team] = { defensive: 0, middle: 0, attacking: 0 }
                  teamZoneTotals[team].defensive += zones.defensive
                  teamZoneTotals[team].middle += zones.middle
                  teamZoneTotals[team].attacking += zones.attacking
                }
              }

              // Detect active zone from all players (same logic as clip summary panel)
              const zoneActivity = { defensive: 0, middle: 0, attacking: 0 }
              for (const zones of Object.values(teamZoneTotals)) {
                zoneActivity.defensive += zones.defensive
                zoneActivity.middle    += zones.middle
                zoneActivity.attacking += zones.attacking
              }
              const detectedZone = (Object.entries(zoneActivity) as [string, number][])
                .sort(([, a], [, b]) => b - a)[0][0] as 'defensive' | 'middle' | 'attacking'
              const clipZoneLabel = clipMode === 'score' ? 'attacking' : 'defensive'
              // Column labels:
              //   middle       → always "midfield"
              //   detectedZone → clipMode label ("attacking third" / "defensive third")
              //   opposite end → inverse label
              const colLabel = (zone: 'defensive' | 'middle' | 'attacking') => {
                const ranges = { defensive: '0–47m', middle: '47–93m', attacking: '93–140m' }
                if (zone === 'middle') return `midfield (${ranges.middle})`
                if (zone === detectedZone) return `${clipZoneLabel} third (${ranges[zone]})`
                const oppositeLabel = clipMode === 'score' ? 'defensive third' : 'attacking third'
                return `${oppositeLabel} (${ranges[zone]})`
              }

              // Depth differential (same calculation as clip summary panel)
              const ellCentY = spatial_summary.per_team['ellistown']?.mean_centroid_y_m
              const oppCentY = spatial_summary.per_team['opposition']?.mean_centroid_y_m
              let depthSentence: string | null = null
              // Start→end centroid depth comparison (same logic as clip summary)
              {
                const tsKeys2 = Object.keys(spatial_timeseries).map(Number).sort((a, b) => a - b)
                const bp2 = tsKeys2.filter(k => {
                  const f = spatial_timeseries[k.toString()]
                  return f?.teams?.['ellistown']?.centroid_y_m != null
                    && f?.teams?.['opposition']?.centroid_y_m != null
                })
                if (bp2.length >= 2) {
                  const f0 = spatial_timeseries[bp2[0].toString()]
                  const f1 = spatial_timeseries[bp2[bp2.length - 1].toString()]
                  const eY0 = f0.teams['ellistown'].centroid_y_m
                  const oY0 = f0.teams['opposition'].centroid_y_m
                  const eY1 = f1.teams['ellistown'].centroid_y_m
                  const oY1 = f1.teams['opposition'].centroid_y_m
                  const oppGs = (eY: number, oY: number) =>
                    detectedZone === 'attacking' ? oY > eY : oY < eY
                  const desc2 = (eY: number, oY: number): string => {
                    const gs = oppGs(eY, oY)
                    const gap = Math.abs(eY - oY).toFixed(1)
                    return gs ? `Opposition ${gap}m goal-side` : `Ellistown ${gap}m goal-side`
                  }
                  depthSentence = `Clip start: ${desc2(eY0, oY0)} · Clip end: ${desc2(eY1, oY1)}`
                }
              }

              return (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  {/* Clip meta */}
                  <div style={{ fontSize: 12, color: '#aaa' }}>
                    {meta.duration_s.toFixed(1)} s &nbsp;·&nbsp; {meta.total_frames} frames &nbsp;·&nbsp; {meta.fps} fps
                  </div>

                  {/* Centroid metrics — 2D separation + y/x components */}
                  {(spatial_summary.centroid_separation_m || (ellCentY != null && oppCentY != null)) && (
                    <div>
                      <strong style={{ fontSize: 13 }}>Team centroids</strong>
                      <table className="debug-table" style={{ fontSize: 12, marginTop: 6 }}>
                        <thead>
                          <tr><th>Metric</th><th>Value</th><th>Note</th></tr>
                        </thead>
                        <tbody>
                          {spatial_summary.centroid_separation_m && <>
                            <tr>
                              <td>2D separation (mean)</td>
                              <td><strong>{spatial_summary.centroid_separation_m.mean} m</strong></td>
                              <td style={{ color: '#888' }}>overall compactness</td>
                            </tr>
                            <tr>
                              <td>2D separation (min / max)</td>
                              <td>{spatial_summary.centroid_separation_m.min} – {spatial_summary.centroid_separation_m.max} m</td>
                              <td style={{ color: '#888' }}>range across clip</td>
                            </tr>
                          </>}
                          {ellCentY != null && oppCentY != null && (() => {
                            const ellCentX = spatial_summary.per_team['ellistown']?.mean_centroid_x_m
                            const oppCentX = spatial_summary.per_team['opposition']?.mean_centroid_x_m
                            const yGap = Math.abs(ellCentY - oppCentY).toFixed(1)
                            const xGap = (ellCentX != null && oppCentX != null)
                              ? Math.abs(ellCentX - oppCentX).toFixed(1)
                              : null
                            const deeper = detectedZone === 'attacking'
                              ? (ellCentY > oppCentY ? 'Ellistown' : 'Opposition')
                              : detectedZone === 'defensive'
                              ? (ellCentY < oppCentY ? 'Ellistown' : 'Opposition')
                              : clipMode === 'score'
                              ? (ellCentY > oppCentY ? 'Ellistown' : 'Opposition')
                              : (ellCentY < oppCentY ? 'Ellistown' : 'Opposition')
                            return <>
                              <tr>
                                <td>Depth gap (y-axis)</td>
                                <td><strong>{yGap} m</strong></td>
                                <td style={{ color: '#888' }}>{deeper} closer to goal</td>
                              </tr>
                              {xGap != null && (
                                <tr>
                                  <td>Lateral gap (x-axis)</td>
                                  <td>{xGap} m</td>
                                  <td style={{ color: '#888' }}>horizontal offset</td>
                                </tr>
                              )}
                              <tr>
                                <td>Ellistown centroid</td>
                                <td>({ellCentX?.toFixed(1)}, {ellCentY.toFixed(1)}) m</td>
                                <td style={{ color: '#888' }}>x, y</td>
                              </tr>
                              <tr>
                                <td>Opposition centroid</td>
                                <td>({oppCentX?.toFixed(1)}, {oppCentY.toFixed(1)}) m</td>
                                <td style={{ color: '#888' }}>x, y</td>
                              </tr>
                            </>
                          })()}
                        </tbody>
                      </table>
                      {depthSentence && (
                        <div style={{ marginTop: 8, fontSize: 12, color: '#ccc', lineHeight: 1.5 }}>
                          {depthSentence}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Team spread */}
                  {Object.keys(spatial_summary.per_team).length > 0 && (
                    <div>
                      <strong style={{ fontSize: 13 }}>Team spread (mean convex hull)</strong>
                      <div style={{ display: 'flex', gap: 24, marginTop: 6, flexWrap: 'wrap' }}>
                        {Object.entries(spatial_summary.per_team)
                          .filter(([t]) => t !== 'unclassified')
                          .map(([team, s]) => (
                            <div key={team}>
                              <span style={{ color: teamColor(team), fontWeight: 600, textTransform: 'capitalize', fontSize: 12 }}>
                                {team}
                              </span>
                              <div style={{ fontSize: 13 }}>{s.mean_spread_m2} m²</div>
                              <div style={{ fontSize: 11, color: '#888' }}>
                                centroid ({s.mean_centroid_x_m}, {s.mean_centroid_y_m}) m
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}

                  {/* Zone balance — all three zones; active zone column highlighted */}
                  {Object.keys(teamZoneTotals).length > 0 && (
                    <div>
                      <strong style={{ fontSize: 13 }}>Zone balance (avg players per frame)</strong>
                      <table className="debug-table" style={{ fontSize: 12, marginTop: 6 }}>
                        <thead>
                          <tr>
                            <th>Team</th>
                            <th style={detectedZone === 'defensive' ? { color: '#FFD700' } : {}}>{colLabel('defensive')}</th>
                            <th style={detectedZone === 'middle'    ? { color: '#FFD700' } : {}}>{colLabel('middle')}</th>
                            <th style={detectedZone === 'attacking' ? { color: '#FFD700' } : {}}>{colLabel('attacking')}</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(teamZoneTotals)
                            .filter(([t]) => t !== 'unclassified')
                            .map(([team, zones]) => (
                              <tr key={team}>
                                <td style={{ color: teamColor(team), fontWeight: 600, textTransform: 'capitalize' }}>{team}</td>
                                <td style={detectedZone === 'defensive' ? { fontWeight: 600 } : { color: '#888' }}>{frameCount > 0 ? (zones.defensive / frameCount).toFixed(1) : 0}</td>
                                <td style={detectedZone === 'middle'    ? { fontWeight: 600 } : { color: '#888' }}>{frameCount > 0 ? (zones.middle    / frameCount).toFixed(1) : 0}</td>
                                <td style={detectedZone === 'attacking' ? { fontWeight: 600 } : { color: '#888' }}>{frameCount > 0 ? (zones.attacking / frameCount).toFixed(1) : 0}</td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                  {/* Per-player distance */}
                  <div>
                    <strong style={{ fontSize: 13 }}>Distance covered</strong>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
                      {Object.entries(per_player)
                        .filter(([, m]) => m.team !== 'referee' && m.team !== 'ignore')
                        .sort(([, a], [, b]) => b.total_distance_m - a.total_distance_m)
                        .map(([tid, m]) => (
                          <span key={tid} style={{
                            background: '#1a1a2e', borderRadius: 6, padding: '3px 8px',
                            fontSize: 12, border: '1px solid #333',
                          }}>
                            <span style={{ color: teamColor(m.team) }}>#{tid}</span>
                            {' '}{m.total_distance_m} m
                          </span>
                        ))}
                    </div>
                  </div>
                </div>
              )
            })()}
          </div>
        </details>
      )}

    </div>
  )
}
