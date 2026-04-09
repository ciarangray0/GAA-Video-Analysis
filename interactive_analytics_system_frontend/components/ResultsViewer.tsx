import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import type { VideoMetadata, PlayerPosition, TeamClassifications, ClassifyTeamsSummary, KpiSummary } from '../types'
import { drawPitch } from '../lib/pitch'
import { PITCH_DISPLAY_WIDTH, PITCH_DISPLAY_HEIGHT, PITCH_CANVAS_W, PITCH_CANVAS_H } from '../lib/pitchConfig'
import { API_URL, classifyTeams, getTeamClassifications, overrideTeamClassification, computeKpis } from '../lib/api'
import TeamClassificationPanel from './TeamClassificationPanel'
import KpiPanel from './KpiPanel'
import ClipSummaryCard from './ClipSummaryCard'


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

              <ClipSummaryCard kpiSummary={kpiSummary} clipMode={clipMode} />
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
        <TeamClassificationPanel
          teamClassifications={teamClassifications}
          classifySummary={classifySummary}
          classifyError={classifyError}
          onOverrideTeam={handleOverrideTeam}
        />
      )}

      {/* KPI Panel */}
      {(kpiSummary || kpiError) && (
        <KpiPanel kpiSummary={kpiSummary} kpiError={kpiError} clipMode={clipMode} />
      )}

    </div>
  )
}
