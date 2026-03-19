import { useState, useRef, useEffect, useCallback } from 'react'
import type { VideoMetadata, PlayerPosition } from '../types'
import { drawPitch } from '../lib/pitch'
import { PITCH_DISPLAY_WIDTH, PITCH_DISPLAY_HEIGHT, PITCH_CANVAS_W, PITCH_CANVAS_H } from '../lib/constants'
import { API_URL } from '../lib/api'


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
  const [isSyncMode, setIsSyncMode] = useState(true)
  const [showBotSortOverlay, setShowBotSortOverlay] = useState(false)
  const [videoObjectUrl, setVideoObjectUrl] = useState<string | null>(null)

  // Debug panel state
  const [showMappingView, setShowMappingView] = useState(false)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const videoPlayerRef = useRef<HTMLVideoElement>(null)
  const animFrameRef = useRef<number>(0)

  // Create object URL for the video file
  useEffect(() => {
    const url = URL.createObjectURL(videoFile)
    setVideoObjectUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [videoFile])

  // Redraw pitch when frame changes
  useEffect(() => {
    if (canvasRef.current && playerPositions.length > 0) {
      drawPitch(canvasRef.current, playerPositions, currentFrame)
    }
  }, [currentFrame, playerPositions])

  const getFramesWithPositions = useCallback(() => {
    const frames = new Set(playerPositions.map(p => p.frame_idx))
    return Array.from(frames).sort((a, b) => a - b)
  }, [playerPositions])

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
    if (frameIdx > processedEndFrame) {
      video.pause()
      setIsPlaying(false)
      return
    }
    onFrameChange(frameIdx)
    animFrameRef.current = requestAnimationFrame(onPlaybackFrame)
  }, [processedFps, processedEndFrame, videoMetadata.fps, onFrameChange])

  const startPlayback = useCallback(() => {
    const video = videoPlayerRef.current
    if (!video || playerPositions.length === 0) return
    video.playbackRate = playbackSpeed
    const fps = processedFps || videoMetadata.fps || 25
    const startTime = processedStartFrame / fps
    // Reset to start if video has ended or is past the end; also honour startTime
    if (video.ended || video.currentTime >= processedEndFrame / fps) {
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
  }, [playbackSpeed, playerPositions.length, processedFps, processedStartFrame, processedEndFrame, videoMetadata.fps, onPlaybackFrame])

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
    if (!isPlaying && isSyncMode && videoPlayerRef.current && playerPositions.length > 0) {
      const video = videoPlayerRef.current
      if (video.readyState >= 2) {
        const timeInSeconds = currentFrame / videoMetadata.fps
        if (Math.abs(video.currentTime - timeInSeconds) > 0.1) {
          video.currentTime = timeInSeconds
        }
      }
    }
  }, [currentFrame, isPlaying, isSyncMode, videoMetadata.fps, playerPositions.length])

  // Cleanup animation frame on unmount
  useEffect(() => {
    return () => { if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current) }
  }, [])


  const framesWithPositions = getFramesWithPositions()

  return (
    <div className="results-section">
      <h2>4. Player Tracking Results</h2>

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
          <button onClick={() => setIsSyncMode(!isSyncMode)} className={`sync-btn ${isSyncMode ? 'active' : ''}`}>
            🔗 {isSyncMode ? 'Sync ON' : 'Sync OFF'}
          </button>
          <button onClick={() => setShowBotSortOverlay(!showBotSortOverlay)} className={`sidebar-toggle ${showBotSortOverlay ? 'active' : ''}`}>
            🎯 BotSort Overlay
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
                  if (isSyncMode && videoPlayerRef.current && !isPlaying) {
                    const frameFromVideo = Math.round(videoPlayerRef.current.currentTime * videoMetadata.fps)
                    if (Math.abs(frameFromVideo - currentFrame) > 1) goToFrame(frameFromVideo)
                  }
                }}
              />
            ) : (
              <div className="video-placeholder"><p>Video not available</p></div>
            )}
          </div>

          <div className="pitch-view-panel">
            <h4>2D Pitch View</h4>
            <canvas ref={canvasRef} width={PITCH_DISPLAY_WIDTH} height={PITCH_DISPLAY_HEIGHT} className="pitch-canvas" />
            <div className="pitch-legend">
              <span>● Each player has a unique color based on their track ID</span>
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
          {playerPositions.filter(p => p.frame_idx === currentFrame).map((pos, idx) => (
            <span key={idx} className="player-badge">
              #{pos.track_id}: ({pos.x_pitch.toFixed(1)}, {pos.y_pitch.toFixed(1)})
              <small>{pos.source}</small>
            </span>
          ))}
          {playerPositions.filter(p => p.frame_idx === currentFrame).length === 0 && (
            <span className="no-players">No players detected in this frame</span>
          )}
        </div>
      </div>

      {/* Debug coordinate table */}
      <div className="debug-coordinates-panel">
        <h4>🔍 Debug: Player Coordinates (Frame {currentFrame})</h4>
        <p className="debug-info">
          Expected ranges: x_pitch: 0-{PITCH_CANVAS_W}, y_pitch: 0-{PITCH_CANVAS_H} →
          Display: 0-{PITCH_DISPLAY_WIDTH} × 0-{PITCH_DISPLAY_HEIGHT}
        </p>
        <div className="debug-table-container">
          <table className="debug-table">
            <thead>
              <tr>
                <th>Track ID</th><th>x_pitch</th><th>y_pitch</th>
                <th>x_display</th><th>y_display</th><th>Source</th><th>Status</th>
              </tr>
            </thead>
            <tbody>
              {playerPositions
                .filter(p => p.frame_idx === currentFrame)
                .sort((a, b) => a.track_id - b.track_id)
                .map((pos, idx) => {
                  const xDisplay = (pos.x_pitch / PITCH_CANVAS_W) * PITCH_DISPLAY_WIDTH
                  const yDisplay = (pos.y_pitch / PITCH_CANVAS_H) * PITCH_DISPLAY_HEIGHT
                  const isOutOfBounds = pos.x_pitch < 0 || pos.x_pitch > PITCH_CANVAS_W || pos.y_pitch < 0 || pos.y_pitch > PITCH_CANVAS_H
                  return (
                    <tr key={idx} className={isOutOfBounds ? 'out-of-bounds' : ''}>
                      <td><strong>#{pos.track_id}</strong></td>
                      <td className={pos.x_pitch < 0 || pos.x_pitch > PITCH_CANVAS_W ? 'bad-value' : ''}>{pos.x_pitch.toFixed(2)}</td>
                      <td className={pos.y_pitch < 0 || pos.y_pitch > PITCH_CANVAS_H ? 'bad-value' : ''}>{pos.y_pitch.toFixed(2)}</td>
                      <td>{xDisplay.toFixed(1)}</td>
                      <td>{yDisplay.toFixed(1)}</td>
                      <td>{pos.source}</td>
                      <td>{isOutOfBounds ? '❌ OUT' : '✅ OK'}</td>
                    </tr>
                  )
                })}
            </tbody>
          </table>
          {playerPositions.filter(p => p.frame_idx === currentFrame).length === 0 && (
            <p className="no-data">No player positions for this frame</p>
          )}
        </div>
        <div className="debug-summary">
          <span>Total players: {playerPositions.filter(p => p.frame_idx === currentFrame).length}</span>
          <span>Out of bounds: {playerPositions.filter(p =>
            p.frame_idx === currentFrame &&
            (p.x_pitch < 0 || p.x_pitch > PITCH_CANVAS_W || p.y_pitch < 0 || p.y_pitch > PITCH_CANVAS_H)
          ).length}</span>
        </div>
      </div>

      {/* Mapping View debug panel */}
      <details className="debug-panel" onToggle={(e) => setShowMappingView((e.target as HTMLDetailsElement).open)}>
        <summary>🗺️ Mapping View</summary>
        {showMappingView && (
          <div className="debug-panel-body">
            <p className="debug-info">
              Frame {currentFrame} —{' '}
              {homographyFrameIndices.includes(currentFrame)
                ? <strong style={{ color: 'green' }}>anchor frame</strong>
                : <span style={{ color: '#888' }}>propagated frame</span>}
            </p>
            <div style={{ width: 425, height: 300, overflow: 'hidden', border: '1px solid #ccc', borderRadius: 4 }}>
              <img
                key={currentFrame}
                src={`${API_URL}/videos/${videoMetadata.video_id}/frames/${currentFrame}/warped`}
                alt={`Warped frame ${currentFrame} with pitch lines`}
                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
              />
            </div>
          </div>
        )}
      </details>

    </div>
  )
}
