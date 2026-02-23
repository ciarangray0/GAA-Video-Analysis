import { useState, useRef, useEffect, useCallback } from 'react'
import Head from 'next/head'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface PitchPoint {
  pitch_id: string
  x_img: number
  y_img: number
}

interface LineAnnotation {
  line_id: string
  u1: number
  v1: number
  u2: number
  v2: number
}

interface PitchAnnotation {
  frame_idx: number
  points: PitchPoint[]
}

interface AnchorFrameAnnotation {
  frame_idx: number
  points: PitchPoint[]
  lines: LineAnnotation[]
}

interface PlayerPosition {
  frame_idx: number
  track_id: number
  x_pitch: number
  y_pitch: number
  source: string
}

interface ProcessResponse {
  video_id: string
  status: string
  player_positions?: PlayerPosition[]
  homography_frames?: number[]
  start_frame?: number
  end_frame?: number
  fps?: number
}

interface VideoMetadata {
  video_id: string
  fps: number
  num_frames: number
  width: number
  height: number
  duration_seconds: number
}

interface AnchorFrame {
  frame_idx: number
  isSkipped: boolean
  points: PitchPoint[]
  lines: LineAnnotation[]
}

export default function Home() {
  // Video upload state
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoMetadata, setVideoMetadata] = useState<VideoMetadata | null>(null)
  const [uploadingVideo, setUploadingVideo] = useState(false)

  // Anchor frame configuration
  const [trimStartSeconds, setTrimStartSeconds] = useState(0)
  const [trimEndSeconds, setTrimEndSeconds] = useState<number | null>(null)
  const [anchorInterval, setAnchorInterval] = useState(1) // 1 = every second
  const [anchorFrames, setAnchorFrames] = useState<AnchorFrame[]>([])

  // Annotation state
  const [currentAnchorIdx, setCurrentAnchorIdx] = useState(0)
  const [frameImageUrl, setFrameImageUrl] = useState<string | null>(null)
  const [loadingFrame, setLoadingFrame] = useState(false)

  // Line annotation state
  const [annotationMode, setAnnotationMode] = useState<'point' | 'line'>('point')
  const [selectedLineId, setSelectedLineId] = useState<string>('20m_top')
  const [pendingLinePoint1, setPendingLinePoint1] = useState<{ x: number; y: number } | null>(null)

  // Available pitch lines for line annotation (matches backend GAA_PITCH_LINES)
  const AVAILABLE_LINES: Record<string, { label: string; y_meters: number }> = {
    '13m_top': { label: '13m Line (Top)', y_meters: 13.0 },
    '20m_top': { label: '20m Line (Top)', y_meters: 20.0 },
    '45m_top': { label: '45m Line (Top)', y_meters: 45.0 },
    '65m_top': { label: '65m Line (Top)', y_meters: 65.0 },
    'halfway': { label: 'Halfway Line', y_meters: 70.0 },
    '65m_bottom': { label: '65m Line (Bottom)', y_meters: 75.0 },
    '45m_bottom': { label: '45m Line (Bottom)', y_meters: 95.0 },
    '20m_bottom': { label: '20m Line (Bottom)', y_meters: 120.0 },
    '13m_bottom': { label: '13m Line (Bottom)', y_meters: 127.0 },
  }

  // Processing state
  const [processing, setProcessing] = useState(false)
  const [status, setStatus] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [playerPositions, setPlayerPositions] = useState<PlayerPosition[]>([])
  const [currentFrame, setCurrentFrame] = useState(0)

  // Results playback state
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(1) // 1x, 0.5x, 2x
  const [isSyncMode, setIsSyncMode] = useState(true)
  const [showHomographySidebar, setShowHomographySidebar] = useState(false)
  const [selectedHomographyFrame, setSelectedHomographyFrame] = useState<number | null>(null)
  const [processedStartFrame, setProcessedStartFrame] = useState(0)
  const [processedEndFrame, setProcessedEndFrame] = useState(0)
  const [homographyFrameIndices, setHomographyFrameIndices] = useState<number[]>([])
  const [processedFps, setProcessedFps] = useState(25)
  const [videoObjectUrl, setVideoObjectUrl] = useState<string | null>(null)
  const [warpedFrameUrl, setWarpedFrameUrl] = useState<string | null>(null)
  const [loadingWarpedFrame, setLoadingWarpedFrame] = useState(false)

  // Step pipeline state
  const [stepAResult, setStepAResult] = useState<{ frames_processed: number; tracks: number; num_detections: number } | null>(null)
  const [stepBResult, setStepBResult] = useState<{ frames: number[]; info: Record<string, any> } | null>(null)
  const [stepCResult, setStepCResult] = useState<{ positions: PlayerPosition[]; total: number } | null>(null)
  const [stepDResult, setStepDResult] = useState<{ frames_generated: number; method: string } | null>(null)
  const [staleSteps, setStaleSteps] = useState<Set<string>>(new Set())
  const [runningStep, setRunningStep] = useState<string | null>(null)
  // Refs that mirror step completion status for use in effects without causing dep loops
  const stepDoneRef = useRef({ B: false, C: false, D: false })

  // Debug log
  const debugLog = useRef<string[]>([])
  const [debugLogEntries, setDebugLogEntries] = useState<string[]>([])
  const [debugLogVisible, setDebugLogVisible] = useState(false)

  // BotSort overlay
  const [showBotSortOverlay, setShowBotSortOverlay] = useState(false)

  // Refs
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameCanvasRef = useRef<HTMLCanvasElement>(null)
  const frameImageRef = useRef<HTMLImageElement | null>(null)
  const videoPlayerRef = useRef<HTMLVideoElement>(null)
  const playbackIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const animFrameRef = useRef<number>(0)
  const resultsFrameCanvasRef = useRef<HTMLCanvasElement>(null)
  const resultsFrameImageRef = useRef<HTMLImageElement | null>(null)
  const importFileRef = useRef<HTMLInputElement>(null)

  // Import/Export and copy status messages
  const [importStatus, setImportStatus] = useState<string>('')
  const [copyStatus, setCopyStatus] = useState<string>('')

  // Backend pitch canvas dimensions (canonical coordinate space)
  // All player positions from backend are in this pixel space
  const PITCH_CANVAS_W = 850
  const PITCH_CANVAS_H = 1400

  // Display canvas dimensions - MUST maintain same aspect ratio as backend canvas
  // Aspect ratio = 850/1400 = 0.607
  // We scale down to fit the UI while preserving exact proportions
  const DISPLAY_SCALE = 0.4  // 40% of backend canvas size
  const PITCH_DISPLAY_WIDTH = Math.round(PITCH_CANVAS_W * DISPLAY_SCALE)   // 340
  const PITCH_DISPLAY_HEIGHT = Math.round(PITCH_CANVAS_H * DISPLAY_SCALE)  // 560

  // Actual GAA pitch dimensions in meters
  const GAA_PITCH_WIDTH = 85.0
  const GAA_PITCH_LENGTH = 140.0

  // Pitch canvas ref for the diagram
  const pitchDiagramRef = useRef<HTMLCanvasElement>(null)

  // Import annotations file input ref
  const importAnnotationsRef = useRef<HTMLInputElement>(null)

  // Pending click state - when user clicks frame, we wait for pitch point selection
  const [pendingFrameClick, setPendingFrameClick] = useState<{ x: number; y: number } | null>(null)

  // All GAA pitch vertices from gaa_pitch_config.py
  const GAA_PITCH_VERTICES: Record<string, [number, number]> = {
    // Corners
    "corner_tl": [0.0, 0.0],
    "corner_tr": [85.0, 0.0],
    "corner_bl": [0.0, 140.0],
    "corner_br": [85.0, 140.0],

    // Goal posts
    "top_goal_lp": [39.25, 0.0],
    "top_goal_rp": [45.75, 0.0],
    "bottom_goal_lp": [39.25, 140.0],
    "bottom_goal_rp": [45.75, 140.0],

    // Goalie box
    "left_box_bottom": [35.5, 135.5],
    "left_box_top": [35.5, 4.5],
    "right_box_bottom": [49.5, 135.5],
    "right_box_top": [49.5, 4.5],

    // 13m box
    "left_13m_box_bottom": [33.0, 127.0],
    "left_13m_box_top": [33.0, 13.0],
    "right_13m_box_bottom": [52.0, 127.0],
    "right_13m_box_top": [52.0, 13.0],
    "left_endline_13m_box_bottom": [33.0, 140.0],
    "left_endline_13m_box_top": [33.0, 0.0],
    "right_endline_13m_box_bottom": [52.0, 140.0],
    "right_endline_13m_box_top": [52.0, 0.0],

    // Small arc
    "left_small_arc_bottom": [29.5, 120.0],
    "left_small_arc_top": [29.5, 20.0],
    "right_small_arc_bottom": [55.5, 120.0],
    "right_small_arc_top": [55.5, 20.0],
    "small_arc_top_top": [42.5, 33.0],
    "small_arc_top_bottom": [42.5, 107.0],

    // 13m line
    "left_13m_line_bottom": [0.0, 127.0],
    "left_13m_line_top": [0.0, 13.0],
    "right_13m_line_bottom": [85.0, 127.0],
    "right_13m_line_top": [85.0, 13.0],

    // 20m line
    "left_20m_line_bottom": [0.0, 120.0],
    "left_20m_line_top": [0.0, 20.0],
    "right_20m_line_bottom": [85.0, 120.0],
    "right_20m_line_top": [85.0, 20.0],

    // 45m line
    "left_45m_line_bottom": [0.0, 95.0],
    "left_45m_line_top": [0.0, 45.0],
    "right_45m_line_bottom": [85.0, 95.0],
    "right_45m_line_top": [85.0, 45.0],

    // 65m line
    "left_65m_line_bottom": [0.0, 75.0],
    "left_65m_line_top": [0.0, 65.0],
    "right_65m_line_bottom": [85.0, 75.0],
    "right_65m_line_top": [85.0, 65.0],

    // Center line (halfway)
    "center_left": [0.0, 70.0],
    "center_right": [85.0, 70.0],
  }

  // Pitch line segments for Type B (line-constrained) annotations
  const PITCH_LINE_SEGMENTS: Array<{ name: string; x1: number; y1: number; x2: number; y2: number }> = [
    { name: 'left_sideline', x1: 0, y1: 0, x2: 0, y2: 140 },
    { name: 'right_sideline', x1: 85, y1: 0, x2: 85, y2: 140 },
    { name: 'top_endline', x1: 0, y1: 0, x2: 85, y2: 0 },
    { name: 'bottom_endline', x1: 0, y1: 140, x2: 85, y2: 140 },
    { name: '13m_top', x1: 0, y1: 13, x2: 85, y2: 13 },
    { name: '13m_bottom', x1: 0, y1: 127, x2: 85, y2: 127 },
    { name: '20m_top', x1: 0, y1: 20, x2: 85, y2: 20 },
    { name: '20m_bottom', x1: 0, y1: 120, x2: 85, y2: 120 },
    { name: '45m_top', x1: 0, y1: 45, x2: 85, y2: 45 },
    { name: '45m_bottom', x1: 0, y1: 95, x2: 85, y2: 95 },
    { name: '65m_top', x1: 0, y1: 65, x2: 85, y2: 65 },
    { name: '65m_bottom', x1: 0, y1: 75, x2: 85, y2: 75 },
    { name: 'halfway', x1: 0, y1: 70, x2: 85, y2: 70 },
    { name: 'left_13m_box_top_v', x1: 33, y1: 0, x2: 33, y2: 13 },
    { name: 'left_13m_box_bottom_v', x1: 33, y1: 127, x2: 33, y2: 140 },
    { name: 'right_13m_box_top_v', x1: 52, y1: 0, x2: 52, y2: 13 },
    { name: 'right_13m_box_bottom_v', x1: 52, y1: 127, x2: 52, y2: 140 },
  ]

  // Helper to convert pitch coordinates to canvas coordinates
  const pitchToCanvas = (pitchX: number, pitchY: number): { x: number; y: number } => {
    const x = (pitchX / GAA_PITCH_WIDTH) * PITCH_DISPLAY_WIDTH
    const y = (pitchY / GAA_PITCH_LENGTH) * PITCH_DISPLAY_HEIGHT
    return { x, y }
  }

  // Helper to get point label for display
  const getPointLabel = (id: string): string => {
    return id.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
  }

  // Wrapped fetch that logs all API calls to the debug panel
  const apiFetch = useCallback(async (url: string, options?: RequestInit): Promise<Response> => {
    const method = options?.method || 'GET'
    const entry1 = `→ ${method} ${url}`
    debugLog.current = [...debugLog.current, entry1]
    setDebugLogEntries([...debugLog.current])
    const start = Date.now()
    try {
      const res = await fetch(url, options)
      const elapsed = Date.now() - start
      const entry2 = `← ${res.status} (${elapsed}ms)`
      debugLog.current = [...debugLog.current, entry2]
      setDebugLogEntries([...debugLog.current])
      return res
    } catch (err) {
      const entry2 = `✗ ${String(err)}`
      debugLog.current = [...debugLog.current, entry2]
      setDebugLogEntries([...debugLog.current])
      throw err
    }
  }, [])

  // Mark downstream steps as stale
  const markStale = useCallback((steps: string[]) => {
    setStaleSteps(prev => {
      const next = new Set(prev)
      steps.forEach(s => next.add(s))
      return next
    })
  }, [])

  const clearStale = useCallback((steps: string[]) => {
    setStaleSteps(prev => {
      const next = new Set(prev)
      steps.forEach(s => next.delete(s))
      return next
    })
  }, [])

  // Step A: Run tracking
  const runStepA = useCallback(async () => {
    if (!videoMetadata) { setError('Please upload a video first'); return }
    setRunningStep('A')
    setError('')
    try {
      const res = await apiFetch(`${API_URL}/videos/${videoMetadata.video_id}/track`, { method: 'POST' })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Tracking failed')
      }
      const data = await res.json()
      // Fetch detections to get total count
      const detRes = await apiFetch(`${API_URL}/videos/${videoMetadata.video_id}/detections`)
      const numDetections = detRes.ok ? (await detRes.json() as any[]).length : 0
      setStepAResult({ frames_processed: data.frames_processed, tracks: data.tracks, num_detections: numDetections })
      markStale(['B', 'C', 'D'])
      clearStale(['A'])
    } catch (err: any) {
      setError(err.message || 'Tracking failed')
    } finally {
      setRunningStep(null)
    }
  }, [videoMetadata, apiFetch, markStale, clearStale])

  // Step B: Compute homographies
  const runStepB = useCallback(async () => {
    if (!videoMetadata) { setError('Please upload a video first'); return }
    if (!stepAResult) { setError('Please run tracking first (Step A)'); return }

    const validAnnotations: AnchorFrameAnnotation[] = anchorFrames
      .filter(af => !af.isSkipped && af.points.length >= 4)
      .map(af => ({ frame_idx: af.frame_idx, points: af.points, lines: af.lines || [] }))

    if (validAnnotations.length === 0) {
      setError('Please annotate at least one anchor frame with 4+ points')
      return
    }

    setRunningStep('B')
    setError('')
    try {
      const hasLines = validAnnotations.some(a => a.lines.length > 0)
      const endpoint = hasLines
        ? `${API_URL}/videos/${videoMetadata.video_id}/homographies/v2`
        : `${API_URL}/videos/${videoMetadata.video_id}/homographies`

      const res = await apiFetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(hasLines ? validAnnotations : validAnnotations.map(a => ({ frame_idx: a.frame_idx, points: a.points })))
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Homography computation failed')
      }
      const data = await res.json()
      setStepBResult({ frames: data.frames || [], info: data.info || {} })
      setHomographyFrameIndices(data.frames || [])
      markStale(['C', 'D'])
      clearStale(['B'])
    } catch (err: any) {
      setError(err.message || 'Homography computation failed')
    } finally {
      setRunningStep(null)
    }
  }, [videoMetadata, stepAResult, anchorFrames, apiFetch, markStale, clearStale])

  // Step C: Map players to pitch
  const runStepC = useCallback(async () => {
    if (!videoMetadata) { setError('Please upload a video first'); return }
    if (!stepBResult) { setError('Please compute homographies first (Step B)'); return }

    setRunningStep('C')
    setError('')
    try {
      const res = await apiFetch(`${API_URL}/videos/${videoMetadata.video_id}/map_players`, { method: 'POST' })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Player mapping failed')
      }
      const positions: PlayerPosition[] = await res.json()
      setStepCResult({ positions, total: positions.length })
      markStale(['D'])
      clearStale(['C'])
    } catch (err: any) {
      setError(err.message || 'Player mapping failed')
    } finally {
      setRunningStep(null)
    }
  }, [videoMetadata, stepBResult, apiFetch, markStale, clearStale])

  // Step D: Interpolate trajectories
  const runStepD = useCallback(async () => {
    if (!videoMetadata) { setError('Please upload a video first'); return }
    if (!stepCResult) { setError('Please map players first (Step C)'); return }

    const startFrame = Math.floor(trimStartSeconds * videoMetadata.fps)
    const endFrame = trimEndSeconds !== null
      ? Math.floor(trimEndSeconds * videoMetadata.fps)
      : videoMetadata.num_frames - 1

    setRunningStep('D')
    setError('')
    try {
      const res = await apiFetch(
        `${API_URL}/videos/${videoMetadata.video_id}/interpolate?start_frame=${startFrame}&end_frame=${endFrame}`,
        { method: 'POST' }
      )
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Interpolation failed')
      }
      const data = await res.json()
      setStepDResult({ frames_generated: data.frames_generated, method: data.method })
      clearStale(['D'])

      // Fetch all player positions and activate results view
      const playersRes = await apiFetch(`${API_URL}/videos/${videoMetadata.video_id}/players`)
      if (playersRes.ok) {
        const allPositions: PlayerPosition[] = await playersRes.json()
        setPlayerPositions(allPositions)
        setProcessedStartFrame(startFrame)
        setProcessedEndFrame(endFrame)
        setProcessedFps(videoMetadata.fps)
        const firstFrame = allPositions.length > 0
          ? Math.min(...allPositions.map(p => p.frame_idx))
          : startFrame
        setCurrentFrame(firstFrame)
        setStatus('Pipeline complete!')
      }
    } catch (err: any) {
      setError(err.message || 'Interpolation failed')
    } finally {
      setRunningStep(null)
    }
  }, [videoMetadata, stepCResult, trimStartSeconds, trimEndSeconds, apiFetch, clearStale])
  const uploadVideo = async () => {
    if (!videoFile) return

    setUploadingVideo(true)
    setError('')

    try {
      const formData = new FormData()
      formData.append('file', videoFile)

      const response = await fetch(`${API_URL}/videos`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Upload failed')
      }

      const metadata: VideoMetadata = await response.json()
      setVideoMetadata(metadata)
      setTrimEndSeconds(metadata.duration_seconds)
      setStatus('Video uploaded successfully!')
    } catch (err: any) {
      setError(err.message || 'Failed to upload video')
    } finally {
      setUploadingVideo(false)
    }
  }

  // Generate anchor frames based on configuration
  const generateAnchorFrames = () => {
    if (!videoMetadata) return

    const fps = videoMetadata.fps
    const startFrame = Math.floor(trimStartSeconds * fps)
    const endFrame = trimEndSeconds !== null
      ? Math.floor(trimEndSeconds * fps)
      : videoMetadata.num_frames - 1

    const frames: AnchorFrame[] = []

    // Generate anchor frames at the specified interval (in seconds)
    for (let seconds = trimStartSeconds; seconds <= (trimEndSeconds || videoMetadata.duration_seconds); seconds += anchorInterval) {
      const frameIdx = Math.floor(seconds * fps)
      if (frameIdx <= endFrame) {
        frames.push({
          frame_idx: frameIdx,
          isSkipped: false,
          points: [],
          lines: []
        })
      }
    }

    setAnchorFrames(frames)
    setCurrentAnchorIdx(0)
    if (frames.length > 0) {
      // Check for saved annotations in localStorage
      const savedKey = videoFile ? `gaa_annotations_${videoFile.name}` : null
      if (savedKey) {
        const saved = localStorage.getItem(savedKey)
        if (saved) {
          try {
            const parsed: AnchorFrame[] = JSON.parse(saved)
            if (confirm(`Found saved annotations for this video (${parsed.length} frames). Restore them?`)) {
              // Merge saved points/lines into newly generated frames
              const merged = frames.map(f => {
                const match = parsed.find(p => p.frame_idx === f.frame_idx)
                return match ? { ...f, isSkipped: match.isSkipped, points: match.points, lines: match.lines || [] } : f
              })
              setAnchorFrames(merged)
              loadFrameImage(merged[0].frame_idx)
              return
            }
          } catch (_) {
            console.warn('Could not restore saved annotations - data may be corrupt')
          }
        }
      }
      loadFrameImage(frames[0].frame_idx)
    }
  }

  // Load frame image from backend
  const loadFrameImage = async (frameIdx: number) => {
    if (!videoMetadata) return

    setLoadingFrame(true)
    setError('')

    try {
      const url = `${API_URL}/videos/${videoMetadata.video_id}/frame/${frameIdx}`
      setFrameImageUrl(url)

      // Create and load the image
      const img = new Image()
      img.crossOrigin = 'anonymous'

      img.onload = () => {
        console.log(`Frame ${frameIdx} loaded: ${img.naturalWidth}x${img.naturalHeight}`)
        frameImageRef.current = img
        setLoadingFrame(false)
        // Draw will be triggered by useEffect when loadingFrame changes
      }

      img.onerror = (e) => {
        console.error(`Failed to load frame ${frameIdx}:`, e)
        setError(`Failed to load frame ${frameIdx}. Check if backend is running.`)
        setLoadingFrame(false)
      }

      // Add cache-busting parameter to avoid stale images
      img.src = `${url}?t=${Date.now()}`
    } catch (err) {
      console.error('Failed to load frame:', err)
      setError('Failed to load frame')
      setLoadingFrame(false)
    }
  }

  // Draw frame with annotation points
  const drawFrameWithPoints = useCallback(() => {
    const canvas = frameCanvasRef.current
    const img = frameImageRef.current
    if (!canvas || !img || anchorFrames.length === 0) return

    // Check if image is actually loaded
    if (!img.complete || img.naturalWidth === 0) {
      console.log('Image not yet loaded, waiting...')
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size to match image (scaled down if needed) - increased max width
    const maxWidth = 1000
    const scale = Math.min(1, maxWidth / img.naturalWidth)
    canvas.width = img.naturalWidth * scale
    canvas.height = img.naturalHeight * scale

    // Draw image
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    const currentAnchor = anchorFrames[currentAnchorIdx]
    const imgScale = canvas.width / img.naturalWidth

    // Draw line annotations for current anchor frame
    if (currentAnchor && currentAnchor.lines) {
      currentAnchor.lines.forEach((line) => {
        const x1 = line.u1 * imgScale
        const y1 = line.v1 * imgScale
        const x2 = line.u2 * imgScale
        const y2 = line.v2 * imgScale

        // Draw dashed line (more subtle)
        ctx.strokeStyle = 'rgba(0, 255, 255, 0.5)'
        ctx.lineWidth = 1.5
        ctx.setLineDash([6, 4])
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.stroke()
        ctx.setLineDash([])

        // Draw endpoints (smaller)
        ctx.fillStyle = 'rgba(0, 255, 255, 0.6)'
        ctx.beginPath()
        ctx.arc(x1, y1, 4, 0, 2 * Math.PI)
        ctx.fill()
        ctx.beginPath()
        ctx.arc(x2, y2, 4, 0, 2 * Math.PI)
        ctx.fill()

        // Draw label at midpoint with subtle background
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

    // Draw pending line point (first click in line mode)
    if (pendingLinePoint1) {
      const x = pendingLinePoint1.x * imgScale
      const y = pendingLinePoint1.y * imgScale

      // Draw subtle pending marker
      ctx.fillStyle = 'rgba(255, 255, 0, 0.7)'
      ctx.beginPath()
      ctx.arc(x, y, 6, 0, 2 * Math.PI)
      ctx.fill()
      ctx.strokeStyle = '#000000'
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw instruction (smaller, more transparent)
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)'
      ctx.fillRect(x - 75, y + 12, 150, 20)
      ctx.fillStyle = '#ffffff'
      ctx.font = '10px Arial'
      ctx.textAlign = 'center'
      ctx.fillText('Click second point on line', x, y + 26)
      ctx.textAlign = 'left'
    }

    // Draw annotation points for current anchor frame
    if (currentAnchor && currentAnchor.points) {
      currentAnchor.points.forEach((point) => {
        const x = point.x_img * imgScale
        const y = point.y_img * imgScale

        // Draw point (smaller, semi-transparent green)
        ctx.fillStyle = 'rgba(0, 255, 0, 0.6)'
        ctx.beginPath()
        ctx.arc(x, y, 5, 0, 2 * Math.PI)
        ctx.fill()

        // Draw border (thinner)
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 1
        ctx.stroke()

        // Draw label with dark background
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
  }, [anchorFrames, currentAnchorIdx, pendingLinePoint1, AVAILABLE_LINES])

  // Handle click on frame to mark a point (first step of annotation)
  const handleFrameClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = frameCanvasRef.current
    const img = frameImageRef.current
    if (!canvas || !img || anchorFrames.length === 0) return
    if (!img.naturalWidth || !img.naturalHeight) return

    const rect = canvas.getBoundingClientRect()

    // Calculate click position relative to canvas display size
    const clickX = e.clientX - rect.left
    const clickY = e.clientY - rect.top

    // Scale from CSS display size to canvas internal size
    const cssToCanvasX = canvas.width / rect.width
    const cssToCanvasY = canvas.height / rect.height

    // Then scale from canvas size to original image size
    const canvasToImageX = img.naturalWidth / canvas.width
    const canvasToImageY = img.naturalHeight / canvas.height

    // Final coordinates in original image space
    const x = clickX * cssToCanvasX * canvasToImageX
    const y = clickY * cssToCanvasY * canvasToImageY

    if (annotationMode === 'line') {
      // Line annotation mode
      if (!pendingLinePoint1) {
        // First click - store first point
        setPendingLinePoint1({ x: Math.round(x), y: Math.round(y) })
      } else {
        // Second click - create line annotation
        const newLine: LineAnnotation = {
          line_id: selectedLineId,
          u1: pendingLinePoint1.x,
          v1: pendingLinePoint1.y,
          u2: Math.round(x),
          v2: Math.round(y)
        }

        setAnchorFrames(prev => {
          const updated = [...prev]
          // Remove existing line of same type if exists
          updated[currentAnchorIdx].lines = updated[currentAnchorIdx].lines.filter(
            l => l.line_id !== selectedLineId
          )
          updated[currentAnchorIdx].lines.push(newLine)
          return updated
        })

        // Clear pending state
        setPendingLinePoint1(null)
      }
    } else {
      // Point annotation mode - store the pending frame click, user must click on pitch diagram
      setPendingFrameClick({ x: Math.round(x), y: Math.round(y) })
    }
  }

  // Cancel line annotation in progress
  const cancelLineAnnotation = () => {
    setPendingLinePoint1(null)
  }

  // Remove line annotation
  const removeLine = (lineIdx: number) => {
    setAnchorFrames(prev => {
      const updated = [...prev]
      updated[currentAnchorIdx].lines = updated[currentAnchorIdx].lines.filter((_, i) => i !== lineIdx)
      return updated
    })
  }

  // Handle click on pitch diagram to complete annotation
  const handlePitchDiagramClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!pendingFrameClick) return

    const canvas = pitchDiagramRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()

    // Calculate click position relative to canvas display size, then scale to internal size
    const cssToCanvasX = canvas.width / rect.width
    const cssToCanvasY = canvas.height / rect.height
    const clickX = (e.clientX - rect.left) * cssToCanvasX
    const clickY = (e.clientY - rect.top) * cssToCanvasY

    // Find closest pitch vertex
    let closestId: string | null = null
    let closestDist = Infinity

    for (const [id, coords] of Object.entries(GAA_PITCH_VERTICES)) {
      const canvasPos = pitchToCanvas(coords[0], coords[1])
      const dist = Math.sqrt(Math.pow(canvasPos.x - clickX, 2) + Math.pow(canvasPos.y - clickY, 2))
      if (dist < closestDist && dist < 20) { // Must be within 20px
        closestDist = dist
        closestId = id
      }
    }

    if (closestId) {
      // Add the annotation point
      const newPoint: PitchPoint = {
        pitch_id: closestId,
        x_img: pendingFrameClick.x,
        y_img: pendingFrameClick.y
      }

      setAnchorFrames(prev => {
        const updated = [...prev]
        // Remove existing point of same type if exists
        updated[currentAnchorIdx].points = updated[currentAnchorIdx].points.filter(
          p => p.pitch_id !== closestId
        )
        updated[currentAnchorIdx].points.push(newPoint)
        return updated
      })

      // Clear pending state
      setPendingFrameClick(null)
      return
    }

    // No vertex found within 20px — try nearest pitch line segment (Type B click)
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
        y_img: pendingFrameClick.y
      }

      setAnchorFrames(prev => {
        const updated = [...prev]
        updated[currentAnchorIdx].points.push(newPoint)
        return updated
      })

      setPendingFrameClick(null)
    }
  }

  // Draw the pitch diagram with all vertices
  const drawPitchDiagram = useCallback(() => {
    const canvas = pitchDiagramRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    canvas.width = PITCH_DISPLAY_WIDTH
    canvas.height = PITCH_DISPLAY_HEIGHT

    // Draw pitch background
    ctx.fillStyle = '#2d5016'
    ctx.fillRect(0, 0, PITCH_DISPLAY_WIDTH, PITCH_DISPLAY_HEIGHT)

    // Draw pitch lines (white)
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 2

    // Outer boundary
    ctx.strokeRect(0, 0, PITCH_DISPLAY_WIDTH, PITCH_DISPLAY_HEIGHT)

    // Draw center line
    const centerY = pitchToCanvas(0, 70).y
    ctx.beginPath()
    ctx.moveTo(0, centerY)
    ctx.lineTo(PITCH_DISPLAY_WIDTH, centerY)
    ctx.stroke()

    // Draw 13m lines
    const line13Top = pitchToCanvas(0, 13).y
    const line13Bottom = pitchToCanvas(0, 127).y
    ctx.beginPath()
    ctx.moveTo(0, line13Top)
    ctx.lineTo(PITCH_DISPLAY_WIDTH, line13Top)
    ctx.moveTo(0, line13Bottom)
    ctx.lineTo(PITCH_DISPLAY_WIDTH, line13Bottom)
    ctx.stroke()

    // Draw 20m lines
    const line20Top = pitchToCanvas(0, 20).y
    const line20Bottom = pitchToCanvas(0, 120).y
    ctx.beginPath()
    ctx.moveTo(0, line20Top)
    ctx.lineTo(PITCH_DISPLAY_WIDTH, line20Top)
    ctx.moveTo(0, line20Bottom)
    ctx.lineTo(PITCH_DISPLAY_WIDTH, line20Bottom)
    ctx.stroke()

    // Draw 45m lines
    const line45Top = pitchToCanvas(0, 45).y
    const line45Bottom = pitchToCanvas(0, 95).y
    ctx.beginPath()
    ctx.moveTo(0, line45Top)
    ctx.lineTo(PITCH_DISPLAY_WIDTH, line45Top)
    ctx.moveTo(0, line45Bottom)
    ctx.lineTo(PITCH_DISPLAY_WIDTH, line45Bottom)
    ctx.stroke()

    // Draw 65m lines
    const line65Top = pitchToCanvas(0, 65).y
    const line65Bottom = pitchToCanvas(0, 75).y
    ctx.beginPath()
    ctx.moveTo(0, line65Top)
    ctx.lineTo(PITCH_DISPLAY_WIDTH, line65Top)
    ctx.moveTo(0, line65Bottom)
    ctx.lineTo(PITCH_DISPLAY_WIDTH, line65Bottom)
    ctx.stroke()

    // Draw 13m box lines (vertical)
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

    // Draw goalie box
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

    // Get already annotated point IDs for this frame
    const currentAnchor = anchorFrames[currentAnchorIdx]
    const annotatedIds = currentAnchor ? currentAnchor.points.map(p => p.pitch_id) : []

    // When pending click, subtly highlight all line segments to show they're clickable
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
      // Reset for vertex drawing
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 2
    }

    // Draw all vertex points
    for (const [id, coords] of Object.entries(GAA_PITCH_VERTICES)) {
      const pos = pitchToCanvas(coords[0], coords[1])
      const isAnnotated = annotatedIds.includes(id)

      // Draw point
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, 6, 0, 2 * Math.PI)

      if (isAnnotated) {
        ctx.fillStyle = '#00ff00' // Green for annotated
      } else if (pendingFrameClick) {
        ctx.fillStyle = '#ffff00' // Yellow when waiting for selection
      } else {
        ctx.fillStyle = '#ff6600' // Orange normally
      }
      ctx.fill()
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 2
      ctx.stroke()
    }

    // If there's a pending click, show instruction (smaller, more subtle)
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
  }, [pendingFrameClick, anchorFrames, currentAnchorIdx, pitchToCanvas, PITCH_LINE_SEGMENTS])

  // Remove annotation point
  const removePoint = (pointIdx: number) => {
    setAnchorFrames(prev => {
      const updated = [...prev]
      updated[currentAnchorIdx].points = updated[currentAnchorIdx].points.filter((_, i) => i !== pointIdx)
      return updated
    })
  }

  // Navigate anchor frames
  const goToAnchorFrame = (idx: number) => {
    if (idx >= 0 && idx < anchorFrames.length) {
      setCurrentAnchorIdx(idx)
      loadFrameImage(anchorFrames[idx].frame_idx)
    }
  }

  // Skip/unskip anchor frame
  const toggleSkipFrame = () => {
    setAnchorFrames(prev => {
      const updated = [...prev]
      updated[currentAnchorIdx].isSkipped = !updated[currentAnchorIdx].isSkipped
      return updated
    })
  }

  // Swap anchor frame for a different frame number
  const swapAnchorFrame = (newFrameIdx: number) => {
    if (!videoMetadata || newFrameIdx < 0 || newFrameIdx >= videoMetadata.num_frames) {
      setError('Invalid frame number')
      return
    }

    setAnchorFrames(prev => {
      const updated = [...prev]
      updated[currentAnchorIdx] = {
        frame_idx: newFrameIdx,
        isSkipped: false,
        points: [], // Clear points when swapping
        lines: []   // Clear lines when swapping
      }
      return updated
    })

    loadFrameImage(newFrameIdx)
  }

  // Export annotations to JSON file
  const exportAnnotations = () => {
    const filename = videoFile?.name || 'unknown'
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

  // Import annotations from JSON file
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
          // Merge imported data into existing anchor frame slots
          const frameIndices = new Set(anchorFrames.map(f => f.frame_idx))
          const importedIndices = imported.map(f => f.frame_idx)
          const hasMismatch = importedIndices.some(idx => !frameIndices.has(idx))
          if (hasMismatch) {
            if (!confirm('Some imported frame indices do not match current anchor frames. Import anyway (matching frames will be updated)?')) return
          }
          const merged = anchorFrames.map(f => {
            const match = imported.find(p => p.frame_idx === f.frame_idx)
            return match ? { ...f, isSkipped: match.isSkipped, points: match.points, lines: match.lines } : f
          })
          setAnchorFrames(merged)
          if (merged.length > 0) loadFrameImage(merged[0].frame_idx)
        } else {
          setAnchorFrames(imported)
          if (imported.length > 0) loadFrameImage(imported[0].frame_idx)
        }
        setCurrentAnchorIdx(0)
      } catch (err: any) {
        alert(`Failed to parse annotation file: ${err?.message || 'Invalid format'}`)
      }
    }
    reader.readAsText(file)
    // Reset input so same file can be re-imported
    e.target.value = ''
  }

  // Copy points from previous non-skipped anchor frame
  const copyFromPrevious = () => {
    let srcIdx = currentAnchorIdx - 1
    while (srcIdx >= 0 && anchorFrames[srcIdx].isSkipped) srcIdx--
    if (srcIdx < 0 || anchorFrames[srcIdx].points.length === 0) return
    setAnchorFrames(prev => {
      const updated = [...prev]
      updated[currentAnchorIdx] = {
        ...updated[currentAnchorIdx],
        points: [...anchorFrames[srcIdx].points],
        lines: [...(anchorFrames[srcIdx].lines || [])],
      }
      return updated
    })
  }

  // Copy points from next non-skipped anchor frame
  const copyFromNext = () => {
    let srcIdx = currentAnchorIdx + 1
    while (srcIdx < anchorFrames.length && anchorFrames[srcIdx].isSkipped) srcIdx++
    if (srcIdx >= anchorFrames.length || anchorFrames[srcIdx].points.length === 0) return
    setAnchorFrames(prev => {
      const updated = [...prev]
      updated[currentAnchorIdx] = {
        ...updated[currentAnchorIdx],
        points: [...anchorFrames[srcIdx].points],
        lines: [...(anchorFrames[srcIdx].lines || [])],
      }
      return updated
    })
  }

  // Process video with annotations
  const processVideo = async () => {
    if (!videoFile || !videoMetadata) {
      setError('Please upload a video first')
      return
    }

    // Get annotated anchor frames (non-skipped with at least 4 points)
    // Use v2 format with lines included
    const validAnnotations: AnchorFrameAnnotation[] = anchorFrames
      .filter(af => !af.isSkipped && af.points.length >= 4)
      .map(af => ({
        frame_idx: af.frame_idx,
        points: af.points,
        lines: af.lines || []
      }))

    if (validAnnotations.length === 0) {
      setError('Please annotate at least one anchor frame with 4+ points')
      return
    }

    // Log line annotations for debugging
    const totalLines = validAnnotations.reduce((sum, a) => sum + a.lines.length, 0)
    console.log(`Processing with ${validAnnotations.length} anchor frames and ${totalLines} line constraints`)

    setProcessing(true)
    setError('')
    setStatus('Processing video...')

    try {
      const formData = new FormData()
      formData.append('file', videoFile)
      formData.append('annotations_json', JSON.stringify(validAnnotations))
      formData.append('start_frame', String(Math.floor(trimStartSeconds * videoMetadata.fps)))
      if (trimEndSeconds !== null) {
        formData.append('end_frame', String(Math.floor(trimEndSeconds * videoMetadata.fps)))
      }

      const response = await fetch(`${API_URL}/process-video`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Processing failed')
      }

      const data: ProcessResponse = await response.json()
      
      if (data.status === 'completed' && data.player_positions) {
        setPlayerPositions(data.player_positions)
        setProcessedStartFrame(data.start_frame || 0)
        setProcessedEndFrame(data.end_frame || 0)
        setHomographyFrameIndices(data.homography_frames || [])
        setProcessedFps(data.fps || videoMetadata.fps)

        // Set initial frame to first frame with positions
        const firstFrameWithPositions = data.player_positions.length > 0
          ? Math.min(...data.player_positions.map(p => p.frame_idx))
          : data.start_frame || 0
        setCurrentFrame(firstFrameWithPositions)

        setStatus('Processing completed!')
        drawPitch(data.player_positions, firstFrameWithPositions)
      } else {
        setError('Processing completed but no positions returned')
      }
    } catch (err: any) {
      setError(err.message || 'Failed to process video')
    } finally {
      setProcessing(false)
    }
  }

  const drawPitch = useCallback((positions: PlayerPosition[], frame: number) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Use the pitch display dimensions for results
    const RESULTS_PITCH_WIDTH = PITCH_DISPLAY_WIDTH
    const RESULTS_PITCH_HEIGHT = PITCH_DISPLAY_HEIGHT

    // Ensure canvas has correct dimensions
    if (canvas.width !== RESULTS_PITCH_WIDTH || canvas.height !== RESULTS_PITCH_HEIGHT) {
      canvas.width = RESULTS_PITCH_WIDTH
      canvas.height = RESULTS_PITCH_HEIGHT
    }

    // Clear canvas with pitch color
    ctx.fillStyle = '#2d5016' // Green pitch color
    ctx.fillRect(0, 0, RESULTS_PITCH_WIDTH, RESULTS_PITCH_HEIGHT)

    // Draw pitch markings
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)'
    ctx.lineWidth = 1

    // Draw outer boundary
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 2
    ctx.strokeRect(2, 2, RESULTS_PITCH_WIDTH - 4, RESULTS_PITCH_HEIGHT - 4)

    // Draw center line (halfway)
    ctx.beginPath()
    ctx.moveTo(0, RESULTS_PITCH_HEIGHT / 2)
    ctx.lineTo(RESULTS_PITCH_WIDTH, RESULTS_PITCH_HEIGHT / 2)
    ctx.stroke()

    // Draw center circle
    ctx.beginPath()
    ctx.arc(RESULTS_PITCH_WIDTH / 2, RESULTS_PITCH_HEIGHT / 2, 40 * DISPLAY_SCALE, 0, 2 * Math.PI)
    ctx.stroke()

    // Draw 13m lines (approximately 9% from each end)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)'
    ctx.lineWidth = 1
    const line13mTop = (13 / GAA_PITCH_LENGTH) * RESULTS_PITCH_HEIGHT
    const line13mBottom = ((GAA_PITCH_LENGTH - 13) / GAA_PITCH_LENGTH) * RESULTS_PITCH_HEIGHT
    ctx.beginPath()
    ctx.moveTo(0, line13mTop)
    ctx.lineTo(RESULTS_PITCH_WIDTH, line13mTop)
    ctx.moveTo(0, line13mBottom)
    ctx.lineTo(RESULTS_PITCH_WIDTH, line13mBottom)
    ctx.stroke()

    // Draw 20m lines
    const line20mTop = (20 / GAA_PITCH_LENGTH) * RESULTS_PITCH_HEIGHT
    const line20mBottom = ((GAA_PITCH_LENGTH - 20) / GAA_PITCH_LENGTH) * RESULTS_PITCH_HEIGHT
    ctx.beginPath()
    ctx.moveTo(0, line20mTop)
    ctx.lineTo(RESULTS_PITCH_WIDTH, line20mTop)
    ctx.moveTo(0, line20mBottom)
    ctx.lineTo(RESULTS_PITCH_WIDTH, line20mBottom)
    ctx.stroke()

    // Draw 45m lines
    const line45mTop = (45 / GAA_PITCH_LENGTH) * RESULTS_PITCH_HEIGHT
    const line45mBottom = ((GAA_PITCH_LENGTH - 45) / GAA_PITCH_LENGTH) * RESULTS_PITCH_HEIGHT
    ctx.beginPath()
    ctx.moveTo(0, line45mTop)
    ctx.lineTo(RESULTS_PITCH_WIDTH, line45mTop)
    ctx.moveTo(0, line45mBottom)
    ctx.lineTo(RESULTS_PITCH_WIDTH, line45mBottom)
    ctx.stroke()

    // Filter positions for current frame
    const framePositions = positions.filter(p => p.frame_idx === frame)

    // Debug: log out-of-bounds positions
    const outOfBounds = framePositions.filter(p =>
      p.x_pitch < 0 || p.x_pitch > PITCH_CANVAS_W ||
      p.y_pitch < 0 || p.y_pitch > PITCH_CANVAS_H
    )
    if (outOfBounds.length > 0) {
      console.warn(`Frame ${frame}: ${outOfBounds.length} out-of-bounds positions:`, outOfBounds)
    }

    // Generate consistent color for each track_id
    const getPlayerColor = (trackId: number): string => {
      // Use golden ratio to spread colors evenly
      const hue = (trackId * 137.508) % 360
      return `hsl(${hue}, 70%, 50%)`
    }

    // Draw player positions
    // Backend returns coordinates in PITCH CANVAS PIXELS (0-850 for x, 0-1400 for y)
    // Scale to display canvas: x_display = (x_pitch / PITCH_CANVAS_W) * DISPLAY_WIDTH
    framePositions.forEach((pos) => {
      const x = (pos.x_pitch / PITCH_CANVAS_W) * RESULTS_PITCH_WIDTH
      const y = (pos.y_pitch / PITCH_CANVAS_H) * RESULTS_PITCH_HEIGHT

      // Check if position is out of bounds (debugging)
      const isOutOfBounds = pos.x_pitch < 0 || pos.x_pitch > PITCH_CANVAS_W ||
                           pos.y_pitch < 0 || pos.y_pitch > PITCH_CANVAS_H

      // Clamp to canvas bounds with padding
      const padding = 8
      const clampedX = Math.max(padding, Math.min(RESULTS_PITCH_WIDTH - padding, x))
      const clampedY = Math.max(padding, Math.min(RESULTS_PITCH_HEIGHT - padding, y))

      // Use unique color per track_id (red border if out of bounds)
      ctx.fillStyle = getPlayerColor(pos.track_id)

      // Draw player as filled circle with border
      ctx.beginPath()
      ctx.arc(clampedX, clampedY, 8, 0, 2 * Math.PI)
      ctx.fill()

      // Red border for out-of-bounds positions, white otherwise
      ctx.strokeStyle = isOutOfBounds ? '#ff0000' : '#ffffff'
      ctx.lineWidth = isOutOfBounds ? 3 : 2
      ctx.stroke()

      // Draw track ID label
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 10px Arial'
      ctx.textAlign = 'center'
      ctx.fillText(pos.track_id.toString(), clampedX, clampedY + 3)
    })

    // Draw frame info in corner
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)'
    ctx.fillRect(5, 5, 120, 50)
    ctx.fillStyle = '#ffffff'
    ctx.font = '14px Arial'
    ctx.textAlign = 'left'
    ctx.fillText(`Frame: ${frame}`, 10, 25)
    ctx.fillText(`Players: ${framePositions.length}`, 10, 45)
  }, [PITCH_DISPLAY_WIDTH, PITCH_DISPLAY_HEIGHT, PITCH_CANVAS_W, PITCH_CANVAS_H, GAA_PITCH_LENGTH, DISPLAY_SCALE])

  // Calculate the valid frame range based on trim settings
  const getValidFrameRange = useCallback(() => {
    if (!videoMetadata) return { startFrame: 0, endFrame: 0 }
    const startFrame = Math.floor(trimStartSeconds * videoMetadata.fps)
    const endFrame = trimEndSeconds
      ? Math.floor(trimEndSeconds * videoMetadata.fps)
      : videoMetadata.num_frames - 1
    return { startFrame, endFrame }
  }, [videoMetadata, trimStartSeconds, trimEndSeconds])

  // Get frames that have player positions
  const getFramesWithPositions = useCallback(() => {
    const frames = new Set(playerPositions.map(p => p.frame_idx))
    return Array.from(frames).sort((a, b) => a - b)
  }, [playerPositions])

  // Load frame image for results view
  const loadResultsFrame = useCallback(async (frameIdx: number) => {
    if (!videoMetadata) return

    const url = `${API_URL}/videos/${videoMetadata.video_id}/frame/${frameIdx}`

    const img = new Image()
    img.crossOrigin = 'anonymous'

    img.onload = () => {
      resultsFrameImageRef.current = img
      const canvas = resultsFrameCanvasRef.current
      if (canvas) {
        const ctx = canvas.getContext('2d')
        if (ctx) {
          const maxWidth = 640
          const scale = Math.min(1, maxWidth / img.naturalWidth)
          canvas.width = img.naturalWidth * scale
          canvas.height = img.naturalHeight * scale
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
        }
      }
    }

    img.src = `${url}?t=${Date.now()}`
  }, [videoMetadata])

  // Playback controls (requestAnimationFrame-based for smooth playback)
  const stopPlayback = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current)
      animFrameRef.current = 0
    }
    if (videoPlayerRef.current) {
      videoPlayerRef.current.pause()
    }
    setIsPlaying(false)
  }, [])

  const onPlaybackFrame = useCallback(() => {
    const video = videoPlayerRef.current
    if (!video || video.paused) {
      setIsPlaying(false)
      return
    }
    const fps = processedFps || videoMetadata?.fps || 25
    const frameIdx = Math.round(video.currentTime * fps)
    // Stop at trim end
    if (frameIdx > processedEndFrame) {
      video.pause()
      setIsPlaying(false)
      return
    }
    setCurrentFrame(frameIdx)
    animFrameRef.current = requestAnimationFrame(onPlaybackFrame)
  }, [processedFps, processedEndFrame, videoMetadata])

  const startPlayback = useCallback(() => {
    const video = videoPlayerRef.current
    if (!video || playerPositions.length === 0) return

    setIsPlaying(true)
    video.playbackRate = playbackSpeed
    // Seek to trim start if not already there
    const fps = processedFps || videoMetadata?.fps || 25
    const startTime = processedStartFrame / fps
    if (video.currentTime < startTime) {
      video.currentTime = startTime
    }
    video.play().catch((err) => { console.warn('Autoplay blocked by browser:', err) })
    animFrameRef.current = requestAnimationFrame(onPlaybackFrame)
  }, [playbackSpeed, playerPositions.length, processedFps, processedStartFrame, videoMetadata, onPlaybackFrame])

  // Update playback rate when speed changes while playing
  useEffect(() => {
    if (isPlaying && videoPlayerRef.current) {
      videoPlayerRef.current.playbackRate = playbackSpeed
    }
  }, [playbackSpeed, isPlaying])

  const togglePlayback = useCallback(() => {
    if (isPlaying) {
      stopPlayback()
    } else {
      startPlayback()
    }
  }, [isPlaying, startPlayback, stopPlayback])

  const goToFrame = useCallback((frameIdx: number) => {
    const framesWithPositions = getFramesWithPositions()
    if (framesWithPositions.length === 0) return

    // Find nearest valid frame
    let nearest = framesWithPositions[0]
    let minDist = Math.abs(frameIdx - nearest)

    for (const f of framesWithPositions) {
      const dist = Math.abs(frameIdx - f)
      if (dist < minDist) {
        minDist = dist
        nearest = f
      }
    }

    setCurrentFrame(nearest)
    // Seek video to this frame when not playing
    if (!isPlaying && videoPlayerRef.current && videoMetadata) {
      videoPlayerRef.current.currentTime = nearest / videoMetadata.fps
    }
  }, [getFramesWithPositions, isPlaying, videoMetadata])

  const skipFrames = useCallback((delta: number) => {
    const framesWithPositions = getFramesWithPositions()
    if (framesWithPositions.length === 0) return

    const currentIdx = framesWithPositions.indexOf(currentFrame)
    const newIdx = Math.max(0, Math.min(framesWithPositions.length - 1, currentIdx + delta))
    setCurrentFrame(framesWithPositions[newIdx])
  }, [currentFrame, getFramesWithPositions])

  // Sync video player with current frame when not in rAF playback
  useEffect(() => {
    if (!isPlaying && isSyncMode && videoPlayerRef.current && videoMetadata && playerPositions.length > 0) {
      const video = videoPlayerRef.current
      if (video.readyState >= 2) {
        const timeInSeconds = currentFrame / videoMetadata.fps
        if (Math.abs(video.currentTime - timeInSeconds) > 0.1) {
          video.currentTime = timeInSeconds
        }
      }
    }
  }, [currentFrame, isPlaying, isSyncMode, videoMetadata, playerPositions.length])

  // Load results frame when current frame changes
  useEffect(() => {
    if (playerPositions.length > 0 && videoMetadata) {
      loadResultsFrame(currentFrame)
    }
  }, [currentFrame, playerPositions.length, videoMetadata, loadResultsFrame])

  // Cleanup playback on unmount
  useEffect(() => {
    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current)
      }
    }
  }, [])

  // Create and cleanup video object URL
  useEffect(() => {
    if (videoFile) {
      const url = URL.createObjectURL(videoFile)
      setVideoObjectUrl(url)
      return () => {
        URL.revokeObjectURL(url)
      }
    } else {
      setVideoObjectUrl(null)
    }
  }, [videoFile])

  // Load warped frame for homography visualization
  const loadWarpedFrame = useCallback(async (frameIdx: number) => {
    if (!videoMetadata) return

    setLoadingWarpedFrame(true)
    try {
      const url = `${API_URL}/videos/${videoMetadata.video_id}/warped-frame/${frameIdx}`
      console.log(`Loading warped frame from: ${url}`)
      // Test if the endpoint exists first
      const response = await fetch(url)
      if (response.ok) {
        const blob = await response.blob()
        const objectUrl = URL.createObjectURL(blob)
        setWarpedFrameUrl(objectUrl)
      } else {
        setWarpedFrameUrl(null)
      }
    } catch (err) {
      console.error('Failed to load warped frame:', err)
      setWarpedFrameUrl(null)
    } finally {
      setLoadingWarpedFrame(false)
    }
  }, [videoMetadata])

  // Load warped frame when selected homography frame changes
  useEffect(() => {
    if (selectedHomographyFrame !== null) {
      loadWarpedFrame(selectedHomographyFrame)
    } else {
      setWarpedFrameUrl(null)
    }
  }, [selectedHomographyFrame, loadWarpedFrame])

  // Cleanup warped frame URL when it changes to prevent memory leaks
  useEffect(() => {
    return () => {
      if (warpedFrameUrl) {
        URL.revokeObjectURL(warpedFrameUrl)
      }
    }
  }, [warpedFrameUrl])

  // Get anchor frames that were used for homography (non-skipped with 4+ points)
  // Use homographyFrameIndices from backend if available, otherwise fall back to local anchorFrames
  const getHomographyFrames = useCallback(() => {
    // If we have homography frames from the backend, use those
    if (homographyFrameIndices.length > 0) {
      return homographyFrameIndices.map(frameIdx => {
        // Find the anchor frame data if it exists
        const anchor = anchorFrames.find(af => af.frame_idx === frameIdx)
        return anchor || { frame_idx: frameIdx, isSkipped: false, points: [], lines: [] }
      })
    }
    // Fall back to local anchor frames (before processing)
    return anchorFrames.filter(af => !af.isSkipped && af.points.length >= 4)
  }, [anchorFrames, homographyFrameIndices])

  // Redraw frame when annotations change or image loads
  useEffect(() => {
    if (!loadingFrame && frameImageRef.current && anchorFrames.length > 0) {
      drawFrameWithPoints()
    }
  }, [anchorFrames, currentAnchorIdx, drawFrameWithPoints, loadingFrame, pendingLinePoint1])

  // Redraw pitch diagram when pending click changes or annotations change
  useEffect(() => {
    if (anchorFrames.length > 0) {
      drawPitchDiagram()
    }
  }, [pendingFrameClick, anchorFrames, currentAnchorIdx, drawPitchDiagram])

  // Redraw pitch with player positions when frame changes
  useEffect(() => {
    if (playerPositions.length > 0) {
      drawPitch(playerPositions, currentFrame)
    }
  }, [currentFrame, playerPositions, drawPitch])

  // Keep stepDoneRef in sync so the annotation stale effect can read it without deps
  useEffect(() => { stepDoneRef.current.B = stepBResult !== null }, [stepBResult])
  useEffect(() => { stepDoneRef.current.C = stepCResult !== null }, [stepCResult])
  useEffect(() => { stepDoneRef.current.D = stepDResult !== null }, [stepDResult])

  // Auto-save annotations to localStorage whenever anchorFrames changes
  useEffect(() => {
    if (anchorFrames.length > 0 && videoFile) {
      localStorage.setItem(`gaa_annotations_${videoFile.name}`, JSON.stringify(anchorFrames))
    }
    // Mark downstream pipeline steps stale when annotations change
    const { B, C, D } = stepDoneRef.current
    if (B || C || D) {
      setStaleSteps(prev => {
        const next = new Set(prev)
        if (B) next.add('B')
        if (C) next.add('C')
        if (D) next.add('D')
        return next
      })
    }
  }, [anchorFrames, videoFile])

  const currentAnchor = anchorFrames[currentAnchorIdx]

  // Coverage quality metrics for current anchor frame
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
    <>
      <Head>
        <title>GAA Video Analysis</title>
        <meta name="description" content="GAA Video Analysis System" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <div className="container">
        <h1>GAA Video Analysis System</h1>

        {/* Step 1: Upload Video */}
        <div className="upload-section">
          <h2>1. Upload Video</h2>
          <div className="file-input">
            <input
              type="file"
              accept="video/mp4"
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) {
                  setVideoFile(file)
                  setVideoMetadata(null)
                  setAnchorFrames([])
                  setError('')
                }
              }}
            />
          </div>
          {videoFile && !videoMetadata && (
            <div>
              <p>Selected: {videoFile.name}</p>
              <button onClick={uploadVideo} disabled={uploadingVideo}>
                {uploadingVideo ? 'Uploading...' : 'Upload & Analyze'}
              </button>
            </div>
          )}

          {videoMetadata && (
            <div className="video-info">
              <h3>Video Information</h3>
              <div className="info-grid">
                <div><strong>Duration:</strong> {videoMetadata.duration_seconds.toFixed(2)}s</div>
                <div><strong>FPS:</strong> {videoMetadata.fps}</div>
                <div><strong>Total Frames:</strong> {videoMetadata.num_frames}</div>
                <div><strong>Resolution:</strong> {videoMetadata.width} x {videoMetadata.height}</div>
              </div>
            </div>
          )}
        </div>

        {/* Step 2: Configure Anchor Frames */}
        {videoMetadata && anchorFrames.length === 0 && (
          <div className="config-section">
            <h2>2. Configure Anchor Frames</h2>
            <p>Set up which frames to use for pitch annotations.</p>

            <div className="config-form">
              <div className="config-row">
                <label>
                  Trim Start (seconds):
                  <input
                    type="number"
                    min={0}
                    max={videoMetadata.duration_seconds}
                    step={0.1}
                    value={trimStartSeconds}
                    onChange={(e) => setTrimStartSeconds(parseFloat(e.target.value) || 0)}
                  />
                </label>
                <label>
                  Trim End (seconds):
                  <input
                    type="number"
                    min={trimStartSeconds}
                    max={videoMetadata.duration_seconds}
                    step={0.1}
                    value={trimEndSeconds ?? videoMetadata.duration_seconds}
                    onChange={(e) => setTrimEndSeconds(parseFloat(e.target.value) || null)}
                  />
                </label>
              </div>

              <div className="config-row">
                <label>
                  Anchor Frame Interval (seconds):
                  <select
                    value={anchorInterval}
                    onChange={(e) => setAnchorInterval(parseFloat(e.target.value))}
                  >
                    <option value={0.5}>Every 0.5 seconds</option>
                    <option value={1}>Every 1 second</option>
                    <option value={2}>Every 2 seconds</option>
                    <option value={5}>Every 5 seconds</option>
                    <option value={10}>Every 10 seconds</option>
                  </select>
                </label>
              </div>

              <div className="config-preview">
                <p>
                  This will generate approximately{' '}
                  <strong>
                    {Math.ceil(((trimEndSeconds ?? videoMetadata.duration_seconds) - trimStartSeconds) / anchorInterval)}
                  </strong>{' '}
                  anchor frames to annotate.
                </p>
              </div>

              <button onClick={generateAnchorFrames}>
                Generate Anchor Frames
              </button>
            </div>
          </div>
        )}

        {/* Step 3: Annotate Anchor Frames */}
        {anchorFrames.length > 0 && playerPositions.length === 0 && (
          <div className="annotation-section">
            <h2>3. Annotate Anchor Frames</h2>

            {/* Anchor frame navigation */}
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
                {/* Coverage quality badge */}
                <span style={{ marginLeft: 12, fontSize: 12 }}>
                  <span style={{ color: coverageColor }}>●</span>{' '}
                  {pointCount} points
                  {currentAnchor.points.length >= 2 && ` | Coverage: ${coveragePercent}%`}
                  {clustered && ' | ⚠ Clustered'}
                </span>
                <div className="anchor-actions">
                  <button
                    onClick={toggleSkipFrame}
                    className={currentAnchor.isSkipped ? 'warning' : ''}
                  >
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
                      onClick={() => {
                        setAnnotationMode('point')
                        setPendingLinePoint1(null)
                      }}
                    >
                      📍 Point Mode
                    </button>
                    <button
                      className={`mode-btn ${annotationMode === 'line' ? 'active' : ''}`}
                      onClick={() => {
                        setAnnotationMode('line')
                        setPendingFrameClick(null)
                      }}
                    >
                      📏 Line Mode
                    </button>
                  </div>

                  {annotationMode === 'line' && (
                    <div className="line-selector">
                      <label>Select Line:</label>
                      <select
                        value={selectedLineId}
                        onChange={(e) => setSelectedLineId(e.target.value)}
                      >
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
                        <button onClick={cancelLineAnnotation} className="cancel-btn">
                          Cancel
                        </button>
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
                      <button
                        onClick={() => setPendingFrameClick(null)}
                        className="cancel-btn"
                      >
                        Cancel
                      </button>
                    </p>
                  ) : (
                    <p>📍 <strong>Point Mode:</strong> Click a point on the video frame, then select the corresponding pitch location on the diagram.</p>
                  )}
                </div>

                {/* Side-by-side frame and pitch diagram */}
                <div className="annotation-workspace">
                  {/* Frame canvas for annotation */}
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

                  {/* Pitch diagram for selecting corresponding point */}
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

                {/* Current points list */}
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
                              <br/>
                              <small>Frame: ({point.x_img}, {point.y_img})</small>
                            </span>
                            <button onClick={() => removePoint(idx)} className="remove-btn">×</button>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* Current lines list */}
                {currentAnchor && currentAnchor.lines && currentAnchor.lines.length > 0 && (
                  <div className="lines-list">
                    <h4>📏 Annotated Lines ({currentAnchor.lines.length}):</h4>
                    <div className="lines-grid">
                      {currentAnchor.lines.map((line, idx) => (
                        <div key={idx} className="line-item">
                          <span>
                            <strong>{AVAILABLE_LINES[line.line_id]?.label || line.line_id}</strong>
                            <br/>
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
              <button
                onClick={() => goToAnchorFrame(currentAnchorIdx - 1)}
                disabled={currentAnchorIdx === 0}
              >
                ← Previous Frame
              </button>
              <button
                onClick={() => goToAnchorFrame(currentAnchorIdx + 1)}
                disabled={currentAnchorIdx === anchorFrames.length - 1}
              >
                Next Frame →
              </button>
            </div>

            {/* Pipeline steps */}
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
                <button onClick={exportAnnotations} className="secondary-btn">
                  ⬇ Export Annotations
                </button>
                <button onClick={() => importAnnotationsRef.current?.click()} className="secondary-btn">
                  ⬆ Import Annotations
                </button>
                <input
                  ref={importAnnotationsRef}
                  type="file"
                  accept=".json"
                  style={{ display: 'none' }}
                  onChange={importAnnotations}
                />
              </div>

              {/* Step A */}
              <div className="pipeline-step">
                <div className="step-header">
                  <h4>Step A: Upload &amp; Run Tracking</h4>
                  {staleSteps.has('A') && <span className="stale-badge">STALE</span>}
                </div>
                <button
                  onClick={runStepA}
                  disabled={!videoMetadata || runningStep !== null}
                  className="process-btn"
                >
                  {runningStep === 'A' ? 'Running...' : 'Upload & Run Tracking'}
                </button>
                {stepAResult && (
                  <div className="step-result">
                    <p>✅ Tracking complete</p>
                    <p><strong>video_id:</strong> {videoMetadata?.video_id}</p>
                    <p><strong>fps:</strong> {videoMetadata?.fps} | <strong>frames:</strong> {videoMetadata?.num_frames}</p>
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
                <button
                  onClick={runStepB}
                  disabled={!stepAResult || runningStep !== null}
                  className="process-btn"
                >
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
                    {/* Warped frame thumbnails */}
                    <div className="warped-thumbs">
                      {stepBResult.frames.map(f => (
                        <div key={f} className="warped-thumb-item">
                          <p className="thumb-label">Frame {f}</p>
                          <div className="thumb-row">
                            <div>
                              <p className="thumb-sublabel">Original</p>
                              <img
                                src={`${API_URL}/videos/${videoMetadata?.video_id}/frame/${f}`}
                                alt={`Original frame ${f}`}
                                className="thumb-img"
                              />
                            </div>
                            <div>
                              <p className="thumb-sublabel">Warped</p>
                              <img
                                src={`${API_URL}/videos/${videoMetadata?.video_id}/frames/${f}/warped`}
                                alt={`Warped frame ${f}`}
                                className="thumb-img"
                              />
                            </div>
                            {stepCResult && (
                              <div>
                                <p className="thumb-sublabel">With Players</p>
                                <img
                                  src={`${API_URL}/videos/${videoMetadata?.video_id}/frames/${f}/warped_with_players`}
                                  alt={`Warped with players frame ${f}`}
                                  className="thumb-img"
                                />
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
                <button
                  onClick={runStepC}
                  disabled={!stepBResult || runningStep !== null}
                  className="process-btn"
                >
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
                <button
                  onClick={runStepD}
                  disabled={!stepCResult || runningStep !== null}
                  className="process-btn"
                >
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
          </div>
        )}

        {/* Status messages */}
        {(status || error) && (
          <div className={`status ${error ? 'error' : 'success'}`}>
            {error || status}
          </div>
        )}

        {/* Loading indicator */}
        {processing && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Processing video... This may take several minutes.</p>
          </div>
        )}

        {/* Step 4: View Results */}
        {playerPositions.length > 0 && (
          <div className="results-section">
            <h2>4. Player Tracking Results</h2>

            {/* Processing info */}
            <div className="processing-info">
              <p>
                <strong>Processed frames:</strong> {processedStartFrame} - {processedEndFrame} |
                <strong> Total detections:</strong> {playerPositions.length} |
                <strong> Unique frames with players:</strong> {getFramesWithPositions().length} |
                <strong> Homography anchors:</strong> {homographyFrameIndices.length}
              </p>
            </div>

            {/* Playback controls */}
            <div className="playback-controls">
              <div className="playback-buttons">
                <button onClick={() => skipFrames(-10)} title="Back 10 frames">⏪</button>
                <button onClick={() => skipFrames(-1)} title="Previous frame">◀</button>
                <button onClick={togglePlayback} className="play-btn">
                  {isPlaying ? '⏸ Pause' : '▶ Play'}
                </button>
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

                <button
                  onClick={() => setIsSyncMode(!isSyncMode)}
                  className={`sync-btn ${isSyncMode ? 'active' : ''}`}
                >
                  🔗 {isSyncMode ? 'Sync ON' : 'Sync OFF'}
                </button>

                <button
                  onClick={() => setShowHomographySidebar(!showHomographySidebar)}
                  className={`sidebar-toggle ${showHomographySidebar ? 'active' : ''}`}
                >
                  📐 Homography Info
                </button>

                <button
                  onClick={() => setShowBotSortOverlay(!showBotSortOverlay)}
                  className={`sidebar-toggle ${showBotSortOverlay ? 'active' : ''}`}
                >
                  🎯 BotSort Overlay
                </button>
                {showBotSortOverlay && (
                  <span className="botsort-placeholder">BotSort overlay: coming soon</span>
                )}
              </div>
            </div>

            {/* Frame slider */}
            <div className="frame-slider">
              <input
                type="range"
                min={getFramesWithPositions()[0] || 0}
                max={getFramesWithPositions()[getFramesWithPositions().length - 1] || 100}
                value={currentFrame}
                onChange={(e) => goToFrame(parseInt(e.target.value))}
                className="slider"
              />
              <span className="frame-info">
                Frame {currentFrame} / {getFramesWithPositions()[getFramesWithPositions().length - 1] || 0}
                {videoMetadata && ` (${(currentFrame / videoMetadata.fps).toFixed(2)}s)`}
              </span>
            </div>

            {/* Main content area */}
            <div className="results-content">
              {/* Homography Sidebar */}
              {showHomographySidebar && (
                <div className="homography-sidebar">
                  <h3>📐 Homography Details</h3>
                  <p className="sidebar-info">
                    Homographies computed from {homographyFrameIndices.length > 0 ? homographyFrameIndices.length : getHomographyFrames().length} anchor frames.
                    Click an anchor frame to see details.
                  </p>

                  <div className="anchor-frame-list">
                    {homographyFrameIndices.length > 0 ? (
                      // Use homography frames from backend
                      homographyFrameIndices.map((frameIdx, idx) => {
                        const anchorData = anchorFrames.find(af => af.frame_idx === frameIdx)
                        return (
                          <div
                            key={idx}
                            className={`anchor-frame-item ${selectedHomographyFrame === frameIdx ? 'selected' : ''}`}
                            onClick={() => setSelectedHomographyFrame(
                              selectedHomographyFrame === frameIdx ? null : frameIdx
                            )}
                          >
                            <span className="frame-badge">Frame {frameIdx}</span>
                            <span className="points-count">
                              {anchorData ? `${anchorData.points.length} points` : 'Computed'}
                            </span>
                          </div>
                        )
                      })
                    ) : (
                      // Fall back to local anchor frames
                      getHomographyFrames().map((af, idx) => (
                        <div
                          key={idx}
                          className={`anchor-frame-item ${selectedHomographyFrame === af.frame_idx ? 'selected' : ''}`}
                          onClick={() => setSelectedHomographyFrame(
                            selectedHomographyFrame === af.frame_idx ? null : af.frame_idx
                          )}
                        >
                          <span className="frame-badge">Frame {af.frame_idx}</span>
                          <span className="points-count">{af.points.length} points</span>
                        </div>
                      ))
                    )}
                  </div>

                  {selectedHomographyFrame !== null && (
                    <div className="homography-detail">
                      <h4>Frame {selectedHomographyFrame} Annotations</h4>

                      {/* Warped frame visualization */}
                      <div className="warped-frame-section">
                        <h5>Warped Frame Preview</h5>
                        {loadingWarpedFrame && <p className="loading-text">Loading warped frame...</p>}
                        {warpedFrameUrl && !loadingWarpedFrame && (
                          <img
                            src={warpedFrameUrl}
                            alt={`Warped frame ${selectedHomographyFrame}`}
                            className="warped-frame-img"
                          />
                        )}
                        {!warpedFrameUrl && !loadingWarpedFrame && (
                          <p className="no-warped-frame">
                            Warped frame preview not available.
                            This shows how the video frame maps to the pitch canvas.
                          </p>
                        )}
                      </div>

                      {(() => {
                        const anchorData = anchorFrames.find(af => af.frame_idx === selectedHomographyFrame)
                        if (anchorData && anchorData.points.length > 0) {
                          return (
                            <div className="point-mapping-list">
                              <h5>Keypoint Correspondences</h5>
                              {anchorData.points.map((point, idx) => (
                                <div key={idx} className="point-mapping">
                                  <span className="pitch-label">{getPointLabel(point.pitch_id)}</span>
                                  <span className="arrow">→</span>
                                  <span className="coords">({Math.round(point.x_img)}, {Math.round(point.y_img)})</span>
                                </div>
                              ))}
                            </div>
                          )
                        } else {
                          return (
                            <p className="no-annotation-data">
                              Keypoint data for this frame is available on the server.
                            </p>
                          )
                        }
                      })()}
                      <p className="homography-note">
                        The warped frame shows the perspective transform applied.
                        Player positions are mapped using this homography.
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Side-by-side view */}
              <div className={`results-main ${showHomographySidebar ? 'with-sidebar' : ''}`}>
                {/* Video frame view */}
                <div className="video-frame-panel">
                  <h4>Video Frame {currentFrame}</h4>
                  {videoObjectUrl && (
                    <video
                      ref={videoPlayerRef}
                      src={videoObjectUrl}
                      className="results-video"
                      muted
                      playsInline
                      onTimeUpdate={() => {
                        if (isSyncMode && videoPlayerRef.current && videoMetadata && !isPlaying) {
                          const frameFromVideo = Math.round(videoPlayerRef.current.currentTime * videoMetadata.fps)
                          if (Math.abs(frameFromVideo - currentFrame) > 1) {
                            goToFrame(frameFromVideo)
                          }
                        }
                      }}
                    />
                  )}
                  {!videoObjectUrl && (
                    <div className="video-placeholder">
                      <p>Video not available</p>
                    </div>
                  )}
                </div>

                {/* 2D Pitch view */}
                <div className="pitch-view-panel">
                  <h4>2D Pitch View</h4>
                  <canvas
                    ref={canvasRef}
                    width={PITCH_DISPLAY_WIDTH}
                    height={PITCH_DISPLAY_HEIGHT}
                    className="pitch-canvas"
                  />
                  <div className="pitch-legend">
                    <span>● Each player has a unique color based on their track ID</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Player list for current frame */}
            <div className="current-frame-players">
              <h4>Players in Frame {currentFrame}</h4>
              <div className="player-list">
                {playerPositions
                  .filter(p => p.frame_idx === currentFrame)
                  .map((pos, idx) => (
                    <span key={idx} className="player-badge">
                      #{pos.track_id}: ({pos.x_pitch.toFixed(1)}, {pos.y_pitch.toFixed(1)})
                      <small>{pos.source}</small>
                    </span>
                  ))
                }
                {playerPositions.filter(p => p.frame_idx === currentFrame).length === 0 && (
                  <span className="no-players">No players detected in this frame</span>
                )}
              </div>
            </div>

            {/* Debug: Detailed coordinate table */}
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
                      <th>Track ID</th>
                      <th>x_pitch</th>
                      <th>y_pitch</th>
                      <th>x_display</th>
                      <th>y_display</th>
                      <th>Source</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {playerPositions
                      .filter(p => p.frame_idx === currentFrame)
                      .sort((a, b) => a.track_id - b.track_id)
                      .map((pos, idx) => {
                        const xDisplay = (pos.x_pitch / PITCH_CANVAS_W) * PITCH_DISPLAY_WIDTH
                        const yDisplay = (pos.y_pitch / PITCH_CANVAS_H) * PITCH_DISPLAY_HEIGHT
                        const isOutOfBounds =
                          pos.x_pitch < 0 || pos.x_pitch > PITCH_CANVAS_W ||
                          pos.y_pitch < 0 || pos.y_pitch > PITCH_CANVAS_H
                        return (
                          <tr key={idx} className={isOutOfBounds ? 'out-of-bounds' : ''}>
                            <td><strong>#{pos.track_id}</strong></td>
                            <td className={pos.x_pitch < 0 || pos.x_pitch > PITCH_CANVAS_W ? 'bad-value' : ''}>
                              {pos.x_pitch.toFixed(2)}
                            </td>
                            <td className={pos.y_pitch < 0 || pos.y_pitch > PITCH_CANVAS_H ? 'bad-value' : ''}>
                              {pos.y_pitch.toFixed(2)}
                            </td>
                            <td>{xDisplay.toFixed(1)}</td>
                            <td>{yDisplay.toFixed(1)}</td>
                            <td>{pos.source}</td>
                            <td>{isOutOfBounds ? '❌ OUT' : '✅ OK'}</td>
                          </tr>
                        )
                      })
                    }
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
          </div>
        )}

        {/* Debug Log Panel (Task 5) */}
        <div className="debug-log-panel">
          <div className="debug-log-header" onClick={() => setDebugLogVisible(v => !v)}>
            <h3>🐛 Debug Log {debugLogEntries.length > 0 && `(${debugLogEntries.length} entries)`}</h3>
            <div className="debug-log-controls">
              <span className="debug-toggle-hint">{debugLogVisible ? '▲ Collapse' : '▼ Expand'}</span>
              <button
                className="secondary-btn"
                onClick={(e) => { e.stopPropagation(); debugLog.current = []; setDebugLogEntries([]) }}
              >
                Clear
              </button>
            </div>
          </div>
          {debugLogVisible && (
            <div className="debug-log-content">
              <div className="debug-pipeline-state">
                <strong>Pipeline state:</strong>{' '}
                video_id: {videoMetadata?.video_id || '—'} |{' '}
                detections: {stepAResult ? stepAResult.num_detections : '—'} |{' '}
                homographies: {stepBResult ? stepBResult.frames.length : '—'} |{' '}
                positions: {stepCResult ? stepCResult.total : '—'} |{' '}
                interpolated frames: {stepDResult ? stepDResult.frames_generated : '—'}
              </div>
              <div className="debug-log-entries">
                {debugLogEntries.length === 0
                  ? <span className="debug-empty">No API calls logged yet.</span>
                  : debugLogEntries.map((entry, i) => (
                    <div key={i} className={`debug-log-entry ${entry.startsWith('←') ? 'response' : entry.startsWith('✗') ? 'error-entry' : 'request'}`}>
                      {entry}
                    </div>
                  ))
                }
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}
