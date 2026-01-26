import { useState, useRef, useEffect, useCallback } from 'react'
import Head from 'next/head'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface PitchPoint {
  pitch_id: string
  x_img: number
  y_img: number
}

interface PitchAnnotation {
  frame_idx: number
  points: PitchPoint[]
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

  // Refs
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameCanvasRef = useRef<HTMLCanvasElement>(null)
  const frameImageRef = useRef<HTMLImageElement | null>(null)
  const videoPlayerRef = useRef<HTMLVideoElement>(null)
  const playbackIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const resultsFrameCanvasRef = useRef<HTMLCanvasElement>(null)
  const resultsFrameImageRef = useRef<HTMLImageElement | null>(null)

  // Backend pitch canvas dimensions (canonical coordinate space)
  // All player positions from backend are in this pixel space
  const PITCH_CANVAS_W = 850
  const PITCH_CANVAS_H = 1450

  // Display canvas dimensions - MUST maintain same aspect ratio as backend canvas
  // Aspect ratio = 850/1450 = 0.5862
  // We scale down to fit the UI while preserving exact proportions
  const DISPLAY_SCALE = 0.4  // 40% of backend canvas size
  const PITCH_DISPLAY_WIDTH = Math.round(PITCH_CANVAS_W * DISPLAY_SCALE)   // 340
  const PITCH_DISPLAY_HEIGHT = Math.round(PITCH_CANVAS_H * DISPLAY_SCALE)  // 580

  // Actual GAA pitch dimensions in meters (145m includes goal area)
  const GAA_PITCH_WIDTH = 85.0
  const GAA_PITCH_LENGTH = 145.0

  // Pitch canvas ref for the diagram
  const pitchDiagramRef = useRef<HTMLCanvasElement>(null)

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
    "bottom_goal_lp": [39.25, 145.0],
    "bottom_goal_rp": [45.75, 145.0],

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

  // Upload video and get metadata
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
          points: []
        })
      }
    }

    setAnchorFrames(frames)
    setCurrentAnchorIdx(0)
    if (frames.length > 0) {
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

    // Draw annotation points for current anchor frame
    const currentAnchor = anchorFrames[currentAnchorIdx]
    if (currentAnchor && currentAnchor.points) {
      const imgScale = canvas.width / img.naturalWidth
      currentAnchor.points.forEach((point) => {
        const x = point.x_img * imgScale
        const y = point.y_img * imgScale

        // Draw point
        ctx.fillStyle = '#ff0000'
        ctx.beginPath()
        ctx.arc(x, y, 8, 0, 2 * Math.PI)
        ctx.fill()

        // Draw border
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 2
        ctx.stroke()

        // Draw label
        ctx.fillStyle = '#ffffff'
        ctx.font = 'bold 12px Arial'
        ctx.fillText(point.pitch_id, x + 12, y + 4)
      })
    }
  }, [anchorFrames, currentAnchorIdx])

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

    // Store the pending frame click - user must now click on pitch diagram
    setPendingFrameClick({ x: Math.round(x), y: Math.round(y) })
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

    // If there's a pending click, show instruction
    if (pendingFrameClick) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
      ctx.fillRect(10, PITCH_DISPLAY_HEIGHT - 40, PITCH_DISPLAY_WIDTH - 20, 30)
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 12px Arial'
      ctx.textAlign = 'center'
      ctx.fillText('Click a point on this pitch diagram', PITCH_DISPLAY_WIDTH / 2, PITCH_DISPLAY_HEIGHT - 20)
      ctx.textAlign = 'left'
    }
  }, [pendingFrameClick, anchorFrames, currentAnchorIdx, pitchToCanvas])

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
        points: [] // Clear points when swapping
      }
      return updated
    })

    loadFrameImage(newFrameIdx)
  }

  // Process video with annotations
  const processVideo = async () => {
    if (!videoFile || !videoMetadata) {
      setError('Please upload a video first')
      return
    }

    // Get annotated anchor frames (non-skipped with at least 4 points)
    const validAnnotations: PitchAnnotation[] = anchorFrames
      .filter(af => !af.isSkipped && af.points.length >= 4)
      .map(af => ({
        frame_idx: af.frame_idx,
        points: af.points
      }))

    if (validAnnotations.length === 0) {
      setError('Please annotate at least one anchor frame with 4+ points')
      return
    }

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

  const drawPitch = (positions: PlayerPosition[], frame: number) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Use the pitch display dimensions for results
    const RESULTS_PITCH_WIDTH = PITCH_DISPLAY_WIDTH
    const RESULTS_PITCH_HEIGHT = PITCH_DISPLAY_HEIGHT

    // Clear canvas
    ctx.fillStyle = '#2d5016' // Green pitch color
    ctx.fillRect(0, 0, RESULTS_PITCH_WIDTH, RESULTS_PITCH_HEIGHT)

    // Draw center line
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(0, RESULTS_PITCH_HEIGHT / 2)
    ctx.lineTo(RESULTS_PITCH_WIDTH, RESULTS_PITCH_HEIGHT / 2)
    ctx.stroke()

    // Draw center circle
    ctx.beginPath()
    ctx.arc(RESULTS_PITCH_WIDTH / 2, RESULTS_PITCH_HEIGHT / 2, 50, 0, 2 * Math.PI)
    ctx.stroke()

    // Filter positions for current frame
    const framePositions = positions.filter(p => p.frame_idx === frame)

    // Draw player positions
    // Backend returns coordinates in PITCH CANVAS PIXELS (0-850 for x, 0-1450 for y)
    // Scale to display canvas: x_display = (x_pitch / PITCH_CANVAS_W) * DISPLAY_WIDTH
    framePositions.forEach((pos, idx) => {
      const x = (pos.x_pitch / PITCH_CANVAS_W) * RESULTS_PITCH_WIDTH
      const y = (pos.y_pitch / PITCH_CANVAS_H) * RESULTS_PITCH_HEIGHT

      // Clamp to canvas bounds
      const clampedX = Math.max(0, Math.min(RESULTS_PITCH_WIDTH, x))
      const clampedY = Math.max(0, Math.min(RESULTS_PITCH_HEIGHT, y))

      // Draw player as circle
      ctx.fillStyle = idx % 2 === 0 ? '#ff0000' : '#0000ff'
      ctx.beginPath()
      ctx.arc(clampedX, clampedY, 8, 0, 2 * Math.PI)
      ctx.fill()

      // Draw track ID
      ctx.fillStyle = '#ffffff'
      ctx.font = '12px Arial'
      ctx.fillText(pos.track_id.toString(), clampedX + 10, clampedY - 10)
    })

    // Draw frame info
    ctx.fillStyle = '#ffffff'
    ctx.font = '16px Arial'
    ctx.fillText(`Frame: ${frame}`, 10, 30)
    ctx.fillText(`Players: ${framePositions.length}`, 10, 50)
  }

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

  // Playback controls
  const stopPlayback = useCallback(() => {
    if (playbackIntervalRef.current) {
      clearInterval(playbackIntervalRef.current)
      playbackIntervalRef.current = null
    }
    setIsPlaying(false)
  }, [])

  const startPlayback = useCallback(() => {
    if (playbackIntervalRef.current) {
      clearInterval(playbackIntervalRef.current)
    }

    const framesWithPositions = getFramesWithPositions()
    if (framesWithPositions.length === 0) return

    setIsPlaying(true)

    const fps = processedFps || videoMetadata?.fps || 25
    const intervalMs = (1000 / fps) / playbackSpeed

    playbackIntervalRef.current = setInterval(() => {
      setCurrentFrame(prev => {
        const currentIdx = framesWithPositions.indexOf(prev)
        const nextIdx = currentIdx + 1

        if (nextIdx >= framesWithPositions.length) {
          // Stop at end
          stopPlayback()
          return prev
        }

        return framesWithPositions[nextIdx]
      })
    }, intervalMs)
  }, [getFramesWithPositions, playbackSpeed, videoMetadata, processedFps, stopPlayback])


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
  }, [getFramesWithPositions])

  const skipFrames = useCallback((delta: number) => {
    const framesWithPositions = getFramesWithPositions()
    if (framesWithPositions.length === 0) return

    const currentIdx = framesWithPositions.indexOf(currentFrame)
    const newIdx = Math.max(0, Math.min(framesWithPositions.length - 1, currentIdx + delta))
    setCurrentFrame(framesWithPositions[newIdx])
  }, [currentFrame, getFramesWithPositions])

  // Sync video player with current frame
  useEffect(() => {
    if (isSyncMode && videoPlayerRef.current && videoMetadata && playerPositions.length > 0) {
      const timeInSeconds = currentFrame / videoMetadata.fps
      if (Math.abs(videoPlayerRef.current.currentTime - timeInSeconds) > 0.1) {
        videoPlayerRef.current.currentTime = timeInSeconds
      }
    }
  }, [currentFrame, isSyncMode, videoMetadata, playerPositions.length])

  // Load results frame when current frame changes
  useEffect(() => {
    if (playerPositions.length > 0 && videoMetadata) {
      loadResultsFrame(currentFrame)
    }
  }, [currentFrame, playerPositions.length, videoMetadata, loadResultsFrame])

  // Cleanup playback on unmount
  useEffect(() => {
    return () => {
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current)
      }
    }
  }, [])

  // Get anchor frames that were used for homography (non-skipped with 4+ points)
  const getHomographyFrames = useCallback(() => {
    return anchorFrames.filter(af => !af.isSkipped && af.points.length >= 4)
  }, [anchorFrames])

  // Redraw frame when annotations change or image loads
  useEffect(() => {
    if (!loadingFrame && frameImageRef.current && anchorFrames.length > 0) {
      drawFrameWithPoints()
    }
  }, [anchorFrames, currentAnchorIdx, drawFrameWithPoints, loadingFrame])

  // Redraw pitch diagram when pending click changes or annotations change
  useEffect(() => {
    if (anchorFrames.length > 0) {
      drawPitchDiagram()
    }
  }, [pendingFrameClick, anchorFrames, currentAnchorIdx, drawPitchDiagram])

  useEffect(() => {
    if (playerPositions.length > 0) {
      drawPitch(playerPositions, currentFrame)
    }
  }, [currentFrame, playerPositions])

  const currentAnchor = anchorFrames[currentAnchorIdx]

  return (
    <>
      <Head>
        <title>GAA Video Analysis</title>
        <meta name="description" content="GAA Video Analysis System" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <div className="container">
        <h1>⚽ GAA Video Analysis System</h1>

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
                  Points: {currentAnchor.points.length}/4+
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
                </div>
              </div>
            )}

            {!currentAnchor?.isSkipped && (
              <>
                {/* Instructions */}
                <div className="annotation-instructions">
                  {pendingFrameClick ? (
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
                    <p>Click a point on the video frame, then select the corresponding pitch location on the diagram.</p>
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
                        className={`frame-canvas ${pendingFrameClick ? 'has-pending' : ''}`}
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
                      {currentAnchor.points.map((point, idx) => (
                        <div key={idx} className="point-item">
                          <span>
                            <strong>{getPointLabel(point.pitch_id)}</strong>
                            <br/>
                            <small>Frame: ({point.x_img}, {point.y_img})</small>
                          </span>
                          <button onClick={() => removePoint(idx)} className="remove-btn">×</button>
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

            {/* Process button */}
            <div className="process-section">
              <div className="annotation-summary">
                <p>
                  <strong>Ready to process:</strong>{' '}
                  {anchorFrames.filter(af => !af.isSkipped && af.points.length >= 4).length} frames annotated |{' '}
                  {anchorFrames.filter(af => af.isSkipped).length} frames skipped |{' '}
                  {anchorFrames.filter(af => !af.isSkipped && af.points.length < 4).length} frames incomplete
                </p>
              </div>
              <button
                onClick={processVideo}
                disabled={processing || anchorFrames.filter(af => !af.isSkipped && af.points.length >= 4).length === 0}
                className="process-btn"
              >
                {processing ? 'Processing...' : 'Process Video'}
              </button>
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
                    Homographies computed from {getHomographyFrames().length} anchor frames.
                    Click an anchor frame to see details.
                  </p>

                  <div className="anchor-frame-list">
                    {getHomographyFrames().map((af, idx) => (
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
                    ))}
                  </div>

                  {selectedHomographyFrame !== null && (
                    <div className="homography-detail">
                      <h4>Frame {selectedHomographyFrame} Annotations</h4>
                      <div className="point-mapping-list">
                        {anchorFrames
                          .find(af => af.frame_idx === selectedHomographyFrame)
                          ?.points.map((point, idx) => (
                            <div key={idx} className="point-mapping">
                              <span className="pitch-label">{getPointLabel(point.pitch_id)}</span>
                              <span className="arrow">→</span>
                              <span className="coords">({point.x_img}, {point.y_img})</span>
                            </div>
                          ))
                        }
                      </div>
                      <p className="homography-note">
                        These pixel-to-pitch mappings define the perspective transform
                        used to project player positions onto the 2D pitch view.
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Side-by-side view */}
              <div className={`results-main ${showHomographySidebar ? 'with-sidebar' : ''}`}>
                {/* Video frame view */}
                <div className="video-frame-panel">
                  <h4>Video Frame</h4>
                  {videoFile && (
                    <video
                      ref={videoPlayerRef}
                      src={URL.createObjectURL(videoFile)}
                      className="results-video"
                      muted
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
                  {!videoFile && (
                    <canvas
                      ref={resultsFrameCanvasRef}
                      className="results-frame-canvas"
                    />
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
                    <span>🔴 Players (even IDs)</span>
                    <span>🔵 Players (odd IDs)</span>
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
          </div>
        )}
      </div>
    </>
  )
}
