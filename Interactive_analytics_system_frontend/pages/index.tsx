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

  // Refs
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameCanvasRef = useRef<HTMLCanvasElement>(null)
  const frameImageRef = useRef<HTMLImageElement | null>(null)

  // Pitch dimensions for display
  const PITCH_DISPLAY_WIDTH = 340
  const PITCH_DISPLAY_HEIGHT = 560

  // Actual GAA pitch dimensions in meters
  const GAA_PITCH_WIDTH = 85.0
  const GAA_PITCH_LENGTH = 140.0

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

    try {
      const url = `${API_URL}/videos/${videoMetadata.video_id}/frame/${frameIdx}`
      setFrameImageUrl(url)

      // Preload the image
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.onload = () => {
        frameImageRef.current = img
        drawFrameWithPoints()
        setLoadingFrame(false)
      }
      img.onerror = () => {
        setError(`Failed to load frame ${frameIdx}`)
        setLoadingFrame(false)
      }
      img.src = url
    } catch (err) {
      setError('Failed to load frame')
      setLoadingFrame(false)
    }
  }

  // Draw frame with annotation points
  const drawFrameWithPoints = useCallback(() => {
    const canvas = frameCanvasRef.current
    const img = frameImageRef.current
    if (!canvas || !img || anchorFrames.length === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size to match image (scaled down if needed)
    const maxWidth = 800
    const scale = Math.min(1, maxWidth / img.width)
    canvas.width = img.width * scale
    canvas.height = img.height * scale

    // Draw image
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    // Draw annotation points for current anchor frame
    const currentAnchor = anchorFrames[currentAnchorIdx]
    if (currentAnchor && currentAnchor.points) {
      currentAnchor.points.forEach((point, idx) => {
        const x = point.x_img * scale
        const y = point.y_img * scale

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

    const rect = canvas.getBoundingClientRect()
    const scaleX = img.width / canvas.width
    const scaleY = img.height / canvas.height

    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY

    // Store the pending frame click - user must now click on pitch diagram
    setPendingFrameClick({ x: Math.round(x), y: Math.round(y) })
  }

  // Handle click on pitch diagram to complete annotation
  const handlePitchDiagramClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!pendingFrameClick) return

    const canvas = pitchDiagramRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const clickY = e.clientY - rect.top

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
        setStatus('Processing completed!')
        setCurrentFrame(0)
        drawPitch(data.player_positions, 0)
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

    // Draw player positions (scale from meters to canvas pixels)
    framePositions.forEach((pos, idx) => {
      const x = (pos.x_pitch / GAA_PITCH_WIDTH) * RESULTS_PITCH_WIDTH
      const y = (pos.y_pitch / GAA_PITCH_LENGTH) * RESULTS_PITCH_HEIGHT

      // Draw player as circle
      ctx.fillStyle = idx % 2 === 0 ? '#ff0000' : '#0000ff'
      ctx.beginPath()
      ctx.arc(x, y, 8, 0, 2 * Math.PI)
      ctx.fill()

      // Draw track ID
      ctx.fillStyle = '#ffffff'
      ctx.font = '12px Arial'
      ctx.fillText(pos.track_id.toString(), x + 10, y - 10)
    })

    // Draw frame info
    ctx.fillStyle = '#ffffff'
    ctx.font = '16px Arial'
    ctx.fillText(`Frame: ${frame}`, 10, 30)
    ctx.fillText(`Players: ${framePositions.length}`, 10, 50)
  }

  // Redraw frame when annotations change
  useEffect(() => {
    if (frameImageRef.current && anchorFrames.length > 0) {
      drawFrameWithPoints()
    }
  }, [anchorFrames, currentAnchorIdx, drawFrameWithPoints])

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
            <h2>4. Player Positions</h2>
            <div className="pitch-container">
              <div>
                <canvas
                  ref={canvasRef}
                  width={PITCH_DISPLAY_WIDTH}
                  height={PITCH_DISPLAY_HEIGHT}
                  className="pitch-canvas"
                />
                <div className="frame-controls">
                  <button onClick={() => setCurrentFrame(Math.max(0, currentFrame - 1))}>
                    Previous Frame
                  </button>
                  <span>
                    Frame {currentFrame} / {playerPositions.length > 0 ? Math.max(...playerPositions.map(p => p.frame_idx)) : 0}
                  </span>
                  <button onClick={() => {
                    const maxFrame = playerPositions.length > 0 ? Math.max(...playerPositions.map(p => p.frame_idx)) : 0
                    setCurrentFrame(Math.min(maxFrame, currentFrame + 1))
                  }}>
                    Next Frame
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  )
}
