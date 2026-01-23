import { useState, useRef, useEffect } from 'react'
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

export default function Home() {
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [annotations, setAnnotations] = useState<PitchAnnotation[]>([])
  const [currentAnnotation, setCurrentAnnotation] = useState({
    frame_idx: 0,
    pitch_id: 'corner_tl',
    x_img: 0,
    y_img: 0
  })
  const [processing, setProcessing] = useState(false)
  const [status, setStatus] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [playerPositions, setPlayerPositions] = useState<PlayerPosition[]>([])
  const [currentFrame, setCurrentFrame] = useState(0)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Pitch dimensions
  const PITCH_WIDTH = 850
  const PITCH_HEIGHT = 1450

  // Pitch point options
  const pitchPointOptions = [
    { value: 'corner_tl', label: 'Top Left Corner' },
    { value: 'corner_tr', label: 'Top Right Corner' },
    { value: 'corner_bl', label: 'Bottom Left Corner' },
    { value: 'corner_br', label: 'Bottom Right Corner' },
    { value: 'top_goal_lp', label: 'Top Goal Left Post' },
    { value: 'top_goal_rp', label: 'Top Goal Right Post' },
    { value: 'bottom_goal_lp', label: 'Bottom Goal Left Post' },
    { value: 'bottom_goal_rp', label: 'Bottom Goal Right Post' },
  ]

  const addAnnotation = () => {
    const frameAnnotations = annotations.find(a => a.frame_idx === currentAnnotation.frame_idx)
    
    if (frameAnnotations) {
      // Add to existing frame
      setAnnotations(annotations.map(a => 
        a.frame_idx === currentAnnotation.frame_idx
          ? { ...a, points: [...a.points, { ...currentAnnotation }] }
          : a
      ))
    } else {
      // Create new frame annotation
      setAnnotations([...annotations, {
        frame_idx: currentAnnotation.frame_idx,
        points: [{ ...currentAnnotation }]
      }])
    }
    
    // Reset form
    setCurrentAnnotation({
      ...currentAnnotation,
      x_img: 0,
      y_img: 0
    })
  }

  const removeAnnotation = (frameIdx: number, pointIdx: number) => {
    setAnnotations(annotations.map(a => 
      a.frame_idx === frameIdx
        ? { ...a, points: a.points.filter((_, i) => i !== pointIdx) }
        : a
    ))
  }

  const processVideo = async () => {
    if (!videoFile) {
      setError('Please select a video file')
      return
    }

    if (annotations.length === 0) {
      setError('Please add at least one annotation')
      return
    }

    setProcessing(true)
    setError('')
    setStatus('Uploading video and processing...')

    try {
      const formData = new FormData()
      formData.append('file', videoFile)
      formData.append('annotations_json', JSON.stringify(annotations))

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

    // Clear canvas
    ctx.fillStyle = '#2d5016' // Green pitch color
    ctx.fillRect(0, 0, PITCH_WIDTH, PITCH_HEIGHT)

    // Draw center line
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(0, PITCH_HEIGHT / 2)
    ctx.lineTo(PITCH_WIDTH, PITCH_HEIGHT / 2)
    ctx.stroke()

    // Draw center circle
    ctx.beginPath()
    ctx.arc(PITCH_WIDTH / 2, PITCH_HEIGHT / 2, 50, 0, 2 * Math.PI)
    ctx.stroke()

    // Filter positions for current frame
    const framePositions = positions.filter(p => p.frame_idx === frame)

    // Draw player positions
    framePositions.forEach((pos, idx) => {
      const x = pos.x_pitch
      const y = pos.y_pitch

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

  useEffect(() => {
    if (playerPositions.length > 0) {
      drawPitch(playerPositions, currentFrame)
    }
  }, [currentFrame, playerPositions])

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
                  setError('')
                }
              }}
            />
          </div>
          {videoFile && (
            <p>Selected: {videoFile.name}</p>
          )}
        </div>

        <div className="annotations-section">
          <h3>2. Add Pitch Annotations</h3>
          <p>Add annotations for anchor frames. You need at least 4 points per frame.</p>
          
          <div className="annotation-form">
            <input
              type="number"
              placeholder="Frame Index"
              value={currentAnnotation.frame_idx}
              onChange={(e) => setCurrentAnnotation({
                ...currentAnnotation,
                frame_idx: parseInt(e.target.value) || 0
              })}
            />
            <select
              value={currentAnnotation.pitch_id}
              onChange={(e) => setCurrentAnnotation({
                ...currentAnnotation,
                pitch_id: e.target.value
              })}
            >
              {pitchPointOptions.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
            <input
              type="number"
              placeholder="X (image)"
              value={currentAnnotation.x_img}
              onChange={(e) => setCurrentAnnotation({
                ...currentAnnotation,
                x_img: parseFloat(e.target.value) || 0
              })}
            />
            <input
              type="number"
              placeholder="Y (image)"
              value={currentAnnotation.y_img}
              onChange={(e) => setCurrentAnnotation({
                ...currentAnnotation,
                y_img: parseFloat(e.target.value) || 0
              })}
            />
            <button onClick={addAnnotation}>Add Point</button>
          </div>

          {annotations.length > 0 && (
            <div className="annotation-list">
              <h4>Current Annotations:</h4>
              {annotations.map((ann) => (
                <div key={ann.frame_idx}>
                  <strong>Frame {ann.frame_idx}:</strong>
                  {ann.points.map((point, idx) => (
                    <div key={idx} className="annotation-item">
                      <span>{point.pitch_id} - ({point.x_img}, {point.y_img})</span>
                      <button onClick={() => removeAnnotation(ann.frame_idx, idx)}>Remove</button>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>

        <div style={{ marginTop: '30px' }}>
          <button onClick={processVideo} disabled={processing || !videoFile || annotations.length === 0}>
            {processing ? 'Processing...' : 'Process Video'}
          </button>
        </div>

        {status && (
          <div className={`status ${error ? 'error' : 'success'}`}>
            {error || status}
          </div>
        )}

        {processing && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Processing video... This may take several minutes.</p>
          </div>
        )}

        {playerPositions.length > 0 && (
          <div className="pitch-container">
            <div>
              <canvas
                ref={canvasRef}
                width={PITCH_WIDTH}
                height={PITCH_HEIGHT}
                className="pitch-canvas"
              />
              <div style={{ marginTop: '20px', textAlign: 'center' }}>
                <button onClick={() => {
                  const maxFrame = Math.max(...playerPositions.map(p => p.frame_idx))
                  setCurrentFrame(Math.max(0, currentFrame - 1))
                }}>
                  Previous Frame
                </button>
                <span style={{ margin: '0 20px' }}>
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
        )}
      </div>
    </>
  )
}
