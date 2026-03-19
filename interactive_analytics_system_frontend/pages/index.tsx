import { useState, useRef, useEffect } from 'react'
import Head from 'next/head'
import type { VideoMetadata, AnchorFrame, PlayerPosition } from '../types'
import VideoUploader from '../components/VideoUploader'
import AnchorFrameAnnotator from '../components/AnchorFrameAnnotator'
import PipelineSteps from '../components/PipelineSteps'
import ResultsViewer from '../components/ResultsViewer'
import DebugLog from '../components/DebugLog'

export default function Home() {
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoMetadata, setVideoMetadata] = useState<VideoMetadata | null>(null)

  const [anchorInterval, setAnchorInterval] = useState(1)
  const [anchorFrames, setAnchorFrames] = useState<AnchorFrame[]>([])
  const [currentAnchorIdx, setCurrentAnchorIdx] = useState(0)

  const [stepAResult, setStepAResult] = useState<{ frames_processed: number; tracks: number; num_detections: number } | null>(null)
  const [stepBResult, setStepBResult] = useState<{ frames: number[]; per_frame_count: number; info: Record<string, any> } | null>(null)
  const [stepCResult, setStepCResult] = useState<{ positions: PlayerPosition[]; total: number } | null>(null)
  const [stepDResult, setStepDResult] = useState<{ frames_generated: number; method: string } | null>(null)
  const [staleSteps, setStaleSteps] = useState<Set<string>>(new Set())
  const [runningSteps, setRunningSteps] = useState<Set<string>>(new Set())
  const stepDoneRef = useRef({ B: false, C: false, D: false })

  const [playerPositions, setPlayerPositions] = useState<PlayerPosition[]>([])
  const [currentFrame, setCurrentFrame] = useState(0)
  const [processedStartFrame, setProcessedStartFrame] = useState(0)
  const [processedEndFrame, setProcessedEndFrame] = useState(0)
  const [homographyFrameIndices, setHomographyFrameIndices] = useState<number[]>([])
  const [processedFps, setProcessedFps] = useState(25)

  const [status, setStatus] = useState('')
  const [error, setError] = useState('')

  const debugLog = useRef<string[]>([])
  const [debugLogEntries, setDebugLogEntries] = useState<string[]>([])

  useEffect(() => { stepDoneRef.current.B = stepBResult !== null }, [stepBResult])
  useEffect(() => { stepDoneRef.current.C = stepCResult !== null }, [stepCResult])
  useEffect(() => { stepDoneRef.current.D = stepDResult !== null }, [stepDResult])

  const markStale = (steps: string[]) => {
    setStaleSteps(prev => { const next = new Set(prev); steps.forEach(s => next.add(s)); return next })
  }

  const clearStale = (steps: string[]) => {
    setStaleSteps(prev => { const next = new Set(prev); steps.forEach(s => next.delete(s)); return next })
  }

  const logApiCall = (entry: string) => {
    debugLog.current = [...debugLog.current, entry]
    setDebugLogEntries([...debugLog.current])
  }

  const handleAnchorFramesChange = (frames: AnchorFrame[]) => {
    setAnchorFrames(frames)
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
  }

  const generateAnchorFrames = () => {
    if (!videoMetadata) return
    const fps = videoMetadata.fps
    const endFrame = videoMetadata.num_frames - 1
    const frames: AnchorFrame[] = []
    for (let seconds = 0; seconds <= videoMetadata.duration_seconds; seconds += anchorInterval) {
      const frameIdx = Math.floor(seconds * fps)
      if (frameIdx <= endFrame) frames.push({ frame_idx: frameIdx, isSkipped: false, points: [], lines: [] })
    }
    setAnchorFrames(frames)
    setCurrentAnchorIdx(0)
    if (frames.length > 0 && videoFile) {
      const savedKey = `gaa_annotations_${videoFile.name}`
      const saved = localStorage.getItem(savedKey)
      if (saved) {
        try {
          const parsed: AnchorFrame[] = JSON.parse(saved)
          if (confirm(`Found saved annotations for this video (${parsed.length} frames). Restore them?`)) {
            const merged = frames.map(f => {
              const match = parsed.find(p => p.frame_idx === f.frame_idx)
              return match ? { ...f, isSkipped: match.isSkipped, points: match.points, lines: match.lines || [] } : f
            })
            setAnchorFrames(merged)
            return
          }
        } catch (_) {
          console.warn('Could not restore saved annotations')
        }
      }
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
        <h1>GAA Video Analysis</h1>
        <div className="app-layout">
          <div className="main-content">

            {/* Step 1: Upload */}
            <div className="step-card">
              <span className="step-card__badge">1</span>
              <span className="step-card__title">Upload Video</span>
              <VideoUploader
                onUploadSuccess={(metadata, file) => {
                  setVideoMetadata(metadata)
                  setVideoFile(file)
                  setAnchorFrames([])
                  setError('')
                  setStatus('Video uploaded successfully!')
                }}
              />
            </div>

            {/* Step 2: Configure Anchor Frames */}
            {videoMetadata && anchorFrames.length === 0 && (
              <div className="step-card">
                <span className="step-card__badge">2</span>
                <span className="step-card__title">Configure Anchor Frames</span>
                <div className="config-section">
                  <p>Set up which frames to use for pitch annotations.</p>
                  <div className="config-form">
                    <div className="config-row">
                      <label>
                        Anchor Frame Interval (seconds):
                        <select value={anchorInterval} onChange={(e) => setAnchorInterval(parseFloat(e.target.value))}>
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
                        <strong>{Math.ceil(videoMetadata.duration_seconds / anchorInterval)}</strong>{' '}
                        anchor frames to annotate.
                      </p>
                    </div>
                    <button onClick={generateAnchorFrames}>Generate Anchor Frames</button>
                  </div>
                </div>
              </div>
            )}

            {/* Step 3: Annotate */}
            {anchorFrames.length > 0 && videoMetadata && (
              <>
                <div className="step-card">
                  <span className="step-card__badge">3</span>
                  <span className="step-card__title">Annotate Frames</span>
                  <AnchorFrameAnnotator
                    videoMetadata={videoMetadata}
                    videoFilename={videoFile?.name}
                    anchorFrames={anchorFrames}
                    currentAnchorIdx={currentAnchorIdx}
                    onAnchorFramesChange={handleAnchorFramesChange}
                    onCurrentIdxChange={setCurrentAnchorIdx}
                  />
                </div>

                {/* Step 4: Pipeline */}
                <div className="step-card">
                  <span className="step-card__badge">4</span>
                  <span className="step-card__title">Run Pipeline</span>
                  <PipelineSteps
                    videoMetadata={videoMetadata}
                    anchorFrames={anchorFrames}
                    stepAResult={stepAResult}
                    stepBResult={stepBResult}
                    stepCResult={stepCResult}
                    stepDResult={stepDResult}
                    staleSteps={staleSteps}
                    runningSteps={runningSteps}
                    onStepAComplete={(result) => setStepAResult(result)}
                    onStepBComplete={(result, frames) => { setStepBResult(result); setHomographyFrameIndices(frames) }}
                    onStepCComplete={(result) => setStepCResult(result)}
                    onStepDComplete={(result, positions, start, end, fps) => {
                      setStepDResult(result)
                      setPlayerPositions(positions)
                      setProcessedStartFrame(start)
                      setProcessedEndFrame(end)
                      setProcessedFps(fps)
                      const firstFrame = positions.length > 0 ? Math.min(...positions.map(p => p.frame_idx)) : start
                      setCurrentFrame(firstFrame)
                    }}
                    onStepsMarkedStale={markStale}
                    onStepsClearedStale={clearStale}
                    onRunningStepChange={(step, action) => {
                      setRunningSteps(prev => {
                        const next = new Set(prev)
                        if (action === 'add') next.add(step)
                        else next.delete(step)
                        return next
                      })
                    }}
                    onError={setError}
                    onStatusChange={setStatus}
                    logApiCall={logApiCall}
                  />
                </div>
              </>
            )}

            {(status || error) && (
              <div className={`status ${error ? 'error' : 'success'}`}>{error || status}</div>
            )}

            {/* Step 5: Results */}
            {playerPositions.length > 0 && videoMetadata && videoFile && (
              <div className="step-card">
                <span className="step-card__badge">5</span>
                <span className="step-card__title">Results</span>
                <ResultsViewer
                  videoMetadata={videoMetadata}
                  videoFile={videoFile}
                  playerPositions={playerPositions}
                  currentFrame={currentFrame}
                  onFrameChange={setCurrentFrame}
                  processedStartFrame={processedStartFrame}
                  processedEndFrame={processedEndFrame}
                  homographyFrameIndices={homographyFrameIndices}
                  processedFps={processedFps}
                />
              </div>
            )}

          </div>
          <div className="activity-sidebar">
            <DebugLog
              entries={debugLogEntries}
              onClear={() => { debugLog.current = []; setDebugLogEntries([]) }}
              videoMetadata={videoMetadata}
              stepAResult={stepAResult}
              stepBResult={stepBResult}
              stepCResult={stepCResult}
              stepDResult={stepDResult}
            />
          </div>
        </div>
      </div>
    </>
  )
}
