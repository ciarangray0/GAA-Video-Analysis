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

  const [trimStartSeconds, setTrimStartSeconds] = useState(0)
  const [trimEndSeconds, setTrimEndSeconds] = useState<number | null>(null)
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

  // Keep stepDoneRef in sync so annotation changes can check which steps are done
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

  // Mark downstream pipeline steps stale when annotations change
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
    const startFrame = Math.floor(trimStartSeconds * fps)
    const endFrame = trimEndSeconds !== null
      ? Math.floor(trimEndSeconds * fps)
      : videoMetadata.num_frames - 1
    const frames: AnchorFrame[] = []
    for (let seconds = trimStartSeconds; seconds <= (trimEndSeconds || videoMetadata.duration_seconds); seconds += anchorInterval) {
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
        <h1>GAA Video Analysis System</h1>

        <VideoUploader
          onUploadSuccess={(metadata, file) => {
            setVideoMetadata(metadata)
            setVideoFile(file)
            setTrimEndSeconds(metadata.duration_seconds)
            setAnchorFrames([])
            setError('')
            setStatus('Video uploaded successfully!')
          }}
        />

        {/* Step 2: Configure Anchor Frames */}
        {videoMetadata && anchorFrames.length === 0 && (
          <div className="config-section">
            <h2>2. Configure Anchor Frames</h2>
            <p>Set up which frames to use for pitch annotations.</p>
            <div className="config-form">
              <div className="config-row">
                <label>
                  Trim Start (seconds):
                  <input type="number" min={0} max={videoMetadata.duration_seconds} step={0.1} value={trimStartSeconds}
                    onChange={(e) => setTrimStartSeconds(parseFloat(e.target.value) || 0)} />
                </label>
                <label>
                  Trim End (seconds):
                  <input type="number" min={trimStartSeconds} max={videoMetadata.duration_seconds} step={0.1}
                    value={trimEndSeconds ?? videoMetadata.duration_seconds}
                    onChange={(e) => setTrimEndSeconds(parseFloat(e.target.value) || null)} />
                </label>
              </div>
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
                  <strong>{Math.ceil(((trimEndSeconds ?? videoMetadata.duration_seconds) - trimStartSeconds) / anchorInterval)}</strong>{' '}
                  anchor frames to annotate.
                </p>
              </div>
              <button onClick={generateAnchorFrames}>Generate Anchor Frames</button>
            </div>
          </div>
        )}

        {/* Step 3: Annotate + Pipeline */}
        {anchorFrames.length > 0 && videoMetadata && (
          <>
            <AnchorFrameAnnotator
              videoMetadata={videoMetadata}
              videoFilename={videoFile?.name}
              anchorFrames={anchorFrames}
              currentAnchorIdx={currentAnchorIdx}
              onAnchorFramesChange={handleAnchorFramesChange}
              onCurrentIdxChange={setCurrentAnchorIdx}
            />
            <PipelineSteps
              videoMetadata={videoMetadata}
              anchorFrames={anchorFrames}
              trimStartSeconds={trimStartSeconds}
              trimEndSeconds={trimEndSeconds}
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
          </>
        )}

        {(status || error) && (
          <div className={`status ${error ? 'error' : 'success'}`}>{error || status}</div>
        )}

        {/* Step 4: Results */}
        {playerPositions.length > 0 && videoMetadata && videoFile && (
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
            anchorFrames={anchorFrames}
          />
        )}

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
    </>
  )
}
