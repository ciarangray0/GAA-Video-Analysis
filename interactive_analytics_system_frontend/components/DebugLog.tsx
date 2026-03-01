import { useState } from 'react'
import type { VideoMetadata, PlayerPosition } from '../types'

interface DebugLogProps {
  entries: string[]
  onClear: () => void
  videoMetadata: VideoMetadata | null
  stepAResult: { frames_processed: number; tracks: number; num_detections: number } | null
  stepBResult: { frames: number[]; info: Record<string, any> } | null
  stepCResult: { positions: PlayerPosition[]; total: number } | null
  stepDResult: { frames_generated: number; method: string } | null
}

export default function DebugLog({
  entries,
  onClear,
  videoMetadata,
  stepAResult,
  stepBResult,
  stepCResult,
  stepDResult,
}: DebugLogProps) {
  const [visible, setVisible] = useState(false)

  return (
    <div className="debug-log-panel">
      <div className="debug-log-header" onClick={() => setVisible(v => !v)}>
        <h3>🐛 Debug Log {entries.length > 0 && `(${entries.length} entries)`}</h3>
        <div className="debug-log-controls">
          <span className="debug-toggle-hint">{visible ? '▲ Collapse' : '▼ Expand'}</span>
          <button
            className="secondary-btn"
            onClick={(e) => { e.stopPropagation(); onClear() }}
          >
            Clear
          </button>
        </div>
      </div>
      {visible && (
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
            {entries.length === 0
              ? <span className="debug-empty">No API calls logged yet.</span>
              : entries.map((entry, i) => (
                <div key={i} className={`debug-log-entry ${entry.startsWith('←') ? 'response' : entry.startsWith('✗') ? 'error-entry' : 'request'}`}>
                  {entry}
                </div>
              ))
            }
          </div>
        </div>
      )}
    </div>
  )
}
