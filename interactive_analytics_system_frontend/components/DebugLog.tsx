import type { VideoMetadata, PlayerPosition } from '../types'

interface DebugLogProps {
  entries: string[]
  onClear: () => void
  videoMetadata: VideoMetadata | null
  stepAResult: { frames_processed: number; tracks: number; num_detections: number } | null
  stepBResult: { frames: number[]; per_frame_count: number; info: Record<string, any> } | null
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
  return (
    <div className="debug-log-panel">
      <div className="debug-log-header">
        <h3>Pipeline Activity</h3>
        <button className="secondary-btn" onClick={onClear}>Clear</button>
      </div>
      <div className="activity-stats">
        <div className="activity-stat">
          <span className="activity-stat__value">{stepAResult ? stepAResult.num_detections : '—'}</span>
          <span className="activity-stat__label">Detections</span>
        </div>
        <div className="activity-stat">
          <span className="activity-stat__value">{stepBResult ? stepBResult.frames.length : '—'}</span>
          <span className="activity-stat__label">Homographies</span>
        </div>
        <div className="activity-stat">
          <span className="activity-stat__value">{stepCResult ? stepCResult.total : '—'}</span>
          <span className="activity-stat__label">Positions</span>
        </div>
        <div className="activity-stat">
          <span className="activity-stat__value">{stepDResult ? stepDResult.frames_generated : '—'}</span>
          <span className="activity-stat__label">Frames</span>
        </div>
      </div>
      <div className="debug-log-content">
        <div className="debug-log-entries">
          {entries.length === 0
            ? <span className="debug-empty">No API calls logged yet.</span>
            : entries.map((entry, i) => (
              <div
                key={i}
                className={`debug-log-entry ${entry.startsWith('←') ? 'response' : entry.startsWith('✗') ? 'error-entry' : 'request'}`}
              >
                {entry}
              </div>
            ))
          }
        </div>
      </div>
    </div>
  )
}
