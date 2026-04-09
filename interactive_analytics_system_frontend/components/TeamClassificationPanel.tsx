import type { TeamClassifications, ClassifyTeamsSummary, TeamName } from '../types'
import { hsvToCss } from '../lib/pitch'

interface TeamClassificationPanelProps {
  teamClassifications: TeamClassifications
  classifySummary: ClassifyTeamsSummary | null
  classifyError: string | null
  onOverrideTeam: (trackId: number, team: string) => void
}

export default function TeamClassificationPanel({
  teamClassifications,
  classifySummary,
  classifyError,
  onOverrideTeam,
}: TeamClassificationPanelProps) {
  return (
    <details className="debug-panel" open>
      <summary>🎽 Team Classifications</summary>
      <div className="debug-panel-body">
        {classifyError && <p style={{ color: '#f88' }}>{classifyError}</p>}

        {classifySummary && (
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 12, fontSize: 13 }}>
            <span>Ellistown: <strong>{classifySummary.num_ellistown}</strong></span>
            <span>Opposition: <strong>{classifySummary.num_opposition}</strong></span>
            <span>Avg confidence: <strong>{(classifySummary.mean_confidence * 100).toFixed(0)}%</strong></span>
            {classifySummary.hsv_cluster_separation !== null && (
              <span>HSV separation: <strong>{classifySummary.hsv_cluster_separation}</strong></span>
            )}
            {classifySummary.low_confidence_tracks.length > 0 && (
              <span style={{ color: '#f88' }}>
                Low-confidence tracks: {classifySummary.low_confidence_tracks.join(', ')}
              </span>
            )}
          </div>
        )}

        {(['ellistown', 'opposition', 'referee', 'ignore'] as TeamName[]).map(team => {
          const trackIds = Object.entries(teamClassifications)
            .filter(([, v]) => v.team === team)
            .map(([k]) => parseInt(k))
            .sort((a, b) => a - b)
          if (trackIds.length === 0) return null
          return (
            <div key={team} style={{ marginBottom: 12 }}>
              <strong style={{ textTransform: 'capitalize' }}>{team}</strong>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 4 }}>
                {trackIds.map(tid => {
                  const cls = teamClassifications[tid.toString()]
                  const [h, s, v] = cls.mean_hsv
                  const swatchColor = hsvToCss(h, s, v)
                  return (
                    <div key={tid} style={{
                      display: 'flex', alignItems: 'center', gap: 6,
                      background: '#1a1a2e', borderRadius: 6, padding: '4px 8px', fontSize: 12,
                    }}>
                      <div style={{
                        width: 14, height: 14, borderRadius: '50%',
                        background: swatchColor, border: '1px solid #555', flexShrink: 0,
                      }} title={`HSV: ${h.toFixed(0)}, ${s.toFixed(0)}, ${v.toFixed(0)}`} />
                      <span>#{tid}</span>
                      <div style={{
                        width: 40, height: 4, background: '#333', borderRadius: 2, overflow: 'hidden',
                      }}>
                        <div style={{
                          width: `${cls.confidence * 100}%`, height: '100%',
                          background: cls.confidence >= 0.6 ? '#4caf50' : '#f88',
                        }} />
                      </div>
                      <select
                        value={cls.team}
                        onChange={(e) => onOverrideTeam(tid, e.target.value)}
                        style={{ fontSize: 11, background: '#222', color: '#fff', border: '1px solid #555', borderRadius: 3 }}
                      >
                        <option value="ellistown">Ellistown</option>
                        <option value="opposition">Opposition</option>
                        <option value="referee">Referee</option>
                        <option value="ignore">Ignore</option>
                      </select>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>
    </details>
  )
}