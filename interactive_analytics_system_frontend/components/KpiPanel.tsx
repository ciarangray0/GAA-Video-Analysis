import type { KpiSummary } from '../types'
import { computeZoneAnalysis, computeDepthSentence, teamColor, ZONE_RANGES } from '../utils/kpiUtils'

interface KpiPanelProps {
  kpiSummary: KpiSummary | null
  kpiError: string | null
  clipMode: 'score' | 'defense'
}

export default function KpiPanel({ kpiSummary, kpiError, clipMode }: KpiPanelProps) {
  return (
    <details className="debug-panel" open>
      <summary>📊 Spatial KPIs</summary>
      <div className="debug-panel-body">
        {kpiError && <p style={{ color: '#f88' }}>{kpiError}</p>}

        {kpiSummary && (() => {
          const { clip_meta: meta, per_player, spatial_summary, spatial_timeseries } = kpiSummary
          const { totals, frameCount, detectedZone } = computeZoneAnalysis(kpiSummary)
          const depthSentence = computeDepthSentence(spatial_timeseries, detectedZone)

          const clipZoneLabel = clipMode === 'score' ? 'attacking' : 'defensive'
          const colLabel = (zone: 'defensive' | 'middle' | 'attacking') => {
            if (zone === 'middle') return `midfield (${ZONE_RANGES.middle})`
            if (zone === detectedZone) return `${clipZoneLabel} third (${ZONE_RANGES[zone]})`
            const oppositeLabel = clipMode === 'score' ? 'defensive third' : 'attacking third'
            return `${oppositeLabel} (${ZONE_RANGES[zone]})`
          }

          const ellCentY = spatial_summary.per_team['ellistown']?.mean_centroid_y_m
          const oppCentY = spatial_summary.per_team['opposition']?.mean_centroid_y_m

          return (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {/* Clip meta */}
              <div style={{ fontSize: 12, color: '#aaa' }}>
                {meta.duration_s.toFixed(1)} s &nbsp;·&nbsp; {meta.total_frames} frames &nbsp;·&nbsp; {meta.fps} fps
              </div>

              {/* Centroid metrics */}
              {(spatial_summary.centroid_separation_m || (ellCentY != null && oppCentY != null)) && (
                <div>
                  <strong style={{ fontSize: 13 }}>Team centroids</strong>
                  <table className="debug-table" style={{ fontSize: 12, marginTop: 6 }}>
                    <thead>
                      <tr><th>Metric</th><th>Value</th><th>Note</th></tr>
                    </thead>
                    <tbody>
                      {spatial_summary.centroid_separation_m && <>
                        <tr>
                          <td>2D separation (mean)</td>
                          <td><strong>{spatial_summary.centroid_separation_m.mean} m</strong></td>
                          <td style={{ color: '#888' }}>overall compactness</td>
                        </tr>
                        <tr>
                          <td>2D separation (min / max)</td>
                          <td>{spatial_summary.centroid_separation_m.min} – {spatial_summary.centroid_separation_m.max} m</td>
                          <td style={{ color: '#888' }}>range across clip</td>
                        </tr>
                      </>}
                      {ellCentY != null && oppCentY != null && (() => {
                        const ellCentX = spatial_summary.per_team['ellistown']?.mean_centroid_x_m
                        const oppCentX = spatial_summary.per_team['opposition']?.mean_centroid_x_m
                        const yGap = Math.abs(ellCentY - oppCentY).toFixed(1)
                        const xGap = (ellCentX != null && oppCentX != null)
                          ? Math.abs(ellCentX - oppCentX).toFixed(1)
                          : null
                        const deeper = detectedZone === 'attacking'
                          ? (ellCentY > oppCentY ? 'Ellistown' : 'Opposition')
                          : detectedZone === 'defensive'
                          ? (ellCentY < oppCentY ? 'Ellistown' : 'Opposition')
                          : clipMode === 'score'
                          ? (ellCentY > oppCentY ? 'Ellistown' : 'Opposition')
                          : (ellCentY < oppCentY ? 'Ellistown' : 'Opposition')
                        return <>
                          <tr>
                            <td>Depth gap (y-axis)</td>
                            <td><strong>{yGap} m</strong></td>
                            <td style={{ color: '#888' }}>{deeper} closer to goal</td>
                          </tr>
                          {xGap != null && (
                            <tr>
                              <td>Lateral gap (x-axis)</td>
                              <td>{xGap} m</td>
                              <td style={{ color: '#888' }}>horizontal offset</td>
                            </tr>
                          )}
                          <tr>
                            <td>Ellistown centroid</td>
                            <td>({ellCentX?.toFixed(1)}, {ellCentY.toFixed(1)}) m</td>
                            <td style={{ color: '#888' }}>x, y</td>
                          </tr>
                          <tr>
                            <td>Opposition centroid</td>
                            <td>({oppCentX?.toFixed(1)}, {oppCentY.toFixed(1)}) m</td>
                            <td style={{ color: '#888' }}>x, y</td>
                          </tr>
                        </>
                      })()}
                    </tbody>
                  </table>
                  {depthSentence && (
                    <div style={{ marginTop: 8, fontSize: 12, color: '#ccc', lineHeight: 1.5 }}>
                      {depthSentence}
                    </div>
                  )}
                </div>
              )}

              {/* Team spread */}
              {Object.keys(spatial_summary.per_team).length > 0 && (
                <div>
                  <strong style={{ fontSize: 13 }}>Team spread (mean convex hull)</strong>
                  <div style={{ display: 'flex', gap: 24, marginTop: 6, flexWrap: 'wrap' }}>
                    {Object.entries(spatial_summary.per_team)
                      .filter(([t]) => t !== 'unclassified')
                      .map(([team, s]) => (
                        <div key={team}>
                          <span style={{ color: teamColor(team), fontWeight: 600, textTransform: 'capitalize', fontSize: 12 }}>
                            {team}
                          </span>
                          <div style={{ fontSize: 13 }}>{s.mean_spread_m2} m²</div>
                          <div style={{ fontSize: 11, color: '#888' }}>
                            centroid ({s.mean_centroid_x_m}, {s.mean_centroid_y_m}) m
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Zone balance */}
              {Object.keys(totals).length > 0 && (
                <div>
                  <strong style={{ fontSize: 13 }}>Zone balance (avg players per frame)</strong>
                  <table className="debug-table" style={{ fontSize: 12, marginTop: 6 }}>
                    <thead>
                      <tr>
                        <th>Team</th>
                        <th style={detectedZone === 'defensive' ? { color: '#FFD700' } : {}}>{colLabel('defensive')}</th>
                        <th style={detectedZone === 'middle'    ? { color: '#FFD700' } : {}}>{colLabel('middle')}</th>
                        <th style={detectedZone === 'attacking' ? { color: '#FFD700' } : {}}>{colLabel('attacking')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(totals)
                        .filter(([t]) => t !== 'unclassified')
                        .map(([team, zones]) => (
                          <tr key={team}>
                            <td style={{ color: teamColor(team), fontWeight: 600, textTransform: 'capitalize' }}>{team}</td>
                            <td style={detectedZone === 'defensive' ? { fontWeight: 600 } : { color: '#888' }}>{frameCount > 0 ? (zones.defensive / frameCount).toFixed(1) : 0}</td>
                            <td style={detectedZone === 'middle'    ? { fontWeight: 600 } : { color: '#888' }}>{frameCount > 0 ? (zones.middle    / frameCount).toFixed(1) : 0}</td>
                            <td style={detectedZone === 'attacking' ? { fontWeight: 600 } : { color: '#888' }}>{frameCount > 0 ? (zones.attacking / frameCount).toFixed(1) : 0}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Per-player distance */}
              <div>
                <strong style={{ fontSize: 13 }}>Distance covered</strong>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
                  {Object.entries(per_player)
                    .filter(([, m]) => m.team !== 'referee' && m.team !== 'ignore')
                    .sort(([, a], [, b]) => b.total_distance_m - a.total_distance_m)
                    .map(([tid, m]) => (
                      <span key={tid} style={{
                        background: '#1a1a2e', borderRadius: 6, padding: '3px 8px',
                        fontSize: 12, border: '1px solid #333',
                      }}>
                        <span style={{ color: teamColor(m.team) }}>#{tid}</span>
                        {' '}{m.total_distance_m} m
                      </span>
                    ))}
                </div>
              </div>
            </div>
          )
        })()}
      </div>
    </details>
  )
}