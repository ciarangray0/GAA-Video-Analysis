import type { KpiSummary } from '../types'
import { computeZoneAnalysis, computeDepthSentence, ZONE_RANGES } from '../utils/kpiUtils'

interface ClipSummaryCardProps {
  kpiSummary: KpiSummary | null
  clipMode: 'score' | 'defense'
}

export default function ClipSummaryCard({ kpiSummary, clipMode }: ClipSummaryCardProps) {
  const cardStyle: React.CSSProperties = {
    background: '#1a2a1a',
    border: '1px solid #3a5a3a',
    borderRadius: 8,
    padding: '14px 16px',
    minWidth: 200,
    maxWidth: 260,
    fontSize: 13,
    lineHeight: 1.6,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  }

  if (!kpiSummary) {
    return (
      <div style={{ ...cardStyle, color: '#666' }}>
        <div style={{ fontSize: 11, color: '#555', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
          Clip summary
        </div>
        Press <strong style={{ color: '#888' }}>Compute KPIs</strong> to see zone balance, team spread, and centroid separation for this clip.
      </div>
    )
  }

  const { totals, frameCount, zoneActivity, detectedZone } = computeZoneAnalysis(kpiSummary)
  const { spatial_summary } = kpiSummary
  const clipZoneLabel = clipMode === 'score' ? 'attacking' : 'defensive'
  const yRangeLabel = ZONE_RANGES[detectedZone]

  // Zone balance sentence
  let zoneSentence = ''
  if (frameCount > 0) {
    const ellNf = (totals['ellistown']?.[detectedZone] ?? 0) / frameCount
    const oppNf = (totals['opposition']?.[detectedZone] ?? 0) / frameCount
    const prefix = `In the ${clipZoneLabel} third (${yRangeLabel})`

    if (ellNf === 0 && oppNf === 0) {
      const totalN = (zoneActivity[detectedZone] / frameCount).toFixed(1)
      zoneSentence = `${prefix}: ${totalN} players avg (classify teams for breakdown)`
    } else {
      const ellN = ellNf.toFixed(1)
      const oppN = oppNf.toFixed(1)
      if (clipMode === 'score') {
        if (oppNf > ellNf)
          zoneSentence = `${prefix}: Opposition defended with a ${oppN}v${ellN} advantage`
        else if (ellNf > oppNf)
          zoneSentence = `${prefix}: Ellistown had a ${ellN}v${oppN} numerical overload`
        else
          zoneSentence = `${prefix}: Teams level at ${ellN}v${oppN}`
      } else {
        if (oppNf > ellNf)
          zoneSentence = `${prefix}: Opposition had a ${oppN}v${ellN} numerical advantage`
        else if (ellNf > oppNf)
          zoneSentence = `${prefix}: Ellistown defended with a ${ellN}v${oppN} advantage`
        else
          zoneSentence = `${prefix}: Teams level at ${ellN}v${oppN}`
      }
    }
  }

  // Spread sentence
  const ellSpread = spatial_summary.per_team['ellistown']?.mean_spread_m2
  const oppSpread = spatial_summary.per_team['opposition']?.mean_spread_m2
  const spreadSentence = (ellSpread != null && oppSpread != null)
    ? `Opposition spread: ${oppSpread} m²  /  Ellistown spread: ${ellSpread} m²`
    : null

  const depthSentence = computeDepthSentence(kpiSummary.spatial_timeseries, detectedZone)

  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 11, color: '#aaa', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        Clip summary · {clipMode === 'score' ? 'Ellistown score' : 'Ellistown defense'}
      </div>

      {zoneSentence && <div style={{ color: '#e0e0e0' }}>{zoneSentence}</div>}
      {spreadSentence && <div style={{ color: '#e0e0e0' }}>{spreadSentence}</div>}
      {depthSentence && <div style={{ color: '#e0e0e0' }}>{depthSentence}</div>}

      {!zoneSentence && !spreadSentence && !depthSentence && (
        <div style={{ color: '#888' }}>Run Compute KPIs to see summary</div>
      )}
    </div>
  )
}