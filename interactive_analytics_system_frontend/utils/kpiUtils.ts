import type { KpiSummary } from '../types'

export const ZONE_RANGES = {
  defensive: '0–47m',
  middle:    '47–93m',
  attacking: '93–140m',
} as const

export function teamColor(team: string): string {
  if (team === 'ellistown') return '#FFD700'
  if (team === 'opposition') return '#4488FF'
  return '#888'
}

export interface ZoneAnalysis {
  /** Cumulative per-team zone player counts across all frames. */
  totals: Record<string, Record<string, number>>
  frameCount: number
  /** Sum across all teams — used to detect which zone the play was in. */
  zoneActivity: { defensive: number; middle: number; attacking: number }
  /** The zone with the most combined player presence. */
  detectedZone: 'defensive' | 'middle' | 'attacking'
}

export function computeZoneAnalysis(kpiSummary: KpiSummary): ZoneAnalysis {
  const totals: Record<string, Record<string, number>> = {}
  let frameCount = 0
  for (const frame of Object.values(kpiSummary.zone_balance_timeseries)) {
    frameCount++
    for (const [team, zones] of Object.entries(frame)) {
      if (!totals[team]) totals[team] = { defensive: 0, middle: 0, attacking: 0 }
      totals[team].defensive += zones.defensive
      totals[team].middle    += zones.middle
      totals[team].attacking += zones.attacking
    }
  }

  const zoneActivity = { defensive: 0, middle: 0, attacking: 0 }
  for (const zones of Object.values(totals)) {
    zoneActivity.defensive += zones.defensive
    zoneActivity.middle    += zones.middle
    zoneActivity.attacking += zones.attacking
  }
  const detectedZone = (Object.entries(zoneActivity) as [string, number][])
    .sort(([, a], [, b]) => b - a)[0][0] as 'defensive' | 'middle' | 'attacking'

  return { totals, frameCount, zoneActivity, detectedZone }
}

/**
 * Compute a sentence describing how the opposition centroid moved relative
 * to the Ellistown centroid from clip start to clip end.
 */
export function computeDepthSentence(
  spatialTimeseries: KpiSummary['spatial_timeseries'],
  detectedZone: 'defensive' | 'middle' | 'attacking',
): string | null {
  const tsKeys = Object.keys(spatialTimeseries).map(Number).sort((a, b) => a - b)
  const bothPresent = tsKeys.filter(k => {
    const f = spatialTimeseries[k.toString()]
    return f?.teams?.['ellistown']?.centroid_y_m != null
      && f?.teams?.['opposition']?.centroid_y_m != null
  })
  if (bothPresent.length < 2) return null

  const f0 = spatialTimeseries[bothPresent[0].toString()]
  const f1 = spatialTimeseries[bothPresent[bothPresent.length - 1].toString()]
  const eY0 = f0.teams['ellistown'].centroid_y_m
  const oY0 = f0.teams['opposition'].centroid_y_m
  const eY1 = f1.teams['ellistown'].centroid_y_m
  const oY1 = f1.teams['opposition'].centroid_y_m

  const oppGoalSide = (eY: number, oY: number) =>
    detectedZone === 'attacking' ? oY > eY : oY < eY
  const desc = (eY: number, oY: number): string => {
    const gs = oppGoalSide(eY, oY)
    const gap = Math.abs(eY - oY).toFixed(1)
    return gs ? `Opposition ${gap}m goal-side` : `Ellistown ${gap}m goal-side`
  }
  return `Clip start: ${desc(eY0, oY0)} · Clip end: ${desc(eY1, oY1)}`
}