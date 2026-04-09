import type {
  VideoMetadata,
  PlayerPosition,
  AnchorFrameAnnotation,
  ClassifyTeamsResponse,
  TeamClassifications,
  KpiSummary,
  AnchorQualityData,
  HomographyComputeResult,
} from '../types'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

/**
 * Upload an MP4 video file and return its metadata (id, fps, dimensions).
 * Throws on non-2xx responses.
 */
export async function uploadVideo(file: File): Promise<VideoMetadata> {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${API_URL}/videos`, { method: 'POST', body: formData })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Upload failed')
  }
  return res.json()
}

/** Run YOLO + BotSort tracking on a video. Returns frame/track counts. */
export async function trackVideo(videoId: string): Promise<{ frames_processed: number; tracks: number }> {
  const res = await fetch(`${API_URL}/videos/${videoId}/track`, { method: 'POST' })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Tracking failed')
  }
  return res.json()
}

/** Fetch raw YOLO+BotSort detections for a video. */
export async function getDetections(videoId: string): Promise<any[]> {
  const res = await fetch(`${API_URL}/videos/${videoId}/detections`)
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Failed to fetch detections')
  }
  return res.json()
}

/** Map player detections to pitch canvas coordinates via per-frame homography. */
export async function mapPlayers(videoId: string): Promise<PlayerPosition[]> {
  const res = await fetch(`${API_URL}/videos/${videoId}/map_players`, { method: 'POST' })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Player mapping failed')
  }
  return res.json()
}

export interface InterpolationParams {
  sgLongWindow?: number   // SG window for tracks >20 frames (default 15)
  sgMidWindow?: number    // SG window for tracks 10-20 frames (default 11)
  maxVelPx?: number       // Max px/frame displacement (default 4, 0 = off)
}

/**
 * Interpolate + smooth player trajectories between anchor frames.
 * Applies Savitzky-Golay smoothing and optional max-velocity clamping.
 */
export async function interpolateTrajectories(
  videoId: string,
  startFrame: number,
  endFrame: number,
  params: InterpolationParams = {},
): Promise<{ frames_generated: number; method: string }> {
  const qs = new URLSearchParams({
    start_frame: String(startFrame),
    end_frame: String(endFrame),
    ...(params.sgLongWindow !== undefined && { sg_long_window: String(params.sgLongWindow) }),
    ...(params.sgMidWindow  !== undefined && { sg_mid_window:  String(params.sgMidWindow) }),
    ...(params.maxVelPx     !== undefined && { max_vel_px:     String(params.maxVelPx) }),
  })
  const res = await fetch(`${API_URL}/videos/${videoId}/interpolate?${qs}`, { method: 'POST' })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Interpolation failed')
  }
  return res.json()
}

/** Fetch all player positions (sparse homography + interpolated) for a video. */
export async function getPlayerPositions(videoId: string): Promise<PlayerPosition[]> {
  const res = await fetch(`${API_URL}/videos/${videoId}/players`)
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Failed to fetch player positions')
  }
  return res.json()
}

/** Run jersey-colour team classification. Returns per-track assignments and summary stats. */
export async function classifyTeams(videoId: string): Promise<ClassifyTeamsResponse> {
  const res = await fetch(`${API_URL}/videos/${videoId}/classify-teams`, { method: 'POST' })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Team classification failed')
  }
  return res.json()
}

/** Fetch stored team classifications. Returns empty object when none saved yet. */
export async function getTeamClassifications(videoId: string): Promise<TeamClassifications> {
  const res = await fetch(`${API_URL}/videos/${videoId}/classify-teams`)
  if (!res.ok) return {}
  const data = await res.json()
  return data.classifications ?? {}
}

/** Override a single track's team assignment. Returns the updated full classification map. */
export async function overrideTeamClassification(
  videoId: string,
  trackId: number,
  team: string,
): Promise<TeamClassifications> {
  const res = await fetch(`${API_URL}/videos/${videoId}/classify-teams`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ track_id: trackId, team }),
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Override failed')
  }
  const data = await res.json()
  return data.classifications ?? {}
}

/**
 * Compute spatial KPIs for a clip (distances, team spread, centroid separation, zone balance).
 * @param endFrame - Optional last frame index; trims the clip before computing.
 */
export async function computeKpis(videoId: string, endFrame?: number): Promise<KpiSummary> {
  const qs = endFrame !== undefined ? `?end_frame=${endFrame}` : ''
  const res = await fetch(`${API_URL}/videos/${videoId}/compute-kpis${qs}`, { method: 'POST' })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'KPI computation failed')
  }
  return res.json()
}

/**
 * Compute anchor homographies (v3 DLT + line constraints) then propagate per-frame
 * via Lucas-Kanade optical flow. Returns anchor frame list and per-frame count.
 */
export async function computeHomographies(
  videoId: string,
  annotations: AnchorFrameAnnotation[],
): Promise<HomographyComputeResult> {
  const res = await fetch(`${API_URL}/videos/${videoId}/homographies/v3`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(annotations),
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Homography computation failed')
  }
  const data = await res.json()
  return {
    frames: data.frames || [],
    per_frame_count: data.per_frame_count ?? (data.frames || []).length,
    info: data.info || {},
  }
}

/** Fetch per-keypoint reprojection errors for each anchor frame. */
export async function getAnchorQuality(videoId: string): Promise<AnchorQualityData> {
  const res = await fetch(`${API_URL}/videos/${videoId}/homographies/anchor-quality`)
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Anchor quality fetch failed')
  }
  return res.json()
}

export { API_URL }
