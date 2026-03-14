import type { VideoMetadata, PlayerPosition, AnchorFrameAnnotation } from '../types'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

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

export async function trackVideo(videoId: string): Promise<{ frames_processed: number; tracks: number }> {
  const res = await fetch(`${API_URL}/videos/${videoId}/track`, { method: 'POST' })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Tracking failed')
  }
  return res.json()
}

export async function getDetections(videoId: string): Promise<any[]> {
  const res = await fetch(`${API_URL}/videos/${videoId}/detections`)
  if (!res.ok) return []
  return res.json()
}

export async function computeHomographiesV2(
  videoId: string,
  annotations: AnchorFrameAnnotation[],
): Promise<{ frames: number[]; info: Record<string, any> }> {
  const res = await fetch(`${API_URL}/videos/${videoId}/homographies/v2`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(annotations),
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Homography computation failed')
  }
  const data = await res.json()
  return { frames: data.frames || [], info: data.info || {} }
}

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

export async function getPlayerPositions(videoId: string): Promise<PlayerPosition[]> {
  const res = await fetch(`${API_URL}/videos/${videoId}/players`)
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Failed to fetch player positions')
  }
  return res.json()
}

export { API_URL }
