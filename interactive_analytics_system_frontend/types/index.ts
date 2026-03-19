export interface PitchPoint {
  pitch_id: string
  x_img: number
  y_img: number
}

export interface LineAnnotation {
  line_id: string
  u1: number
  v1: number
  u2: number
  v2: number
}

export interface AnchorFrameAnnotation {
  frame_idx: number
  points: PitchPoint[]
  lines: LineAnnotation[]
}

export interface PlayerPosition {
  frame_idx: number
  track_id: number
  x_pitch: number
  y_pitch: number
  source: string
}

export interface VideoMetadata {
  video_id: string
  fps: number
  num_frames: number
  width: number
  height: number
  duration_seconds: number
}

export interface AnchorFrame {
  frame_idx: number
  isSkipped: boolean
  points: PitchPoint[]
  lines: LineAnnotation[]
}

export type TeamName = 'ellistown' | 'opposition' | 'referee' | 'ignore'

export interface TrackClassification {
  team: TeamName
  confidence: number
  mean_hsv: [number, number, number]
}

export type TeamClassifications = Record<string, TrackClassification>

export interface ClassifyTeamsSummary {
  num_ellistown: number
  num_opposition: number
  num_referee: number
  mean_confidence: number
  low_confidence_tracks: number[]
  hsv_cluster_separation: number | null
}

export interface ClassifyTeamsResponse {
  classifications: TeamClassifications
  summary: ClassifyTeamsSummary
}
