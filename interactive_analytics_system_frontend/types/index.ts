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

export interface PlayerDistanceMetrics {
  team: string
  total_distance_m: number
}

export interface TeamSpatialMetrics {
  centroid_x_m: number
  centroid_y_m: number
  spread_m2: number
  num_players_visible: number
}

export interface SpatialFrame {
  teams: Record<string, TeamSpatialMetrics>
  centroid_separation_m: number | null
}

export interface ZoneCounts {
  defensive: number
  middle: number
  attacking: number
}

export interface SpatialSummary {
  centroid_separation_m: { mean: number; min: number; max: number }
  per_team: Record<string, { mean_spread_m2: number; mean_centroid_x_m: number; mean_centroid_y_m: number }>
}

export interface ClipMeta {
  fps: number
  duration_s: number
  total_frames: number
}

export interface KpiSummary {
  per_player: Record<string, PlayerDistanceMetrics>
  spatial_timeseries: Record<string, SpatialFrame>
  zone_balance_timeseries: Record<string, Record<string, ZoneCounts>>
  spatial_summary: SpatialSummary
  clip_meta: ClipMeta
}

export interface AnchorQualityPoint {
  pitch_id: string
  x_img: number
  y_img: number
  error_px: number
  verdict: 'good' | 'high' | 'outlier'
  impact: 'helpful' | 'marginal' | 'harmful'
}

export interface AnchorQualityFrame {
  frame_idx: number
  n_keypoints: number
  n_lines: number
  mean_error_px: number
  max_error_px: number
  n_outlier_points: number
  n_helpful_points: number
  overall_quality: 'good' | 'warning' | 'bad'
  keypoints: AnchorQualityPoint[]
  recommendation: string
}

export interface AnchorQualityData {
  anchors: AnchorQualityFrame[]
}

export interface HomographyComputeResult {
  frames: number[]
  per_frame_count: number
  info: Record<string, any>
}
