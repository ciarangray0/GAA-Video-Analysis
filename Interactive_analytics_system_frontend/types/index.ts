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

export interface PitchAnnotation {
  frame_idx: number
  points: PitchPoint[]
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

export interface ProcessResponse {
  video_id: string
  status: string
  player_positions?: PlayerPosition[]
  homography_frames?: number[]
  start_frame?: number
  end_frame?: number
  fps?: number
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
