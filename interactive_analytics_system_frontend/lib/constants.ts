export const PITCH_CANVAS_W = 850
export const PITCH_CANVAS_H = 1400
export const DISPLAY_SCALE = 0.4
export const PITCH_DISPLAY_WIDTH = Math.round(PITCH_CANVAS_W * DISPLAY_SCALE)   // 340
export const PITCH_DISPLAY_HEIGHT = Math.round(PITCH_CANVAS_H * DISPLAY_SCALE)  // 560
export const GAA_PITCH_WIDTH = 85.0
export const GAA_PITCH_LENGTH = 140.0

export const AVAILABLE_LINES: Record<string, { label: string; y_meters: number }> = {
  '13m_top': { label: '13m Line (Top)', y_meters: 13.0 },
  '20m_top': { label: '20m Line (Top)', y_meters: 20.0 },
  '45m_top': { label: '45m Line (Top)', y_meters: 45.0 },
  '65m_top': { label: '65m Line (Top)', y_meters: 65.0 },
  'halfway': { label: 'Halfway Line', y_meters: 70.0 },
  '65m_bottom': { label: '65m Line (Bottom)', y_meters: 75.0 },
  '45m_bottom': { label: '45m Line (Bottom)', y_meters: 95.0 },
  '20m_bottom': { label: '20m Line (Bottom)', y_meters: 120.0 },
  '13m_bottom': { label: '13m Line (Bottom)', y_meters: 127.0 },
}

export const GAA_PITCH_VERTICES: Record<string, [number, number]> = {
  // Corners
  "corner_tl": [0.0, 0.0],
  "corner_tr": [85.0, 0.0],
  "corner_bl": [0.0, 140.0],
  "corner_br": [85.0, 140.0],

  // Goal posts
  "top_goal_lp": [39.25, 0.0],
  "top_goal_rp": [45.75, 0.0],
  "bottom_goal_lp": [39.25, 140.0],
  "bottom_goal_rp": [45.75, 140.0],

  // Goalie box
  "left_box_bottom": [35.5, 135.5],
  "left_box_top": [35.5, 4.5],
  "right_box_bottom": [49.5, 135.5],
  "right_box_top": [49.5, 4.5],

  // 13m box
  "left_13m_box_bottom": [33.0, 127.0],
  "left_13m_box_top": [33.0, 13.0],
  "right_13m_box_bottom": [52.0, 127.0],
  "right_13m_box_top": [52.0, 13.0],
  "left_endline_13m_box_bottom": [33.0, 140.0],
  "left_endline_13m_box_top": [33.0, 0.0],
  "right_endline_13m_box_bottom": [52.0, 140.0],
  "right_endline_13m_box_top": [52.0, 0.0],

  // Small arc
  "left_small_arc_bottom": [29.5, 120.0],
  "left_small_arc_top": [29.5, 20.0],
  "right_small_arc_bottom": [55.5, 120.0],
  "right_small_arc_top": [55.5, 20.0],
  "small_arc_top_top": [42.5, 33.0],
  "small_arc_top_bottom": [42.5, 107.0],

  // 13m line
  "left_13m_line_bottom": [0.0, 127.0],
  "left_13m_line_top": [0.0, 13.0],
  "right_13m_line_bottom": [85.0, 127.0],
  "right_13m_line_top": [85.0, 13.0],

  // 20m line
  "left_20m_line_bottom": [0.0, 120.0],
  "left_20m_line_top": [0.0, 20.0],
  "right_20m_line_bottom": [85.0, 120.0],
  "right_20m_line_top": [85.0, 20.0],

  // 45m line
  "left_45m_line_bottom": [0.0, 95.0],
  "left_45m_line_top": [0.0, 45.0],
  "right_45m_line_bottom": [85.0, 95.0],
  "right_45m_line_top": [85.0, 45.0],

  // 65m line
  "left_65m_line_bottom": [0.0, 75.0],
  "left_65m_line_top": [0.0, 65.0],
  "right_65m_line_bottom": [85.0, 75.0],
  "right_65m_line_top": [85.0, 65.0],

  // Center line (halfway)
  "center_left": [0.0, 70.0],
  "center_right": [85.0, 70.0],
}

export const PITCH_LINE_SEGMENTS: Array<{ name: string; x1: number; y1: number; x2: number; y2: number }> = [
  { name: 'left_sideline', x1: 0, y1: 0, x2: 0, y2: 140 },
  { name: 'right_sideline', x1: 85, y1: 0, x2: 85, y2: 140 },
  { name: 'top_endline', x1: 0, y1: 0, x2: 85, y2: 0 },
  { name: 'bottom_endline', x1: 0, y1: 140, x2: 85, y2: 140 },
  { name: '13m_top', x1: 0, y1: 13, x2: 85, y2: 13 },
  { name: '13m_bottom', x1: 0, y1: 127, x2: 85, y2: 127 },
  { name: '20m_top', x1: 0, y1: 20, x2: 85, y2: 20 },
  { name: '20m_bottom', x1: 0, y1: 120, x2: 85, y2: 120 },
  { name: '45m_top', x1: 0, y1: 45, x2: 85, y2: 45 },
  { name: '45m_bottom', x1: 0, y1: 95, x2: 85, y2: 95 },
  { name: '65m_top', x1: 0, y1: 65, x2: 85, y2: 65 },
  { name: '65m_bottom', x1: 0, y1: 75, x2: 85, y2: 75 },
  { name: 'halfway', x1: 0, y1: 70, x2: 85, y2: 70 },
  { name: 'left_13m_box_top_v', x1: 33, y1: 0, x2: 33, y2: 13 },
  { name: 'left_13m_box_bottom_v', x1: 33, y1: 127, x2: 33, y2: 140 },
  { name: 'right_13m_box_top_v', x1: 52, y1: 0, x2: 52, y2: 13 },
  { name: 'right_13m_box_bottom_v', x1: 52, y1: 127, x2: 52, y2: 140 },
]
