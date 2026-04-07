// Pitch physical dimensions — matches backend gaa_pitch_config.py
export const GAA_PITCH_WIDTH  = 85.0    // meters (x)
export const GAA_PITCH_LENGTH = 140.0   // meters (y)
export const PITCH_SCALE      = 10      // px / m

// Canvas dimensions (derived)
export const PITCH_CANVAS_W = GAA_PITCH_WIDTH  * PITCH_SCALE   // 850
export const PITCH_CANVAS_H = GAA_PITCH_LENGTH * PITCH_SCALE   // 1400

// Display scaling for the annotation / results canvas
export const DISPLAY_SCALE        = 0.4
export const PITCH_DISPLAY_WIDTH  = Math.round(PITCH_CANVAS_W * DISPLAY_SCALE)  // 340
export const PITCH_DISPLAY_HEIGHT = Math.round(PITCH_CANVAS_H * DISPLAY_SCALE)  // 560

// ---------------------------------------------------------------------------
// Vertices — matches GAA_PITCH_VERTICES in gaa_pitch_config.py
// [x_meters, y_meters]
// ---------------------------------------------------------------------------
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
  "left_box_bottom":  [35.5, 135.5],
  "left_box_top":     [35.5, 4.5],
  "right_box_bottom": [49.5, 135.5],
  "right_box_top":    [49.5, 4.5],

  // 13m box
  "left_13m_box_bottom":         [33.0, 127.0],
  "left_13m_box_top":            [33.0, 13.0],
  "right_13m_box_bottom":        [52.0, 127.0],
  "right_13m_box_top":           [52.0, 13.0],
  "left_endline_13m_box_bottom": [33.0, 140.0],
  "left_endline_13m_box_top":    [33.0, 0.0],
  "right_endline_13m_box_bottom":[52.0, 140.0],
  "right_endline_13m_box_top":   [52.0, 0.0],

  // Small arc
  "left_small_arc_bottom":  [29.5, 120.0],
  "left_small_arc_top":     [29.5, 20.0],
  "right_small_arc_bottom": [55.5, 120.0],
  "right_small_arc_top":    [55.5, 20.0],
  "small_arc_top_top":      [42.5, 33.0],
  "small_arc_top_bottom":   [42.5, 107.0],

  // 13m line
  "left_13m_line_bottom":  [0.0,  127.0],
  "left_13m_line_top":     [0.0,  13.0],
  "right_13m_line_bottom": [85.0, 127.0],
  "right_13m_line_top":    [85.0, 13.0],

  // 20m line
  "left_20m_line_bottom":  [0.0,  120.0],
  "left_20m_line_top":     [0.0,  20.0],
  "right_20m_line_bottom": [85.0, 120.0],
  "right_20m_line_top":    [85.0, 20.0],

  // 45m line
  "left_45m_line_bottom":  [0.0,  95.0],
  "left_45m_line_top":     [0.0,  45.0],
  "right_45m_line_bottom": [85.0, 95.0],
  "right_45m_line_top":    [85.0, 45.0],

  // 65m line
  "left_65m_line_bottom":  [0.0,  75.0],
  "left_65m_line_top":     [0.0,  65.0],
  "right_65m_line_bottom": [85.0, 75.0],
  "right_65m_line_top":    [85.0, 65.0],

  // Center line (halfway)
  "center_left":  [0.0,  70.0],
  "center_right": [85.0, 70.0],
}

// ---------------------------------------------------------------------------
// Horizontal lines — matches GAA_PITCH_LINES in gaa_pitch_config.py
// name → y_meters
// ---------------------------------------------------------------------------
export const GAA_PITCH_LINES: Record<string, number> = {
  'endline_top':             0.0,
  'small_rectangle_top':     4.5,   // goal-area box line
  '13m_top':                13.0,
  '20m_top':                20.0,
  '45m_top':                45.0,
  '65m_top':                65.0,
  'halfway':                70.0,
  '65m_bottom':             75.0,
  '45m_bottom':             95.0,
  '20m_bottom':            120.0,
  '13m_bottom':            127.0,
  'small_rectangle_bottom': 135.5,  // goal-area box line
  'endline_bottom':        140.0,
}

// ---------------------------------------------------------------------------
// Vertical lines — matches GAA_PITCH_SIDELINES in gaa_pitch_config.py
// name → x_meters
// ---------------------------------------------------------------------------
export const GAA_PITCH_SIDELINES: Record<string, number> = {
  'left_sideline':   0.0,
  'right_sideline':  85.0,
  '13m_box_left':    33.0,
  '13m_box_right':   52.0,
  'small_arc_left':  29.5,
  'small_arc_right': 55.5,
}

// ---------------------------------------------------------------------------
// Symmetric line pairs (top_y, bottom_y) used when drawing the pitch diagram.
// Derived from GAA_PITCH_LINES
// ---------------------------------------------------------------------------
export const PITCH_SYMMETRIC_LINE_PAIRS: ReadonlyArray<readonly [number, number]> = [
  [GAA_PITCH_LINES['13m_top'],  GAA_PITCH_LINES['13m_bottom']],
  [GAA_PITCH_LINES['20m_top'],  GAA_PITCH_LINES['20m_bottom']],
  [GAA_PITCH_LINES['45m_top'],  GAA_PITCH_LINES['45m_bottom']],
  [GAA_PITCH_LINES['65m_top'],  GAA_PITCH_LINES['65m_bottom']],
]

// ---------------------------------------------------------------------------
// AVAILABLE_LINES — line IDs that can be used as annotation constraints.
// Subset of GAA_PITCH_LINES + GAA_PITCH_SIDELINES with UI labels.
// ---------------------------------------------------------------------------
export const AVAILABLE_LINES: Record<string, {
  label: string
  y_meters?: number
  x_meters?: number
  orientation?: 'horizontal' | 'vertical'
}> = {
  'endline_top':     { label: 'End Line (Top)',        y_meters: GAA_PITCH_LINES['endline_top'] },
  '13m_top':         { label: '13m Line (Top)',         y_meters: GAA_PITCH_LINES['13m_top'] },
  '20m_top':         { label: '20m Line (Top)',         y_meters: GAA_PITCH_LINES['20m_top'] },
  '45m_top':         { label: '45m Line (Top)',         y_meters: GAA_PITCH_LINES['45m_top'] },
  '65m_top':         { label: '65m Line (Top)',         y_meters: GAA_PITCH_LINES['65m_top'] },
  'halfway':         { label: 'Halfway Line',           y_meters: GAA_PITCH_LINES['halfway'] },
  '65m_bottom':      { label: '65m Line (Bottom)',      y_meters: GAA_PITCH_LINES['65m_bottom'] },
  '45m_bottom':      { label: '45m Line (Bottom)',      y_meters: GAA_PITCH_LINES['45m_bottom'] },
  '20m_bottom':      { label: '20m Line (Bottom)',      y_meters: GAA_PITCH_LINES['20m_bottom'] },
  '13m_bottom':      { label: '13m Line (Bottom)',      y_meters: GAA_PITCH_LINES['13m_bottom'] },
  'endline_bottom':  { label: 'End Line (Bottom)',      y_meters: GAA_PITCH_LINES['endline_bottom'] },
  'left_sideline':   { label: 'Left Sideline',          x_meters: GAA_PITCH_SIDELINES['left_sideline'],  orientation: 'vertical' },
  'right_sideline':  { label: 'Right Sideline',         x_meters: GAA_PITCH_SIDELINES['right_sideline'], orientation: 'vertical' },
  '13m_box_left':    { label: '13m Box Line (Left)',    x_meters: GAA_PITCH_SIDELINES['13m_box_left'],   orientation: 'vertical' },
  '13m_box_right':   { label: '13m Box Line (Right)',   x_meters: GAA_PITCH_SIDELINES['13m_box_right'],  orientation: 'vertical' },
  'small_arc_left':  { label: 'Small Arc Line (Left)',  x_meters: GAA_PITCH_SIDELINES['small_arc_left'], orientation: 'vertical' },
  'small_arc_right': { label: 'Small Arc Line (Right)', x_meters: GAA_PITCH_SIDELINES['small_arc_right'],orientation: 'vertical' },
}

// ---------------------------------------------------------------------------
// Line segments — used to draw and hit-test selectable pitch lines in the
// annotation canvas.  Coordinates are in meters.
// ---------------------------------------------------------------------------
export const PITCH_LINE_SEGMENTS: Array<{ name: string; x1: number; y1: number; x2: number; y2: number }> = [
  { name: 'left_sideline',         x1: GAA_PITCH_SIDELINES['left_sideline'],  y1: 0,                              x2: GAA_PITCH_SIDELINES['left_sideline'],  y2: GAA_PITCH_LENGTH },
  { name: 'right_sideline',        x1: GAA_PITCH_SIDELINES['right_sideline'], y1: 0,                              x2: GAA_PITCH_SIDELINES['right_sideline'], y2: GAA_PITCH_LENGTH },
  { name: 'top_endline',           x1: 0,                                     y1: GAA_PITCH_LINES['endline_top'], x2: GAA_PITCH_WIDTH,                       y2: GAA_PITCH_LINES['endline_top'] },
  { name: 'bottom_endline',        x1: 0,                                     y1: GAA_PITCH_LINES['endline_bottom'], x2: GAA_PITCH_WIDTH,                    y2: GAA_PITCH_LINES['endline_bottom'] },
  { name: '13m_top',               x1: 0, y1: GAA_PITCH_LINES['13m_top'],    x2: GAA_PITCH_WIDTH, y2: GAA_PITCH_LINES['13m_top'] },
  { name: '13m_bottom',            x1: 0, y1: GAA_PITCH_LINES['13m_bottom'], x2: GAA_PITCH_WIDTH, y2: GAA_PITCH_LINES['13m_bottom'] },
  { name: '20m_top',               x1: 0, y1: GAA_PITCH_LINES['20m_top'],    x2: GAA_PITCH_WIDTH, y2: GAA_PITCH_LINES['20m_top'] },
  { name: '20m_bottom',            x1: 0, y1: GAA_PITCH_LINES['20m_bottom'], x2: GAA_PITCH_WIDTH, y2: GAA_PITCH_LINES['20m_bottom'] },
  { name: '45m_top',               x1: 0, y1: GAA_PITCH_LINES['45m_top'],    x2: GAA_PITCH_WIDTH, y2: GAA_PITCH_LINES['45m_top'] },
  { name: '45m_bottom',            x1: 0, y1: GAA_PITCH_LINES['45m_bottom'], x2: GAA_PITCH_WIDTH, y2: GAA_PITCH_LINES['45m_bottom'] },
  { name: '65m_top',               x1: 0, y1: GAA_PITCH_LINES['65m_top'],    x2: GAA_PITCH_WIDTH, y2: GAA_PITCH_LINES['65m_top'] },
  { name: '65m_bottom',            x1: 0, y1: GAA_PITCH_LINES['65m_bottom'], x2: GAA_PITCH_WIDTH, y2: GAA_PITCH_LINES['65m_bottom'] },
  { name: 'halfway',               x1: 0, y1: GAA_PITCH_LINES['halfway'],    x2: GAA_PITCH_WIDTH, y2: GAA_PITCH_LINES['halfway'] },
  { name: 'left_13m_box_top_v',    x1: GAA_PITCH_SIDELINES['13m_box_left'],  y1: 0,                            x2: GAA_PITCH_SIDELINES['13m_box_left'],  y2: GAA_PITCH_LINES['13m_top'] },
  { name: 'left_13m_box_bottom_v', x1: GAA_PITCH_SIDELINES['13m_box_left'],  y1: GAA_PITCH_LINES['13m_bottom'], x2: GAA_PITCH_SIDELINES['13m_box_left'],  y2: GAA_PITCH_LENGTH },
  { name: 'right_13m_box_top_v',   x1: GAA_PITCH_SIDELINES['13m_box_right'], y1: 0,                            x2: GAA_PITCH_SIDELINES['13m_box_right'], y2: GAA_PITCH_LINES['13m_top'] },
  { name: 'right_13m_box_bottom_v',x1: GAA_PITCH_SIDELINES['13m_box_right'], y1: GAA_PITCH_LINES['13m_bottom'], x2: GAA_PITCH_SIDELINES['13m_box_right'], y2: GAA_PITCH_LENGTH },
]
