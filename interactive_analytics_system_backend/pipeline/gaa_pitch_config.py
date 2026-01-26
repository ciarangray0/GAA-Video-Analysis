# gaa_pitch_config.py

GAA_PITCH_LENGTH = 140.0  # meters  (y)
GAA_PITCH_WIDTH  = 85.0   # meters (x)

GAA_PITCH_VERTICES = {
    # Corners
    "corner_tl": (0.0, 0.0),
    "corner_tr": (85.0, 0.0),
    "corner_bl": (0.0, 140.0),
    "corner_br": (85.0, 140.0),

    # Goal centers
    "top_goal_lp": (39.25, 0.0),
    "top_goal_rp": (45.75, 0.0),
    "bottom_goal_lp": (39.25, 140.0),
    "bottom_goal_rp": (45.75, 140.0),

    # goalie box
    "left_box_bottom": (35.5, GAA_PITCH_LENGTH - 4.5),
    "left_box_top": (35.5, 4.5),
    "right_box_bottom": (49.5, GAA_PITCH_LENGTH - 4.5),
    "right_box_top": (49.5, 4.5),

    # 13m box
    "left_13m_box_bottom": (33.0, GAA_PITCH_LENGTH - 13.0),
    "left_13m_box_top": (33.0, 13.0),
    "right_13m_box_bottom": (52.0, GAA_PITCH_LENGTH - 13.0),
    "right_13m_box_top": (52.0, 13.0),
    "left_endline_13m_box_bottom": (33.0, GAA_PITCH_LENGTH),
    "left_endline_13m_box_top": (33.0, 0.0),
    "right_endline_13m_box_bottom": (52.0, GAA_PITCH_LENGTH),
    "right_endline_13m_box_top": (52.0, 0.0),

    # small arc
    "left_small_arc_bottom": (29.5, GAA_PITCH_LENGTH - 20.0),
    "left_small_arc_top": (29.5, 20.0),
    "right_small_arc_bottom": (55.5, GAA_PITCH_LENGTH - 20.0),
    "right_small_arc_top": (55.5, 20.0),
    "small_arc_top_top": (42.5, 33.),
    "small_arc_top_bottom": (42.5, GAA_PITCH_LENGTH - 33.0),

    # 13 m line
    "left_13m_line_bottom": (0.0, GAA_PITCH_LENGTH - 13.0),
    "left_13m_line_top": (0.0, 13.0),
    "right_13m_line_bottom": (85.0, GAA_PITCH_LENGTH - 13.0),
    "right_13m_line_top": (85.0, 13.0),

    # 20 m line
    "left_20m_line_bottom": (0.0, GAA_PITCH_LENGTH - 20.0),
    "left_20m_line_top": (0.0, 20.0),
    "right_20m_line_bottom": (85.0, GAA_PITCH_LENGTH - 20.0),
    "right_20m_line_top": (85.0, 20.0),

    # 45 m line
    "left_45m_line_bottom": (0.0, GAA_PITCH_LENGTH - 45.0),
    "left_45m_line_top": (0.0, 45.0),
    "right_45m_line_bottom": (85.0, GAA_PITCH_LENGTH - 45.0),
    "right_45m_line_top": (85.0, 45.0),

    # 65 m line
    "left_65m_line_bottom": (0.0, GAA_PITCH_LENGTH - 65.0),
    "left_65m_line_top": (0.0, 65.0),
    "right_65m_line_bottom": (85.0, GAA_PITCH_LENGTH - 65.0),
    "right_65m_line_top": (85.0, 65.0)
}

# =============================================================================
# GAA Pitch Horizontal Lines (for line-constrained homography)
# =============================================================================
# These are horizontal lines that cross the full width of the pitch.
# Used as additional constraints when computing homography in regions
# where point intersections are not visible (e.g., midfield).
#
# The Y-values are in meters from the top goal line.
# See pipeline/line_constraints.py for usage.

GAA_PITCH_LINES = {
    # Top half of pitch (near goal at Y=0)
    "endline_top": 0.0,
    "small_rectangle_top": 4.5,      # Goal area line
    "13m_top": 13.0,
    "20m_top": 20.0,
    "45m_top": 45.0,
    "65m_top": 65.0,

    # Halfway line (pitch is 140m, halfway at 70m)
    "halfway": 72.5,

    # Bottom half of pitch (near goal at Y=140m)
    "65m_bottom": 75.0,              # 140 - 65 = 75
    "45m_bottom": 95.0,              # 140 - 45 = 95
    "20m_bottom": 120.0,             # 140 - 20 = 120
    "13m_bottom": 127.0,             # 140 - 13 = 127
    "small_rectangle_bottom": 135.5, # 140 - 4.5 = 135.5
    "endline_bottom": 140.0,
}

