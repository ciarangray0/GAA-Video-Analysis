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
    "bottom_goal_lp": (39.25, 145.0),
    "bottom_goal_rp": (45.75, 145.0),

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
