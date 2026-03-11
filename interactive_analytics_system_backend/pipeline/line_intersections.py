"""Auto-derive pitch keypoints from annotated line intersections.

For each pair of annotated lines where one is horizontal and one is vertical,
this module computes their intersection in image space and maps the result to a
known pitch_id via INTERSECTION_MAP.

This removes the need for users to click individual keypoints that sit on the
intersection of two annotatable lines (corners, 13m-box corners, etc.).

Algorithm
---------
For each (horizontal_line, vertical_line) pair in the annotations:
1. Form the homogeneous line  l = p1 × p2  from the two image-space endpoints.
2. Intersect: pt = l_h × l_v  (cross product of homogeneous lines).
3. Skip degenerate results where |pt[2]| < 1e-6 (point at infinity / parallel lines).
4. Look up (vertical_id, horizontal_id) in INTERSECTION_MAP.
5. Emit PitchPoint(pitch_id=..., x_img=pt[0]/pt[2], y_img=pt[1]/pt[2]).

Usage
-----
    from pipeline.line_intersections import compute_line_intersections
    from pipeline.schemas import LineAnnotation

    points = compute_line_intersections(line_annotations)
"""

from __future__ import annotations

from typing import List

import numpy as np

from pipeline.schemas import LineAnnotation, PitchPoint

# ---------------------------------------------------------------------------
# Intersection map: (vertical_id, horizontal_id) → pitch_id
# Keys are in canonical order: vertical first, then horizontal.
# ---------------------------------------------------------------------------

INTERSECTION_MAP: dict[tuple[str, str], str] = {
    # left_sideline (x=0) × horizontal lines — 11 points
    ("left_sideline", "endline_top"):    "corner_tl",
    ("left_sideline", "13m_top"):        "left_13m_line_top",
    ("left_sideline", "20m_top"):        "left_20m_line_top",
    ("left_sideline", "45m_top"):        "left_45m_line_top",
    ("left_sideline", "65m_top"):        "left_65m_line_top",
    ("left_sideline", "halfway"):        "center_left",
    ("left_sideline", "65m_bottom"):     "left_65m_line_bottom",
    ("left_sideline", "45m_bottom"):     "left_45m_line_bottom",
    ("left_sideline", "20m_bottom"):     "left_20m_line_bottom",
    ("left_sideline", "13m_bottom"):     "left_13m_line_bottom",
    ("left_sideline", "endline_bottom"): "corner_bl",

    # right_sideline (x=85) × horizontal lines — 11 points
    ("right_sideline", "endline_top"):    "corner_tr",
    ("right_sideline", "13m_top"):        "right_13m_line_top",
    ("right_sideline", "20m_top"):        "right_20m_line_top",
    ("right_sideline", "45m_top"):        "right_45m_line_top",
    ("right_sideline", "65m_top"):        "right_65m_line_top",
    ("right_sideline", "halfway"):        "center_right",
    ("right_sideline", "65m_bottom"):     "right_65m_line_bottom",
    ("right_sideline", "45m_bottom"):     "right_45m_line_bottom",
    ("right_sideline", "20m_bottom"):     "right_20m_line_bottom",
    ("right_sideline", "13m_bottom"):     "right_13m_line_bottom",
    ("right_sideline", "endline_bottom"): "corner_br",

    # 13m_box_left (x=33) × horizontal lines — 4 points
    ("13m_box_left", "endline_top"):    "left_endline_13m_box_top",
    ("13m_box_left", "13m_top"):        "left_13m_box_top",
    ("13m_box_left", "13m_bottom"):     "left_13m_box_bottom",
    ("13m_box_left", "endline_bottom"): "left_endline_13m_box_bottom",

    # 13m_box_right (x=52) × horizontal lines — 4 points
    ("13m_box_right", "endline_top"):    "right_endline_13m_box_top",
    ("13m_box_right", "13m_top"):        "right_13m_box_top",
    ("13m_box_right", "13m_bottom"):     "right_13m_box_bottom",
    ("13m_box_right", "endline_bottom"): "right_endline_13m_box_bottom",

    # small_arc_left (x=29.5) × horizontal lines — 2 points
    ("small_arc_left", "20m_top"):    "left_small_arc_top",
    ("small_arc_left", "20m_bottom"): "left_small_arc_bottom",

    # small_arc_right (x=55.5) × horizontal lines — 2 points
    ("small_arc_right", "20m_top"):    "right_small_arc_top",
    ("small_arc_right", "20m_bottom"): "right_small_arc_bottom",
}

# ---------------------------------------------------------------------------
# Known vertical and horizontal line sets (mirrors line_constraints.py)
# ---------------------------------------------------------------------------

_VERTICAL_LINE_IDS: frozenset[str] = frozenset({
    "left_sideline",
    "right_sideline",
    "13m_box_left",
    "13m_box_right",
    "small_arc_left",
    "small_arc_right",
})

_HORIZONTAL_LINE_IDS: frozenset[str] = frozenset({
    "endline_top",
    "13m_top",
    "20m_top",
    "45m_top",
    "65m_top",
    "halfway",
    "65m_bottom",
    "45m_bottom",
    "20m_bottom",
    "13m_bottom",
    "endline_bottom",
})


def _homogeneous_line(u1: float, v1: float, u2: float, v2: float) -> np.ndarray:
    """Return the homogeneous line l = p1 × p2 through two image points."""
    p1 = np.array([u1, v1, 1.0], dtype=np.float64)
    p2 = np.array([u2, v2, 1.0], dtype=np.float64)
    return np.cross(p1, p2)


def compute_line_intersections(lines: List[LineAnnotation]) -> List[PitchPoint]:
    """Derive pitch keypoints from annotated line pairs.

    For every pair where one line is horizontal and one is vertical, the
    function computes their image-space intersection and emits a PitchPoint
    if the pair appears in INTERSECTION_MAP.

    Unrecognized pairs and degenerate intersections (|w| < 1e-6) are silently
    skipped.

    Args:
        lines: List of LineAnnotation objects from the annotator.

    Returns:
        List of PitchPoint objects — one per recognised intersection.
    """
    results: list[PitchPoint] = []

    verticals   = [l for l in lines if l.line_id in _VERTICAL_LINE_IDS]
    horizontals = [l for l in lines if l.line_id in _HORIZONTAL_LINE_IDS]

    for v_ann in verticals:
        # Guard against degenerate annotation (same point clicked twice)
        if v_ann.u1 == v_ann.u2 and v_ann.v1 == v_ann.v2:
            continue
        l_v = _homogeneous_line(v_ann.u1, v_ann.v1, v_ann.u2, v_ann.v2)

        for h_ann in horizontals:
            if h_ann.u1 == h_ann.u2 and h_ann.v1 == h_ann.v2:
                continue
            l_h = _homogeneous_line(h_ann.u1, h_ann.v1, h_ann.u2, h_ann.v2)

            # Homogeneous intersection
            pt = np.cross(l_v, l_h)
            if abs(pt[2]) < 1e-6:
                continue  # degenerate / parallel

            key = (v_ann.line_id, h_ann.line_id)
            pitch_id = INTERSECTION_MAP.get(key)
            if pitch_id is None:
                continue

            results.append(PitchPoint(
                pitch_id=pitch_id,
                x_img=float(pt[0] / pt[2]),
                y_img=float(pt[1] / pt[2]),
            ))

    return results
