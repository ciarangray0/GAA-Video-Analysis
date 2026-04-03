"""Rendering utilities for the video analysis pipeline."""
import cv2
import numpy as np

from pipeline.config import OUT_W, OUT_H
from pipeline.gaa_pitch_config import GAA_PITCH_WIDTH, GAA_PITCH_LENGTH

# Reference lines for warped frame overlays: (label, y_meters, bgr_colour)
_REFERENCE_LINES = [
    ("13m",     13.0, (0, 200, 255)),
    ("20m",     20.0, (0, 255, 0)),
    ("45m",     45.0, (255, 128, 0)),
    ("65m",     65.0, (128, 0, 255)),
    ("halfway", 70.0, (255, 255, 0)),
    ("65m",     75.0, (128, 0, 255)),
    ("45m",     95.0, (255, 128, 0)),
    ("20m",    120.0, (0, 255, 0)),
    ("13m",    127.0, (0, 200, 255)),
]

_LINE_ALPHA = 0.45  # overlay opacity for pitch reference lines


def draw_reference_lines(warped: np.ndarray) -> None:
    """Draw pitch reference lines onto a warped canvas image (in-place, semi-transparent)."""
    overlay = warped.copy()

    # Horizontal reference lines (dashed)
    for label, y_m, colour in _REFERENCE_LINES:
        y_px = int(y_m / GAA_PITCH_LENGTH * OUT_H)
        x, dash_len = 0, 20
        while x < OUT_W:
            cv2.line(overlay, (x, y_px), (min(x + dash_len, OUT_W), y_px), colour, 2)
            x += dash_len * 2
        cv2.putText(overlay, label, (4, y_px - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    # 20m semicircles (radius 13m = 130px, curving into pitch)
    arc_r = int(13 / GAA_PITCH_LENGTH * OUT_H)
    arc_cx = OUT_W // 2
    colour_20m = (0, 255, 0)
    cv2.ellipse(overlay, (arc_cx, int(20 / GAA_PITCH_LENGTH * OUT_H)), (arc_r, arc_r), 0, 0, 180, colour_20m, 2)
    cv2.ellipse(overlay, (arc_cx, int(120 / GAA_PITCH_LENGTH * OUT_H)), (arc_r, arc_r), 0, 180, 360, colour_20m, 2)

    # 13m box vertical lines (x=33m, x=52m, from endline to 13m line)
    box_colour = (255, 255, 255)
    box13_lx = int(33 / GAA_PITCH_WIDTH * OUT_W)
    box13_rx = int(52 / GAA_PITCH_WIDTH * OUT_W)
    y13_top = int(13 / GAA_PITCH_LENGTH * OUT_H)
    y13_bot = int(127 / GAA_PITCH_LENGTH * OUT_H)
    cv2.line(overlay, (box13_lx, 0),      (box13_lx, y13_top), box_colour, 2)
    cv2.line(overlay, (box13_rx, 0),      (box13_rx, y13_top), box_colour, 2)
    cv2.line(overlay, (box13_lx, y13_bot), (box13_lx, OUT_H),  box_colour, 2)
    cv2.line(overlay, (box13_rx, y13_bot), (box13_rx, OUT_H),  box_colour, 2)

    # Small (goalie) box (x=35.5m–49.5m, depth=4.5m from each endline)
    boxs_lx = int(35.5 / GAA_PITCH_WIDTH * OUT_W)
    boxs_rx = int(49.5 / GAA_PITCH_WIDTH * OUT_W)
    ys_top = int(4.5 / GAA_PITCH_LENGTH * OUT_H)
    ys_bot = int(135.5 / GAA_PITCH_LENGTH * OUT_H)
    cv2.line(overlay, (boxs_lx, 0),      (boxs_lx, ys_top),  box_colour, 2)
    cv2.line(overlay, (boxs_lx, ys_top), (boxs_rx, ys_top),  box_colour, 2)
    cv2.line(overlay, (boxs_rx, 0),      (boxs_rx, ys_top),  box_colour, 2)
    cv2.line(overlay, (boxs_lx, OUT_H),  (boxs_lx, ys_bot),  box_colour, 2)
    cv2.line(overlay, (boxs_lx, ys_bot), (boxs_rx, ys_bot),  box_colour, 2)
    cv2.line(overlay, (boxs_rx, OUT_H),  (boxs_rx, ys_bot),  box_colour, 2)

    cv2.addWeighted(overlay, _LINE_ALPHA, warped, 1 - _LINE_ALPHA, 0, warped)
