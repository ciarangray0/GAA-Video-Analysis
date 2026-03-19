#!/usr/bin/env python3
"""
Jersey HSV Inspector
====================
Interactive tool to inspect HSV values of bounding box regions in video frames.
Use this to calibrate the thresholds in pipeline/team_classifier.py.

Controls:
    Click (no drag)   — inspect HSV at a single pixel
    Click + drag      — draw a bounding box and inspect full stats
    n / p             — next / previous frame (step 1)
    N / P             — jump forward / backward 30 frames
    q                 — quit
"""

import cv2
import numpy as np
import sys

# ── Hardcode your clip and starting frame here ───────────────────────────────
VIDEO_PATH   = "/Users/ciarangray/Desktop/FYP/FYP-resources/data/good clip/040 002429_-_Scores_For.mp4"
START_FRAME  = 45
# ─────────────────────────────────────────────────────────────────────────────

# ── Mirror the constants from team_classifier ────────────────────────────────
YELLOW_HUE_MIN      = 14
YELLOW_HUE_MAX      = 28
YELLOW_SAT_MIN      = 100
GREY_SAT_THRESHOLD  = 30
JERSEY_CROP_FRACTION = 0.5
MIN_YELLOW_FRACTION  = 0.15
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY_W = 1280

# ── Shared state ─────────────────────────────────────────────────────────────
state = {
    "drawing": False,
    "x0": 0, "y0": 0,
    "x1": 0, "y1": 0,
    "box":   None,   # finalised box in display coords, or None
    "pixel": None,   # (x, y) of a single-pixel click in display coords
    "scale": 1.0,
}


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["drawing"] = True
        state["x0"] = state["x1"] = x
        state["y0"] = state["y1"] = y

    elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
        state["x1"] = x
        state["y1"] = y

    elif event == cv2.EVENT_LBUTTONUP:
        state["drawing"] = False
        state["x1"] = x
        state["y1"] = y
        x0, x1 = sorted([state["x0"], state["x1"]])
        y0, y1 = sorted([state["y0"], state["y1"]])
        if x1 - x0 > 4 and y1 - y0 > 4:
            state["box"]   = (x0, y0, x1, y1)
            state["pixel"] = None
        else:
            state["pixel"] = (state["x0"], state["y0"])
            state["box"]   = None


# ── Analysis ─────────────────────────────────────────────────────────────────

def pixel_hsv(frame: np.ndarray, fx: int, fy: int) -> dict:
    """Return HSV values for a single pixel in frame coords."""
    fh, fw = frame.shape[:2]
    fx = max(0, min(fx, fw - 1))
    fy = max(0, min(fy, fh - 1))
    bgr = frame[fy, fx]
    px  = np.uint8([[bgr]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    in_yellow = (YELLOW_HUE_MIN <= h <= YELLOW_HUE_MAX and s >= YELLOW_SAT_MIN)
    print(f"\n[pixel] H={h}  S={s}  V={v}  "
          f"{'✓ in yellow range' if in_yellow else '✗ not yellow'}")
    return {"h": h, "s": s, "v": v, "in_yellow": in_yellow}


def region_stats(region_bgr: np.ndarray, label: str) -> dict:
    """Compute HSV stats and yellow fraction for a BGR region."""
    if region_bgr.size == 0:
        return {}

    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    n_total = len(hsv)

    mean_full   = hsv.mean(axis=0)
    median_full = np.median(hsv, axis=0)

    sat_mask = hsv[:, 1] >= GREY_SAT_THRESHOLD
    n_masked = int(sat_mask.sum())

    if n_masked > 0:
        hsv_m         = hsv[sat_mask]
        mean_masked   = hsv_m.mean(axis=0)
        median_masked = np.median(hsv_m, axis=0)

        hue_bins = (hsv_m[:, 0] // 5 * 5).astype(int)
        counts   = {}
        for h in hue_bins:
            counts[h] = counts.get(h, 0) + 1
        mode_hue = max(counts, key=counts.get)

        yellow_mask = (
            (hsv_m[:, 0] >= YELLOW_HUE_MIN) &
            (hsv_m[:, 0] <= YELLOW_HUE_MAX) &
            (hsv_m[:, 1] >= YELLOW_SAT_MIN)
        )
        pct_yellow = 100.0 * yellow_mask.sum() / n_masked
    else:
        mean_masked = median_masked = np.zeros(3)
        mode_hue    = 0
        pct_yellow  = 0.0

    passes = pct_yellow / 100.0 >= MIN_YELLOW_FRACTION
    classification = "ELLISTOWN" if passes else "opposition"

    result = {
        "label":          label,
        "n_total":        n_total,
        "n_masked":       n_masked,
        "mean_full":      mean_full,
        "median_full":    median_full,
        "mean_masked":    mean_masked,
        "median_masked":  median_masked,
        "mode_hue":       mode_hue,
        "pct_yellow":     pct_yellow,
        "classification": classification,
    }

    print(f"\n{'='*54}")
    print(f"  {label}")
    print(f"{'='*54}")
    print(f"  Pixels: {n_total} total / {n_masked} after sat>{GREY_SAT_THRESHOLD} mask "
          f"({100*n_masked/max(n_total,1):.0f}% kept)")
    print(f"  Full box — mean   H={mean_full[0]:.1f}  S={mean_full[1]:.1f}  V={mean_full[2]:.1f}")
    print(f"  Full box — median H={median_full[0]:.1f}  S={median_full[1]:.1f}  V={median_full[2]:.1f}")
    if n_masked > 0:
        print(f"  Masked   — mean   H={mean_masked[0]:.1f}  S={mean_masked[1]:.1f}  V={mean_masked[2]:.1f}")
        print(f"  Masked   — median H={median_masked[0]:.1f}  S={median_masked[1]:.1f}  V={median_masked[2]:.1f}")
        print(f"  Mode hue (bin 5°): {mode_hue}–{mode_hue+5}")
        print(f"  Yellow pixel %: {pct_yellow:.1f}%  (threshold: {MIN_YELLOW_FRACTION*100:.0f}%)")
    print(f"  → {classification}")
    print(f"{'='*54}")

    return result


# ── Info panel ────────────────────────────────────────────────────────────────

def build_panel(pixel_info: dict, box_results: list, panel_w=440) -> np.ndarray:
    lines = []  # (text, colour, scale, bold)

    def add(text, colour=(200, 200, 200), scale=0.42, bold=False):
        lines.append((text, colour, scale, bold))

    # Header
    add(f"Yellow H={YELLOW_HUE_MIN}-{YELLOW_HUE_MAX}  S>={YELLOW_SAT_MIN}  "
        f"mask-sat>{GREY_SAT_THRESHOLD}", (200, 200, 80))
    add(f"Jersey crop: top {JERSEY_CROP_FRACTION:.0%}   "
        f"min yellow: {MIN_YELLOW_FRACTION*100:.0f}%", (160, 160, 160))
    lines.append(None)

    # Single-pixel section
    if pixel_info:
        h, s, v = pixel_info["h"], pixel_info["s"], pixel_info["v"]
        add("[ Single pixel ]", (200, 200, 80), scale=0.47, bold=True)
        add(f"  H = {h}   S = {s}   V = {v}")
        add(f"  Sat mask (>{GREY_SAT_THRESHOLD}): {'pass' if s >= GREY_SAT_THRESHOLD else 'masked out'}")
        in_y = pixel_info["in_yellow"]
        add(f"  In yellow range: {'YES' if in_y else 'no'}",
            (0, 220, 80) if in_y else (180, 180, 180))
        lines.append(None)

    # Box sections
    for r in box_results:
        if not r:
            continue
        c = (80, 220, 80) if r["classification"] == "ELLISTOWN" else (80, 80, 220)
        add(f"[ {r['label']} ]", c, scale=0.47, bold=True)

        mf = r["mean_full"]
        mm = r["mean_masked"]
        md = r["median_masked"]
        add(f"  full mean   H={mf[0]:.0f}  S={mf[1]:.0f}  V={mf[2]:.0f}")
        add(f"  masked mean H={mm[0]:.0f}  S={mm[1]:.0f}  V={mm[2]:.0f}")
        add(f"  masked med  H={md[0]:.0f}  S={md[1]:.0f}  V={md[2]:.0f}")
        add(f"  mode hue: {r['mode_hue']}-{r['mode_hue']+5}   "
            f"pixels kept: {r['n_masked']}/{r['n_total']}")

        yc = (0, 200, 80) if r["pct_yellow"] >= MIN_YELLOW_FRACTION * 100 else (80, 80, 220)
        add(f"  yellow px: {r['pct_yellow']:.1f}%  (need {MIN_YELLOW_FRACTION*100:.0f}%)", yc)

        vc = (0, 220, 80) if r["classification"] == "ELLISTOWN" else (80, 80, 220)
        add(f"  → {r['classification']}", vc, scale=0.5, bold=True)
        lines.append(None)

    add("click=pixel  drag=box  n/p=±1  N/P=±30  q=quit",
        (110, 110, 110), scale=0.37)

    # Measure height
    line_h = 18
    total_h = max(400, len(lines) * line_h + 20)
    panel = np.full((total_h, panel_w, 3), (25, 25, 35), dtype=np.uint8)

    y = 18
    for item in lines:
        if item is None:
            y += 6
            continue
        text, colour, scale, bold = item
        cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, 2 if bold else 1, cv2.LINE_AA)
        y += int(scale * 40)

    return panel


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open: {VIDEO_PATH}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx    = max(0, min(START_FRAME, total_frames - 1))

    WIN = "Jersey HSV Inspector"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, mouse_callback)

    pixel_info  = {}
    box_results = []
    prev_box    = None
    prev_pixel  = None

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        scale     = DISPLAY_W / frame.shape[1]
        display_h = int(frame.shape[0] * scale)
        display   = cv2.resize(frame, (DISPLAY_W, display_h))
        state["scale"] = scale

        # Live rubber-band rect while dragging
        if state["drawing"]:
            cv2.rectangle(display,
                          (state["x0"], state["y0"]),
                          (state["x1"], state["y1"]),
                          (0, 255, 255), 1)

        # ── Process single-pixel click ────────────────────────────────────────
        if state["pixel"] and state["pixel"] != prev_pixel:
            prev_pixel = state["pixel"]
            dx, dy = state["pixel"]
            fx = int(dx / scale)
            fy = int(dy / scale)
            pixel_info  = pixel_hsv(frame, fx, fy)
            box_results = []
            prev_box    = None
            state["box"] = None

        # ── Process completed box ─────────────────────────────────────────────
        if state["box"] and state["box"] != prev_box:
            prev_box = state["box"]
            dx0, dy0, dx1, dy1 = state["box"]
            fx0 = int(dx0 / scale)
            fy0 = int(dy0 / scale)
            fx1 = int(dx1 / scale)
            fy1 = int(dy1 / scale)

            full_crop  = frame[fy0:fy1, fx0:fx1]
            crop_fy1   = fy0 + max(1, int((fy1 - fy0) * JERSEY_CROP_FRACTION))
            crop_region = frame[fy0:crop_fy1, fx0:fx1]

            box_results = [
                region_stats(full_crop,   "Full box"),
                region_stats(crop_region, f"Top {JERSEY_CROP_FRACTION:.0%} crop (classifier)"),
            ]
            pixel_info = {}
            prev_pixel = None
            state["pixel"] = None

        # ── Draw overlays ─────────────────────────────────────────────────────

        # Pixel crosshair
        if state["pixel"]:
            dx, dy = state["pixel"]
            cv2.drawMarker(display, (dx, dy), (0, 255, 255),
                           cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)

        # Box + crop line
        if state["box"]:
            dx0, dy0, dx1, dy1 = state["box"]
            cv2.rectangle(display, (dx0, dy0), (dx1, dy1), (0, 255, 0), 2)
            crop_dy1 = dy0 + max(1, int((dy1 - dy0) * JERSEY_CROP_FRACTION))
            cv2.line(display, (dx0, crop_dy1), (dx1, crop_dy1), (0, 200, 255), 1)
            cv2.putText(display, f"top {JERSEY_CROP_FRACTION:.0%}",
                        (dx0 + 2, crop_dy1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)

        # Live hover HSV in bottom-left corner
        mx, my = state["x1"], state["y1"]
        if 0 <= mx < DISPLAY_W and 0 <= my < display_h and not state["drawing"]:
            hx = int(mx / scale)
            hy = int(my / scale)
            fh2, fw2 = frame.shape[:2]
            if 0 <= hx < fw2 and 0 <= hy < fh2:
                px  = np.uint8([[frame[hy, hx]]])
                hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
                cv2.putText(display,
                            f"H={hsv[0]}  S={hsv[1]}  V={hsv[2]}",
                            (8, display_h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 80), 1, cv2.LINE_AA)

        cv2.putText(display, f"Frame {frame_idx}/{total_frames-1}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        panel    = build_panel(pixel_info, box_results, panel_w=440)
        ph = panel.shape[0]
        if ph < display_h:
            pad   = np.full((display_h - ph, 440, 3), (25, 25, 35), dtype=np.uint8)
            panel = np.vstack([panel, pad])
        elif ph > display_h:
            panel = panel[:display_h]

        combined = np.hstack([display, panel])
        cv2.imshow(WIN, combined)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            frame_idx = min(frame_idx + 1, total_frames - 1)
        elif key == ord('p'):
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord('N'):
            frame_idx = min(frame_idx + 30, total_frames - 1)
        elif key == ord('P'):
            frame_idx = max(frame_idx - 30, 0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
