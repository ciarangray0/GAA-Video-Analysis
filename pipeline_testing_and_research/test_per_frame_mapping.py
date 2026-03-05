"""
Test: map players at every frame using propagated per-frame H,
vs current approach (only anchor frames, then interpolate).

This directly tests the expert's suggestion:
  "use every (or every other) frame so that the inter-frame motion is small"

Since we don't have YOLO detections for this test, we simulate players
by sampling points from the annotated lines — these are known pitch
positions, so we can verify mapping accuracy.

Run from: interactive_analytics_system_backend/
"""
import sys, os, json, re
import numpy as np
import cv2

sys.path.insert(0, ".")

from pipeline.config import OUT_W, OUT_H, K1
from pipeline.gaa_pitch_config import GAA_PITCH_VERTICES
from pipeline.rendering import distorted_homography_warp
from pipeline.homography import map_pixel_to_pitch
from pipeline.constrained_homography import build_constrained_per_frame_H

ANNOTATIONS_FILE = "/Users/ciarangray/Downloads/v4_annotations_040 002429_-_Scores_For.mp4_1772652456228.json"
VIDEO_FILE       = "/Users/ciarangray/Desktop/FYP/FYP-resources/data/Veo highlights Straffan IFC/040 002429_-_Scores_For.mp4"
OUTPUT_DIR       = "/Users/ciarangray/Desktop/FYP/GAA-Video-Analysis/interactive_analytics_system_backend"
START_FRAME, END_FRAME = 0, 145

PITCH_METERS_W, PITCH_METERS_H = 85.0, 140.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

def meters_to_canvas(x_m, y_m):
    return x_m / PITCH_METERS_W * OUT_W, y_m / PITCH_METERS_H * OUT_H

def resolve_pitch_id(pid):
    if pid in GAA_PITCH_VERTICES: return GAA_PITCH_VERTICES[pid]
    m = re.match(r'^line_.+_x([-\d.]+)_y([-\d.]+)$', pid)
    if m: return float(m.group(1)), float(m.group(2))
    raise ValueError(pid)

def extract_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read(); cap.release()
    return f if ret else None

# ── Load annotations + build anchor Hs ───────────────────────────────────────
with open(ANNOTATIONS_FILE) as f: data = json.load(f)
frames_raw = data["anchorFrames"]
if isinstance(frames_raw, dict): frames_raw = list(frames_raw.values())

# Replace the anchor H computation block (the for loop over frames_raw)
# with this — uses the new line-constrained method for every anchor

from pipeline.line_constraints import compute_line_constrained_homography

anchor_H     = {}
anchor_lines = {}

for a in frames_raw:
    fidx = a["frame_idx"]
    if a.get("isSkipped"): continue
    pts   = a.get("points", [])
    lines = a.get("lines", [])
    if len(pts) < 4: continue

    pi = np.float32([[p["x_img"], p["y_img"]] for p in pts])
    pc = np.float32([meters_to_canvas(*resolve_pitch_id(p["pitch_id"])) for p in pts])

    try:
        H, info = compute_line_constrained_homography(
            pi, pc, lines,
            num_samples_per_line=15,
            max_iterations=5,
            keypoint_weight=3,
            prefer_line_pts_for_init=True,
            min_line_pts_for_init=4,
        )
        if H is not None:
            anchor_H[fidx]     = H
            anchor_lines[fidx] = lines
            print(f"  Anchor {fidx}: line_init={info['used_line_pts_for_init']} "
                  f"valid_lines={info['valid_lines']} "
                  f"synthetic_pts={info['synthetic_points']}")
    except Exception as e:
        print(f"  Anchor {fidx}: FAILED — {e}")

print(f"Anchor frames: {sorted(anchor_H.keys())}")

# ── Build per-frame H using constrained propagation ───────────────────────────
print("\nBuilding per-frame Hs (constrained propagation)...")
# Build anchor_annotations dict from frames_raw (needed for optical flow tracking)
anchor_annotations = {a["frame_idx"]: a for a in frames_raw if not a.get("isSkipped")}

# Pass it into the function
per_frame_H, inter_analysis = build_constrained_per_frame_H(
    VIDEO_FILE, anchor_H, START_FRAME, END_FRAME,
    anchor_annotations=anchor_annotations,
    verbose=True,
    correction_weight=5.0,
    regularisation_weight=1.0,
)
print(f"Per-frame H coverage: {len(per_frame_H)} frames")

# ── Simulate "players" using annotated line points ────────────────────────────
# For each anchor frame, use the annotated line points as simulated player feet.
# These have known canvas Y positions, so we can measure mapping accuracy.
# We then ask: for frames BETWEEN anchors, how accurately does each approach
# place a point that was visible at the nearest anchor?

anchor_list = sorted(anchor_H.keys())
expected_y_45m = 45.0 / PITCH_METERS_H * OUT_H   # 450px
expected_y_20m = 20.0 / PITCH_METERS_H * OUT_H
expected_y_0m = 0.0# 200px

print(f"\nSimulated player mapping accuracy")
print(f"Using annotated 0m_top line points as known-position 'players'")
print(f"Expected canvas Y for 0m_top = {expected_y_20m:.1f}px")
print()
print(f"{'frame':>6}  {'per_frame_y':>12}  {'nearest_anc_y':>14}  "
      f"{'pf_err':>8}  {'na_err':>8}  {'improvement':>12}  note")
print("-" * 80)

results = []

for a in frames_raw:
    fidx = a["frame_idx"]
    if a.get("isSkipped"): continue

    # Find 20m_top line annotation for this frame
    line_0 = next((l for l in a.get("lines",[]) if l["line_id"]=="20m_top"), None)
    if line_0 is None: continue

    # For each frame in the segment starting at this anchor,
    # simulate a player standing on the 20m line at the midpoint
    # of the annotated line. Map it using both methods.
    # (In reality the player position in image space would change per frame,
    # but for this test we use the anchor-frame image coords to measure
    # how stable the H is across frames.)

    mid_img_x = (line_0["u1"] + line_0["u2"]) / 2
    mid_img_y = (line_0["v1"] + line_0["v2"]) / 2

    # Find the segment this anchor covers
    idx_in_list = anchor_list.index(fidx) if fidx in anchor_list else -1
    if idx_in_list == -1: continue

    next_anchor = anchor_list[idx_in_list+1] if idx_in_list+1 < len(anchor_list) else END_FRAME
    frames_to_test = range(fidx, min(next_anchor, END_FRAME+1))

    for f in frames_to_test:
        # Per-frame H
        H_pf = per_frame_H.get(f)
        # Nearest anchor H
        nearest = min(anchor_list, key=lambda aa: abs(aa-f))
        H_na = anchor_H[nearest]

        if H_pf is None: continue

        _, y_pf = map_pixel_to_pitch(mid_img_x, mid_img_y, H_pf)
        _, y_na = map_pixel_to_pitch(mid_img_x, mid_img_y, H_na)

        err_pf = abs(y_pf - expected_y_20m)
        err_na = abs(y_na - expected_y_20m)
        improvement = err_na - err_pf

        note = "← ANCHOR" if f in anchor_H else ""
        results.append({
            "frame": f, "y_pf": y_pf, "y_na": y_na,
            "err_pf": err_pf, "err_na": err_na,
            "improvement": improvement
        })

        print(f"{f:>6}  {y_pf:>12.1f}  {y_na:>14.1f}  "
              f"{err_pf:>8.1f}  {err_na:>8.1f}  "
              f"{'+' if improvement>0 else ''}{improvement:>11.1f}  {note}")

# ── Summary stats ─────────────────────────────────────────────────────────────
if results:
    errs_pf = [r["err_pf"] for r in results]
    errs_na = [r["err_na"] for r in results]
    print()
    print("=" * 60)
    print("SUMMARY — 20m_top line mapping error across all frames")
    print(f"  Per-frame H:    median={np.median(errs_pf):.1f}px  "
          f"mean={np.mean(errs_pf):.1f}px  max={np.max(errs_pf):.1f}px")
    print(f"  Nearest anchor: median={np.median(errs_na):.1f}px  "
          f"mean={np.mean(errs_na):.1f}px  max={np.max(errs_na):.1f}px")
    pct_better = 100*sum(1 for r in results if r["improvement"]>2)/len(results)
    print(f"  Per-frame H better on {pct_better:.0f}% of frames (by >2px)")
    print()

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "per_frame_mapping_test.csv")
    with open(csv_path, "w") as cf:
        cf.write("frame,per_frame_y,nearest_anchor_y,expected_y,pf_err,na_err,improvement\n")
        for r in results:
            cf.write(f"{r['frame']},{r['y_pf']:.2f},{r['y_na']:.2f},"
                     f"{expected_y_20m:.2f},{r['err_pf']:.2f},{r['err_na']:.2f},"
                     f"{r['improvement']:.2f}\n")
    print(f"Saved: {csv_path}")

# ── Save warp overlay video for visual inspection ─────────────────────────────
# Create a short video showing warped frames + mapped line positions
# for both methods side-by-side, around the problematic anchor at frame 58
print("\nGenerating comparison video (frames 44-72, around frame 58)...")

PITCH_LINE_DEFS = {
    "45m_top": (45.0, (0,255,255)),
    "20m_top": (20.0, (200,0,200)),
    "0m_top":  (0.0,  (255,200,0)),
}

fourcc   = cv2.VideoWriter_fourcc(*'MJPG')
vid_path = os.path.join(OUTPUT_DIR, "per_frame_vs_anchor_comparison.avi")
frame_h_out = int(OUT_H * 0.4)
frame_w_out = int(OUT_W * 0.4)
video_out = cv2.VideoWriter(vid_path, fourcc, 5.0,
                             (frame_w_out*2 + 20, frame_h_out + 40))

if not video_out.isOpened():
    print(f"ERROR: VideoWriter failed to open. Try changing codec.")
    print(f"  Current codec: {fourcc}")
    print(f"  Output path:   {vid_path}")
    # Fall back to MJPEG .avi
    vid_path = vid_path.replace('.mp4', '.avi')
    fourcc   = cv2.VideoWriter_fourcc(*'MJPG')
    video_out = cv2.VideoWriter(vid_path, fourcc, 5.0,
                                 (frame_w_out*2 + 20, frame_h_out + 40))
    print(f"  Fell back to MJPEG: {vid_path}")

for f in range(0, 145):
    img = extract_frame(VIDEO_FILE, f)
    if img is None: continue

    H_pf = per_frame_H.get(f)
    nearest = min(anchor_list, key=lambda aa: abs(aa-f))
    H_na = anchor_H[nearest]

    panels = []
    for H_use, label in [(H_pf, f"per-frame f={f}"),
                          (H_na, f"nearest anchor={nearest}")]:
        if H_use is None:
            panels.append(np.zeros((frame_h_out, frame_w_out, 3), dtype=np.uint8))
            continue
        warped = distorted_homography_warp(img, H_use, OUT_W, OUT_H, K1)
        # Draw expected lines
        for lid, (ym, col) in PITCH_LINE_DEFS.items():
            yp = int(ym/PITCH_METERS_H*OUT_H)
            cv2.line(warped, (0,yp), (OUT_W,yp), col, 3)
        cv2.putText(warped, label, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        panels.append(cv2.resize(warped, (frame_w_out, frame_h_out)))

    divider = np.full((frame_h_out, 20, 3), 50, dtype=np.uint8)
    row = np.hstack([panels[0], divider, panels[1]])
    # Add frame number bar
    bar = np.zeros((40, row.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"Frame {f}  |  LEFT=per-frame H  RIGHT=nearest anchor H",
                (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    frame_out = np.vstack([row, bar])
    video_out.write(frame_out)

video_out.release()
print(f"Saved comparison video: {vid_path}")
print("Watch this — if per-frame H keeps lines aligned through the zoom/pan")
print("at frame 58 while nearest-anchor jumps, the approach is working.")