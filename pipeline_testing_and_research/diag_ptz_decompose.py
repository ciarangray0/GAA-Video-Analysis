"""
DIAGNOSTIC: Are anchor PTZ decompositions self-consistent?

If focal lengths jump wildly (e.g. 500→2141→800) or rotations are not
monotonic across anchors, PTZ interpolation will produce garbage.
This tells you whether the decompose_H_to_ptz approach is even salvageable.

Run from: interactive_analytics_system_backend/
"""
import sys, os, json, re
sys.path.insert(0, ".")
import numpy as np
from pipeline.line_constraints import compute_line_constrained_homography
from pipeline.gaa_pitch_config import GAA_PITCH_VERTICES
from pipeline.config import OUT_W, OUT_H

ANNOTATIONS_FILE = "/Users/ciarangray/Downloads/v4_annotations_040 002429_-_Scores_For.mp4_1772652456228.json"
PITCH_METERS_W, PITCH_METERS_H = 85.0, 140.0

def meters_to_canvas(x_m, y_m):
    return x_m / PITCH_METERS_W * OUT_W, y_m / PITCH_METERS_H * OUT_H

def resolve_pitch_id(pid):
    if pid in GAA_PITCH_VERTICES: return GAA_PITCH_VERTICES[pid]
    m = re.match(r'^line_.+_x([-\d.]+)_y([-\d.]+)$', pid)
    if m: return float(m.group(1)), float(m.group(2))
    raise ValueError(pid)

# Load anchor Hs
with open(ANNOTATIONS_FILE) as f: data = json.load(f)
frames_raw = data["anchorFrames"]
if isinstance(frames_raw, dict): frames_raw = list(frames_raw.values())

anchor_H = {}
for a in frames_raw:
    fidx = a["frame_idx"]
    if a.get("isSkipped"): continue
    pts = a.get("points", [])
    lines = a.get("lines", [])
    if len(pts) < 4: continue
    pi = np.float32([[p["x_img"], p["y_img"]] for p in pts])
    pc = np.float32([meters_to_canvas(*resolve_pitch_id(p["pitch_id"])) for p in pts])
    try:
        H, _ = compute_line_constrained_homography(
            pi, pc, lines, num_samples_per_line=15, max_iterations=5,
            keypoint_weight=3, prefer_line_pts_for_init=True, min_line_pts_for_init=4,
        )
        if H is not None:
            anchor_H[fidx] = H
    except Exception as e:
        print(f"Anchor {fidx}: FAILED — {e}")

print(f"Loaded {len(anchor_H)} anchor Hs: {sorted(anchor_H.keys())}\n")

# ── Test 1: current broken decompose_H_to_ptz ────────────────────────────────
print("=== Current ptz_homography.decompose_H_to_ptz ===")
try:
    from pipeline.ptz_homography import decompose_H_to_ptz
    for fidx in sorted(anchor_H.keys()):
        H = anchor_H[fidx]
        f, R = decompose_H_to_ptz(H)
        pan  = np.degrees(np.arctan2(R[0, 2], R[2, 2]))
        tilt = np.degrees(np.arcsin(np.clip(-R[1, 2], -1, 1)))
        det  = np.linalg.det(R)
        print(f"  Anchor {fidx:3d}: f={f:8.1f}  pan={pan:+8.2f}°  tilt={tilt:+7.2f}°  det(R)={det:+.4f}")
    print()
    print("VERDICT:")
    print("  Good: f values are all in 500-2500 range and don't jump >500 between adjacent anchors")
    print("  Good: pan values are monotonically changing (camera panning one direction)")
    print("  Good: det(R) is close to 1.0 for all anchors (valid rotation matrix)")
    print("  Bad:  f jumping wildly, pan non-monotonic, det(R) far from 1.0 → decomp is broken\n")
except Exception as e:
    print(f"  IMPORT FAILED: {e}\n")

# ── Test 2: alternative decompose using H directly (no K assumption) ─────────
# This measures whether the Hs THEMSELVES are geometrically consistent,
# independent of the broken K assumption in ptz_homography.py.
print("=== Direct H consistency check (no K assumption) ===")
anchor_list = sorted(anchor_H.keys())
for i in range(len(anchor_list) - 1):
    a1 = anchor_list[i]
    a2 = anchor_list[i + 1]
    H1 = anchor_H[a1]
    H2 = anchor_H[a2]

    # Relative H between the two anchors: H_rel maps a1 image coords to a2 image coords
    H_rel = H2 @ np.linalg.inv(H1)
    H_rel /= H_rel[2, 2]

    # Measure: how much does H_rel rotate/scale/translate the image?
    # Project 4 corners through H_rel and measure displacement
    corners = np.float32([[0,0],[1920,0],[1920,1080],[0,1080]])
    corners_h = np.hstack([corners, np.ones((4,1))])
    proj = (H_rel @ corners_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    displacements = np.linalg.norm(proj - corners, axis=1)

    print(f"  {a1:3d}→{a2:3d}: corner displacements (px): "
          f"min={displacements.min():.1f}  mean={displacements.mean():.1f}  max={displacements.max():.1f}")

print()
print("VERDICT:")
print("  Consistent: displacements are similar magnitude across all segments")
print("  Inconsistent: one segment has 5x the displacement of another → anchor H for that segment is wrong\n")

# ── Test 3: what does a CORRECT PTZ decomposition look like? ─────────────────
# The correct model for H: image → canvas is NOT H = K R K^-1.
# It's H = M_canvas @ R @ K^-1  where M_canvas encodes the pitch-to-canvas mapping.
# We can extract R without knowing K by using the known pitch geometry.
print("=== Better PTZ: extract pan angle from vanishing point of pitch sidelines ===")
# The left and right sidelines (x=0 and x=85m) are parallel in world space.
# Their vanishing point in the image gives us the camera's pan direction.
# We can compute this from the anchor H directly.
for fidx in sorted(anchor_H.keys()):
    H = anchor_H[fidx]
    H_inv = np.linalg.inv(H)

    # Left sideline in canvas: x=0, y from 0 to 1400 → canvas points (0,0) to (0,1400)
    # Project back to image to get the sideline in image space
    p1_canvas = np.array([0.0, 0.0, 1.0])
    p2_canvas = np.array([0.0, 1400.0, 1.0])
    p1_img = H_inv @ p1_canvas; p1_img /= p1_img[2]
    p2_img = H_inv @ p2_canvas; p2_img /= p2_img[2]

    # Right sideline
    p3_canvas = np.array([850.0, 0.0, 1.0])
    p4_canvas = np.array([850.0, 1400.0, 1.0])
    p3_img = H_inv @ p3_canvas; p3_img /= p3_img[2]
    p4_img = H_inv @ p4_canvas; p4_img /= p4_img[2]

    # Line through p1,p2 in homogeneous form: l = p1 × p2
    l_left  = np.cross(p1_img, p2_img)
    l_right = np.cross(p3_img, p4_img)
    # Vanishing point = intersection of the two parallel lines
    vp = np.cross(l_left, l_right)
    if abs(vp[2]) > 1e-8:
        vp /= vp[2]
        # Horizontal displacement from image center tells us pan
        pan_px = vp[0] - 960  # relative to cx=960
        print(f"  Anchor {fidx:3d}: VP=({vp[0]:.0f}, {vp[1]:.0f})  pan_offset={pan_px:+.0f}px from centre")
    else:
        print(f"  Anchor {fidx:3d}: sidelines parallel in image (VP at infinity) — camera facing straight on")

print()
print("VERDICT:")
print("  Good: pan_offset changes smoothly and monotonically across anchors")
print("  Good: VP y-coordinate is consistent (same tilt) across anchors")
print("  This is the data the PTZ interpolation should be based on — NOT the broken K decomp")