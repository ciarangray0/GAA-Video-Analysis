"""Per-frame PTZ estimation, optical-flow propagation, and homography mapping.

This module provides *additive* functionality on top of the existing sparse
anchor-frame homography pipeline.  All existing behaviour is preserved.

Public API
----------
compute_optical_flow_keypoint_propagation(video_path, anchor_frame_idx,
    target_frame_idx, keypoints)  → (propagated_pts, confidence)
solve_homography_from_propagated_points(pts_image, pts_canvas,
    confidence, min_confidence)  → H | None
detect_zoom_from_motion_vectors(frames_window)  → zoom_scale
parametric_ptz_estimation(frames_window)  → List[dict]
generate_per_frame_mappings(video_id, anchor_frames, method, options)  → job_id
compare_homography_heatmap(H1, H2, grid_size, out_w, out_h)  → dict
"""

from __future__ import annotations

import base64
import logging
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.config import OUT_W, OUT_H
from pipeline.homography import compute_homography

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal fallback job registry (used when store is unavailable)
# ---------------------------------------------------------------------------
_jobs: Dict[str, dict] = {}


def _get_jobs() -> Dict[str, dict]:
    """Return the shared jobs registry from store, with fallback to module-level dict."""
    try:
        from store import store  # noqa: PLC0415
        return store.jobs  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        return _jobs


# ---------------------------------------------------------------------------
# Frame-loading helpers
# ---------------------------------------------------------------------------

def _read_frame_gray(video_path: str, frame_idx: int) -> np.ndarray:
    """Read a single frame from *video_path* at *frame_idx* as an 8-bit grey image."""
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame {frame_idx} from {video_path!r}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    finally:
        cap.release()


def _read_frames_gray(video_path: str, start: int, end: int) -> List[np.ndarray]:
    """Read contiguous frames [start, end] inclusive from *video_path* as grey images."""
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(end - start + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()
    return frames


# ---------------------------------------------------------------------------
# Core public functions
# ---------------------------------------------------------------------------

def compute_optical_flow_keypoint_propagation(
    video_path: str,
    anchor_frame_idx: int,
    target_frame_idx: int,
    keypoints: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Track anchor keypoints to a target frame with Lucas-Kanade optical flow.

    Sequential frame-by-frame propagation is used to reduce drift when the
    gap between *anchor_frame_idx* and *target_frame_idx* is large.

    Args:
        video_path: Path to the source video file.
        anchor_frame_idx: Frame index at which *keypoints* were annotated.
        target_frame_idx: Frame index to propagate the keypoints to.
        keypoints: (N, 2) float32 array of (x, y) image coordinates at the
            anchor frame.

    Returns:
        propagated_pts: (N, 2) float32 array of tracked positions at
            *target_frame_idx*.
        confidence: (N,) float32 array in ``[0, 1]`` — 1.0 means successfully
            tracked, 0.0 means lost.
    """
    pts = np.array(keypoints, dtype=np.float32).reshape(-1, 1, 2)
    n_pts = len(pts)

    lk_params: dict = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    step = 1 if target_frame_idx >= anchor_frame_idx else -1
    frame_range = range(anchor_frame_idx, target_frame_idx, step)

    if len(frame_range) == 0:
        return pts.reshape(-1, 2), np.ones(n_pts, dtype=np.float32)

    prev_gray = _read_frame_gray(video_path, anchor_frame_idx)
    current_pts = pts.copy()
    accumulated_status = np.ones((n_pts, 1), dtype=np.uint8)

    for idx in frame_range:
        next_idx = idx + step
        next_gray = _read_frame_gray(video_path, next_idx)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, next_gray, current_pts, None, **lk_params
        )

        if next_pts is None:
            break

        if status is None:
            status = np.zeros((n_pts, 1), dtype=np.uint8)

        accumulated_status &= status.reshape(-1, 1)

        # Keep failed points at last known position for subsequent frames
        tracked_mask = accumulated_status.ravel() == 1
        current_pts[tracked_mask] = next_pts[tracked_mask]
        prev_gray = next_gray

    confidence = accumulated_status.ravel().astype(np.float32)
    return current_pts.reshape(-1, 2), confidence


def solve_homography_from_propagated_points(
    pts_image: np.ndarray,
    pts_canvas: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    min_confidence: float = 0.5,
) -> Optional[np.ndarray]:
    """Solve a homography from (possibly noisy) propagated keypoints.

    Optionally filters out low-confidence matches before RANSAC.

    Args:
        pts_image: (N, 2) float array of image coordinates.
        pts_canvas: (N, 2) float array of corresponding pitch-canvas coordinates.
        confidence: Optional (N,) confidence scores in ``[0, 1]``.
        min_confidence: Points below this confidence threshold are excluded.

    Returns:
        3×3 homography matrix, or ``None`` if fewer than 4 points remain after
        filtering.
    """
    pts_image = np.array(pts_image, dtype=np.float32)
    pts_canvas = np.array(pts_canvas, dtype=np.float32)

    if confidence is not None:
        mask = np.array(confidence) >= min_confidence
        pts_image = pts_image[mask]
        pts_canvas = pts_canvas[mask]

    if len(pts_image) < 4:
        return None

    return compute_homography(pts_image, pts_canvas)


def detect_zoom_from_motion_vectors(frames_window: List[np.ndarray]) -> float:
    """Estimate a zoom scale from a short window of greyscale frames.

    Sparse optical flow is computed on a regular grid between the first and
    last frames.  The radial component of each motion vector is analysed to
    estimate whether the scene is zooming in (scale > 1) or out (scale < 1).

    Args:
        frames_window: List of at least 2 greyscale ``(H, W)`` frames in
            temporal order.

    Returns:
        Estimated zoom scale (float). Returns ``1.0`` on failure.
    """
    if len(frames_window) < 2:
        return 1.0

    first = frames_window[0].astype(np.uint8)
    last = frames_window[-1].astype(np.uint8)

    h, w = first.shape[:2]
    grid_step = max(h, w) // 8

    ys, xs = np.mgrid[grid_step : h - grid_step : grid_step,
                      grid_step : w - grid_step : grid_step]
    src_pts = (
        np.column_stack([xs.ravel(), ys.ravel()])
        .astype(np.float32)
        .reshape(-1, 1, 2)
    )

    if len(src_pts) < 4:
        return 1.0

    lk_params: dict = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    dst_pts, status, _ = cv2.calcOpticalFlowPyrLK(first, last, src_pts, None, **lk_params)

    if dst_pts is None or status is None:
        return 1.0

    ok = status.ravel().astype(bool)
    src_ok = src_pts.reshape(-1, 2)[ok]
    dst_ok = dst_pts.reshape(-1, 2)[ok]

    if len(src_ok) < 4:
        return 1.0

    cx, cy = w / 2.0, h / 2.0
    r_src = np.sqrt((src_ok[:, 0] - cx) ** 2 + (src_ok[:, 1] - cy) ** 2)
    r_dst = np.sqrt((dst_ok[:, 0] - cx) ** 2 + (dst_ok[:, 1] - cy) ** 2)

    valid = r_src > 5.0
    if not np.any(valid):
        return 1.0

    zoom_scale = float(np.median(r_dst[valid] / r_src[valid]))
    return float(np.clip(zoom_scale, 0.1, 10.0))


def parametric_ptz_estimation(frames_window: List[np.ndarray]) -> List[Dict[str, float]]:
    """Estimate per-frame PTZ parameters for a sequence of greyscale frames.

    For each consecutive pair of frames a partial-affine model is fitted using
    RANSAC-based ``cv2.estimateAffinePartial2D``.  The translation components
    give approximate pan/tilt, and the scale gives zoom.  Accumulated values
    relative to the first frame are returned.

    Args:
        frames_window: List of greyscale frames in temporal order.

    Returns:
        List of dicts ``{"pan": float, "tilt": float, "zoom": float}``, one
        per input frame (same length as *frames_window*).  The first frame
        always returns zero-motion ``pan=0, tilt=0, zoom=1``.
    """
    results: List[Dict[str, float]] = [{"pan": 0.0, "tilt": 0.0, "zoom": 1.0}]

    if len(frames_window) < 2:
        return results

    lk_params: dict = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    feature_params: dict = dict(
        maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7
    )

    prev = frames_window[0]
    prev_pts = cv2.goodFeaturesToTrack(prev, mask=None, **feature_params)

    cum_pan, cum_tilt, cum_zoom = 0.0, 0.0, 1.0

    for frame in frames_window[1:]:
        if prev_pts is None or len(prev_pts) < 4:
            results.append({"pan": cum_pan, "tilt": cum_tilt, "zoom": cum_zoom})
            prev = frame
            prev_pts = cv2.goodFeaturesToTrack(prev, mask=None, **feature_params)
            continue

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev, frame, prev_pts, None, **lk_params
        )

        pan, tilt, zoom = 0.0, 0.0, 1.0

        if next_pts is not None and status is not None:
            ok = status.ravel().astype(bool)
            src = prev_pts.reshape(-1, 2)[ok]
            dst = next_pts.reshape(-1, 2)[ok]

            if len(src) >= 4:
                M, inliers = cv2.estimateAffinePartial2D(
                    src.reshape(-1, 1, 2),
                    dst.reshape(-1, 1, 2),
                    method=cv2.RANSAC,
                )
                if M is not None and inliers is not None and int(inliers.sum()) >= 4:
                    pan = float(M[0, 2])
                    tilt = float(M[1, 2])
                    zoom = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))

            # Keep well-tracked points to reduce re-detection cost
            prev_pts = dst.reshape(-1, 1, 2) if ok.any() else None
        else:
            prev_pts = None

        cum_pan += pan
        cum_tilt += tilt
        cum_zoom *= zoom
        results.append({"pan": cum_pan, "tilt": cum_tilt, "zoom": cum_zoom})
        prev = frame

    return results


def generate_per_frame_mappings(
    video_id: str,
    anchor_frames: Dict[int, Dict],
    method: str = "flow",
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """Asynchronously generate and cache per-frame homographies.

    Launches a background thread to compute per-frame homographies by one of
    three methods:

    - ``'flow'``: Propagate anchor keypoints with optical flow and re-solve H.
    - ``'interpolate'``: Linearly interpolate between anchor homography matrices.
    - ``'ptz'``: Estimate PTZ parameters and compose H from the PTZ delta.

    Results are stored in ``store.homographies_cache_per_frame[video_id]``.

    Args:
        video_id: Identifier of the video in the store.
        anchor_frames: Mapping ``frame_idx → {keypoints_image, keypoints_canvas, H}``.
            ``keypoints_image`` / ``keypoints_canvas`` are lists of ``[x, y]``
            pairs; ``H`` is an optional 3×3 matrix (required for
            ``'interpolate'`` and helpful for ``'ptz'``).
        method: One of ``'flow'``, ``'interpolate'``, or ``'ptz'``.
        options: Reserved for future configuration; currently unused.

    Returns:
        A *job_id* string that can be polled via ``GET /api/jobs/{job_id}``.
    """
    if options is None:
        options = {}

    job_id = str(uuid.uuid4())
    jobs = _get_jobs()
    jobs[job_id] = {
        "job_id": job_id,
        "video_id": video_id,
        "status": "queued",
        "method": method,
        "progress": 0,
        "total": 0,
        "error": None,
    }

    thread = threading.Thread(
        target=_run_generate_per_frame_mappings,
        args=(job_id, video_id, anchor_frames, method, options),
        daemon=True,
    )
    thread.start()
    return job_id


def compare_homography_heatmap(
    H1: np.ndarray,
    H2: np.ndarray,
    grid_size: int = 20,
    out_w: int = OUT_W,
    out_h: int = OUT_H,
) -> Dict[str, Any]:
    """Compare two homographies by projecting a uniform grid through each.

    For every grid point on the pitch canvas the function applies each
    homography and measures the displacement between the resulting image-space
    positions.

    Args:
        H1: First 3×3 homography matrix (canvas → image direction).
        H2: Second 3×3 homography matrix (canvas → image direction).
        grid_size: Number of sample points per axis.
        out_w: Canvas width in pixels.
        out_h: Canvas height in pixels.

    Returns:
        dict with keys:

        - ``mean_displacement``: Mean displacement across the grid (px).
        - ``max_displacement``: Maximum displacement across the grid (px).
        - ``grid_displacements``: List of per-point displacements (length
          ``grid_size ** 2``).
        - ``heatmap``: Base64-encoded PNG heatmap image.
    """
    H1 = np.array(H1, dtype=np.float64)
    H2 = np.array(H2, dtype=np.float64)

    xs = np.linspace(0, out_w - 1, grid_size)
    ys = np.linspace(0, out_h - 1, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    # Homogeneous coordinates: 3 × (grid_size²)
    pts = np.stack(
        [grid_x.ravel(), grid_y.ravel(), np.ones(grid_size ** 2, dtype=np.float64)]
    )

    try:
        p1 = H1 @ pts
        p1 /= p1[2:3] + 1e-9
        p2 = H2 @ pts
        p2 /= p2[2:3] + 1e-9
    except np.linalg.LinAlgError:
        n = grid_size ** 2
        return {
            "mean_displacement": 0.0,
            "max_displacement": 0.0,
            "grid_displacements": [0.0] * n,
            "heatmap": "",
        }

    displacements: List[float] = np.sqrt(
        (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    ).tolist()

    disp_grid = np.array(displacements, dtype=np.float32).reshape(grid_size, grid_size)

    d_min, d_max = float(disp_grid.min()), float(disp_grid.max())
    if d_max > d_min:
        norm = ((disp_grid - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        norm = np.zeros((grid_size, grid_size), dtype=np.uint8)

    heatmap_img = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    heatmap_resized = cv2.resize(
        heatmap_img, (out_w // 4, out_h // 4), interpolation=cv2.INTER_NEAREST
    )

    _, buf = cv2.imencode(".png", heatmap_resized)
    heatmap_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return {
        "mean_displacement": float(np.mean(displacements)),
        "max_displacement": float(np.max(displacements)),
        "grid_displacements": displacements,
        "heatmap": heatmap_b64,
    }


# ---------------------------------------------------------------------------
# Background worker helpers
# ---------------------------------------------------------------------------

def _run_generate_per_frame_mappings(
    job_id: str,
    video_id: str,
    anchor_frames: Dict[int, Dict],
    method: str,
    options: Dict[str, Any],
) -> None:
    """Worker function executed in a background daemon thread."""
    jobs = _get_jobs()
    jobs[job_id]["status"] = "running"

    try:
        from store import store  # noqa: PLC0415

        video_meta = store.videos.get(video_id)
        if video_meta is None:
            raise ValueError(f"Video {video_id!r} not found in store")

        video_path: str = video_meta["path"]
        num_frames: int = int(video_meta.get("num_frames", 0))
        sorted_anchors = sorted(anchor_frames.keys())

        if not sorted_anchors:
            raise ValueError("No anchor frames provided")

        jobs[job_id]["total"] = num_frames

        if method == "interpolate":
            per_frame = _generate_interpolated(sorted_anchors, anchor_frames, num_frames)
        elif method == "ptz":
            per_frame = _generate_ptz(
                video_path, sorted_anchors, anchor_frames, num_frames, job_id, jobs
            )
        else:  # default: 'flow'
            per_frame = _generate_flow(
                video_path, sorted_anchors, anchor_frames, num_frames, job_id, jobs
            )

        if not hasattr(store, "homographies_cache_per_frame"):
            store.homographies_cache_per_frame = {}  # type: ignore[attr-defined]
        store.homographies_cache_per_frame[video_id] = per_frame

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = num_frames
        logger.info(
            "per-frame mappings done: video=%s method=%s frames=%d",
            video_id, method, len(per_frame),
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception("generate_per_frame_mappings failed: %s", exc)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)


def _generate_interpolated(
    sorted_anchors: List[int],
    anchor_frames: Dict[int, Dict],
    num_frames: int,
) -> Dict[int, np.ndarray]:
    """Linearly interpolate anchor homographies for every frame."""
    anchor_Hs: Dict[int, np.ndarray] = {}
    for fidx in sorted_anchors:
        H = anchor_frames[fidx].get("H")
        if H is not None:
            anchor_Hs[fidx] = np.array(H, dtype=np.float64)

    if not anchor_Hs:
        return {}

    sorted_H_anchors = sorted(anchor_Hs.keys())
    per_frame: Dict[int, np.ndarray] = {}

    for frame_idx in range(num_frames):
        if frame_idx in anchor_Hs:
            per_frame[frame_idx] = anchor_Hs[frame_idx]
            continue

        before = [k for k in sorted_H_anchors if k <= frame_idx]
        after = [k for k in sorted_H_anchors if k > frame_idx]

        if not before:
            per_frame[frame_idx] = anchor_Hs[sorted_H_anchors[0]]
        elif not after:
            per_frame[frame_idx] = anchor_Hs[sorted_H_anchors[-1]]
        else:
            k0, k1 = before[-1], after[0]
            t = (frame_idx - k0) / float(k1 - k0)
            per_frame[frame_idx] = (1.0 - t) * anchor_Hs[k0] + t * anchor_Hs[k1]

    return per_frame


def _generate_flow(
    video_path: str,
    sorted_anchors: List[int],
    anchor_frames: Dict[int, Dict],
    num_frames: int,
    job_id: str,
    jobs: Dict[str, dict],
) -> Dict[int, np.ndarray]:
    """Propagate anchor keypoints with optical flow and re-solve H per frame."""
    per_frame: Dict[int, np.ndarray] = {}

    for frame_idx in range(num_frames):
        nearest = min(sorted_anchors, key=lambda a: abs(a - frame_idx))
        anchor_data = anchor_frames[nearest]
        kpts_image = np.array(anchor_data.get("keypoints_image", []), dtype=np.float32)
        kpts_canvas = np.array(anchor_data.get("keypoints_canvas", []), dtype=np.float32)

        if len(kpts_image) < 4:
            H_anchor = anchor_data.get("H")
            if H_anchor is not None:
                per_frame[frame_idx] = np.array(H_anchor, dtype=np.float64)
        elif frame_idx == nearest:
            H_anchor = anchor_data.get("H")
            if H_anchor is not None:
                per_frame[frame_idx] = np.array(H_anchor, dtype=np.float64)
            else:
                H = solve_homography_from_propagated_points(kpts_image, kpts_canvas)
                if H is not None:
                    per_frame[frame_idx] = H
        else:
            try:
                prop_pts, conf = compute_optical_flow_keypoint_propagation(
                    video_path, nearest, frame_idx, kpts_image
                )
                H = solve_homography_from_propagated_points(prop_pts, kpts_canvas, conf)
                if H is not None:
                    per_frame[frame_idx] = H
                else:
                    H_anchor = anchor_data.get("H")
                    if H_anchor is not None:
                        per_frame[frame_idx] = np.array(H_anchor, dtype=np.float64)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Flow propagation failed for frame %d: %s", frame_idx, exc)
                H_anchor = anchor_data.get("H")
                if H_anchor is not None:
                    per_frame[frame_idx] = np.array(H_anchor, dtype=np.float64)

        jobs[job_id]["progress"] = frame_idx + 1

    return per_frame


def _generate_ptz(
    video_path: str,
    sorted_anchors: List[int],
    anchor_frames: Dict[int, Dict],
    num_frames: int,
    job_id: str,
    jobs: Dict[str, dict],
) -> Dict[int, np.ndarray]:
    """Estimate PTZ parameters per segment and compose per-frame homographies."""
    per_frame: Dict[int, np.ndarray] = {}

    for seg_idx, anchor_idx in enumerate(sorted_anchors):
        next_anchor = (
            sorted_anchors[seg_idx + 1]
            if seg_idx + 1 < len(sorted_anchors)
            else num_frames - 1
        )
        seg_end = min(next_anchor, num_frames - 1)

        seg_frames = _read_frames_gray(video_path, anchor_idx, seg_end)
        if not seg_frames:
            continue

        ptz_params = parametric_ptz_estimation(seg_frames)

        anchor_data = anchor_frames[anchor_idx]
        H_anchor = anchor_data.get("H")
        if H_anchor is None:
            kpts_image = np.array(anchor_data.get("keypoints_image", []), dtype=np.float32)
            kpts_canvas = np.array(anchor_data.get("keypoints_canvas", []), dtype=np.float32)
            if len(kpts_image) >= 4:
                H_anchor = solve_homography_from_propagated_points(kpts_image, kpts_canvas)

        if H_anchor is None:
            continue

        H_anchor_arr = np.array(H_anchor, dtype=np.float64)

        for local_idx, ptz in enumerate(ptz_params):
            global_idx = anchor_idx + local_idx
            if global_idx > seg_end:
                break

            pan = ptz["pan"]
            tilt = ptz["tilt"]
            zoom = ptz["zoom"]

            # Approximate motion homography from PTZ delta (pan/tilt translation + zoom scale)
            T = np.array(
                [[zoom, 0.0, pan], [0.0, zoom, tilt], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            try:
                per_frame[global_idx] = H_anchor_arr @ np.linalg.inv(T)
            except np.linalg.LinAlgError:
                per_frame[global_idx] = H_anchor_arr

        jobs[job_id]["progress"] = seg_end

    return per_frame
