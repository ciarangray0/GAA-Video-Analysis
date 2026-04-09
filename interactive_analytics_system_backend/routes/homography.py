"""Homography computation and quality endpoints."""
import asyncio
import logging
from functools import partial
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from pipeline.homography import compute_homographies_with_lines_v3, resolve_pitch_coordinates
from pipeline.constrained_homography import build_optical_flow_per_frame_H
from pipeline.schemas import AnchorFrameAnnotation
from pipeline.config import OUT_W, OUT_H
from pipeline.gaa_pitch_config import GAA_PITCH_WIDTH, GAA_PITCH_LENGTH, GAA_PITCH_LINES
from pipeline.persistence import (
    save_homography_dict, load_homography_dict,
    save_annotations, load_annotations,
)
from store import store
from routes.deps import get_video_or_404

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/videos/{video_id}/homographies/v3")
async def compute_homographies_v3(
    video_id: str,
    annotations: List[AnchorFrameAnnotation],
    num_samples_per_line: int = Query(10, ge=2, le=50, description="Points to sample per line"),
    ransac_iterations: int = Query(2000, ge=100, le=10000, description="RANSAC trials for keypoint-only H₀"),
    ransac_threshold: float = Query(5.0, ge=1.0, le=50.0, description="RANSAC inlier threshold in canvas pixels"),
    keypoint_weight: float = Query(20.0, ge=1.0, le=100.0, description="Weight multiplier for keypoints vs line samples"),
):
    """
    Compute anchor homographies using DLT line constraints, then propagate
    per-frame via Lucas-Kanade optical flow with drift correction and SG smoothing.
    """
    video_info = get_video_or_404(video_id)

    annotations_dict = {
        ann.frame_idx: {"keypoints": ann.points, "lines": ann.lines}
        for ann in annotations
    }

    try:
        anchor_homographies, computation_info = await asyncio.to_thread(
            partial(
                compute_homographies_with_lines_v3,
                annotations_dict,
                num_samples_per_line=num_samples_per_line,
                ransac_iterations=ransac_iterations,
                ransac_threshold=ransac_threshold,
                keypoint_weight=keypoint_weight,
            )
        )
    except Exception as e:
        logger.error(f"v3 DLT homography computation failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"v3 homography computation failed: {str(e)}")

    if not anchor_homographies:
        frame_errors = {str(k): v.get("error", "unknown") for k, v in computation_info.items()}
        logger.warning(f"v3: no valid homographies for {video_id}. Per-frame: {frame_errors}")
        raise HTTPException(
            status_code=400,
            detail=(
                "No valid v3 homographies computed. Each annotated frame needs at least 4 keypoints. "
                f"Per-frame errors: {frame_errors}"
            ),
        )

    store.v3_anchor_H_cache[video_id] = anchor_homographies
    save_homography_dict(video_id, "v3_anchor_homographies", anchor_homographies)
    save_annotations(video_id, annotations_dict)

    try:
        per_frame_hs, of_info = await asyncio.to_thread(
            partial(
                build_optical_flow_per_frame_H,
                video_info["path"],
                anchor_homographies,
                total_frames=video_info["num_frames"],
            )
        )
        logger.info(
            f"v3 optical flow: {of_info.get('num_frames')} frames, "
            f"{len(of_info.get('failed_frames', []))} failed OF pairs, "
            f"drift norms: {of_info.get('drift_at_anchors')}"
        )
    except Exception as e:
        logger.error(f"v3 per-frame H propagation failed for video {video_id}: {e}")
        per_frame_hs = anchor_homographies

    store.v3_per_frame_H_cache[video_id] = per_frame_hs
    save_homography_dict(video_id, "v3_homographies", per_frame_hs)

    logger.info(
        f"v3 DLT: computed {len(anchor_homographies)} anchor Hs + {len(per_frame_hs)} per-frame Hs "
        f"for video {video_id}"
    )

    return {
        "frames": sorted(anchor_homographies.keys()),
        "per_frame_count": len(per_frame_hs),
        "info": {str(k): v for k, v in computation_info.items()},
    }


@router.get("/line-constraints/available-lines")
async def get_available_line_ids():
    """Get available line IDs and their Y positions (meters) for line annotations."""
    return {
        "lines": GAA_PITCH_LINES,
        "description": {
            "13m_top": "13 meter line (top/near goal)",
            "20m_top": "20 meter line (top)",
            "45m_top": "45 meter line (top)",
            "65m_top": "65 meter line (top)",
            "halfway": "Halfway line (70m)",
            "65m_bottom": "65 meter line (bottom)",
            "45m_bottom": "45 meter line (bottom)",
            "20m_bottom": "20 meter line (bottom)",
            "13m_bottom": "13 meter line (bottom/far goal)",
        },
    }


@router.get("/videos/{video_id}/homographies/anchor-quality")
async def get_anchor_quality(video_id: str):
    """Compute per-keypoint reprojection error for each anchor frame."""
    get_video_or_404(video_id)

    annotations = load_annotations(video_id)
    if annotations is None:
        raise HTTPException(status_code=400, detail="No annotations found. Compute homographies first.")

    anchor_hs = store.v3_anchor_H_cache.get(video_id) or load_homography_dict(video_id, "v3_anchor_homographies")
    if not anchor_hs:
        raise HTTPException(status_code=400, detail="No anchor homographies found. Compute homographies first.")

    def _to_canvas(pitch_id: str):
        x_m, y_m = resolve_pitch_coordinates(pitch_id)
        return x_m / GAA_PITCH_WIDTH * OUT_W, y_m / GAA_PITCH_LENGTH * OUT_H

    anchors = []
    for frame_idx in sorted(anchor_hs.keys()):
        H = anchor_hs[frame_idx]
        ann = annotations.get(frame_idx, {})
        keypoints_raw = ann.get("keypoints", [])
        lines_raw = ann.get("lines", [])

        kp_results = []
        for kp in keypoints_raw:
            if isinstance(kp, dict):
                pitch_id, x_img, y_img = kp.get("pitch_id", ""), float(kp.get("x_img", 0)), float(kp.get("y_img", 0))
            else:
                pitch_id, x_img, y_img = getattr(kp, "pitch_id", ""), float(getattr(kp, "x_img", 0)), float(getattr(kp, "y_img", 0))

            p = np.array([x_img, y_img, 1.0], dtype=np.float64)
            projected = H @ p
            if abs(projected[2]) > 1e-12:
                projected /= projected[2]
            x_pred, y_pred = float(projected[0]), float(projected[1])

            try:
                x_exp, y_exp = _to_canvas(pitch_id)
            except ValueError:
                continue

            error_px = float(np.sqrt((x_pred - x_exp) ** 2 + (y_pred - y_exp) ** 2))
            if error_px > 30:
                verdict, impact = "outlier", "harmful"
            elif error_px > 15:
                verdict, impact = "high", "marginal"
            else:
                verdict, impact = "good", "helpful"

            kp_results.append({
                "pitch_id": pitch_id, "x_img": x_img, "y_img": y_img,
                "error_px": round(error_px, 2), "verdict": verdict, "impact": impact,
            })

        if not kp_results:
            continue

        errors = [kp["error_px"] for kp in kp_results]
        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))
        n_outliers = sum(1 for kp in kp_results if kp["verdict"] == "outlier")
        n_helpful = sum(1 for kp in kp_results if kp["impact"] == "helpful")

        if n_outliers > 0 or mean_err > 30:
            overall = "bad"
        elif mean_err > 15:
            overall = "warning"
        else:
            overall = "good"

        recommendation = (
            f"Remove or re-annotate {n_outliers} outlier point(s) listed above." if n_outliers > 0
            else "Consider improving marginal keypoints for better accuracy." if overall == "warning"
            else "Homography quality looks good."
        )

        anchors.append({
            "frame_idx": frame_idx,
            "n_keypoints": len(kp_results),
            "n_lines": len(lines_raw),
            "mean_error_px": round(mean_err, 2),
            "max_error_px": round(max_err, 2),
            "n_outlier_points": n_outliers,
            "n_helpful_points": n_helpful,
            "overall_quality": overall,
            "keypoints": kp_results,
            "recommendation": recommendation,
        })

    return {"anchors": anchors}