"""Rendering utilities for the video analysis pipeline."""
import cv2
import numpy as np


def distorted_homography_warp(img, H, out_w, out_h, k1):
    """
    Warp an image using homography with radial distortion.

    This matches the notebook's distorted_homography_warp function exactly.
    Uses vectorized operations for performance.
    """
    h, w = img.shape[:2]
    H_inv = np.linalg.inv(H)

    cx, cy = out_w / 2, out_h / 2

    # Create coordinate grids
    xs, ys = np.meshgrid(np.arange(out_w), np.arange(out_h))

    # Apply radial distortion
    dx = xs - cx
    dy = ys - cy
    r2 = dx**2 + dy**2

    xs_d = xs + dx * k1 * r2
    ys_d = ys + dy * k1 * r2

    # Create homogeneous coordinates
    ones = np.ones_like(xs_d)
    pts_d = np.stack([xs_d, ys_d, ones], axis=-1)

    # Transform back to source image coordinates
    pts_flat = pts_d.reshape(-1, 3)
    src_flat = (H_inv @ pts_flat.T).T
    src_flat = src_flat[:, :2] / src_flat[:, 2:3]

    # Reshape to grid
    map_x = src_flat[:, 0].reshape(out_h, out_w).astype(np.float32)
    map_y = src_flat[:, 1].reshape(out_h, out_w).astype(np.float32)

    # Remap
    warped = cv2.remap(img, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)

    return warped
