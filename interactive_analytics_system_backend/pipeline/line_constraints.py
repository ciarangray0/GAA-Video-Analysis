"""Point-sampling utilities for homography line constraints."""

from typing import Dict
import numpy as np

from pipeline.gaa_pitch_config import GAA_PITCH_LINES, GAA_PITCH_SIDELINES


def get_available_lines() -> Dict[str, float]:
    """Return dict of available line IDs and their Y values in meters."""
    return GAA_PITCH_LINES.copy()


# =============================================================================
# Point Sampling
# =============================================================================

def sample_points_on_line(
    u1: float, v1: float,
    u2: float, v2: float,
    num_samples: int = 10
) -> np.ndarray:
    """Sample N points uniformly along a line segment in image space.

    Args:
        u1, v1: First endpoint in image pixels
        u2, v2: Second endpoint in image pixels
        num_samples: Number of points to sample (including endpoints)

    Returns:
        Nx2 array of image points [(u, v), ...]
    """
    if num_samples < 2:
        num_samples = 2

    t_values = np.linspace(0.0, 1.0, num_samples)
    u_samples = (1 - t_values) * u1 + t_values * u2
    v_samples = (1 - t_values) * v1 + t_values * v2
    return np.column_stack([u_samples, v_samples]).astype(np.float32)
