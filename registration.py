import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RegistrationConf:
    """All configuration required to run the depth-map registration pipeline."""
    registration_max_iters: int = 1000
    registration_inlier_th: float = 0.04
    registration_early_exit: float = 0.95
    registration_min_inlier_count: int = 50
    registration_dmap_sample_points: int = 3000
    registration_max_shift: float = 0.4

def compute_robust_depth_scale_and_shift(
    src_depth_samples: np.ndarray,
    dst_depth_samples: np.ndarray,
    conf: RegistrationConf,
) -> tuple[float, float, int]:
    """
    Compute robust scale and shift from source depth samples to destination depth samples.
    Args:
        src_depth_samples: Nx1 array of source depth samples
        dst_depth_samples: Nx1 array of destination depth samples
        conf: RegistrationConf containing registration parameters
    Returns:
        scale: float
        shift: float
        inliers: int
    """

    max_iterations = conf.registration_max_iters
    inlier_threshold = conf.registration_inlier_th
    early_exit_ratio = conf.registration_early_exit
    inlier_count_th = conf.registration_min_inlier_count
    shift_th = conf.registration_max_shift

    N = len(src_depth_samples)
    if N != len(dst_depth_samples):
        raise ValueError(
            f"⚠️ Source and destination depth samples must have the same length: {N} != {len(dst_depth_samples)}"
        )
    if N < inlier_count_th:
        raise ValueError(f"⚠️ Need at least {inlier_count_th} points to compute scale and shift transform")

    # Filter invalid/degenerate samples up front.
    finite_mask = np.isfinite(src_depth_samples) & np.isfinite(dst_depth_samples)
    if not finite_mask.all():
        logger.debug("Depth samples contain NaN/Inf values; filtering invalid samples")
    src_depth_samples = src_depth_samples[finite_mask]
    dst_depth_samples = dst_depth_samples[finite_mask]

    positive_mask = (src_depth_samples > 1e-6) & (dst_depth_samples > 1e-6)
    if not positive_mask.all():
        logger.debug("Depth samples contain non-positive values; filtering invalid samples")
    src_depth_samples = src_depth_samples[positive_mask]
    dst_depth_samples = dst_depth_samples[positive_mask]

    N = len(src_depth_samples)
    if N < inlier_count_th:
        return 1.0, 0.0, 0

    best_inliers = 0
    best_scale = 1.0
    best_shift = 0.0

    inlier_set: list[int] = []

    # RANSAC loop
    min_src_std = 1e-6
    for iteration in range(max_iterations):
        if len(inlier_set) > 0.1 * N and iteration < 100:
            sample_indices = np.random.choice(inlier_set, size=3, replace=False)
        else:
            sample_indices = np.random.choice(N, size=3, replace=False)

        src_sample = src_depth_samples[sample_indices]
        dst_sample = dst_depth_samples[sample_indices]
        if np.std(src_sample) < min_src_std:
            continue

        # Compute scale and shift from sample
        scale, shift = compute_depth_scale_and_shift(src_sample, dst_sample)
        if abs(shift) > shift_th:
            continue

        # Transform all source depth samples
        transformed_depth_samples = scale * src_depth_samples + shift

        # compute the error in terms of abs percentage change wrt destination depth samples
        denom = np.maximum(np.abs(dst_depth_samples), 1e-3)
        error = np.abs(transformed_depth_samples - dst_depth_samples) / denom
        inliers = error < inlier_threshold
        num_inliers = np.sum(inliers)
        if num_inliers <= 3:
            continue

        # Update best model if this is better
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_scale, best_shift = scale, shift
            inlier_set = [k for k in range(N) if inliers[k]]

            # Refine using all inliers
            best_scale, best_shift = compute_depth_scale_and_shift(
                src_depth_samples[inliers], dst_depth_samples[inliers]
            )
            transformed_depth_samples = best_scale * src_depth_samples + best_shift
            denom = np.maximum(np.abs(dst_depth_samples), 1e-3)
            error = np.abs(transformed_depth_samples - dst_depth_samples) / denom
            inliers = error < inlier_threshold
            num_inliers = np.sum(inliers)
            best_inliers = num_inliers
            inlier_set = [k for k in range(N) if inliers[k]]

        if best_inliers > early_exit_ratio * N:
            break

    if best_inliers < inlier_count_th or not np.isfinite(best_scale) or not np.isfinite(best_shift):
        fallback_scale, fallback_shift, fallback_inliers = compute_robust_depth_scale_and_shift_fallback(
            src_depth_samples, dst_depth_samples
        )
        return fallback_scale, fallback_shift, fallback_inliers

    return best_scale, best_shift, best_inliers

def compute_depth_scale_and_shift(src_depth_samples: np.ndarray, dst_depth_samples: np.ndarray) -> tuple[float, float]:
    """
    Compute scale and shift from source depth samples to destination depth samples. The transform maps src_depth_samples
    to dst_depth_samples: dst_depth_samples = scale * src_depth_samples + shift

    Args:
        src_depth_samples: Nx1 array of source depth samples
        dst_depth_samples: Nx1 array of destination depth samples
    Returns:
        scale: float
        shift: float
    """

    if len(src_depth_samples) != len(dst_depth_samples):
        raise ValueError("Source and destination depth samples must have the same length")
    if len(src_depth_samples) < 3 or len(dst_depth_samples) < 3:
        raise ValueError("Need at least 3 points to compute scale and shift transform")

    # Compute centroids
    src_centroid = np.mean(src_depth_samples)
    dst_centroid = np.mean(dst_depth_samples)

    # Center the depth samples
    src_centered = src_depth_samples - src_centroid
    dst_centered = dst_depth_samples - dst_centroid
    src_centered = src_centered.reshape(-1, 1)
    dst_centered = dst_centered.reshape(-1, 1)

    # Compute scale
    scale = np.sqrt(np.mean(dst_centered**2, axis=0)) / np.sqrt(np.mean(src_centered**2, axis=0))[0]

    # Compute translation
    shift = dst_centroid - scale * src_centroid

    return float(scale[0]), float(shift[0])


def compute_robust_depth_scale_and_shift_fallback(
    src_depth_samples: np.ndarray,
    dst_depth_samples: np.ndarray,
    trim_ratio: float = 0.1,
) -> tuple[float, float, int]:
    """Fallback method for robust scale and shift estimation.

    dst_depth_samples = scale * src_depth_samples + shift

    The method:
    1) Estimate per-sample scale and shift using median-centered samples.
    2) Sort scales and shifts, trim outliers.
    3) Compute a single robust scale and shift from inliers.

    Args:
        src_depth_samples: Nx1 array of source depth samples
        dst_depth_samples: Nx1 array of destination depth samples
        trim_ratio: ratio of outliers to trim
    Returns:
        scale: float
        shift: float
        inliers: int
    """
    N = len(src_depth_samples)
    if N != len(dst_depth_samples):
        raise ValueError("Source and destination depth samples must have the same length")
    if N < 3:
        raise ValueError("Need at least 3 points to compute scale and shift transform")

    # Check for NaN/Inf in inputs
    valid_mask = np.isfinite(src_depth_samples) & np.isfinite(dst_depth_samples)
    if not valid_mask.all():
        logger.warning("Depth samples contain NaN/Inf values; filtering invalid samples for robust estimation")
    src = src_depth_samples[valid_mask].astype(np.float64)
    dst = dst_depth_samples[valid_mask].astype(np.float64)
    if len(src) < 3:
        return 1.0, 0.0, 0

    # Median-center to get per-sample scale/shift estimates.
    src_med = np.median(src)
    dst_med = np.median(dst)
    src_delta = src - src_med
    dst_delta = dst - dst_med

    # Avoid division by tiny deltas.
    delta_mask = np.abs(src_delta) > 1e-6
    src_valid = src[delta_mask]
    dst_valid = dst[delta_mask]
    if len(src_valid) < 3:
        return 1.0, 0.0, 0

    scale_candidates = dst_delta[delta_mask] / src_delta[delta_mask]
    shift_candidates = dst_valid - scale_candidates * src_valid

    # Sort scales and shifts and trim outliers.
    scales_sorted = np.sort(scale_candidates)
    shifts_sorted = np.sort(shift_candidates)
    lower_idx = int(trim_ratio * (len(scales_sorted) - 1))
    upper_idx = int((1.0 - trim_ratio) * (len(scales_sorted) - 1))
    scale_low, scale_high = scales_sorted[lower_idx], scales_sorted[upper_idx]
    shift_low, shift_high = shifts_sorted[lower_idx], shifts_sorted[upper_idx]

    inlier_mask = (
        (scale_candidates >= scale_low)
        & (scale_candidates <= scale_high)
        & (shift_candidates >= shift_low)
        & (shift_candidates <= shift_high)
    )
    if np.sum(inlier_mask) < 3:
        # Fall back to median-based estimates if trimming is too aggressive.
        robust_scale = float(np.median(scale_candidates))
        robust_shift = float(np.median(dst_valid - robust_scale * src_valid))
        return robust_scale, robust_shift, int(len(scale_candidates))

    src_inliers = src_valid[inlier_mask]
    dst_inliers = dst_valid[inlier_mask]

    # Robust final estimates: median scale then median shift on inliers.
    robust_scale = float(np.median(scale_candidates[inlier_mask]))
    robust_shift = float(np.median(dst_inliers - robust_scale * src_inliers))
    return robust_scale, robust_shift, int(np.sum(inlier_mask))


