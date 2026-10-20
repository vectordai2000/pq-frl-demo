"""Client-side differential privacy (the "P" of PQ-FRL).

Each round every UAV releases two objects:

1. its actor update  ``delta_theta = theta_n - Theta_G``, and
2. the scalar average local reward ``R_avg`` used as quality signal.

Both are clipped to a public bound and perturbed with calibrated Gaussian noise
(Eq. 12-13).  We use the standard Gaussian mechanism

    sigma = sqrt(2 ln(1.25 / delta)) * sensitivity / eps

with an optional *per-coordinate* normalisation (``sqrt(dim)``) so that a small
demo network stays trainable; ``dp_noise_scale == 1.0`` reproduces the textbook
calibration, small values emulate the looser calibration that is common in
practical FRL deployments (see README "Privacy calibration").
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def gaussian_sigma(sensitivity: float, eps: float, delta: float,
                   dim: int | None = None, noise_scale: float = 1.0) -> float:
    """Noise standard deviation of the Gaussian mechanism."""
    if eps <= 0 or sensitivity <= 0:
        return 0.0
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / eps
    if dim:
        sigma /= math.sqrt(dim)
    return float(noise_scale * sigma)


def clip_vector(vec: np.ndarray, bound: float) -> np.ndarray:
    """L2 clipping, Eq. (12)."""
    norm = float(np.linalg.norm(vec))
    if norm <= bound or norm == 0.0:
        return vec.copy()
    return vec * (bound / norm)


def privatize_vector(vec: np.ndarray, bound: float, cfg, rng: np.random.Generator
                     ) -> Tuple[np.ndarray, Dict[str, float]]:
    """Clip + noised actor update, Eq. (12)-(13)."""
    clipped = clip_vector(vec, bound)
    sigma = gaussian_sigma(bound, cfg.eps, cfg.delta, dim=vec.size,
                           noise_scale=cfg.dp_noise_scale)
    noise = rng.normal(0.0, sigma, size=vec.size)
    stats = {
        "delta_norm": float(np.linalg.norm(vec)),
        "clipped_norm": float(np.linalg.norm(clipped)),
        "sigma": sigma,
        "noise_norm": float(np.linalg.norm(noise)),
    }
    return clipped + noise, stats


def privatize_scalar(value: float, bound: float, cfg, rng: np.random.Generator
                     ) -> Tuple[float, Dict[str, float]]:
    """Clip + noised average local reward (quality report)."""
    clipped = float(np.clip(value, -bound, bound))
    # a clipped scalar in [-bound, bound] has L2 sensitivity 2 * bound
    sigma = gaussian_sigma(2.0 * bound, cfg.eps, cfg.delta, dim=None,
                           noise_scale=cfg.dp_noise_scale)
    stats = {"reward": float(value), "sigma_r": sigma}
    return clipped + float(rng.normal(0.0, sigma)), stats


def compose_eps(eps_round: float, rounds: int, delta: float) -> float:
    """Advanced-composition bound for ``rounds`` identical (eps, delta)-DP steps.

    eps_total = sqrt(2 T ln(1/delta)) eps + T eps (e^eps - 1)
    """
    if rounds <= 0 or eps_round <= 0:
        return 0.0
    return float(math.sqrt(2.0 * rounds * math.log(1.0 / delta)) * eps_round
                 + rounds * eps_round * (math.exp(eps_round) - 1.0))
