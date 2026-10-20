"""Aggregation rules for federated reinforcement learning.

* ``uniform``    -- FedAvg: every client counts the same (Eq. FedAvg baseline).
* ``reward``     -- PQ-FRL quality-aware aggregation: softmax over the
                    DP-protected average local reward (Eq. 14).
* ``reputation`` -- DP-FedRep baseline: softmax over an EMA of update
                    consistency (cosine similarity with the round mean).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _softmax(scores: np.ndarray) -> np.ndarray:
    z = scores - scores.max()
    e = np.exp(z)
    return e / e.sum()


def uniform_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def reward_weights(rewards: Sequence[float], eta: float, clip: float) -> np.ndarray:
    """Eq. (14) with the paper's numerical-stability safeguard."""
    scores = np.clip(np.asarray(rewards, dtype=np.float64), -clip, clip)
    return _softmax(eta * scores)


def reputation_weights(reputation: Sequence[float], eta: float) -> np.ndarray:
    return _softmax(eta * np.asarray(reputation, dtype=np.float64))


def update_reputation(reputation: np.ndarray, deltas: Sequence[np.ndarray],
                      beta: float) -> np.ndarray:
    """EMA of cosine similarity between a client's update and the round mean."""
    stacked = np.stack([d.ravel() for d in deltas])          # (N, P)
    mean = stacked.mean(axis=0)
    mean_norm = np.linalg.norm(mean) + 1e-12
    new = reputation.copy()
    for i, d in enumerate(stacked):
        cos = float(d @ mean / ((np.linalg.norm(d) + 1e-12) * mean_norm))
        new[i] = beta * reputation[i] + (1.0 - beta) * cos
    return new


def compute_weights(rule: str, rewards: Sequence[float], reputation: np.ndarray,
                    eta: float, score_clip: float) -> np.ndarray:
    n = len(rewards)
    if rule == "uniform":
        return uniform_weights(n)
    if rule == "reward":
        return reward_weights(rewards, eta, score_clip)
    if rule == "reputation":
        return reputation_weights(reputation, eta)
    raise ValueError(f"unknown weighting rule {rule!r}")


def diagnostics(weights: np.ndarray) -> Dict[str, float]:
    """Concentration of the aggregation weights (useful for the demo figures)."""
    return {
        "w_max": float(weights.max()),
        "w_min": float(weights.min()),
        "w_entropy": float(-(weights * np.log(weights + 1e-12)).sum()),
    }
