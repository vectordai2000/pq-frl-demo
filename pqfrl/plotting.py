"""Figure generation for the PQ-FRL demo (matplotlib, non-interactive backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALGO_STYLE: Dict[str, Dict[str, object]] = {
    "pq-frl":     dict(color="#c62828", marker="D", label="PQ-FRL (Ours)"),
    "q-frl":      dict(color="#2e7d32", marker="^", label="Q-FRL (non-private)"),
    "dp-fedavg":  dict(color="#1565c0", marker="v", label="DP-FedAvg"),
    "dp-fedrep":  dict(color="#6a1b9a", marker="s", label="DP-FedRep (SOTA)"),
    "fedavg":     dict(color="#616161", marker="o", label="Standard FedAvg"),
    "local-only": dict(color="#8d6e63", marker="x", label="Local-Only"),
}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.8,
    })


def smooth(values: Sequence[float], window: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------- curves
def plot_convergence(series: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                     path: str | Path, title: str = "Convergence under the Non-IID scenario",
                     smooth_window: int = 2) -> Path:
    """``series``: algorithm -> (rounds, mean cost, std cost)."""
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for algo, (rounds, mean, std) in series.items():
        style = ALGO_STYLE.get(algo, {"color": "k", "marker": "o", "label": algo})
        rounds = np.asarray(rounds)
        mean = smooth(mean, smooth_window)
        std = smooth(std, smooth_window)
        ax.plot(rounds, mean, color=style["color"], marker=style["marker"],
                markevery=max(1, len(rounds) // 8), markersize=4, label=style["label"])
        ax.fill_between(rounds, mean - std, mean + std, color=style["color"], alpha=0.13, linewidth=0)
    ax.set_xlabel("FRL aggregation round")
    ax.set_ylabel("Average system cost  $C(t)$")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=2)
    return _save(fig, path)


def plot_curve(x: Sequence[float], series: Dict[str, Tuple[np.ndarray, np.ndarray]],
               path: str | Path, xlabel: str, ylabel: str, title: str,
               xscale: str = "linear") -> Path:
    """Generic metric-vs-parameter curve (epsilon, eta, swarm size)."""
    _style()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = np.asarray(x, dtype=float)
    for algo, (mean, err) in series.items():
        style = ALGO_STYLE.get(algo, {"color": "k", "marker": "o", "label": algo})
        ax.errorbar(x, mean, yerr=err, color=style["color"], marker=style["marker"],
                    markersize=5, capsize=3, label=style["label"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale(xscale)
    ax.legend(loc="best")
    return _save(fig, path)


def plot_grouped_bars(groups: Sequence[str], series: Dict[str, Tuple[np.ndarray, np.ndarray]],
                      path: str | Path, ylabel: str, title: str) -> Path:
    """Grouped dot plot: ``series`` maps algorithm -> (mean per group, std per group)."""
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    n_groups = len(groups)
    n_algos = len(series)
    idx = np.arange(n_groups)
    for i, (algo, (mean, err)) in enumerate(series.items()):
        style = ALGO_STYLE.get(algo, {"color": "k", "marker": "o", "label": algo})
        offset = (i - (n_algos - 1) / 2) * (0.9 / max(n_algos, 1))
        ax.errorbar(idx + offset, mean, yerr=err, fmt=style.get("marker", "o"),
                    color=style["color"], markersize=6, capsize=3, elinewidth=1.2,
                    label=style["label"])
    ax.set_xticks(idx)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
    return _save(fig, path)


# ---------------------------------------------------------------------- diagnostics
def plot_weights(weights: Sequence[Sequence[float]], path: str | Path,
                 title: str = "Quality-aware aggregation weights",
                 window: int = 9) -> Path:
    """Aggregation weight of every UAV over the FRL rounds (moving average)."""
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    w = np.asarray(weights, dtype=float)
    rounds = np.arange(1, len(w) + 1)
    cmap = plt.get_cmap("viridis")
    for i in range(w.shape[1]):
        ax.plot(rounds, smooth(w[:, i], window),
                color=cmap(i / max(w.shape[1] - 1, 1)), label=f"UAV {i + 1}", linewidth=1.6)
    ax.axhline(1.0 / w.shape[1], color="k", linestyle=":", linewidth=1.0,
               label="uniform (FedAvg)")
    ax.set_xlabel("FRL aggregation round")
    ax.set_ylabel("weight  $w_n$")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5, fontsize=8)
    return _save(fig, path)


def plot_trajectory(gu_pos: Sequence[Sequence[float]], trajectory: Sequence[Sequence[Sequence[float]]],
                    path: str | Path, area: float = 1000.0,
                    title: str = "Learned trajectories") -> Path:
    """2-D map with GU positions, hotspot quadrants and UAV flight paths."""
    _style()
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    gu = np.asarray(gu_pos, dtype=float)
    traj = np.asarray(trajectory, dtype=float)          # (T+1, N, 2)

    ax.add_patch(plt.Rectangle((0, area / 2), area / 2, area / 2, facecolor="#ffcdd2",
                               alpha=0.35, edgecolor="none"))
    ax.add_patch(plt.Rectangle((area / 2, 0), area / 2, area / 2, facecolor="#bbdefb",
                               alpha=0.35, edgecolor="none"))
    ax.scatter(gu[:, 0], gu[:, 1], s=22, c="#37474f", marker="o", zorder=5,
               label="ground user")

    cmap = plt.get_cmap("plasma")
    for n in range(traj.shape[1]):
        color = cmap(n / max(traj.shape[1] - 1, 1))
        ax.plot(traj[:, n, 0], traj[:, n, 1], color=color, linewidth=1.6, label=f"UAV {n + 1}")
        ax.scatter(traj[0, n, 0], traj[0, n, 1], s=55, facecolors="white",
                   edgecolors=color, linewidths=1.6, zorder=6)
        ax.scatter(traj[-1, n, 0], traj[-1, n, 1], s=45, color=color, marker="*", zorder=7)

    ax.set_xlim(0, area)
    ax.set_ylim(0, area)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=2, fontsize=7.5)
    return _save(fig, path)


def plot_components(metrics: Dict[str, Dict[str, float]], path: str | Path) -> Path:
    """Component-wise breakdown (latency / energy / TDR), paper Table II style."""
    _style()
    algos = list(metrics)
    fields = [("latency", "Avg. latency [s]", 1.0),
              ("energy_j", "Avg. energy [J]", 1.0),
              ("tdr", "Task drop rate [%]", 100.0)]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
    for ax, (key, label, scale) in zip(axes, fields):
        values = [metrics[a][key] * scale for a in algos]
        colors = [ALGO_STYLE.get(a, {"color": "gray"})["color"] for a in algos]
        ax.bar(range(len(algos)), values, color=colors, alpha=0.85)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels([ALGO_STYLE.get(a, {"label": a})["label"].split(" (")[0] for a in algos],
                           rotation=35, ha="right", fontsize=7.5)
        ax.set_title(label)
    fig.suptitle("Component-wise performance breakdown", y=1.02, fontsize=11)
    return _save(fig, path)
