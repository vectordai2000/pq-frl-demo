"""Render an animated rollout of the trained PQ-FRL swarm as a GIF.

The script trains a short PQ-FRL run (or reuses the result), then renders a
deterministic mission: UAVs moving over the service area, the ground users they
serve each slot, dropped tasks, and the instantaneous system cost.

    python make_animation.py                       # ~1 min on CPU
    python make_animation.py --rounds 150 --slots 300 --fps 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from pqfrl.aggregators import diagnostics
from pqfrl.config import Config, apply_algorithm, apply_preset
from pqfrl.env import UAVMECEnv
from pqfrl.server import FederatedTrainer

UAV_CMAP = plt.get_cmap("plasma")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render a PQ-FRL rollout animation")
    p.add_argument("--rounds", type=int, default=100, help="FRL rounds used to train the policy")
    p.add_argument("--slots", type=int, default=240, help="mission length in slots")
    p.add_argument("--stride", type=int, default=2, help="render every N-th slot")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results/figures/demo_rollout.gif")
    p.add_argument("--preset", default="demo", choices=("demo", "paper"))
    return p


@torch.no_grad()
def collect(trainer: FederatedTrainer, cfg: Config, slots: int) -> tuple[list[dict], np.ndarray]:
    env = UAVMECEnv(cfg, seed=cfg.seed + 7)
    state = env.reset(seed=cfg.seed + 7)
    actor = trainer.global_actor
    records: list[dict] = []
    for _ in range(slots):
        positions = env.uav_pos.copy()
        actions = np.stack([
            actor(torch.as_tensor(state[i], dtype=torch.float32).unsqueeze(0)).numpy()[0]
            for i in range(env.n)
        ])
        state, _, info = env.step(actions)
        records.append({"positions": positions, "info": info})
    return records, env.gu_pos.copy()


def render(records: list[dict], gu_pos: np.ndarray, cfg: Config, out: Path,
           stride: int, fps: int) -> list[Path]:
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5})
    fig = Figure(figsize=(5.4, 5.4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.08, 0.26, 0.88, 0.66])
    ax_cost = fig.add_axes([0.08, 0.055, 0.88, 0.14])

    area = cfg.area
    trails: list[np.ndarray] = [[] for _ in range(cfg.n_uav)]
    costs: list[float] = []
    images: list[Image.Image] = []

    for idx, rec in enumerate(records):
        pos, info = rec["positions"], rec["info"]
        costs.append(info["cost"])
        for i in range(cfg.n_uav):
            trails[i].append(pos[i].copy())

        if idx % stride and idx != len(records) - 1:
            continue

        ax.clear()
        ax.set_xlim(0, area)
        ax.set_ylim(0, area)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#fafafa")
        if cfg.scenario == "hotspot":
            ax.add_patch(plt.Rectangle((0, area / 2), area / 2, area / 2,
                                       facecolor="#ffcdd2", alpha=0.30, edgecolor="none"))
            ax.add_patch(plt.Rectangle((area / 2, 0), area / 2, area / 2,
                                       facecolor="#bbdefb", alpha=0.30, edgecolor="none"))

        ax.scatter(gu_pos[:, 0], gu_pos[:, 1], s=16, c="#546e7a", zorder=2)
        active = np.asarray(info["active_gu"], dtype=int)
        if active.size:
            ax.scatter(gu_pos[active, 0], gu_pos[active, 1], s=38, c="#ff9800",
                       edgecolors="white", linewidths=0.6, zorder=3, label="pending task")
        for k, n in info["links"]:
            ax.plot([gu_pos[k, 0], pos[n, 0]], [gu_pos[k, 1], pos[n, 1]],
                    color=UAV_CMAP(n / max(cfg.n_uav - 1, 1)), linewidth=0.9, alpha=0.75, zorder=2)
        dropped = info["dropped_gu"]
        if dropped:
            ax.scatter(gu_pos[dropped, 0], gu_pos[dropped, 1], s=46, marker="x",
                       c="#d32f2f", linewidths=1.2, zorder=4, label="dropped")

        for i in range(cfg.n_uav):
            trail = np.asarray(trails[i])
            color = UAV_CMAP(i / max(cfg.n_uav - 1, 1))
            if len(trail) > 1:
                ax.plot(trail[-40:, 0], trail[-40:, 1], color=color, linewidth=1.1, alpha=0.85)
            ax.scatter(pos[i, 0], pos[i, 1], s=70, color=color, edgecolors="white",
                       linewidths=1.0, zorder=5)
            ax.annotate(f"U{i + 1}", (pos[i, 0], pos[i, 1]), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color=color, weight="bold")

        ax.set_title(f"PQ-FRL mission  |  t = {idx * cfg.slot:4.1f} s   "
                     f"cost = {info['cost']:.2f}   TDR = {100 * info['tdr']:.0f}%")
        handles = [plt.Line2D([], [], marker="o", color="w", markerfacecolor="#546e7a",
                              markersize=4, label="idle user"),
                   plt.Line2D([], [], marker="o", color="w", markerfacecolor="#ff9800",
                              markersize=5, label="pending task"),
                   plt.Line2D([], [], marker="x", color="#d32f2f", markersize=5, label="dropped")]
        ax.legend(handles=handles, loc="upper right", fontsize=6.8, frameon=True, framealpha=0.85)

        ax_cost.clear()
        ax_cost.plot(costs, color="#c62828", linewidth=1.2)
        ax_cost.set_xlim(0, len(records))
        ax_cost.set_ylim(0, max(2.0, float(np.max(costs)) * 1.1))
        ax_cost.set_ylabel("cost", fontsize=7)
        ax_cost.set_xlabel("slot", fontsize=7)
        ax_cost.tick_params(labelsize=6.5)
        ax_cost.grid(alpha=0.25, linewidth=0.5)

        canvas.draw()
        img = Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3])
        images.append(img)

    out.parent.mkdir(parents=True, exist_ok=True)
    palette_imgs = [im.convert("P", palette=Image.ADAPTIVE, colors=128) for im in images]
    palette_imgs[0].save(out, save_all=True, append_images=palette_imgs[1:],
                         duration=int(1000 / max(fps, 1)), loop=0, optimize=True)
    print(f"wrote {out}  ({len(images)} frames, {out.stat().st_size / 1e6:.1f} MB)")
    return [out]


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(4)

    cfg = Config()
    apply_preset(cfg, args.preset)
    apply_algorithm(cfg, "pq-frl")
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.rounds = args.rounds
    cfg.eval_every = 0

    print(f"training PQ-FRL for {cfg.rounds} rounds ...")
    trainer = FederatedTrainer(cfg)
    trainer.run()
    weights = np.asarray(trainer.history.weights[-1])
    print("last aggregation weights:", np.round(weights, 2), diagnostics(weights))

    print("rendering rollout ...")
    records, gu_pos = collect(trainer, cfg, args.slots)
    render(records, gu_pos, cfg, Path(args.out), args.stride, args.fps)


if __name__ == "__main__":
    main()
