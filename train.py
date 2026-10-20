"""Train a single PQ-FRL / baseline run and save its curves.

Examples
--------
    python train.py --algo pq-frl                     # demo preset, ~1-2 min on CPU
    python train.py --algo dp-fedavg --rounds 300
    python train.py --algo pq-frl --preset paper --rounds 500 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pqfrl import plotting
from pqfrl.config import ALGORITHMS, Config, apply_algorithm, apply_preset
from pqfrl.dp import compose_eps
from pqfrl.server import FederatedTrainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PQ-FRL demo: single training run")
    p.add_argument("--algo", default="pq-frl", choices=ALGORITHMS, help="algorithm / baseline")
    p.add_argument("--preset", default="demo", choices=("demo", "paper"),
                   help="parameter preset (demo = CPU-sized, paper = Table I)")
    p.add_argument("--scenario", default=None, choices=("iid", "hotspot"),
                   help="GU distribution (paper Sec. V-A)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=None, help="FRL aggregation rounds")
    p.add_argument("--tg", type=int, default=None, help="time slots per round")
    p.add_argument("--n-uav", type=int, default=None, help="number of UAVs N")
    p.add_argument("--eps", type=float, default=None, help="per-round privacy budget")
    p.add_argument("--eta", type=float, default=None, help="softmax temperature")
    p.add_argument("--arrival-rate", type=float, default=None, help="tasks/s per GU")
    p.add_argument("--t-max", type=float, default=None, help="task deadline [s]")
    p.add_argument("--dp-noise-scale", type=float, default=None,
                   help="noise calibration factor (1.0 = textbook Gaussian mechanism)")
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-slots", type=int, default=None)
    p.add_argument("--device", default=None, help="auto | cpu | cuda")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--no-plot", action="store_true", help="skip the trajectory figure")
    return p


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(max(args.threads, 1))

    cfg = Config()
    apply_preset(cfg, args.preset)
    apply_algorithm(cfg, args.algo)
    for name in ("scenario", "rounds", "tg", "n_uav", "eps", "eta", "arrival_rate",
                 "t_max", "dp_noise_scale", "eval_every", "eval_slots", "device"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    cfg.seed = args.seed

    out_dir = Path(args.out) if args.out else (
        Path("results") / f"{cfg.algorithm}_{cfg.scenario}_seed{cfg.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pq-frl] {cfg.algorithm} | scenario={cfg.scenario} | N={cfg.n_uav} "
          f"| rounds={cfg.rounds} | eps={cfg.eps} | eta={cfg.eta} | device={cfg.device}")
    started = time.time()
    trainer = FederatedTrainer(cfg)
    result = trainer.run()
    result["config"] = cfg.to_dict()
    result["privacy"] = {
        "eps_round": cfg.eps,
        "delta": cfg.delta,
        "eps_total_advanced_composition": compose_eps(cfg.eps, cfg.rounds, cfg.delta),
        "dp_noise_scale": cfg.dp_noise_scale,
        "clip_norm_calibrated": trainer.history.clip_norm[-1] if trainer.history.clip_norm else 0.0,
    }

    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    if not args.no_plot:
        # a longer deterministic rollout makes the learned flight paths visible
        rollout = trainer.evaluate(slots=max(cfg.eval_slots, 400), record_trajectory=True)
        plotting.plot_trajectory(rollout["gu_pos"], rollout["trajectory"],
                                 out_dir / "trajectory.png", cfg.area,
                                 title=f"{cfg.algorithm.upper()} trajectories ({cfg.scenario})")

    h, ev = result["history"], result["eval"]
    ev_rounds, ev_cost = h["eval_rounds"], h["eval_cost"]
    print(f"\nfinished in {time.time() - started:.1f} s")
    if ev_rounds:
        print(f"eval cost: round 0 = {ev_cost[0]:.3f} -> round {ev_rounds[-1]} = {ev_cost[-1]:.3f}")
    print(f"final eval: cost={ev['cost']:.3f} | latency={ev['latency']:.3f} s | "
          f"energy={ev['energy_j']:.1f} J | TDR={100 * ev['tdr']:.1f}% | "
          f"speed={ev['speed']:.1f} m/s | offload={100 * ev['offload_rate']:.0f}%")
    print(f"artifacts written to {out_dir}/")


if __name__ == "__main__":
    main()
