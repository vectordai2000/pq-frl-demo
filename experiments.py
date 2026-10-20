"""Generate the demo figures with the PQ-FRL demo.

    python experiments.py                 # quick suite (a few minutes on CPU)
    python experiments.py --figures convergence,privacy --seeds 3 --rounds 300
    python experiments.py --full          # larger suite used for the README figures

Every training run is cached under ``results/runs`` so the script can be
re-executed or partially rerun without repeating expensive work.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from pqfrl import plotting
from pqfrl.config import ALGORITHMS, Config, apply_algorithm, apply_preset
from pqfrl.server import FederatedTrainer

FIGURES = ("convergence", "scenario", "privacy", "scalability", "eta", "components")


# ---------------------------------------------------------------------- utilities
def run_cached(cfg: Config, runs_dir: Path, device: str, threads: int) -> dict:
    key_fields = ("algorithm", "seed", "scenario", "eps", "n_uav", "eta", "rounds",
                  "tg", "arrival_rate", "t_max", "dp_noise_scale", "hidden", "actor_lr")
    key = "_".join(f"{k}={getattr(cfg, k)}" for k in key_fields)
    path = runs_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    cfg.device = device
    torch.set_num_threads(max(threads, 1))
    trainer = FederatedTrainer(cfg)
    result = trainer.run()
    runs_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")
    return result


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def eval_curve(result: dict) -> tuple[np.ndarray, np.ndarray]:
    h = result["history"]
    return np.asarray(h["eval_rounds"], dtype=float), np.asarray(h["eval_cost"], dtype=float)


def interp_curve(rounds: np.ndarray, cost: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, rounds, cost)


def curve_stats(results: List[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = []
    for r in results:
        rounds, cost = eval_curve(r)
        curves.append((rounds, cost))
    grid = curves[0][0]
    stacked = np.stack([interp_curve(rd, c, grid) for rd, c in curves])
    return grid, stacked.mean(axis=0), stacked.std(axis=0)


def final_costs(results: List[dict]) -> tuple[float, float]:
    return mean_std([r["eval"]["cost"] for r in results])


# ---------------------------------------------------------------------- experiments
def exp_convergence(opts, runs_dir: Path) -> dict:
    series, raw = {}, {}
    for algo in ALGORITHMS:
        results = []
        for seed in range(opts.seeds):
            cfg = Config()
            apply_preset(cfg, opts.preset)
            apply_algorithm(cfg, algo)
            cfg.seed, cfg.rounds, cfg.scenario = seed, opts.rounds, "hotspot"
            cfg.eval_every, cfg.eval_slots = opts.eval_every, opts.eval_slots
            results.append(run_cached(cfg, runs_dir, opts.device, opts.threads))
            print(f"  {algo:10s} seed {seed} done", flush=True)
        rounds, mean, std = curve_stats(results)
        series[algo] = (rounds, mean, std)
        raw[algo] = results
    path = plotting.plot_convergence(series, opts.out / "figures" / "demo_convergence.png")
    print(f"  -> {path}")

    weights_path = plotting.plot_weights(
        raw["pq-frl"][0]["history"]["weights"],
        opts.out / "figures" / "demo_weights.png",
        title="PQ-FRL aggregation weights, 9-round moving average (seed 0)")
    print(f"  -> {weights_path}")
    return {
        "kind": "convergence",
        "series": {a: [s[0].tolist(), s[1].tolist(), s[2].tolist()] for a, s in series.items()},
        "final": {a: final_costs(raw[a]) for a in raw},
    }


def exp_scenario(opts, runs_dir: Path) -> dict:
    algos = ["pq-frl", "q-frl", "dp-fedavg", "fedavg", "local-only"]
    groups = ["IID", "Non-IID\n(hotspot)"]
    series, raw = {}, {}
    for algo in algos:
        mean, err = [], []
        for scenario in ("iid", "hotspot"):
            costs = []
            for seed in range(opts.seeds):
                cfg = Config()
                apply_preset(cfg, opts.preset)
                apply_algorithm(cfg, algo)
                cfg.seed, cfg.rounds, cfg.scenario = seed, opts.rounds, scenario
                cfg.eval_every, cfg.eval_slots = opts.eval_every, opts.eval_slots
                costs.append(run_cached(cfg, runs_dir, opts.device, opts.threads)["eval"]["cost"])
                print(f"  {algo:10s} {scenario:8s} seed {seed} done", flush=True)
            m, s = mean_std(costs)
            mean.append(m)
            err.append(s)
            raw.setdefault(algo, {})[scenario] = costs
        series[algo] = (np.asarray(mean), np.asarray(err))
    path = plotting.plot_grouped_bars(groups, series, opts.out / "figures" / "demo_iid_vs_hotspot.png",
                                      ylabel="Converged system cost",
                                      title="Impact of Non-IID data on the converged cost")
    print(f"  -> {path}")
    return {"kind": "scenario", "groups": groups,
            "series": {a: [s[0].tolist(), s[1].tolist()] for a, s in series.items()},
            "raw": raw}


def exp_privacy(opts, runs_dir: Path) -> dict:
    eps_list = [0.5, 1.0, 2.0, 4.0]
    algos = ["pq-frl", "dp-fedavg", "dp-fedrep"]
    series, raw = {}, {}
    for algo in algos:
        mean, err, raw[algo] = [], [], []
        for eps in eps_list:
            costs = []
            for seed in range(opts.seeds):
                cfg = Config()
                apply_preset(cfg, opts.preset)
                apply_algorithm(cfg, algo)
                cfg.seed, cfg.rounds, cfg.scenario, cfg.eps = seed, opts.rounds, "hotspot", eps
                cfg.eval_every, cfg.eval_slots = opts.eval_every, opts.eval_slots
                costs.append(run_cached(cfg, runs_dir, opts.device, opts.threads)["eval"]["cost"])
                print(f"  {algo:10s} eps={eps:<4} seed {seed} done", flush=True)
            m, s = mean_std(costs)
            mean.append(m)
            err.append(s)
            raw[algo].append(costs)
        series[algo] = (np.asarray(mean), np.asarray(err))
    path = plotting.plot_curve(eps_list, series, opts.out / "figures" / "demo_privacy_budget.png",
                               xlabel="Per-round privacy budget  $\\epsilon$",
                               ylabel="Converged system cost",
                               title="Impact of the privacy budget",
                               xscale="log")
    print(f"  -> {path}")
    return {"kind": "privacy", "x": eps_list,
            "series": {a: [s[0].tolist(), s[1].tolist()] for a, s in series.items()},
            "raw": raw}


def exp_scalability(opts, runs_dir: Path) -> dict:
    n_list = [4, 8, 16]
    algos = ["pq-frl", "dp-fedavg", "dp-fedrep", "fedavg"]
    if opts.algos:
        algos = [a for a in algos if a in opts.algos]
    series, raw = {}, {}
    for algo in algos:
        mean, err, raw[algo] = [], [], []
        for n in n_list:
            costs = []
            for seed in range(opts.seeds):
                cfg = Config()
                apply_preset(cfg, opts.preset)
                apply_algorithm(cfg, algo)
                cfg.seed, cfg.rounds, cfg.scenario, cfg.n_uav = seed, opts.rounds, "hotspot", n
                cfg.eval_every, cfg.eval_slots = opts.eval_every, opts.eval_slots
                costs.append(run_cached(cfg, runs_dir, opts.device, opts.threads)["eval"]["cost"])
                print(f"  {algo:10s} N={n:<3} seed {seed} done", flush=True)
            m, s = mean_std(costs)
            mean.append(m)
            err.append(s)
            raw[algo].append(costs)
        series[algo] = (np.asarray(mean), np.asarray(err))
    path = plotting.plot_curve(n_list, series, opts.out / "figures" / "demo_scalability.png",
                               xlabel="Number of UAVs  $N$",
                               ylabel="Converged system cost",
                               title="Scalability with the swarm size")
    print(f"  -> {path}")
    return {"kind": "scalability", "x": n_list,
            "series": {a: [s[0].tolist(), s[1].tolist()] for a, s in series.items()},
            "raw": raw}


def exp_eta(opts, runs_dir: Path) -> dict:
    eta_list = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    costs, raw = [], []
    for eta in eta_list:
        runs = []
        for seed in range(opts.seeds):
            cfg = Config()
            apply_preset(cfg, opts.preset)
            apply_algorithm(cfg, "pq-frl")
            cfg.seed, cfg.rounds, cfg.scenario, cfg.eta = seed, opts.rounds, "hotspot", eta
            cfg.eval_every, cfg.eval_slots = opts.eval_every, opts.eval_slots
            runs.append(run_cached(cfg, runs_dir, opts.device, opts.threads)["eval"]["cost"])
            print(f"  pq-frl eta={eta:<5} seed {seed} done", flush=True)
        m, s = mean_std(runs)
        costs.append((m, s))
        raw.append(runs)
    mean = np.asarray([c[0] for c in costs])
    err = np.asarray([c[1] for c in costs])
    path = plotting.plot_curve(eta_list, {"pq-frl": (mean, err)},
                               opts.out / "figures" / "demo_eta_sensitivity.png",
                               xlabel="Softmax temperature  $\\eta$",
                               ylabel="Converged system cost",
                               title="Sensitivity to the weighting temperature",
                               xscale="log")
    print(f"  -> {path}")
    return {"kind": "eta", "x": eta_list,
            "series": {"pq-frl": [mean.tolist(), err.tolist()]}, "raw": raw}


def exp_components(opts, runs_dir: Path) -> dict:
    """Component-wise breakdown of the final state (paper Table II)."""
    metrics = {}
    for algo in ALGORITHMS:
        evals = []
        for seed in range(opts.seeds):
            cfg = Config()
            apply_preset(cfg, opts.preset)
            apply_algorithm(cfg, algo)
            cfg.seed, cfg.rounds, cfg.scenario = seed, opts.rounds, "hotspot"
            cfg.eval_every, cfg.eval_slots = 0, opts.eval_slots
            evals.append(run_cached(cfg, runs_dir, opts.device, opts.threads)["eval"])
            print(f"  {algo:10s} seed {seed} done", flush=True)
        metrics[algo] = {
            "cost": float(np.mean([e["cost"] for e in evals])),
            "latency": float(np.mean([e["latency"] for e in evals])),
            "energy_j": float(np.mean([e["energy_j"] for e in evals])),
            "tdr": float(np.mean([e["tdr"] for e in evals])),
        }
    path = plotting.plot_components(metrics, opts.out / "figures" / "demo_components.png")
    print(f"  -> {path}")
    return {"kind": "components", "metrics": metrics}


EXPERIMENTS = {
    "convergence": exp_convergence,
    "scenario": exp_scenario,
    "privacy": exp_privacy,
    "scalability": exp_scalability,
    "eta": exp_eta,
    "components": exp_components,
}


# ---------------------------------------------------------------------- reporting
def write_markdown(summary: Dict[str, dict], path: Path) -> None:
    lines = ["# Demo results", "",
             "Generated by `experiments.py` with the **demo** preset. Absolute values differ",
             "from the paper (reduced model/round budget); the relative ordering is the",
             "quantity under test.", ""]

    comp = summary.get("components", {}).get("metrics")
    if comp:
        lines += ["## Component-wise breakdown (paper Table II)", "",
                  "| Algorithm | Avg. cost | Avg. latency [s] | Avg. energy [J] | TDR [%] |",
                  "|---|---|---|---|---|"]
        for algo, m in comp.items():
            lines.append(f"| {plotting.ALGO_STYLE.get(algo, {}).get('label', algo)} "
                         f"| {m['cost']:.2f} | {m['latency']:.2f} | {m['energy_j']:.1f} "
                         f"| {100 * m['tdr']:.1f} |")
        lines.append("")

    if "privacy" in summary:
        s = summary["privacy"]
        lines += ["## Privacy budget sweep", "",
                  "| $\\epsilon_{round}$ | " + " | ".join(
                      plotting.ALGO_STYLE.get(a, {}).get("label", a) for a in s["series"]) + " |",
                  "|---|" + "---|" * len(s["series"])]
        for i, eps in enumerate(s["x"]):
            cells = [f"{s['series'][a][0][i]:.2f}" for a in s["series"]]
            lines.append(f"| {eps} | " + " | ".join(cells) + " |")
        lines.append("")

    if "scalability" in summary:
        s = summary["scalability"]
        lines += ["## Swarm-size sweep", "",
                  "| $N$ | " + " | ".join(
                      plotting.ALGO_STYLE.get(a, {}).get("label", a) for a in s["series"]) + " |",
                  "|---|" + "---|" * len(s["series"])]
        for i, n in enumerate(s["x"]):
            cells = [f"{s['series'][a][0][i]:.2f}" for a in s["series"]]
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")

    if "convergence" in summary:
        s = summary["convergence"]["final"]
        lines += ["## Final converged cost (Non-IID)", "",
                  "| Algorithm | Cost (mean ± std) |", "|---|---|"]
        for algo, (m, sd) in s.items():
            lines.append(f"| {plotting.ALGO_STYLE.get(algo, {}).get('label', algo)} "
                         f"| {m:.2f} ± {sd:.2f} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="PQ-FRL demo figures")
    p.add_argument("--figures", default="all",
                   help=f"comma separated subset of {FIGURES} or 'all'")
    p.add_argument("--preset", default="demo", choices=("demo", "paper"))
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-slots", type=int, default=80)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="results")
    p.add_argument("--full", action="store_true", help="larger run (more seeds/rounds)")
    p.add_argument("--force", action="store_true", help="ignore the run cache")
    p.add_argument("--algos", default="", help="comma separated algorithm subset for sweeps")
    opts = p.parse_args()

    if opts.full:
        opts.seeds = max(opts.seeds, 3)
        opts.rounds = max(opts.rounds, 300)
    opts.algos = tuple(a.strip() for a in opts.algos.split(',') if a.strip())
    opts.out = Path(opts.out)
    (opts.out / "figures").mkdir(parents=True, exist_ok=True)
    runs_dir = opts.out / "runs"
    if opts.force and runs_dir.exists():
        for f in runs_dir.glob("*.json"):
            f.unlink()

    names = FIGURES if opts.figures == "all" else tuple(
        n.strip() for n in opts.figures.split(",") if n.strip())
    summary: Dict[str, dict] = {}
    started = time.time()
    if names == ("none",):
        # rebuild summary.md from the JSON files already on disk
        for name in FIGURES:
            path = opts.out / f"{name}.json"
            if path.exists():
                summary[name] = json.loads(path.read_text(encoding="utf-8"))
        names = ()
    for name in names:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown figure {name!r}; choose from {FIGURES}")
        print(f"\n=== {name} ===", flush=True)
        summary[name] = EXPERIMENTS[name](opts, runs_dir)
        (opts.out / f"{name}.json").write_text(json.dumps(summary[name], indent=1),
                                               encoding="utf-8")
        print(f"  elapsed {time.time() - started:.0f} s", flush=True)

    write_markdown(summary, opts.out / "summary.md")
    print(f"\ndone in {time.time() - started:.0f} s -> {opts.out}/summary.md")


if __name__ == "__main__":
    main()
