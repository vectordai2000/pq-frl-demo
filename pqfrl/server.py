"""Federated training loop of PQ-FRL and its baselines (Algorithm 1)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from .agents import Actor, DDPGAgent, flat_params, load_flat_params
from .aggregators import compute_weights, diagnostics, update_reputation
from .config import ALGO_DEFAULTS, Config
from .dp import privatize_scalar, privatize_vector
from .env import UAVMECEnv


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass
class History:
    """Per-round training curves plus the final evaluation."""

    rounds: List[int] = field(default_factory=list)
    cost: List[float] = field(default_factory=list)
    latency: List[float] = field(default_factory=list)
    energy_j: List[float] = field(default_factory=list)
    tdr: List[float] = field(default_factory=list)
    weights: List[List[float]] = field(default_factory=list)
    rewards: List[List[float]] = field(default_factory=list)
    delta_norm: List[float] = field(default_factory=list)
    noise_norm: List[float] = field(default_factory=list)
    sigma: List[float] = field(default_factory=list)
    clip_norm: List[float] = field(default_factory=list)
    actor_loss: List[float] = field(default_factory=list)
    critic_loss: List[float] = field(default_factory=list)
    eval_rounds: List[int] = field(default_factory=list)
    eval_cost: List[float] = field(default_factory=list)
    eval_latency: List[float] = field(default_factory=list)
    eval_energy_j: List[float] = field(default_factory=list)
    eval_tdr: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return self.__dict__

    def final_cost(self, tail: float = 0.1) -> float:
        k = max(1, int(len(self.cost) * tail))
        return float(np.mean(self.cost[-k:]))


class FederatedTrainer:
    """Runs one algorithm (PQ-FRL or a baseline) on one environment instance."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.federated, self.weighting, self.use_dp = self._resolve_algorithm(cfg.algorithm)
        self.device = resolve_device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed + 17)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.env = UAVMECEnv(cfg, seed=cfg.seed)
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        self.n = self.env.n

        # all clients start from the same global actor (standard FRL initialisation)
        torch.manual_seed(cfg.seed + 7)
        self.global_actor = Actor(self.state_dim, self.action_dim,
                                  cfg.hidden, self.env.k_loc).to(self.device)
        self.agents = [
            DDPGAgent(self.state_dim, self.action_dim, cfg, self.device, self.env.k_loc)
            for _ in range(self.n)
        ]
        for agent in self.agents:
            agent.sync_from_global(self.global_actor)
            agent.actor_target.load_state_dict(self.global_actor.state_dict())

        self.reputation = np.zeros(self.n)
        self.clip_norm = float(cfg.clip_norm)
        self.history = History()
        self.runtime_s = 0.0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _resolve_algorithm(name: str):
        if name not in ALGO_DEFAULTS:
            raise ValueError(f"unknown algorithm {name!r}")
        federated = ALGO_DEFAULTS[name][2]
        weighting = ALGO_DEFAULTS[name][1]
        use_dp = ALGO_DEFAULTS[name][0]
        return federated, weighting, use_dp

    # ------------------------------------------------------------------ training
    def run(self) -> Dict[str, object]:
        cfg = self.cfg
        started = time.time()
        state = self.env.reset(seed=cfg.seed)
        if cfg.eval_every > 0:
            self._log_eval(0)
        for round_idx in range(1, cfg.rounds + 1):
            if self.federated:
                global_vec = flat_params(self.global_actor)
                for agent in self.agents:
                    agent.sync_from_global(self.global_actor)
                    agent.actor_target.load_state_dict(self.global_actor.state_dict())
            else:
                global_vec = None

            slot_cost, slot_lat, slot_eng, slot_tdr = [], [], [], []
            avg_reward = np.zeros(self.n)
            actor_loss, critic_loss = 0.0, 0.0
            for _ in range(cfg.tg):
                actions = np.zeros((self.n, self.action_dim), dtype=np.float32)
                for i, agent in enumerate(self.agents):
                    actions[i], _ = agent.act(state[i], noise=cfg.expl_noise)
                next_state, rewards, info = self.env.step(actions)
                for i, agent in enumerate(self.agents):
                    agent.buffer.add(state[i], actions[i], rewards[i], next_state[i])
                state = next_state
                avg_reward += rewards
                slot_cost.append(info["cost"])
                slot_lat.append(info["latency"])
                slot_eng.append(info["energy_j"])
                slot_tdr.append(info["tdr"])
            # local DDPG updates (the paper samples a batch every slot; the demo
            # performs `updates_per_round` steps after collecting the round)
            for _ in range(cfg.updates_per_round):
                for agent in self.agents:
                    al, cl = agent.update()
                    actor_loss += al
                    critic_loss += cl
            n_updates = max(cfg.updates_per_round, 1)
            actor_loss /= n_updates
            critic_loss /= n_updates
            avg_reward /= cfg.tg

            # ---------------- secure upload (P) + quality-aware aggregation (Q)
            delta_stats = {"delta_norm": 0.0, "noise_norm": 0.0, "sigma": 0.0}
            if self.federated:
                raw = [agent.delta(global_vec) for agent in self.agents]
                # the paper fixes Cclip = 5.0 for its larger models; a compact
                # demo network produces much smaller updates, so the bound is
                # calibrated from the first informative round unless --clip-norm
                # is supplied (the replay buffer needs a few rounds to warm up).
                norms = [float(np.linalg.norm(v)) for v in raw]
                renew = (self.clip_norm <= 0.0 and max(norms) > 1e-8) or (
                    cfg.clip_norm <= 0.0 and cfg.clip_refresh > 0
                    and round_idx % cfg.clip_refresh == 0 and max(norms) > 1e-8)
                if renew:
                    self.clip_norm = float(3.0 * np.median(norms) + 1e-12)
                bound = self.clip_norm if self.clip_norm > 0.0 else 1.0
                self.history.clip_norm.append(bound)
                deltas = []
                for vec in raw:
                    if self.use_dp:
                        vec, stats = privatize_vector(vec, bound, cfg, self.rng)
                        delta_stats["delta_norm"] += stats["delta_norm"] / self.n
                        delta_stats["noise_norm"] += stats["noise_norm"] / self.n
                        delta_stats["sigma"] = stats["sigma"]
                    else:
                        delta_stats["delta_norm"] += float(np.linalg.norm(vec)) / self.n
                    deltas.append(vec)

                scores = avg_reward.astype(float)
                if self.use_dp:
                    scores = np.array([
                        privatize_scalar(s, cfg.reward_clip, cfg, self.rng)[0] for s in scores
                    ])

                if self.weighting == "reputation":
                    if round_idx == 1:
                        self.reputation = np.zeros(self.n)
                    else:
                        self.reputation = update_reputation(
                            self.reputation, deltas, cfg.reputation_beta)

                weights = compute_weights(self.weighting, scores, self.reputation,
                                          cfg.eta, cfg.score_clip)
                update = np.zeros_like(global_vec)
                for w, dv in zip(weights, deltas):
                    update += w * dv
                new_vec = global_vec + update
                load_flat_params(self.global_actor, new_vec)

                self.history.weights.append([float(w) for w in weights])
                self.history.rewards.append([float(s) for s in scores])
                self.history.delta_norm.append(delta_stats["delta_norm"])
                self.history.noise_norm.append(delta_stats["noise_norm"])
                self.history.sigma.append(delta_stats["sigma"])
            else:
                self.history.weights.append([1.0 / self.n] * self.n)
                self.history.rewards.append([float(s) for s in avg_reward])
                self.history.delta_norm.append(0.0)
                self.history.noise_norm.append(0.0)
                self.history.sigma.append(0.0)
                self.history.clip_norm.append(0.0)

            self.history.rounds.append(round_idx)
            self.history.cost.append(float(np.mean(slot_cost)))
            self.history.latency.append(float(np.mean(slot_lat)))
            self.history.energy_j.append(float(np.mean(slot_eng)))
            self.history.tdr.append(float(np.mean(slot_tdr)))
            self.history.actor_loss.append(actor_loss / cfg.tg)
            self.history.critic_loss.append(critic_loss / cfg.tg)
            if cfg.eval_every > 0 and round_idx % cfg.eval_every == 0:
                self._log_eval(round_idx)

        self.runtime_s = time.time() - started
        evaluation = self.evaluate(record_trajectory=True)
        return {
            "algorithm": cfg.algorithm,
            "seed": cfg.seed,
            "scenario": cfg.scenario,
            "n_uav": cfg.n_uav,
            "eps": cfg.eps,
            "eta": cfg.eta,
            "rounds": cfg.rounds,
            "runtime_s": self.runtime_s,
            "history": self.history.as_dict(),
            "final_cost": self.history.final_cost(),
            "eval": evaluation,
        }

    # ------------------------------------------------------------------ evaluation
    def _log_eval(self, round_idx: int, slots: int | None = None) -> None:
        metrics = self.evaluate(slots=slots, record_trajectory=False)
        self.history.eval_rounds.append(round_idx)
        self.history.eval_cost.append(metrics["cost"])
        self.history.eval_latency.append(metrics["latency"])
        self.history.eval_energy_j.append(metrics["energy_j"])
        self.history.eval_tdr.append(metrics["tdr"])

    @torch.no_grad()
    def evaluate(self, slots: int | None = None, seed: int | None = None,
                 record_trajectory: bool = False) -> Dict[str, object]:
        """Deterministic rollout (no exploration noise, no learning).

        The evaluation environment re-uses the training GU layout but resets the
        swarm to its initial formation, so the metric measures the learned policy
        rather than the accidental position drift of the training rollouts.
        """
        cfg = self.cfg
        slots = int(slots or cfg.eval_slots)
        seed = cfg.seed if seed is None else int(seed)
        env = UAVMECEnv(cfg, seed=seed)
        state = env.reset(seed=seed)

        if self.federated:
            actors = [self.global_actor] * self.n
        else:
            actors = [agent.actor for agent in self.agents]

        trajectory = [env.uav_pos.copy()]
        costs, lats, engs, tdrs = [], [], [], []
        speeds, accepts, task_counts = [], [], []
        for _ in range(slots):
            actions = np.stack([
                actors[i](torch.as_tensor(state[i], dtype=torch.float32,
                                          device=self.device).unsqueeze(0)).cpu().numpy()[0]
                for i in range(self.n)
            ])
            state, _, info = env.step(actions)
            if record_trajectory:
                trajectory.append(env.uav_pos.copy())
            costs.append(info["cost"])
            lats.append(info["latency"])
            engs.append(info["energy_j"])
            tdrs.append(info["tdr"])
            speeds.append(info["speed"])
            task_counts.append(info["n_tasks"])
            if info["n_tasks"] > 0:
                accepts.append(info["n_offloaded"] / info["n_tasks"])

        return {
            "cost": float(np.mean(costs)),
            "latency": float(np.mean(lats)),
            "energy_j": float(np.mean(engs)),
            "tdr": float(np.mean(tdrs)),
            "speed": float(np.mean(speeds)),
            "offload_rate": float(np.mean(accepts)) if accepts else 0.0,
            "tasks_per_slot": float(np.mean(task_counts)),
            "gu_pos": env.gu_pos.tolist(),
            "trajectory": np.asarray(trajectory).tolist(),
        }
