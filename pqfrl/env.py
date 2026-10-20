"""UAV-assisted MEC environment.

Implements the system model of Section III of the paper:

* discrete time slots ``tau`` and rotary-wing flight energy (Eq. 4-6),
* OFDMA uplink with an LoS-dominated air-to-ground channel (Eq. 2-3),
* local / edge computing with a latency deadline ``Tmax`` (Sec. III-C),
* the immediate system cost ``C(t)`` (Eq. 7) and the shaped reward (Eq. 11).

The environment is deliberately small and dependency-free (NumPy only) so that
the whole FRL loop can be understood and modified in a few minutes.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .config import Config


class UAVMECEnv:
    """Multi-UAV MEC environment exposed as a vectorised multi-agent MDP."""

    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.n = int(cfg.n_uav)
        self.k = int(cfg.n_gu)
        self.k_loc = int(min(cfg.k_local, cfg.n_gu))

        # Pre-compute constants that never change.
        self.n0 = 1e-3 * 10 ** (cfg.n0_dbm / 10.0)     # noise PSD [W/Hz]
        self.g0 = 10 ** (cfg.g0_db / 10.0)             # channel gain at 1 m
        self.hover_power = cfg.p0 + cfg.p_i            # [W], used for reference
        self.a2 = cfg.d0 * cfg.rho_air * cfg.solidity * cfg.rotor_area

        self.state_dim = 2 + 2 * (self.n - 1) + 6 * self.k_loc
        self.action_dim = 2 + 2 * self.k_loc

        self.uav_pos = np.zeros((self.n, 2))
        self.gu_pos = np.zeros((self.k, 2))
        self.tasks = np.zeros((self.k, 2))             # (D_k, c_k)
        self.active = np.zeros(self.k, dtype=bool)
        self.time = 0
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset GU layout, UAV positions and the task stream."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.gu_pos = self._sample_gu(self.cfg.scenario)
        angles = np.arange(self.n) * 2.0 * np.pi / max(self.n, 1)
        self.uav_pos = np.stack(
            [0.5 * self.cfg.area + 0.35 * self.cfg.area * np.cos(angles),
             0.5 * self.cfg.area + 0.35 * self.cfg.area * np.sin(angles)],
            axis=1,
        )
        self.time = 0
        self._generate_tasks()
        return self._build_state()

    def _sample_gu(self, scenario: str) -> np.ndarray:
        cfg, area = self.cfg, self.cfg.area
        if scenario == "iid":
            pos = self.rng.uniform(0.05 * area, 0.95 * area, size=(self.k, 2))
        elif scenario == "hotspot":
            # Sec. V-A2: two clusters in the top-left and bottom-right quadrants.
            half = self.k // 2
            left = self.rng.uniform([0.05 * area, 0.55 * area],
                                    [0.45 * area, 0.95 * area], size=(half, 2))
            right = self.rng.uniform([0.55 * area, 0.05 * area],
                                     [0.95 * area, 0.45 * area], size=(self.k - half, 2))
            pos = np.concatenate([left, right], axis=0)
        else:
            raise ValueError(f"unknown scenario {scenario!r} (use 'iid' or 'hotspot')")
        return pos

    def _generate_tasks(self) -> None:
        """Bernoulli thinning of a Poisson task stream (rate per GU per second)."""
        cfg = self.cfg
        p = 1.0 - np.exp(-cfg.arrival_rate * cfg.slot)
        self.active = self.rng.random(self.k) < p
        self.tasks = np.stack(
            [self.rng.uniform(cfg.data_min, cfg.data_max, self.k),
             self.rng.uniform(cfg.cycles_min, cfg.cycles_max, self.k)],
            axis=1,
        )

    # ------------------------------------------------------------------ physics helpers
    def _dist3d(self, uav_xy: np.ndarray) -> np.ndarray:
        """3-D UAV-GU distances, shape (N, K)."""
        d2 = ((uav_xy[:, None, :] - self.gu_pos[None, :, :]) ** 2).sum(-1)
        return np.sqrt(d2 + self.cfg.altitude ** 2)

    def flight_power(self, v: np.ndarray) -> np.ndarray:
        """Rotary-wing propulsion power, Eq. (4)."""
        cfg = self.cfg
        v = np.abs(np.asarray(v, dtype=np.float64))
        profile = cfg.p0 * (1.0 + 3.0 * v ** 2 / cfg.u_tip ** 2)
        drag = 0.5 * self.a2 * v ** 3
        induced = cfg.p_i * np.sqrt(
            np.sqrt(1.0 + v ** 4 / (4.0 * cfg.v0 ** 4)) - v ** 2 / (2.0 * cfg.v0 ** 2)
        )
        return profile + drag + induced

    # ------------------------------------------------------------------ MDP interface
    def _candidates(self, dist: np.ndarray) -> np.ndarray:
        """Indices of the ``k_loc`` nearest GUs for every UAV, shape (N, k_loc)."""
        return np.argsort(dist, axis=1)[:, : self.k_loc]

    def _build_state(self) -> np.ndarray:
        cfg, area = self.cfg, self.cfg.area
        dist = self._dist3d(self.uav_pos)
        cand = self._candidates(dist)
        self._cand = cand

        state = np.zeros((self.n, self.state_dim), dtype=np.float32)
        state[:, 0:2] = self.uav_pos / area - 0.5
        for n in range(self.n):
            rel = (self.uav_pos - self.uav_pos[n]) / area
            mask = np.arange(self.n) != n
            state[n, 2: 2 + 2 * (self.n - 1)] = rel[mask].reshape(-1)

            idx = cand[n]
            base = 2 + 2 * (self.n - 1)
            state[n, base + 0::6] = (self.gu_pos[idx, 0] - self.uav_pos[n, 0]) / area
            state[n, base + 1::6] = (self.gu_pos[idx, 1] - self.uav_pos[n, 1]) / area
            state[n, base + 2::6] = dist[n, idx] / area
            state[n, base + 3::6] = self.active[idx]
            state[n, base + 4::6] = self.tasks[idx, 0] / cfg.data_max
            state[n, base + 5::6] = self.tasks[idx, 1] / cfg.cycles_max
        return state

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply one slot of joint actions.

        ``actions`` has shape (N, action_dim) with values already squashed by the
        actor: speed in [0, 1] (scaled by Vmax), heading in [-1, 1] (scaled by pi),
        offloading probabilities in [0, 1] and CPU-allocation demands in [0, 1].
        """
        cfg = self.cfg
        actions = np.asarray(actions, dtype=np.float64)
        vel = np.clip(actions[:, 0], 0.0, 1.0) * cfg.v_max
        heading = actions[:, 1] * np.pi
        accept = actions[:, 2: 2 + self.k_loc] >= 0.5
        f_demand = np.clip(actions[:, 2 + self.k_loc: 2 + 2 * self.k_loc], 0.0, 1.0)

        # -------- 1) assign every active task to the nearest UAV that accepts it
        owner = -np.ones(self.k, dtype=int)
        rank_of = np.zeros(self.k, dtype=int)
        for j in range(self.k_loc):
            for n in range(self.n):
                k = self._cand[n, j]
                if self.active[k] and owner[k] < 0 and accept[n, j]:
                    owner[k] = n
                    rank_of[k] = j
        served = np.where(owner >= 0)[0]

        # -------- 2) CPU allocation (capacity constraint sum_k f_k <= Fmax)
        f_alloc = np.zeros(self.k)
        f_used = np.zeros(self.n)
        for n in range(self.n):
            mine = served[owner[served] == n]
            if mine.size == 0:
                continue
            demand = f_demand[n, rank_of[mine]]
            if demand.sum() < 1e-6:          # degenerate preference -> equal split
                demand = np.ones_like(demand)
            share = demand / demand.sum()
            f_alloc[mine] = cfg.f_uav * share
            f_used[n] = f_alloc[mine].sum()

        # -------- 3) latency, deadline and energy of the slot
        dist = self._dist3d(self.uav_pos)
        local_latency = self.tasks[:, 1] / cfg.f_loc
        latency = local_latency.copy()
        dropped = np.zeros(self.k, dtype=bool)
        energy_comp = cfg.zeta * f_used ** 3 * cfg.slot
        for k in served:
            n = owner[k]
            n_served = max(int((owner[served] == n).sum()), 1)
            bandwidth = cfg.b_up / n_served                     # B_k = Bup / |K_n(t)|
            h = self.g0 * dist[n, k] ** (-cfg.alpha)            # Eq. (2)
            snr = cfg.p_tx * h / (self.n0 * bandwidth)
            rate = bandwidth * np.log2(1.0 + snr)               # Eq. (3)
            trans = self.tasks[k, 0] / max(rate, 1.0)
            edge = trans + self.tasks[k, 1] / max(f_alloc[k], 1.0)  # T^mec = trans + exec
            latency[k] = edge
            dropped[k] = edge > cfg.t_max                       # Tmax violation

        served_mask = np.zeros(self.k, dtype=bool)
        served_mask[served] = True
        local_mask = self.active & ~served_mask
        # paper Sec. III-C3: a task is dropped if it cannot be completed in time,
        # no matter whether it was computed locally or at the edge
        dropped[local_mask] = local_latency[local_mask] > cfg.t_max
        completed = self.active & ~dropped
        n_tasks_slot = max(int(self.active.sum()), 1)

        energy_fly = self.flight_power(vel) * cfg.slot          # Eq. (5)
        energy = energy_fly + energy_comp                       # Eq. (6)
        mean_energy = float(energy.mean())

        # -------- 4) reward shaping, Eq. (11)
        # Each UAV is also charged for the tasks in its own coverage cell that
        # nobody offloaded.  This extra alignment term (documented in the README)
        # makes the shaped local reward a decomposition of the global cost C(t),
        # so greedy "accept everything" is not rewarded when the UAV is congested,
        # and "reject everything" is not rewarded either.
        assoc = np.argmin(dist, axis=0)      # nearest UAV of every GU
        rewards = np.zeros(self.n)
        for n in range(self.n):
            mine = served[owner[served] == n]
            if mine.size:
                good = (latency[mine] < local_latency[mine]) & (~dropped[mine])
                rewards[n] += cfg.reward_success * float(good.sum())
                served_ok = mine[~dropped[mine]]
                rewards[n] -= cfg.w_lat * float(np.minimum(latency[served_ok], cfg.t_max).sum())
            local_mine = local_mask & (assoc == n)
            if local_mine.any():
                local_ok = local_mine & ~dropped
                rewards[n] -= cfg.w_lat * float(np.minimum(local_latency[local_ok], cfg.t_max).sum())
            failed = int(dropped[mine].sum() + dropped[local_mine].sum())
            rewards[n] -= cfg.w_drop * cfg.drop_penalty * failed / n_tasks_slot
            rewards[n] -= cfg.w_eng * (energy[n] / cfg.energy_unit)

        # soft anti-collision penalty (constraint d_nm >= dmin)
        pair = np.linalg.norm(self.uav_pos[:, None, :] - self.uav_pos[None, :, :], axis=-1)
        pair[np.diag_indices(self.n)] = np.inf
        violations = (pair < cfg.d_min).sum(axis=1)
        rewards -= cfg.collision_penalty * violations

        # -------- 5) global system cost, Eq. (7)
        if completed.any():
            cost_lat = float(latency[completed].mean())
        else:
            cost_lat = 0.0
        tdr = float(dropped.sum() / n_tasks_slot)
        cost = (cfg.w_lat * cost_lat
                + cfg.w_eng * (mean_energy / cfg.energy_unit)
                + cfg.w_drop * 10.0 * tdr)

        info = {
            "cost": cost,
            "latency": cost_lat,
            "energy_j": mean_energy,
            "energy_fly_j": float(energy_fly.mean()),
            "energy_comp_j": float(energy_comp.mean()),
            "tdr": tdr,
            "n_tasks": int(self.active.sum()),
            "n_dropped": int(dropped.sum()),
            "n_offloaded": int(served.size),
            "speed": float(vel.mean()),
            "cpu_load": float((f_used / cfg.f_uav).mean()),
            "collisions": int(violations.sum() // 2),
            # per-slot assignment details, handy for diagnostics and animations
            "active_gu": [int(k) for k in np.where(self.active)[0]],
            "links": [(int(k), int(owner[k])) for k in served],
            "dropped_gu": [int(k) for k in np.where(dropped)[0]],
        }

        # -------- 6) move and roll the environment forward
        self.uav_pos = self.uav_pos + np.stack(
            [vel * cfg.slot * np.cos(heading), vel * cfg.slot * np.sin(heading)], axis=1
        )
        self.uav_pos = np.clip(self.uav_pos, 1.0, cfg.area - 1.0)
        self.time += 1
        self._generate_tasks()
        return self._build_state(), rewards, info

    # ------------------------------------------------------------------ convenience
    def sample_action(self) -> np.ndarray:
        return self.rng.uniform(0.0, 1.0, size=(self.n, self.action_dim))

    def truth(self) -> Dict[str, Any]:
        """Ground-truth quantities used by the privacy diagnostics."""
        return {"uav_pos": self.uav_pos.copy(), "gu_pos": self.gu_pos.copy()}
