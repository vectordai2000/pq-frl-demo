"""Configuration objects for the PQ-FRL demo.

The defaults mirror Table I of the paper as closely as a CPU-sized demo
allows; every deviation is marked with ``# demo:`` and explained in the README.
All physical units are SI (m, s, Hz, W, J, bit, cycle).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict


@dataclass
class Config:
    """Flat configuration shared by the environment, agents and FRL server."""

    # ------------------------------------------------------------------ experiment
    algorithm: str = "pq-frl"       # pq-frl | q-frl | dp-fedavg | dp-fedrep | fedavg | local-only
    seed: int = 0
    rounds: int = 200               # FRL aggregation rounds (paper: 5000)
    tg: int = 10                    # time slots per aggregation round
    eval_slots: int = 80            # slots used by the deterministic evaluation
    eval_every: int = 20            # log a deterministic evaluation every N rounds
    device: str = "cpu"             # cpu | cuda | auto (CPU is faster for this tiny model)
    out_dir: str = "results/run"

    # ------------------------------------------------------------------ scenario
    scenario: str = "hotspot"       # iid | hotspot  (paper Sec. V-A)
    n_uav: int = 4                  # N
    n_gu: int = 20                  # K
    area: float = 1000.0            # side length of the square service area [m]
    altitude: float = 100.0         # H [m]
    slot: float = 0.1               # tau [s]
    v_max: float = 20.0             # Vmax [m/s]
    d_min: float = 50.0             # dmin [m]
    k_local: int = 5                # demo: candidate GUs per UAV (paper uses the dynamic set K_n(t))

    # ------------------------------------------------------------------ compute
    f_uav: float = 10e9             # Fmax [cycles/s]
    f_loc: float = 1e9              # Floc [cycles/s]
    t_max: float = 0.5              # Tmax [s] (demo: 0.5 s so the deadline binds; paper: 5.0 s)
    zeta: float = 1e-28             # effective capacitance coefficient
    cycles_min: float = 0.5e9       # c_k range [cycles]
    cycles_max: float = 1.0e9

    # ------------------------------------------------------------------ communication
    b_up: float = 10e6              # Bup [Hz]
    p_tx: float = 1.0               # Pk [W]  (30 dBm)
    n0_dbm: float = -174.0          # noise PSD [dBm/Hz]
    g0_db: float = -40.0            # channel gain at 1 m [dB]
    alpha: float = 2.0              # path-loss exponent

    # ------------------------------------------------------------------ rotary-wing energy
    p0: float = 79.86               # blade profile power [W]
    p_i: float = 88.63              # induced power in hover [W]
    u_tip: float = 120.0            # tip speed of rotor blade [m/s]
    v0: float = 4.03                # mean rotor induced velocity in hover [m/s]
    d0: float = 0.6                 # fuselage drag ratio
    rho_air: float = 1.225          # air density [kg/m^3]
    solidity: float = 0.05          # rotor solidity
    rotor_area: float = 0.503       # rotor disk area [m^2]

    # ------------------------------------------------------------------ task / reward
    arrival_rate: float = 2.0       # demo: tasks/s per GU (paper: 0.5)
    data_min: float = 1.0e6         # Dk range [bit]
    data_max: float = 2.0e6
    reward_success: float = 2.0     # Rsuccess in Eq. (11)
    drop_penalty: float = 5.0       # Pdrop in Eq. (11)
    collision_penalty: float = 0.5  # demo: soft penalty for violating dmin
    w_lat: float = 0.5              # wT
    w_eng: float = 0.3              # wE
    w_drop: float = 0.2             # wP
    energy_unit: float = 10.0       # demo: cost/reward energy normalisation [J] (see README)

    # ------------------------------------------------------------------ DDPG
    hidden: int = 64                # demo: paper uses (256, 128)
    actor_lr: float = 1e-3          # demo: paper uses 1e-4 (CPU-friendly)
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005              # soft target update
    batch: int = 64
    buffer: int = 20000
    expl_noise: float = 0.10        # Gaussian action noise during local training
    updates_per_round: int = 10     # gradient steps per client per aggregation round

    # ------------------------------------------------------------------ differential privacy
    eps: float = 1.0                # per-round privacy budget
    delta: float = 1e-5
    clip_norm: float = 0.0          # Cclip; 0 = adaptive (paper: 5.0)
    clip_refresh: int = 20          # demo: recalibrate the adaptive bound every N rounds
    reward_clip: float = 1.0        # CR, applied to the average local reward
    dp_noise_scale: float = 0.05     # 1.0 = textbook Gaussian mechanism (see README)

    # ------------------------------------------------------------------ aggregation
    weighting: str = "reward"       # reward | uniform | reputation
    eta: float = 0.5                # softmax temperature (Eq. 14)
    reputation_beta: float = 0.9    # EMA factor of DP-FedRep
    score_clip: float = 3.0         # clip the noisy reward before softmax (paper's safeguard)

    # ------------------------------------------------------------------ bookkeeping
    log_every: int = 10
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in payload.items() if k in known}
        return cls(**clean)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


ALGORITHMS = ("pq-frl", "q-frl", "dp-fedavg", "dp-fedrep", "fedavg", "local-only")

ALGO_DEFAULTS = {
    # name: (differential privacy, weighting rule, federated?)
    "pq-frl": (True, "reward", True),
    "q-frl": (False, "reward", True),
    "dp-fedavg": (True, "uniform", True),
    "dp-fedrep": (True, "reputation", True),
    "fedavg": (False, "uniform", True),
    "local-only": (False, "uniform", False),
}


# Two parameter sets: the faithful Table I of the paper and the reduced demo
# used for the published figures (CPU-sized, deadline kept binding).
PRESETS = {
    "paper": {
        "n_uav": 4, "n_gu": 20, "area": 1000.0, "altitude": 100.0, "slot": 0.1,
        "v_max": 20.0, "d_min": 50.0, "f_uav": 10e9, "f_loc": 1e9, "t_max": 5.0,
        "b_up": 10e6, "arrival_rate": 0.5, "hidden": 256,
        "actor_lr": 1e-4, "critic_lr": 1e-3, "rounds": 5000, "tg": 10,
        "eval_every": 50, "eval_slots": 200, "eps": 1.0, "eta": 0.5,
        "dp_noise_scale": 1.0, "clip_norm": 5.0,
    },
    "demo": {
        "n_uav": 4, "n_gu": 20, "area": 1000.0, "altitude": 100.0, "slot": 0.1,
        "v_max": 20.0, "d_min": 50.0, "f_uav": 10e9, "f_loc": 1e9, "t_max": 0.5,
        "b_up": 10e6, "arrival_rate": 2.0, "hidden": 64,
        "actor_lr": 1e-3, "critic_lr": 1e-3, "rounds": 200, "tg": 10,
        "eval_every": 20, "eval_slots": 80, "eps": 1.0, "eta": 0.5,
        "dp_noise_scale": 0.05, "clip_norm": 0.0,
        "reward_clip": 1.0, "clip_refresh": 20,
    },
}


def apply_preset(cfg: Config, name: str) -> Config:
    """Overwrite ``cfg`` with a named parameter set."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {tuple(PRESETS)}")
    for key, value in PRESETS[name].items():
        setattr(cfg, key, value)
    return cfg


def apply_algorithm(cfg: Config, algorithm: str) -> Config:
    """Set the privacy / weighting switches implied by an algorithm name."""
    if algorithm not in ALGO_DEFAULTS:
        raise ValueError(f"unknown algorithm {algorithm!r}; choose from {ALGORITHMS}")
    _, weighting, _ = ALGO_DEFAULTS[algorithm]
    cfg.algorithm = algorithm
    cfg.weighting = weighting
    return cfg
