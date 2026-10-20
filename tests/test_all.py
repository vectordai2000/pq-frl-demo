"""Fast sanity checks for the PQ-FRL demo (no pytest required).

Run with:  python tests/test_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pqfrl.agents import DDPGAgent, Actor
from pqfrl.aggregators import compute_weights
from pqfrl.config import Config, apply_algorithm
from pqfrl.dp import clip_vector, compose_eps, gaussian_sigma, privatize_vector
from pqfrl.env import UAVMECEnv
from pqfrl.server import FederatedTrainer


def test_env_dimensions():
    cfg = Config(n_uav=3, n_gu=10, k_local=4)
    env = UAVMECEnv(cfg, seed=0)
    state = env.reset(seed=0)
    assert state.shape == (3, 2 + 2 * 2 + 6 * 4)
    actions = np.full((3, env.action_dim), 0.5)
    state, rewards, info = env.step(actions)
    assert state.shape == (3, env.state_dim)
    assert rewards.shape == (3,)
    assert 0.0 <= info["tdr"] <= 1.0
    assert np.isfinite(info["cost"]) and np.isfinite(rewards).all()


def test_local_tasks_miss_deadline():
    """If every task is rejected and local computing is too slow, all tasks drop."""
    cfg = Config(t_max=0.01, cycles_min=5e9, cycles_max=6e9, f_loc=1e9, arrival_rate=50.0)
    env = UAVMECEnv(cfg, seed=1)
    env.reset(seed=1)
    actions = np.zeros((env.n, env.action_dim))
    actions[:, 0] = 0.0
    actions[:, 2: 2 + env.k_loc] = 0.0          # reject everything
    _, _, info = env.step(actions)
    assert info["tdr"] > 0.9, info["tdr"]
    assert info["cost"] > 0


def test_flight_power_model():
    cfg = Config()
    env = UAVMECEnv(cfg, seed=0)
    power = env.flight_power(np.array([0.0, 5.0, 10.0, 20.0]))
    assert np.all(power > 0)
    # hovering power is P0 + Pi (induced power at v = 0)
    assert abs(power[0] - (cfg.p0 + cfg.p_i)) < 1e-6
    # the rotary-wing model is non-monotonic: cruising is cheaper than hovering
    assert power[2] < power[0]


def test_dp_mechanism():
    vec = np.ones(1000)
    clipped = clip_vector(vec, 5.0)
    assert abs(np.linalg.norm(clipped) - 5.0) < 1e-9

    cfg = Config(eps=1.0, delta=1e-5, dp_noise_scale=0.05)
    rng = np.random.default_rng(0)
    noisy, stats = privatize_vector(clipped, 5.0, cfg, rng)
    assert stats["sigma"] == gaussian_sigma(5.0, cfg.eps, cfg.delta, dim=1000,
                                            noise_scale=cfg.dp_noise_scale)
    assert np.isfinite(noisy).all()

    assert compose_eps(1.0, 10, 1e-5) > compose_eps(1.0, 5, 1e-5) > 0.0


def test_aggregation_weights():
    uniform = compute_weights("uniform", [1.0, 2.0, 3.0], np.zeros(3), 0.5, 3.0)
    assert np.allclose(uniform, 1.0 / 3)

    weighted = compute_weights("reward", [0.0, 1.0, 2.0], np.zeros(3), 1.0, 3.0)
    assert np.allclose(weighted.sum(), 1.0)
    assert weighted[2] > weighted[1] > weighted[0]


def test_actor_action_ranges():
    actor = Actor(state_dim=10, action_dim=12, hidden=16, k_loc=5)
    out = actor(torch.zeros(4, 10))
    assert out.shape == (4, 12)
    assert torch.all(out[:, 0] >= 0) and torch.all(out[:, 0] <= 1)
    assert torch.all(out[:, 1].abs() <= 1)


def test_ddpg_learns_toy_target():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = Config(hidden=32, actor_lr=1e-3, critic_lr=1e-3, batch=32)
    agent = DDPGAgent(8, 4, cfg, torch.device("cpu"), 1)
    state = np.zeros(8, dtype=np.float32)
    target = np.array([0.9, 0.5, 0.8, 0.2])
    reward = lambda a: -float(np.sum((a - target) ** 2)) * 5.0
    for _ in range(1500):
        action, _ = agent.act(state, noise=0.3)
        agent.buffer.add(state, action, reward(action), state, False)
        agent.update()
    final, _ = agent.act(state, noise=0.0)
    assert reward(final) > -0.6, reward(final)


def test_short_training_run():
    torch.set_num_threads(1)
    cfg = Config()
    apply_algorithm(cfg, "pq-frl")
    cfg.rounds, cfg.tg, cfg.eval_every, cfg.eval_slots = 3, 2, 1, 5
    result = FederatedTrainer(cfg).run()
    assert len(result["history"]["cost"]) == 3
    assert result["runtime_s"] > 0
    assert np.isfinite(result["eval"]["cost"])


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
