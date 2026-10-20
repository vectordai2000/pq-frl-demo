"""DDPG actors/critics and the straight-through relaxation used by PQ-FRL.

The actor outputs a *mixed* action vector:

===========  =================  =============================================
slice        meaning            squashing
===========  =================  =============================================
0            speed v_n          sigmoid -> [0, 1] -> [0, Vmax]
1            heading theta_n    tanh    -> [-1, 1] -> [-pi, pi)
2 : 2+L      beta_{k,n}         sigmoid -> [0, 1]  (offloading probability)
2+L : 2+2L   f_{k,n}            sigmoid -> [0, 1]  (CPU demand share)
===========  =================  =============================================

``beta`` is binarised by the environment with a hard threshold.  We keep the
*continuous* value in the replay buffer and feed it to the critic, so the critic
gradient flows to the relaxed variable -- i.e. a straight-through estimator
(STE) for the non-differentiable threshold (Sec. IV-A).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    """Three-layer MLP with per-group output squashing."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int, k_loc: int):
        super().__init__()
        self.k_loc = k_loc
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        # small initial weights keep the initial policy close to deterministic
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.net(state)
        speed = torch.sigmoid(raw[:, 0:1])
        heading = torch.tanh(raw[:, 1:2])
        beta = torch.sigmoid(raw[:, 2: 2 + self.k_loc])
        cpu = torch.sigmoid(raw[:, 2 + self.k_loc:])
        return torch.cat([speed, heading, beta, cpu], dim=1)


class Critic(nn.Module):
    """Q(s, a) with the action injected at the first hidden layer."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.fc_s = nn.Linear(state_dim, hidden)
        self.fc_a = nn.Linear(action_dim, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.act = nn.ReLU()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc_s(state) + self.fc_a(action))
        return self.fc2(self.act(self.fc1(x)))


class ReplayBuffer:
    """Fixed-capacity FIFO buffer (the paper's D_n)."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity, action_dim), dtype=np.float32)
        self.r = np.zeros((capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, s, a, r, s2, done=False) -> None:
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch)
        to = lambda x: torch.as_tensor(x[idx], device=device)
        return to(self.s), to(self.a), to(self.r), to(self.s2), to(self.d)


# ---------------------------------------------------------------------- helpers
def flat_params(module: nn.Module) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()]).cpu().numpy().astype(np.float64)


def load_flat_params(module: nn.Module, vec: np.ndarray) -> None:
    with torch.no_grad():
        offset = 0
        for p in module.parameters():
            n = p.numel()
            p.copy_(torch.as_tensor(vec[offset: offset + n], dtype=p.dtype, device=p.device).view_as(p))
            offset += n


class DDPGAgent:
    """Local DDPG learner of one UAV (actor is federated, critic is private)."""

    def __init__(self, state_dim: int, action_dim: int, cfg, device: torch.device, k_loc: int):
        self.cfg = cfg
        self.device = device
        self.k_loc = k_loc
        self.actor = Actor(state_dim, action_dim, cfg.hidden, k_loc).to(device)
        self.actor_target = Actor(state_dim, action_dim, cfg.hidden, k_loc).to(device)
        self.critic = Critic(state_dim, action_dim, cfg.hidden).to(device)
        self.critic_target = Critic(state_dim, action_dim, cfg.hidden).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.buffer = ReplayBuffer(cfg.buffer, state_dim, action_dim)
        self.actor_loss = 0.0
        self.critic_loss = 0.0

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, state: np.ndarray, noise: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Return (action for the environment, action stored in the buffer)."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(s).cpu().numpy()[0]
        if noise > 0:
            a = a + np.random.normal(0.0, noise, size=a.shape)
            a[0] = np.clip(a[0], 0.0, 1.0)
            a[1] = np.clip(a[1], -1.0, 1.0)
            a[2:] = np.clip(a[2:], 0.0, 1.0)
        return a, a.copy()

    # ------------------------------------------------------------------ learning
    def update(self) -> Tuple[float, float]:
        if self.buffer.size < self.cfg.batch:
            return 0.0, 0.0
        s, a, r, s2, d = self.buffer.sample(self.cfg.batch, self.device)

        with torch.no_grad():
            a2 = self.actor_target(s2)
            target_q = self.critic_target(s2, a2)
            y = r + self.cfg.gamma * (1.0 - d) * target_q

        q = self.critic(s, a)
        critic_loss = nn.functional.mse_loss(q, y)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        self.actor_loss = float(actor_loss.item())
        self.critic_loss = float(critic_loss.item())
        return self.actor_loss, self.critic_loss

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        tau = self.cfg.tau
        with torch.no_grad():
            for tp, p in zip(target.parameters(), source.parameters()):
                tp.mul_(1.0 - tau).add_(p, alpha=tau)

    # ------------------------------------------------------------------ federation
    def sync_from_global(self, global_actor: Actor) -> None:
        self.actor.load_state_dict(global_actor.state_dict())

    def delta(self, global_vec: np.ndarray) -> np.ndarray:
        """Delta theta_n = theta_n - Theta_G  (Eq. 12 numerator)."""
        return flat_params(self.actor) - global_vec

    def apply_global(self, global_actor: Actor) -> None:
        self.sync_from_global(global_actor)
