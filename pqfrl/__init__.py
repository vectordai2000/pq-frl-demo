"""PQ-FRL: a compact demo of privacy-preserving, quality-aware federated RL.

Reference: Z. Dai, H. Xie, S. Wei, C. Hu, "PQ-FRL: A Privacy-Preserving and
Quality-Aware Federated Reinforcement Learning for UAV-Assisted Edge Computing",
IEEE Internet of Things Journal, 2026.
"""

from .config import ALGORITHMS, ALGO_DEFAULTS, Config, apply_algorithm
from .env import UAVMECEnv
from .server import FederatedTrainer

__version__ = "0.1.0"
__all__ = ["Config", "ALGORITHMS", "ALGO_DEFAULTS", "apply_algorithm",
           "UAVMECEnv", "FederatedTrainer"]
