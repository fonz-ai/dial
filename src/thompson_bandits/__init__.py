"""Thompson Sampling bandits with pluggable storage backends.

Quick start::

    from thompson_bandits import ThompsonBandit, InMemoryStore

    store = InMemoryStore(arm_ids=["a", "b", "c"])
    bandit = ThompsonBandit(store)

    arm = bandit.select()
    bandit.update(arm, reward=0.7)
"""

from thompson_bandits.bandit import ThompsonBandit, cost_aware_reward
from thompson_bandits.stores import ArmStore, InMemoryStore, SQLiteStore
from thompson_bandits.types import ArmStats, ArmSummary, BanditConfig, BanditSummary

__all__ = [
    "ThompsonBandit",
    "ArmStore",
    "InMemoryStore",
    "SQLiteStore",
    "ArmStats",
    "ArmSummary",
    "BanditConfig",
    "BanditSummary",
    "cost_aware_reward",
]
