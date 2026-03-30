"""Core Thompson Sampling bandit logic.

No SQLite dependency -- all persistence goes through an ArmStore.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from thompson_bandits.types import ArmStats, ArmSummary, BanditConfig, BanditSummary

if TYPE_CHECKING:
    from thompson_bandits.stores import ArmStore


class ThompsonBandit:
    """Thompson Sampling bandit with Beta posteriors.

    Each arm maintains a Beta(alpha, beta) distribution. On ``select()``,
    a sample is drawn from every arm's posterior and the arm with the
    highest sample is returned. On ``update()``, the selected arm's
    posterior is updated with the observed reward.

    Supports *Discounted Thompson Sampling*: if ``config.discount`` is set,
    alpha and beta are decayed by that factor before each update, letting
    the bandit track non-stationary reward distributions.

    Example::

        from thompson_bandits import ThompsonBandit, InMemoryStore

        store = InMemoryStore(arm_ids=["a", "b", "c"])
        bandit = ThompsonBandit(store)

        arm = bandit.select()      # sample and pick best arm
        bandit.update(arm, 0.8)    # observe reward
    """

    def __init__(
        self,
        store: ArmStore,
        config: BanditConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.store = store
        self.config = config or BanditConfig()
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select(self) -> str:
        """Sample from each arm's Beta posterior and return the best arm_id.

        Raises:
            ValueError: If the store contains no arms.
        """
        arms = self.store.get_all_arms()
        if not arms:
            raise ValueError("No arms in store. Add arms before calling select().")

        samples = [
            (arm.arm_id, self.rng.beta(arm.alpha, arm.beta))
            for arm in arms
        ]
        return max(samples, key=lambda x: x[1])[0]

    def update(self, arm_id: str, reward: float) -> None:
        """Update the posterior for *arm_id* given an observed reward in [0, 1].

        If discounting is enabled, alpha and beta are decayed *before* the
        new observation is incorporated.

        Raises:
            ValueError: If reward is outside [0, 1].
        """
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"Reward must be in [0, 1], got {reward}")

        if self.config.discount is not None:
            self.store.decay(arm_id, self.config.discount)

        self.store.update_stats(
            arm_id=arm_id,
            alpha_delta=reward,
            beta_delta=1.0 - reward,
            reward=reward,
        )

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_arms(self) -> list[ArmStats]:
        """Return current stats for all arms."""
        return self.store.get_all_arms()

    def get_arm(self, arm_id: str) -> ArmStats | None:
        """Return stats for a single arm, or None if it doesn't exist."""
        return self.store.get_stats(arm_id)

    def get_summary(self) -> BanditSummary:
        """Return a serialization-friendly summary of bandit state."""
        arms = self.store.get_all_arms()
        return BanditSummary(
            arms=[
                ArmSummary(
                    arm_id=a.arm_id,
                    alpha=a.alpha,
                    beta=a.beta,
                    mean=round(a.mean, 4),
                    pulls=a.pulls,
                )
                for a in arms
            ],
            best_arm=max(arms, key=lambda a: a.mean).arm_id if arms else None,
            total_pulls=sum(a.pulls for a in arms),
        )


# ======================================================================
# Utility functions (domain-independent reward helpers)
# ======================================================================


def cost_aware_reward(
    raw_reward: float,
    cost: float,
    baseline_cost: float = 1.0,
) -> float:
    """Scale a reward by cost efficiency.

    ``reward_effective = raw_reward * (baseline_cost / cost)``

    The result is clamped to [0, 1] to remain valid for Beta updates.

    Args:
        raw_reward: Original reward in [0, 1].
        cost: Observed cost (e.g., tokens used, wall-clock time).
        baseline_cost: Expected cost for a "typical" episode. Episodes with
            lower cost get a boost; higher cost get penalized.

    Returns:
        Adjusted reward in [0, 1].
    """
    if cost <= 0:
        return raw_reward
    effective = raw_reward * (baseline_cost / cost)
    return max(0.0, min(1.0, effective))
