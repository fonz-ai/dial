"""Shared types for Thompson Sampling bandits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ArmStats:
    """State of one arm in the bandit.

    Represents a Beta(alpha, beta) posterior distribution.
    """

    arm_id: str
    alpha: float = 1.0
    beta: float = 1.0
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean(self) -> float:
        """Expected value of the Beta distribution: alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of the Beta distribution."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))


@dataclass
class BanditConfig:
    """Configuration for a Thompson Sampling bandit.

    Attributes:
        discount: Optional decay factor in (0, 1) for Discounted Thompson Sampling.
            Before each update, existing alpha and beta are multiplied by this factor,
            allowing the bandit to adapt when the best arm shifts over time.
            None means standard (non-discounted) Thompson Sampling.
        prior_alpha: Initial alpha for new arms (default 1.0 = uniform prior).
        prior_beta: Initial beta for new arms (default 1.0 = uniform prior).
    """

    discount: float | None = None
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def __post_init__(self) -> None:
        if self.discount is not None and not (0.0 < self.discount < 1.0):
            raise ValueError(f"Discount must be in (0, 1), got {self.discount}")
        if self.prior_alpha <= 0:
            raise ValueError(f"prior_alpha must be positive, got {self.prior_alpha}")
        if self.prior_beta <= 0:
            raise ValueError(f"prior_beta must be positive, got {self.prior_beta}")


@dataclass
class BanditSummary:
    """Summary of bandit state for logging and analysis."""

    arms: list[ArmSummary]
    best_arm: str | None
    total_pulls: int


@dataclass
class ArmSummary:
    """Summary of a single arm."""

    arm_id: str
    alpha: float
    beta: float
    mean: float
    pulls: int
