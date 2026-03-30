"""Full integration tests: bandit + stores end-to-end."""

import sqlite3

import numpy as np
import pytest

from thompson_bandits import (
    BanditConfig,
    InMemoryStore,
    SQLiteStore,
    ThompsonBandit,
    cost_aware_reward,
)

ARM_IDS = ["strategy_a", "strategy_b", "strategy_c"]


class TestEndToEndInMemory:
    """Complete bandit lifecycle using InMemoryStore."""

    def test_full_lifecycle(self):
        store = InMemoryStore(arm_ids=ARM_IDS)
        bandit = ThompsonBandit(store, rng=np.random.default_rng(42))

        # Run 200 rounds: strategy_b is best
        rewards = {"strategy_a": 0.4, "strategy_b": 0.85, "strategy_c": 0.3}
        for _ in range(200):
            arm = bandit.select()
            bandit.update(arm, rewards[arm])

        summary = bandit.get_summary()
        assert summary.best_arm == "strategy_b"
        assert summary.total_pulls == 200

    def test_custom_priors_bias_exploration(self):
        """Strong prior on one arm biases early selections."""
        store = InMemoryStore(
            arm_ids=["weak", "strong"],
            prior_alpha=1.0,
            prior_beta=1.0,
        )
        # Manually boost "strong" arm's prior
        store._arms["strong"].alpha = 10.0
        store._arms["strong"].beta = 2.0

        bandit = ThompsonBandit(store, rng=np.random.default_rng(0))
        selections = [bandit.select() for _ in range(50)]
        # "strong" should be selected much more often early on
        strong_count = selections.count("strong")
        assert strong_count > 30


class TestEndToEndSQLite:
    """Complete bandit lifecycle using SQLiteStore."""

    def test_full_lifecycle(self):
        conn = sqlite3.connect(":memory:")
        store = SQLiteStore(conn, arm_ids=ARM_IDS)
        bandit = ThompsonBandit(store, rng=np.random.default_rng(42))

        rewards = {"strategy_a": 0.4, "strategy_b": 0.85, "strategy_c": 0.3}
        for _ in range(200):
            arm = bandit.select()
            bandit.update(arm, rewards[arm])

        summary = bandit.get_summary()
        assert summary.best_arm == "strategy_b"
        assert summary.total_pulls == 200

    def test_persistence_across_bandit_instances(self):
        """State survives creating a new bandit on the same store."""
        conn = sqlite3.connect(":memory:")
        store = SQLiteStore(conn, arm_ids=ARM_IDS)

        bandit1 = ThompsonBandit(store, rng=np.random.default_rng(42))
        for _ in range(50):
            bandit1.update("strategy_a", 0.9)

        # New bandit on same store
        bandit2 = ThompsonBandit(store, rng=np.random.default_rng(42))
        arm = bandit2.get_arm("strategy_a")
        assert arm is not None
        assert arm.pulls == 50
        assert arm.mean > 0.8

    def test_file_based_persistence(self, tmp_path):
        """State persists to disk and can be reloaded."""
        db_path = tmp_path / "bandit.db"

        # Session 1: create and train
        store1 = SQLiteStore.from_path(db_path, arm_ids=ARM_IDS)
        bandit1 = ThompsonBandit(store1, rng=np.random.default_rng(42))
        for _ in range(30):
            bandit1.update("strategy_b", 0.8)

        # Session 2: reopen from disk
        store2 = SQLiteStore.from_path(db_path)
        bandit2 = ThompsonBandit(store2)
        arm = bandit2.get_arm("strategy_b")
        assert arm is not None
        assert arm.pulls == 30


class TestDiscountedEndToEnd:
    """Discounted Thompson Sampling over a full lifecycle."""

    def test_adapts_to_regime_change(self):
        """Discounted bandit correctly switches preference after regime change."""
        store = InMemoryStore(arm_ids=["old_best", "new_best"])
        config = BanditConfig(discount=0.95)
        bandit = ThompsonBandit(
            store, config=config, rng=np.random.default_rng(42)
        )

        # Phase 1: old_best is better
        for _ in range(100):
            bandit.update("old_best", 0.9)
            bandit.update("new_best", 0.3)

        # Phase 2: new_best becomes better
        for _ in range(100):
            bandit.update("old_best", 0.3)
            bandit.update("new_best", 0.9)

        summary = bandit.get_summary()
        assert summary.best_arm == "new_best"


class TestCostAwareIntegration:
    """Integration of cost_aware_reward with the bandit."""

    def test_cost_efficient_arm_wins(self):
        """Arm with same raw reward but lower cost should win."""
        store = InMemoryStore(arm_ids=["cheap", "expensive"])
        bandit = ThompsonBandit(store, rng=np.random.default_rng(42))

        rng = np.random.default_rng(7)
        for _ in range(200):
            raw_reward = rng.uniform(0.5, 0.7)
            # "cheap" gets cost-adjusted boost
            bandit.update("cheap", cost_aware_reward(raw_reward, 1000, 2500))
            bandit.update(
                "expensive", cost_aware_reward(raw_reward, 5000, 2500)
            )

        summary = bandit.get_summary()
        assert summary.best_arm == "cheap"


class TestArmStatsProperties:
    """Verify ArmStats mean/variance match Beta distribution formulas."""

    def test_uniform_prior(self):
        from thompson_bandits import ArmStats

        arm = ArmStats(arm_id="test", alpha=1.0, beta=1.0)
        assert arm.mean == pytest.approx(0.5)
        assert arm.variance == pytest.approx(1.0 / 12.0)

    def test_strong_posterior(self):
        from thompson_bandits import ArmStats

        arm = ArmStats(arm_id="test", alpha=10.0, beta=2.0)
        assert arm.mean == pytest.approx(10.0 / 12.0)
        expected_var = (10.0 * 2.0) / (12.0**2 * 13.0)
        assert arm.variance == pytest.approx(expected_var)
