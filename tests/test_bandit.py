"""Core bandit tests using InMemoryStore."""

import numpy as np
import pytest

from thompson_bandits import (
    BanditConfig,
    InMemoryStore,
    ThompsonBandit,
    cost_aware_reward,
)

ARM_IDS = ["equal", "relevance_heavy", "recency_heavy"]


@pytest.fixture
def bandit():
    """Bandit with three arms and a fixed RNG seed."""
    store = InMemoryStore(arm_ids=ARM_IDS)
    return ThompsonBandit(store, rng=np.random.default_rng(42))


@pytest.fixture
def bandit_discounted():
    """Bandit with discount=0.9 and a fixed RNG seed."""
    store = InMemoryStore(arm_ids=ARM_IDS)
    config = BanditConfig(discount=0.9)
    return ThompsonBandit(store, config=config, rng=np.random.default_rng(42))


class TestArmSelection:
    def test_select_returns_valid_arm(self, bandit):
        selected = bandit.select()
        assert selected in ARM_IDS

    def test_all_arms_get_selected_eventually(self, bandit):
        """With uniform priors, all arms should be explored."""
        selected = set()
        for _ in range(100):
            selected.add(bandit.select())
        assert selected == set(ARM_IDS)

    def test_select_raises_on_empty_store(self):
        store = InMemoryStore(arm_ids=[])
        b = ThompsonBandit(store)
        with pytest.raises(ValueError, match="No arms"):
            b.select()

    def test_deterministic_with_same_seed(self):
        """Same seed produces same sequence of selections."""
        store1 = InMemoryStore(arm_ids=ARM_IDS)
        store2 = InMemoryStore(arm_ids=ARM_IDS)
        b1 = ThompsonBandit(store1, rng=np.random.default_rng(123))
        b2 = ThompsonBandit(store2, rng=np.random.default_rng(123))
        selections1 = [b1.select() for _ in range(20)]
        selections2 = [b2.select() for _ in range(20)]
        assert selections1 == selections2


class TestArmUpdate:
    def test_update_increments_alpha_beta(self, bandit):
        bandit.update("equal", 0.8)

        arm = bandit.get_arm("equal")
        assert arm is not None
        assert arm.alpha == pytest.approx(1.0 + 0.8)
        assert arm.beta == pytest.approx(1.0 + 0.2)
        assert arm.pulls == 1
        assert arm.total_reward == pytest.approx(0.8)

    def test_update_rejects_invalid_reward(self, bandit):
        with pytest.raises(ValueError, match="Reward must be in"):
            bandit.update("equal", 1.5)
        with pytest.raises(ValueError, match="Reward must be in"):
            bandit.update("equal", -0.1)

    def test_convergence_toward_best_arm(self, bandit):
        """After many high-reward updates, one arm should dominate."""
        for _ in range(100):
            bandit.update("relevance_heavy", 0.9)
            bandit.update("equal", 0.4)
            bandit.update("recency_heavy", 0.3)

        arms = bandit.get_arms()
        best = max(arms, key=lambda a: a.mean)
        assert best.arm_id == "relevance_heavy"
        assert best.mean > 0.8

    def test_update_unknown_arm_raises(self, bandit):
        with pytest.raises(KeyError, match="not_an_arm"):
            bandit.update("not_an_arm", 0.5)


class TestDiscountedThompsonSampling:
    def test_discount_reduces_alpha_beta(self, bandit_discounted):
        bandit_discounted.update("equal", 0.8)

        arm = bandit_discounted.get_arm("equal")
        assert arm is not None
        # Discounted: alpha = 1.0 * 0.9 + 0.8 = 1.7
        #             beta  = 1.0 * 0.9 + 0.2 = 1.1
        assert arm.alpha == pytest.approx(1.7)
        assert arm.beta == pytest.approx(1.1)

    def test_no_discount_standard_update(self, bandit):
        bandit.update("equal", 0.8)
        arm = bandit.get_arm("equal")
        assert arm is not None
        assert arm.alpha == pytest.approx(1.8)
        assert arm.beta == pytest.approx(1.2)

    def test_discount_enables_drift_adaptation(self):
        """Discounted TS adapts faster when the best arm shifts."""
        rng_seed = 99

        # Standard bandit (no discount)
        store_std = InMemoryStore(arm_ids=ARM_IDS)
        bandit_std = ThompsonBandit(
            store_std, rng=np.random.default_rng(rng_seed)
        )
        # Phase 1: "equal" is best
        for _ in range(50):
            bandit_std.update("equal", 0.9)
            bandit_std.update("relevance_heavy", 0.3)
        # Phase 2: "relevance_heavy" becomes best
        for _ in range(50):
            bandit_std.update("equal", 0.3)
            bandit_std.update("relevance_heavy", 0.9)

        arms_std = bandit_std.get_arms()
        equal_std = next(a for a in arms_std if a.arm_id == "equal")
        relev_std = next(a for a in arms_std if a.arm_id == "relevance_heavy")

        # Discounted bandit
        store_disc = InMemoryStore(arm_ids=ARM_IDS)
        config = BanditConfig(discount=0.95)
        bandit_disc = ThompsonBandit(
            store_disc, config=config, rng=np.random.default_rng(rng_seed)
        )
        for _ in range(50):
            bandit_disc.update("equal", 0.9)
            bandit_disc.update("relevance_heavy", 0.3)
        for _ in range(50):
            bandit_disc.update("equal", 0.3)
            bandit_disc.update("relevance_heavy", 0.9)

        arms_disc = bandit_disc.get_arms()
        equal_disc = next(a for a in arms_disc if a.arm_id == "equal")
        relev_disc = next(a for a in arms_disc if a.arm_id == "relevance_heavy")

        disc_gap = relev_disc.mean - equal_disc.mean
        std_gap = relev_std.mean - equal_std.mean
        assert disc_gap > std_gap, (
            f"Discounted bandit should adapt faster. "
            f"disc_gap={disc_gap:.4f}, std_gap={std_gap:.4f}"
        )

    def test_discount_preserves_mean_ratio(self):
        """Decaying alpha and beta by the same factor preserves alpha/(alpha+beta)."""
        store = InMemoryStore(arm_ids=ARM_IDS)
        bandit = ThompsonBandit(store)
        bandit.update("equal", 0.8)
        bandit.update("equal", 0.6)
        bandit.update("equal", 0.7)

        arm = bandit.get_arm("equal")
        assert arm is not None
        alpha_before = arm.alpha
        beta_before = arm.beta
        ratio_before = alpha_before / (alpha_before + beta_before)

        # Apply decay directly
        store.decay("equal", 0.95)

        arm_after = bandit.get_arm("equal")
        assert arm_after is not None
        ratio_after = arm_after.alpha / (arm_after.alpha + arm_after.beta)

        assert ratio_after == pytest.approx(ratio_before, abs=1e-10)
        # Absolute values should be smaller after decay
        assert arm_after.alpha < alpha_before
        assert arm_after.beta < beta_before

    def test_discount_invalid_value_rejected(self):
        with pytest.raises(ValueError, match="Discount must be in"):
            BanditConfig(discount=1.0)
        with pytest.raises(ValueError, match="Discount must be in"):
            BanditConfig(discount=0.0)
        with pytest.raises(ValueError, match="Discount must be in"):
            BanditConfig(discount=1.5)


class TestSummary:
    def test_summary_structure(self, bandit):
        summary = bandit.get_summary()
        assert len(summary.arms) == 3
        assert summary.total_pulls == 0
        assert summary.best_arm is not None

    def test_summary_after_updates(self, bandit):
        bandit.update("equal", 0.9)
        bandit.update("recency_heavy", 0.1)
        summary = bandit.get_summary()
        assert summary.total_pulls == 2
        assert summary.best_arm == "equal"


class TestCostAwareReward:
    def test_same_cost_same_reward(self):
        assert cost_aware_reward(0.6, 2500, 2500) == pytest.approx(0.6)

    def test_lower_cost_higher_reward(self):
        result = cost_aware_reward(0.5, 1250, 2500)
        assert result == pytest.approx(1.0)

    def test_higher_cost_lower_reward(self):
        result = cost_aware_reward(0.8, 5000, 2500)
        assert result == pytest.approx(0.4)

    def test_clamp_to_unit_interval(self):
        assert cost_aware_reward(0.9, 500, 2500) == 1.0
        assert cost_aware_reward(0.0, 500, 2500) == 0.0

    def test_zero_cost_passthrough(self):
        assert cost_aware_reward(0.7, 0, 2500) == 0.7
