"""Tests for ArmStore implementations."""

import sqlite3

import pytest

from thompson_bandits import InMemoryStore, SQLiteStore
from thompson_bandits.stores import ArmStore

ARM_IDS = ["a", "b", "c"]


# ------------------------------------------------------------------
# Shared store contract tests (parametrized over implementations)
# ------------------------------------------------------------------


@pytest.fixture
def memory_store():
    return InMemoryStore(arm_ids=ARM_IDS)


@pytest.fixture
def sqlite_store():
    conn = sqlite3.connect(":memory:")
    return SQLiteStore(conn, arm_ids=ARM_IDS)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, memory_store, sqlite_store):
    """Parametrized fixture that runs each test against both stores."""
    if request.param == "memory":
        return memory_store
    return sqlite_store


class TestArmStoreProtocol:
    def test_implements_protocol(self, store):
        assert isinstance(store, ArmStore)


class TestGetStats:
    def test_returns_arm(self, store):
        stats = store.get_stats("a")
        assert stats is not None
        assert stats.arm_id == "a"
        assert stats.alpha == 1.0
        assert stats.beta == 1.0
        assert stats.pulls == 0
        assert stats.total_reward == 0.0

    def test_returns_none_for_missing(self, store):
        assert store.get_stats("nonexistent") is None


class TestGetAllArms:
    def test_returns_all_sorted(self, store):
        arms = store.get_all_arms()
        assert len(arms) == 3
        assert [a.arm_id for a in arms] == ["a", "b", "c"]

    def test_empty_store(self):
        empty = InMemoryStore()
        assert empty.get_all_arms() == []


class TestUpdateStats:
    def test_increments_correctly(self, store):
        store.update_stats("a", alpha_delta=0.7, beta_delta=0.3, reward=0.7)
        stats = store.get_stats("a")
        assert stats is not None
        assert stats.alpha == pytest.approx(1.7)
        assert stats.beta == pytest.approx(1.3)
        assert stats.pulls == 1
        assert stats.total_reward == pytest.approx(0.7)

    def test_multiple_updates_accumulate(self, store):
        store.update_stats("b", alpha_delta=0.5, beta_delta=0.5, reward=0.5)
        store.update_stats("b", alpha_delta=0.8, beta_delta=0.2, reward=0.8)
        stats = store.get_stats("b")
        assert stats is not None
        assert stats.alpha == pytest.approx(1.0 + 0.5 + 0.8)
        assert stats.beta == pytest.approx(1.0 + 0.5 + 0.2)
        assert stats.pulls == 2
        assert stats.total_reward == pytest.approx(1.3)

    def test_raises_on_missing_arm(self, store):
        with pytest.raises(KeyError):
            store.update_stats("missing", 0.5, 0.5, 0.5)


class TestDecay:
    def test_multiplies_alpha_beta(self, store):
        # Set up known state
        store.update_stats("a", alpha_delta=1.0, beta_delta=1.0, reward=0.5)
        # Now alpha=2.0, beta=2.0
        store.decay("a", 0.5)
        stats = store.get_stats("a")
        assert stats is not None
        assert stats.alpha == pytest.approx(1.0)
        assert stats.beta == pytest.approx(1.0)
        # Pulls and total_reward unchanged by decay
        assert stats.pulls == 1
        assert stats.total_reward == pytest.approx(0.5)

    def test_preserves_mean(self, store):
        store.update_stats("a", alpha_delta=3.0, beta_delta=1.0, reward=0.75)
        before = store.get_stats("a")
        assert before is not None
        mean_before = before.alpha / (before.alpha + before.beta)

        store.decay("a", 0.8)
        after = store.get_stats("a")
        assert after is not None
        mean_after = after.alpha / (after.alpha + after.beta)

        assert mean_after == pytest.approx(mean_before, abs=1e-10)

    def test_raises_on_missing_arm(self, store):
        with pytest.raises(KeyError):
            store.decay("missing", 0.9)


# ------------------------------------------------------------------
# InMemoryStore-specific tests
# ------------------------------------------------------------------


class TestInMemoryStoreSpecific:
    def test_custom_priors(self):
        store = InMemoryStore(arm_ids=["x"], prior_alpha=2.0, prior_beta=3.0)
        stats = store.get_stats("x")
        assert stats is not None
        assert stats.alpha == 2.0
        assert stats.beta == 3.0

    def test_no_arm_ids_creates_empty(self):
        store = InMemoryStore()
        assert store.get_all_arms() == []


# ------------------------------------------------------------------
# SQLiteStore-specific tests
# ------------------------------------------------------------------


class TestSQLiteStoreSpecific:
    def test_from_path(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = SQLiteStore.from_path(db_path, arm_ids=["x", "y"])
        assert len(store.get_all_arms()) == 2
        stats = store.get_stats("x")
        assert stats is not None
        assert stats.alpha == 1.0

    def test_custom_table_name(self):
        conn = sqlite3.connect(":memory:")
        store = SQLiteStore(conn, arm_ids=["a"], table_name="my_arms")
        # Verify the custom table was created
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='my_arms'"
        ).fetchall()
        assert len(tables) == 1
        assert store.get_stats("a") is not None

    def test_idempotent_init(self):
        """Creating a SQLiteStore twice with the same arms doesn't duplicate."""
        conn = sqlite3.connect(":memory:")
        SQLiteStore(conn, arm_ids=["a", "b"])
        SQLiteStore(conn, arm_ids=["a", "b"])  # should not raise
        rows = conn.execute("SELECT COUNT(*) FROM bandit_arms").fetchone()
        assert rows[0] == 2

    def test_custom_priors(self):
        conn = sqlite3.connect(":memory:")
        store = SQLiteStore(
            conn, arm_ids=["x"], prior_alpha=2.0, prior_beta=3.0
        )
        stats = store.get_stats("x")
        assert stats is not None
        assert stats.alpha == 2.0
        assert stats.beta == 3.0

    def test_persistence_across_store_instances(self):
        """Data persists when a new SQLiteStore is created on the same connection."""
        conn = sqlite3.connect(":memory:")
        store1 = SQLiteStore(conn, arm_ids=["a"])
        store1.update_stats("a", alpha_delta=2.0, beta_delta=0.5, reward=0.8)

        store2 = SQLiteStore(conn)  # No arm_ids — should see existing data
        stats = store2.get_stats("a")
        assert stats is not None
        assert stats.alpha == pytest.approx(3.0)
        assert stats.pulls == 1
