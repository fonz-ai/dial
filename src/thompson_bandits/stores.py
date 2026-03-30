"""Arm storage backends for Thompson Sampling bandits.

Defines the ArmStore protocol and two implementations:
  - InMemoryStore: dict-based, for testing and lightweight usage
  - SQLiteStore: persistent storage backed by a SQLite database
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable

from thompson_bandits.types import ArmStats


@runtime_checkable
class ArmStore(Protocol):
    """Protocol for bandit arm persistence.

    Implementations must provide get/update/list operations for arm statistics.
    All mutations are expected to be durable by the time the method returns.
    """

    def get_stats(self, arm_id: str) -> ArmStats | None:
        """Return stats for a single arm, or None if it doesn't exist."""
        ...

    def update_stats(
        self,
        arm_id: str,
        alpha_delta: float,
        beta_delta: float,
        reward: float,
    ) -> None:
        """Increment alpha, beta, pulls, and total_reward for an arm.

        Args:
            arm_id: Identifier for the arm.
            alpha_delta: Amount to add to alpha.
            beta_delta: Amount to add to beta.
            reward: The raw reward value (added to total_reward; pulls incremented by 1).
        """
        ...

    def get_all_arms(self) -> list[ArmStats]:
        """Return stats for all arms, sorted by arm_id."""
        ...

    def decay(self, arm_id: str, factor: float) -> None:
        """Multiply alpha and beta by factor (for discounted Thompson Sampling).

        Args:
            arm_id: Identifier for the arm.
            factor: Multiplicative decay factor in (0, 1).
        """
        ...


class InMemoryStore:
    """Dict-backed arm store. Fast, ephemeral, good for testing.

    Arms are created on first access with the specified priors.
    """

    def __init__(
        self,
        arm_ids: list[str] | None = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._arms: dict[str, ArmStats] = {}
        if arm_ids:
            for arm_id in arm_ids:
                self._arms[arm_id] = ArmStats(
                    arm_id=arm_id,
                    alpha=prior_alpha,
                    beta=prior_beta,
                )

    def get_stats(self, arm_id: str) -> ArmStats | None:
        return self._arms.get(arm_id)

    def update_stats(
        self,
        arm_id: str,
        alpha_delta: float,
        beta_delta: float,
        reward: float,
    ) -> None:
        arm = self._arms.get(arm_id)
        if arm is None:
            raise KeyError(f"Arm '{arm_id}' not found in store")
        arm.alpha += alpha_delta
        arm.beta += beta_delta
        arm.pulls += 1
        arm.total_reward += reward

    def get_all_arms(self) -> list[ArmStats]:
        return sorted(self._arms.values(), key=lambda a: a.arm_id)

    def decay(self, arm_id: str, factor: float) -> None:
        arm = self._arms.get(arm_id)
        if arm is None:
            raise KeyError(f"Arm '{arm_id}' not found in store")
        arm.alpha *= factor
        arm.beta *= factor


# ---------------------------------------------------------------------------
# SQL used by SQLiteStore
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS bandit_arms (
    arm_id       TEXT PRIMARY KEY,
    alpha        REAL NOT NULL DEFAULT 1.0,
    beta         REAL NOT NULL DEFAULT 1.0,
    pulls        INTEGER NOT NULL DEFAULT 0,
    total_reward REAL NOT NULL DEFAULT 0.0,
    last_updated TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class SQLiteStore:
    """SQLite-backed arm store for durable persistence.

    Creates a ``bandit_arms`` table in the provided database if it does not
    already exist. Each instance operates on a single SQLite connection.

    Can be initialized in two ways:

    1. **From a path** (owns the connection)::

           store = SQLiteStore.from_path("bandits.db", arm_ids=["a", "b"])

    2. **From an existing connection** (caller owns the connection)::

           store = SQLiteStore(conn, arm_ids=["a", "b"])
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        arm_ids: list[str] | None = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        table_name: str = "bandit_arms",
    ) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._table = table_name
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta

        # Ensure table exists
        self._conn.execute(
            _CREATE_TABLE_SQL.replace("bandit_arms", self._table)
        )
        self._conn.commit()

        # Seed arms if provided
        if arm_ids:
            for arm_id in arm_ids:
                self._conn.execute(
                    f"INSERT OR IGNORE INTO {self._table} "
                    "(arm_id, alpha, beta) VALUES (?, ?, ?)",
                    (arm_id, prior_alpha, prior_beta),
                )
            self._conn.commit()

    @classmethod
    def from_path(
        cls,
        db_path: str | Path,
        arm_ids: list[str] | None = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        table_name: str = "bandit_arms",
    ) -> "SQLiteStore":
        """Create a store from a database file path.

        The connection is created with WAL mode and foreign keys enabled.
        """
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return cls(
            conn,
            arm_ids=arm_ids,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            table_name=table_name,
        )

    def get_stats(self, arm_id: str) -> ArmStats | None:
        row = self._conn.execute(
            f"SELECT arm_id, alpha, beta, pulls, total_reward "
            f"FROM {self._table} WHERE arm_id = ?",
            (arm_id,),
        ).fetchone()
        if row is None:
            return None
        return ArmStats(
            arm_id=row["arm_id"],
            alpha=row["alpha"],
            beta=row["beta"],
            pulls=row["pulls"],
            total_reward=row["total_reward"],
        )

    def update_stats(
        self,
        arm_id: str,
        alpha_delta: float,
        beta_delta: float,
        reward: float,
    ) -> None:
        cursor = self._conn.execute(
            f"UPDATE {self._table} "
            "SET alpha = alpha + ?, beta = beta + ?, "
            "pulls = pulls + 1, total_reward = total_reward + ?, "
            "last_updated = datetime('now') "
            "WHERE arm_id = ?",
            (alpha_delta, beta_delta, reward, arm_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"Arm '{arm_id}' not found in store")
        self._conn.commit()

    def get_all_arms(self) -> list[ArmStats]:
        rows = self._conn.execute(
            f"SELECT arm_id, alpha, beta, pulls, total_reward "
            f"FROM {self._table} ORDER BY arm_id",
        ).fetchall()
        return [
            ArmStats(
                arm_id=row["arm_id"],
                alpha=row["alpha"],
                beta=row["beta"],
                pulls=row["pulls"],
                total_reward=row["total_reward"],
            )
            for row in rows
        ]

    def decay(self, arm_id: str, factor: float) -> None:
        cursor = self._conn.execute(
            f"UPDATE {self._table} "
            "SET alpha = alpha * ?, beta = beta * ? "
            "WHERE arm_id = ?",
            (factor, factor, arm_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"Arm '{arm_id}' not found in store")
        self._conn.commit()
