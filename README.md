# dial

**Online weight optimization via Thompson Sampling.** Learns optimal configurations from outcome feedback — no grid search, no manual tuning.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-47_passing-brightgreen.svg)]()

```
pip install kusp-dial
```

## What it does

Dial treats configuration selection as an online learning problem. You define options ("arms"), observe outcomes, and the system learns which option works best — adapting over time as conditions change.

```python
from thompson_bandits import ThompsonBandit, InMemoryStore

store = InMemoryStore(arm_ids=["strategy_a", "strategy_b", "strategy_c"])
bandit = ThompsonBandit(store)

arm = bandit.select()           # pick the best option (exploration + exploitation)
bandit.update(arm, reward=0.8)  # observe outcome, update beliefs
```

50 observations to converge. Zero hyperparameters to set.

## Why

Most systems treat configuration as a one-time decision. Set weights, pick a strategy, move on. But the best choice changes — user behavior drifts, data distributions shift, what worked in January fails in March.

Dial uses [Thompson Sampling](https://en.wikipedia.org/wiki/Thompson_sampling) (1933) to balance exploration and exploitation automatically. Each option maintains a Beta distribution that gets sharper with evidence. The system samples from these distributions to make decisions, naturally exploring uncertain options while exploiting known winners.

### Use cases

- **Retrieval weight tuning** — learn the optimal blend of relevance, recency, and importance for RAG systems
- **Model routing** — discover which LLM performs best for different query types
- **Prompt selection** — A/B test prompt variants with automatic convergence
- **Feature flags** — gradual rollout with reward-based promotion
- **Any multi-option decision** where you can measure outcomes

## Features

- **Beta posteriors** — each arm maintains a `Beta(alpha, beta)` distribution updated with observed rewards
- **Discounted Thompson Sampling** — optional decay factor tracks non-stationary environments where the best arm shifts over time
- **Cost-aware rewards** — built-in `cost_aware_reward()` scales outcomes by resource efficiency
- **Pluggable storage** — `InMemoryStore` for testing, `SQLiteStore` for persistence, or implement the `ArmStore` protocol for anything else
- **Dependency injection** — core bandit logic has zero SQLite dependency
- **Type-safe** — full type annotations, `runtime_checkable` Protocol

## Storage backends

### In-memory (ephemeral)

```python
from thompson_bandits import InMemoryStore

store = InMemoryStore(arm_ids=["a", "b", "c"], prior_alpha=1.0, prior_beta=1.0)
```

### SQLite (persistent)

```python
from thompson_bandits import SQLiteStore

# From a file path (store owns the connection)
store = SQLiteStore.from_path("bandits.db", arm_ids=["a", "b", "c"])

# From an existing connection (you own the connection)
import sqlite3
conn = sqlite3.connect("bandits.db")
store = SQLiteStore(conn, arm_ids=["a", "b", "c"])
```

### Custom storage

Implement the `ArmStore` protocol — any class with the right methods works, no inheritance required:

```python
from thompson_bandits import ArmStore, ArmStats

class RedisStore:
    def get_stats(self, arm_id: str) -> ArmStats | None: ...
    def update_stats(self, arm_id: str, alpha_delta: float, beta_delta: float, reward: float) -> None: ...
    def get_all_arms(self) -> list[ArmStats]: ...
    def decay(self, arm_id: str, factor: float) -> None: ...
```

## Non-stationary environments

When the best option changes over time, enable discounting:

```python
from thompson_bandits import ThompsonBandit, InMemoryStore, BanditConfig

config = BanditConfig(discount=0.95)  # decay factor in (0, 1)
bandit = ThompsonBandit(store, config=config)
```

Before each update, existing evidence is decayed by the discount factor. Recent observations carry more weight than old ones.

## Cost-aware optimization

When options have different costs (tokens, latency, dollars), scale rewards accordingly:

```python
from thompson_bandits import cost_aware_reward

raw_reward = 0.9
token_cost = 1500
baseline_cost = 1000

adjusted = cost_aware_reward(raw_reward, cost=token_cost, baseline_cost=baseline_cost)
bandit.update(arm, reward=adjusted)
```

## Inspecting state

```python
summary = bandit.get_summary()
print(summary.best_arm)      # arm with highest posterior mean
print(summary.total_pulls)   # total observations across all arms

for arm in summary.arms:
    print(f"{arm.arm_id}: mean={arm.mean:.3f}, pulls={arm.pulls}")
```

## Research

Dial extracts the Thompson Sampling engine from a published research experiment on gradient-free retrieval weight learning:

> DiRocco, A. (2026). *Gradient-Free Retrieval Weight Learning via Thompson Sampling with LLM Self-Assessment.* [kusp-dev/retrieval-weight-experiment](https://github.com/kusp-dev/retrieval-weight-experiment)

The experiment ran 1,200 episodes across 4 conditions and demonstrated that Thompson Sampling converges to effective retrieval weight configurations in ~50 queries, achieving +41% NDCG@5 over fixed-weight baselines.

## Development

```bash
git clone https://github.com/fonz-ai/dial.git
cd dial
pip install -e ".[dev]"
pytest
```

## License

MIT
