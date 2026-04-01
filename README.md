# dial

**Online weight optimization via Thompson Sampling.** Learns optimal configurations from outcome feedback — no grid search, no manual tuning. Converges in ~50 observations. [+41% NDCG@5](https://github.com/kusp-dev/retrieval-weight-experiment) over fixed-weight baselines in controlled experiments.

[![CI](https://github.com/fonz-ai/dial/actions/workflows/ci.yml/badge.svg)](https://github.com/fonz-ai/dial/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

```
pip install fonz-dial
```

## Quick start

```python
from thompson_bandits import ThompsonBandit, InMemoryStore

store = InMemoryStore(arm_ids=["relevance_heavy", "balanced", "recency_heavy"])
bandit = ThompsonBandit(store)

# Run the loop: select → observe → update
for query in queries:
    arm = bandit.select()
    reward = run_query(query, strategy=arm)
    bandit.update(arm, reward=reward)

print(bandit.get_summary())
```

After 50 iterations:

```
BanditSummary(
  best_arm='relevance_heavy',
  total_pulls=50,
  arms=[
    ArmSummary(arm_id='balanced',        mean=0.5765, pulls=11),
    ArmSummary(arm_id='recency_heavy',   mean=0.4210, pulls=8),
    ArmSummary(arm_id='relevance_heavy', mean=0.8903, pulls=31),
  ]
)
```

The bandit explores all three options early, then converges — 31 of 50 pulls on the winner, without you telling it which arm is best.

## Why Dial?

**vs. grid search / random search** — Those require running every combination upfront. Dial learns online, one observation at a time. No batch experiments needed.

**vs. manual tuning** — Manual weights are a guess that stays frozen. Dial adapts when the best option shifts — user behavior drifts, data distributions change, what worked in January fails in March.

**vs. contextual bandits (LinUCB, neural)** — Those need feature engineering and thousands of observations. Dial works with 50 observations and zero features. Start with Dial; graduate to contextual bandits when you have the data to justify them.

**vs. Bayesian optimization (Optuna, Ax)** — Those optimize over continuous parameter spaces. Dial optimizes over discrete options (strategies, presets, model choices). Different problem shape.

### Use cases

- **Retrieval weight tuning** — learn the optimal blend of relevance, recency, and importance for RAG systems
- **Model routing** — discover which LLM performs best for different query types
- **Prompt selection** — A/B test prompt variants with automatic convergence
- **Feature flag rollout** — promote variants based on measured outcomes
- **Any multi-option decision** where you can observe a reward signal

## Features

- **Beta posteriors** — each arm maintains a `Beta(alpha, beta)` distribution updated with observed rewards
- **Discounted Thompson Sampling** — optional decay factor for non-stationary environments where the best arm shifts over time
- **Cost-aware rewards** — built-in `cost_aware_reward()` scales outcomes by resource efficiency
- **Pluggable storage** — `InMemoryStore` for testing, `SQLiteStore` for persistence, or implement the `ArmStore` protocol for anything else
- **Zero SQLite dependency in core** — bandit logic talks only to the `ArmStore` protocol
- **Type-safe** — full annotations, `runtime_checkable` Protocol

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
print(summary.best_arm)      # 'relevance_heavy'
print(summary.total_pulls)   # 50

for arm in summary.arms:
    print(f"{arm.arm_id}: mean={arm.mean:.3f}, pulls={arm.pulls}")
# balanced:        mean=0.577, pulls=11
# recency_heavy:   mean=0.421, pulls=8
# relevance_heavy: mean=0.890, pulls=31
```

## Warm-start transfer

When you have prior knowledge (from a previous experiment, a related task, or domain expertise), encode it as informative priors instead of starting from uniform:

```python
from thompson_bandits import ThompsonBandit, InMemoryStore, BanditConfig

# Previous experiment found relevance_heavy won ~63% of pulls.
# Encode that as Beta(6.3, 3.7) instead of the default Beta(1, 1).
config = BanditConfig(prior_alpha=1.0, prior_beta=1.0)
store = InMemoryStore(arm_ids=["relevance_heavy", "balanced", "recency_heavy"])

# Override priors for the arm with known history
arm = store.get_stats("relevance_heavy")
arm.alpha = 6.3
arm.beta = 3.7

bandit = ThompsonBandit(store, config=config)
```

The bandit starts biased toward the prior winner but remains open to switching if the data disagrees. With shrinkage (e.g., scaling the prior by 0.15), the prior influence fades within ~20 observations.

## API reference

### ThompsonBandit

```python
ThompsonBandit(store: ArmStore, config: BanditConfig | None = None, rng: numpy.random.Generator | None = None)
```

The main entry point. Wraps any `ArmStore` backend with Thompson Sampling logic.

- **`select() -> str`** — Draws a sample from each arm's Beta posterior and returns the arm_id with the highest sample. Raises `ValueError` if the store is empty.
- **`update(arm_id: str, reward: float) -> None`** — Updates the posterior for the given arm. Reward must be in `[0, 1]`. If discounting is enabled, existing evidence is decayed before the new observation is applied.
- **`get_summary() -> BanditSummary`** — Returns a serialization-friendly snapshot of all arms.
- **`get_arm(arm_id: str) -> ArmStats | None`** — Returns stats for one arm, or `None`.
- **`get_arms() -> list[ArmStats]`** — Returns stats for all arms.

### BanditConfig

```python
BanditConfig(discount: float | None = None, prior_alpha: float = 1.0, prior_beta: float = 1.0)
```

- **`discount`** — Decay factor in `(0, 1)` for non-stationary environments. `None` disables discounting.
- **`prior_alpha`**, **`prior_beta`** — Initial Beta parameters for new arms. Defaults give a uniform `Beta(1, 1)` prior.

### ArmStats

```python
ArmStats(arm_id: str, alpha: float = 1.0, beta: float = 1.0, pulls: int = 0, total_reward: float = 0.0)
```

State of one arm. Also exposes two computed properties:

- **`mean -> float`** — Expected value: `alpha / (alpha + beta)`.
- **`variance -> float`** — Variance of the Beta distribution.

### BanditSummary / ArmSummary

```python
BanditSummary(arms: list[ArmSummary], best_arm: str | None, total_pulls: int)
ArmSummary(arm_id: str, alpha: float, beta: float, mean: float, pulls: int)
```

Returned by `get_summary()`. `best_arm` is the arm_id with the highest mean, or `None` if no arms exist.

### InMemoryStore

```python
InMemoryStore(arm_ids: list[str] | None = None, prior_alpha: float = 1.0, prior_beta: float = 1.0)
```

Dict-backed, ephemeral. Good for testing and short-lived processes.

### SQLiteStore

```python
SQLiteStore(conn: sqlite3.Connection, arm_ids: list[str] | None = None, prior_alpha: float = 1.0, prior_beta: float = 1.0, table_name: str = "bandit_arms")
SQLiteStore.from_path(db_path: str | Path, ...) -> SQLiteStore  # creates connection with WAL mode
```

Persistent storage. `from_path` owns the connection; the constructor form lets you pass an existing connection. Creates the table if it doesn't exist. Safe to re-initialize — existing arms are preserved (`INSERT OR IGNORE`).

### ArmStore protocol

```python
class ArmStore(Protocol):
    def get_stats(self, arm_id: str) -> ArmStats | None: ...
    def update_stats(self, arm_id: str, alpha_delta: float, beta_delta: float, reward: float) -> None: ...
    def get_all_arms(self) -> list[ArmStats]: ...
    def decay(self, arm_id: str, factor: float) -> None: ...
```

`runtime_checkable`. Implement these four methods on any class to create a custom backend — no inheritance required.

### cost_aware_reward

```python
cost_aware_reward(raw_reward: float, cost: float, baseline_cost: float = 1.0) -> float
```

Scales a reward by cost efficiency: `raw_reward * (baseline_cost / cost)`, clamped to `[0, 1]`. Cheaper-than-baseline episodes get a boost; more expensive ones get penalized. Zero cost passes through unchanged.

## Research

Dial extracts the Thompson Sampling engine from a research experiment on gradient-free retrieval weight learning. The experiment ran 1,200 episodes across 4 conditions on a $50/month API budget.

<details>
<summary>Citation (BibTeX)</summary>

```bibtex
@article{dirocco2026gradient,
  title   = {Gradient-Free Retrieval Weight Learning via Thompson Sampling
             with LLM Self-Assessment},
  author  = {DiRocco, Alfonso},
  year    = {2026},
  url     = {https://github.com/kusp-dev/retrieval-weight-experiment},
  note    = {1,200 episodes, 4 conditions, +41\% NDCG@5 over fixed baselines}
}
```

</details>

## Development

```bash
git clone https://github.com/fonz-ai/dial.git
cd dial
uv sync --extra dev
uv run pytest tests/ -v
uv run ruff check src/ tests/
```

## License

MIT
