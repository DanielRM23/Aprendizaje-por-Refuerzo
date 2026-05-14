# Aprendizaje por Refuerzo — IIMAS, UNAM (2025-2)

Graduate coursework in Reinforcement Learning at the Instituto de Investigaciones en
Matemáticas Aplicadas y en Sistemas (IIMAS), UNAM. All implementations are in Python
using NumPy, PyTorch, Gymnasium, and MinAtar.

---

## Topics & Assignments

### Task 1 — Dynamic Programming (100/100)
Implementation of classical DP algorithms for MDPs from scratch:
- **Value Iteration** and **Policy Iteration** applied to two environments:
  - **FrozenLake** (Gymnasium, `is_slippery=True`)
  - **Tic-Tac-Toe** against a random opponent (γ=1)
- Explicit reporting of optimal value function **v\*(s)** and optimal policy **π\*(a|s)**
  for each algorithm/environment combination.

### Task 2 — Model-Free & Model-Based RL (90/100)
Implementation and benchmarking of two algorithms:
- **Actor-Critic with baselines** and linear function approximation
- **Dyna-Q** (model-based planning)

Trained and evaluated on three environments:
- **MountainCar**, **CartPole**, **Breakout** (MinAtar)

Results reported as mean ± std over 10 independent experiments per
(algorithm, environment) pair, with reward curves (steps vs. average reward).
Includes critical reviews of four landmark RL papers (Q-Learning, AlphaGo,
Rainbow, Dyna-2).

### Task 3 — Deep RL & Evolutionary Methods (90/100)
From-scratch implementation of three algorithms:
- **DDQN** (Double Deep Q-Network)
- **PPO** (Proximal Policy Optimization)
- **NES/GA** (Natural Evolution Strategies / Genetic Algorithm)

Benchmarked on **MinAtar Breakout** and an **Atari** environment. Each method
compared against its Stable Baselines3 counterpart. Full statistical comparison
following rigorous methodology: multiple independent runs, convergence plots,
descriptive statistics, **Wilcoxon rank-sum test** for pairwise comparison, and
**critical difference diagrams** for global ranking.

### Task 4 — Multi-Objective Reinforcement Learning (85/100)
Implementation of multi-objective optimization and RL algorithms:
- **NSGA-III** (non-dominated sorting genetic algorithm)
- **Pareto Q-learning** (scalarized multi-objective RL)
- **Descent method** for multi-objective optimization
- **NSGA-II**, **MOEA/D**, **SMS-EMOA**, **MONES** *(extras)*

Tested on **WFG1** (n=24, k=3) and **MO-LunarLander** (mo-gymnasium).
Compared against Pymoo (NSGA-III) and MORL-Baselines (Pareto Q-learning)
using **hypervolume** as the performance metric. Statistical comparison
following the same rigorous protocol as Task 3.

### Task 5 — Custom Environments & Full Benchmark (425/100)
Open-ended capstone task combining environment design and algorithm comparison:
- **Adapted a single-objective environment to multi-objective** setting
- **Designed a custom environment** with numerical reward signals
- **Full comparative benchmark** of all methods developed during the course on
  both custom environments
- *Extra:* training from expert demonstrations (imitation learning)
- *Extra:* video explanation of a course exercise
- *Extra:* novel RL method with original components
- *Extra:* replication of a published RL paper

---

## Final Project — Multi-Objective Portfolio Optimization with RL
Stock portfolio optimization framed as a **multi-objective reinforcement learning**
problem, combining FinRL's trading framework with NSGA-III for Pareto-optimal
portfolio construction.

### Problem Setup
- **Objective 1:** Maximize cumulative return
- **Objective 2:** Minimize portfolio risk (volatility)
- Assets: S&P500 stocks downloaded via FinRL/Yahoo Finance
- Features: MACD, RSI, CCI, Bollinger Bands, moving averages, VIX, turbulence

### Methods
- **FinRL training pipeline** for single-objective baseline (PPO-based agent)
- **NSGA-III** for multi-objective Pareto front exploration
- **Backtesting framework** to evaluate out-of-sample performance
- Results stored in `results/` and `trained_models/`

### Key Files
- `FinRL_Train.ipynb` — agent training pipeline
- `FinRL_Backtest.ipynb` — backtesting and performance evaluation
- `Manipulacion_Data.ipynb` — data preprocessing and feature engineering
- `NSGA3.py` — NSGA-III implementation
- `Proyecto.ipynb` — full project notebook
- `resultados_nsga3.csv` — Pareto front results

---

## Stack
- **Language:** Python
- **Libraries:** NumPy, PyTorch, Gymnasium, MinAtar, Stable Baselines3,
  mo-gymnasium, morl-baselines, Pymoo, FinRL, Matplotlib, Pandas
- **Tools:** Git, Jupyter Notebooks

---

## Structure
```
Tarea1/   # Dynamic programming: Value Iteration, Policy Iteration
Tarea2/   # Actor-Critic, Dyna-Q: MountainCar, CartPole, Breakout
Tarea3/   # DDQN, PPO, NES/GA: MinAtar, Atari, statistical comparison
Tarea4/   # NSGA-III, Pareto Q-learning, MOEA/D: WFG1, MO-LunarLander
Tarea5/   # Custom environments, full benchmark, extras
Proyecto/ # Multi-objective portfolio optimization with FinRL + NSGA-III
```

---

*Graduate course — Posgrado en Ciencia e Ingeniería de la Computación, IIMAS, UNAM.*
