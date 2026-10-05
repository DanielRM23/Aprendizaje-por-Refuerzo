# Multi-Objective Portfolio Optimization with Reinforcement Learning

Final project for the graduate course *Aprendizaje por Refuerzo* (IIMAS, UNAM, 2025-2).
Stock portfolio optimization framed as a **multi-objective reinforcement learning**
problem, combining FinRL's trading framework with **NSGA-III** for Pareto-optimal
portfolio construction.

**Start here:** [`Proyecto_RojoMata/ProyectoFinal-RL-RojoMata.ipynb`](Proyecto_RojoMata/ProyectoFinal-RL-RojoMata.ipynb)
is the self-contained final notebook — fully narrated, with all results. The
accompanying [report](Proyecto_RojoMata/Reporte_Proyecto_RojoMata.pdf) and
[presentation](Proyecto_RojoMata/Presentacion_Proyecto_RojoMata.pdf) (PDF) cover
the same work in writeup and slide form.

The notebooks at the top level of this folder (`FinRL_Train.ipynb`,
`FinRL_Backtest.ipynb`, `Manipulacion_Data.ipynb`, `Proyecto.ipynb`, `NSGA3.py`)
are the development notebooks used to build the pipeline piece by piece, kept
here for transparency and reference.

## Problem Setup
- **Objective 1:** Maximize cumulative return
- **Objective 2:** Minimize portfolio risk (volatility)
- Assets: S&P 500 stocks, downloaded via FinRL / Yahoo Finance
- Features: MACD, RSI, CCI, Bollinger Bands, moving averages, VIX, turbulence

## Methods
- **FinRL training pipeline** for a single-objective PPO baseline
- **NSGA-III** to evolve scalarization weights and explore the Pareto front
  of return-vs-risk trade-offs
- **Backtesting** against classical PPO, mean-variance optimization (MVO),
  and the Dow Jones Index

## Results
The top NSGA-III-derived configuration outperformed classical PPO,
mean-variance optimization, and the Dow Jones Index in cumulative return
while maintaining comparable volatility. See `Imagenes/` for the strategy
comparison and Pareto front plots, and `resultados_nsga3.csv` for the
full set of evolved configurations.

## Stack
Python, FinRL, Stable Baselines3, Pymoo (NSGA-III), Pandas, NumPy, Matplotlib.

## Folder Contents
```
FinRL_Train.ipynb          # Agent training pipeline (development)
FinRL_Backtest.ipynb       # Backtesting and performance evaluation (development)
Manipulacion_Data.ipynb    # Data preprocessing and feature engineering (development)
NSGA3.py                   # NSGA-III implementation
Proyecto.ipynb             # Full project notebook (development)
resultados_nsga3.csv       # Pareto front results
Imagenes/                  # Result plots (Pareto front, strategy comparison)
Proyecto_RojoMata/         # Final notebook + report + presentation (start here)
Anteproyecto/              # Initial project proposal
Articulos/                 # Reference paper
```
