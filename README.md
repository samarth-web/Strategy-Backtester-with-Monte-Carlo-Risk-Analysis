# Statistical Arbitrage Backtester – Pairs Trading

A quantitative finance project that implements a **pairs trading strategy** with full backtesting, performance analytics, and **Monte Carlo risk analysis**.

> "If two historically related assets diverge more than usual, can we profit when they converge again?"

## Features

- **Pairs Trading Strategy** – Estimates hedge ratios via OLS regression, computes spreads and z-scores, and generates entry/exit signals
- **Backtesting Engine** – Simulates daily trading with transaction costs and position sizing
- **Comprehensive Metrics** – Total return, CAGR, Sharpe ratio, Sortino ratio, max drawdown, win rate, profit factor, and more
- **Monte Carlo Risk Analysis** – Bootstraps trade returns across 10,000+ simulations to estimate VaR, CVaR, and probability of loss
- **Parameter Optimization** – Grid search across entry/exit thresholds and lookback windows
- **Walk-Forward Testing** – Train/test split to validate out-of-sample performance
- **Statistical Tests** – Cointegration (Engle-Granger) and stationarity (ADF) tests
- **Visualization** – Price charts, spread/z-score plots, equity curves, drawdown charts, and Monte Carlo distributions

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── downloader.py     # Data downloading & preprocessing
│   ├── strategy.py       # Spread, z-score, and signal generation
│   ├── backtester.py     # Backtesting engine
│   ├── metrics.py        # Performance metrics calculation
│   ├── montecarlo.py     # Monte Carlo risk simulation
│   ├── plots.py          # Visualization functions
│   └── utils.py          # Statistical tests, optimization, walk-forward
├── tests/
│   └── test_backtester.py
├── data/                  # Output directory for plots and data
├── notebooks/             # Jupyter notebooks for exploration
├── main.py                # CLI entry point
├── requirements.txt
└── pyproject.toml
```

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- `yfinance` – Market data download
- `pandas` / `numpy` – Data manipulation
- `statsmodels` – OLS regression, cointegration tests
- `scipy` – Statistical functions
- `matplotlib` / `seaborn` – Visualization
- `scikit-learn` – Additional utilities

## Quick Start

### Basic Run (Visa & Mastercard)

```bash
python main.py
```

### Custom Pair and Parameters

```bash
python main.py --ticker1 KO --ticker2 PEP --entry 2.5 --exit 0.25 --lookback 90
```

### With Parameter Optimization

```bash
python main.py --optimize
```

### With Walk-Forward Testing

```bash
python main.py --walk-forward --train-end 2021-12-31
```

### Full Example

```bash
python main.py \
    --ticker1 V --ticker2 MA \
    --start 2015-01-01 --end 2025-01-01 \
    --entry 2.0 --exit 0.5 --lookback 60 \
    --capital 100000 --mc-sims 10000 \
    --optimize --walk-forward
```

## How It Works

### 1. Hedge Ratio Estimation

Uses OLS regression to find the optimal hedge ratio (β):

```
spread = Asset₁ − β × Asset₂
```

### 2. Z-Score Calculation

Rolling z-score standardizes the spread:

```
z = (spread − rolling_mean) / rolling_std
```

### 3. Trading Rules

| Condition | Action |
|-----------|--------|
| z < −entry_threshold | **Long spread** (buy Asset₁, sell Asset₂) |
| z > +entry_threshold | **Short spread** (sell Asset₁, buy Asset₂) |
| \|z\| < exit_threshold | **Exit** position |

### 4. Monte Carlo Risk Analysis

Bootstraps historical trade returns 10,000 times to estimate:
- **95% Value at Risk (VaR)** – Worst expected loss at 95% confidence
- **Conditional VaR (CVaR)** – Expected loss beyond VaR
- **Probability of Loss** – Chance of negative total return
- **Worst Drawdown** – Maximum peak-to-trough decline across all simulations

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker1` | `V` | First ticker symbol |
| `--ticker2` | `MA` | Second ticker symbol |
| `--start` | `2015-01-01` | Start date |
| `--end` | `2025-01-01` | End date |
| `--entry` | `2.0` | Z-score entry threshold |
| `--exit` | `0.5` | Z-score exit threshold |
| `--lookback` | `60` | Rolling window (days) |
| `--capital` | `100000` | Initial capital ($) |
| `--mc-sims` | `10000` | Monte Carlo simulations |
| `--optimize` | off | Run parameter grid search |
| `--walk-forward` | off | Run walk-forward test |
| `--train-end` | `2021-12-31` | Training period end date |
| `--csv` | None | Load from CSV instead of downloading |
| `--output-dir` | `data` | Output directory for plots |
| `--no-plots` | off | Skip plot generation |

## Running Tests

```bash
python -m pytest tests/ -v
```

## Recommended Pairs

| Pair | Rationale |
|------|-----------|
| Visa (V) & Mastercard (MA) | Payment networks, similar business models |
| Coca-Cola (KO) & Pepsi (PEP) | Beverage industry competitors |
| Shell (SHEL) & BP (BP) | Oil & gas majors |
| Ford (F) & GM (GM) | Auto manufacturers |
| JPMorgan (JPM) & Bank of America (BAC) | Large banks |

## Performance Metrics

The backtester calculates:

- **Total Return** – Overall profit/loss percentage
- **CAGR** – Compound Annual Growth Rate
- **Sharpe Ratio** – Risk-adjusted return (annualized)
- **Sortino Ratio** – Downside risk-adjusted return
- **Maximum Drawdown** – Largest peak-to-trough decline
- **Win Rate** – Percentage of profitable trades
- **Profit Factor** – Gross profits / gross losses
- **Average Holding Period** – Mean trade duration in days

## License

See [LICENSE](LICENSE) for details.
