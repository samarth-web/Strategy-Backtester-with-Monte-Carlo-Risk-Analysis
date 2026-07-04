"""Tests for the pairs trading backtester."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import estimate_hedge_ratio, compute_spread, compute_zscore, generate_signals
from src.backtester import run_backtest
from src.metrics import calculate_metrics
from src.montecarlo import run_monte_carlo
from src.utils import (
    test_cointegration as check_cointegration,
    test_stationarity as check_stationarity,
    compute_correlation,
    walk_forward_split,
)


@pytest.fixture
def sample_prices():
    """Create synthetic price data for testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Create cointegrated pair
    base = np.cumsum(np.random.randn(n) * 0.5) + 100
    noise = np.random.randn(n) * 2
    asset1 = base + noise
    asset2 = base * 0.8 + np.random.randn(n) * 1.5 + 20

    prices = pd.DataFrame({"ASSET1": asset1, "ASSET2": asset2}, index=dates)
    return prices


@pytest.fixture
def sample_spread(sample_prices):
    """Create a spread from sample prices."""
    hedge_ratio = estimate_hedge_ratio(
        sample_prices["ASSET1"], sample_prices["ASSET2"]
    )
    return compute_spread(sample_prices["ASSET1"], sample_prices["ASSET2"], hedge_ratio)


class TestStrategy:
    """Tests for strategy module."""

    def test_estimate_hedge_ratio(self, sample_prices):
        """Hedge ratio should be a reasonable positive number."""
        hr = estimate_hedge_ratio(sample_prices["ASSET1"], sample_prices["ASSET2"])
        assert isinstance(hr, float)
        assert 0 < hr < 5  # reasonable range

    def test_compute_spread(self, sample_prices):
        """Spread should have same length as input."""
        spread = compute_spread(sample_prices["ASSET1"], sample_prices["ASSET2"])
        assert len(spread) == len(sample_prices)

    def test_compute_spread_with_hedge_ratio(self, sample_prices):
        """Spread with explicit hedge ratio."""
        spread = compute_spread(sample_prices["ASSET1"], sample_prices["ASSET2"], hedge_ratio=1.0)
        expected = sample_prices["ASSET1"] - 1.0 * sample_prices["ASSET2"]
        pd.testing.assert_series_equal(spread, expected, check_names=False)

    def test_compute_zscore(self, sample_spread):
        """Z-score should be standardized."""
        zscore = compute_zscore(sample_spread, lookback=60)
        # After warmup period, z-score should have reasonable values
        valid = zscore.dropna()
        assert len(valid) > 0
        assert valid.std() > 0

    def test_generate_signals_values(self, sample_spread):
        """Signals should only contain -1, 0, 1."""
        zscore = compute_zscore(sample_spread, lookback=60)
        signals = generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.5)
        unique_vals = set(signals.unique())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_generate_signals_length(self, sample_spread):
        """Signals should have same length as input."""
        zscore = compute_zscore(sample_spread, lookback=60)
        signals = generate_signals(zscore)
        assert len(signals) == len(zscore)


class TestBacktester:
    """Tests for backtester module."""

    def test_run_backtest_returns_result(self, sample_prices):
        """Backtest should return a BacktestResult."""
        hedge_ratio = estimate_hedge_ratio(sample_prices["ASSET1"], sample_prices["ASSET2"])
        spread = compute_spread(sample_prices["ASSET1"], sample_prices["ASSET2"], hedge_ratio)
        zscore = compute_zscore(spread, lookback=60)
        signals = generate_signals(zscore)
        result = run_backtest(sample_prices, signals, spread, zscore)

        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
        assert result.equity_curve.iloc[0] == 100000.0

    def test_backtest_equity_positive(self, sample_prices):
        """Equity should remain positive (no bankruptcy)."""
        hedge_ratio = estimate_hedge_ratio(sample_prices["ASSET1"], sample_prices["ASSET2"])
        spread = compute_spread(sample_prices["ASSET1"], sample_prices["ASSET2"], hedge_ratio)
        zscore = compute_zscore(spread, lookback=60)
        signals = generate_signals(zscore)
        result = run_backtest(sample_prices, signals, spread, zscore)

        # Equity should stay positive for reasonable parameters
        assert result.equity_curve.min() > 0

    def test_no_signal_no_trade(self, sample_prices):
        """With all-zero signals, no trades should occur."""
        spread = compute_spread(sample_prices["ASSET1"], sample_prices["ASSET2"], hedge_ratio=1.0)
        zscore = compute_zscore(spread, lookback=60)
        signals = pd.Series(0.0, index=sample_prices.index)
        result = run_backtest(sample_prices, signals, spread, zscore)

        assert len(result.trades) == 0


class TestMetrics:
    """Tests for metrics module."""

    def test_calculate_metrics_keys(self, sample_prices):
        """Metrics should contain all expected keys."""
        hedge_ratio = estimate_hedge_ratio(sample_prices["ASSET1"], sample_prices["ASSET2"])
        spread = compute_spread(sample_prices["ASSET1"], sample_prices["ASSET2"], hedge_ratio)
        zscore = compute_zscore(spread, lookback=60)
        signals = generate_signals(zscore)
        result = run_backtest(sample_prices, signals, spread, zscore)
        metrics = calculate_metrics(result)

        expected_keys = [
            "total_return", "cagr", "annual_volatility", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "n_trades", "win_rate",
            "avg_trade_pnl", "profit_factor", "avg_holding_period_days"
        ]
        for key in expected_keys:
            assert key in metrics

    def test_max_drawdown_negative(self, sample_prices):
        """Max drawdown should be negative or zero."""
        hedge_ratio = estimate_hedge_ratio(sample_prices["ASSET1"], sample_prices["ASSET2"])
        spread = compute_spread(sample_prices["ASSET1"], sample_prices["ASSET2"], hedge_ratio)
        zscore = compute_zscore(spread, lookback=60)
        signals = generate_signals(zscore)
        result = run_backtest(sample_prices, signals, spread, zscore)
        metrics = calculate_metrics(result)

        assert metrics["max_drawdown"] <= 0


class TestMonteCarlo:
    """Tests for Monte Carlo module."""

    def test_monte_carlo_basic(self):
        """Monte Carlo should run and return expected keys."""
        trade_returns = [100, -50, 200, -30, 150, -80, 120, -40, 90, -60]
        results = run_monte_carlo(trade_returns, n_simulations=1000)

        assert "prob_loss" in results
        assert "var_95" in results
        assert "cvar_95" in results
        assert "terminal_wealth" in results
        assert len(results["terminal_wealth"]) == 1000

    def test_monte_carlo_no_trades(self):
        """Monte Carlo with empty trades should handle gracefully."""
        results = run_monte_carlo([], n_simulations=100)
        assert results["prob_loss"] == 0.0

    def test_monte_carlo_all_positive(self):
        """If all trades are positive, probability of loss should be low."""
        trade_returns = [100, 200, 150, 300, 250]
        results = run_monte_carlo(trade_returns, n_simulations=5000)
        assert results["prob_loss"] == 0.0

    def test_monte_carlo_reproducibility(self):
        """Same seed should give same results."""
        trades = [100, -50, 200, -30, 150]
        r1 = run_monte_carlo(trades, n_simulations=100, random_seed=42)
        r2 = run_monte_carlo(trades, n_simulations=100, random_seed=42)
        np.testing.assert_array_equal(r1["terminal_wealth"], r2["terminal_wealth"])


class TestUtils:
    """Tests for utility functions."""

    def test_cointegration(self, sample_prices):
        """Cointegration test should return expected keys."""
        result = check_cointegration(sample_prices["ASSET1"], sample_prices["ASSET2"])
        assert "test_statistic" in result
        assert "p_value" in result
        assert "is_cointegrated" in result

    def test_stationarity(self, sample_spread):
        """Stationarity test should return expected keys."""
        result = check_stationarity(sample_spread)
        assert "adf_statistic" in result
        assert "p_value" in result
        assert "is_stationary" in result

    def test_correlation(self, sample_prices):
        """Correlation should be between -1 and 1."""
        corr = compute_correlation(sample_prices["ASSET1"], sample_prices["ASSET2"])
        assert -1 <= corr <= 1

    def test_walk_forward_split(self, sample_prices):
        """Walk-forward split should produce non-overlapping sets."""
        train, test = walk_forward_split(sample_prices, train_end="2020-12-31")
        assert len(train) > 0
        assert len(test) > 0
        assert train.index[-1] <= test.index[0]
