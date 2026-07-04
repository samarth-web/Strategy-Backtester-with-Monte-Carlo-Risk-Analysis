"""Utility functions for pairs trading backtester."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller


def test_cointegration(series1: pd.Series, series2: pd.Series, significance: float = 0.05) -> dict:
    """Test for cointegration between two price series using the Engle-Granger method.

    Parameters
    ----------
    series1 : pd.Series
        First price series.
    series2 : pd.Series
        Second price series.
    significance : float
        Significance level for the test.

    Returns
    -------
    dict
        Dictionary with test statistic, p-value, and whether the pair is cointegrated.
    """
    score, pvalue, _ = coint(series1, series2)
    return {
        "test_statistic": float(score),
        "p_value": float(pvalue),
        "is_cointegrated": pvalue < significance,
        "significance_level": significance,
    }


def test_stationarity(series: pd.Series, significance: float = 0.05) -> dict:
    """Test for stationarity using the Augmented Dickey-Fuller test.

    Parameters
    ----------
    series : pd.Series
        Time series to test.
    significance : float
        Significance level for the test.

    Returns
    -------
    dict
        Dictionary with ADF statistic, p-value, and stationarity result.
    """
    series_clean = series.dropna()
    result = adfuller(series_clean)
    return {
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "is_stationary": result[1] < significance,
        "critical_values": result[4],
    }


def compute_correlation(series1: pd.Series, series2: pd.Series) -> float:
    """Compute Pearson correlation between two series.

    Parameters
    ----------
    series1 : pd.Series
        First series.
    series2 : pd.Series
        Second series.

    Returns
    -------
    float
        Correlation coefficient.
    """
    return float(series1.corr(series2))


def compute_rolling_correlation(
    series1: pd.Series, series2: pd.Series, window: int = 60
) -> pd.Series:
    """Compute rolling correlation between two series.

    Parameters
    ----------
    series1 : pd.Series
        First series.
    series2 : pd.Series
        Second series.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling correlation series.
    """
    return series1.rolling(window).corr(series2)


def optimize_parameters(
    prices: pd.DataFrame,
    entry_range: list[float] | None = None,
    exit_range: list[float] | None = None,
    lookback_range: list[int] | None = None,
) -> pd.DataFrame:
    """Grid search over strategy parameters to find optimal settings.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with two asset price columns.
    entry_range : list[float] or None
        Entry threshold values to test.
    exit_range : list[float] or None
        Exit threshold values to test.
    lookback_range : list[int] or None
        Lookback window values to test.

    Returns
    -------
    pd.DataFrame
        DataFrame of parameter combinations and resulting metrics.
    """
    from .strategy import estimate_hedge_ratio, compute_spread, compute_zscore, generate_signals
    from .backtester import run_backtest
    from .metrics import calculate_metrics

    if entry_range is None:
        entry_range = [1.5, 2.0, 2.5, 3.0]
    if exit_range is None:
        exit_range = [0.0, 0.25, 0.5, 0.75]
    if lookback_range is None:
        lookback_range = [30, 60, 90, 120]

    ticker1, ticker2 = prices.columns[0], prices.columns[1]
    hedge_ratio = estimate_hedge_ratio(prices[ticker1], prices[ticker2])
    spread = compute_spread(prices[ticker1], prices[ticker2], hedge_ratio)

    results = []

    for lookback in lookback_range:
        zscore = compute_zscore(spread, lookback=lookback)
        for entry in entry_range:
            for exit_val in exit_range:
                if exit_val >= entry:
                    continue
                signals = generate_signals(zscore, entry_threshold=entry, exit_threshold=exit_val)
                bt_result = run_backtest(prices, signals, spread, zscore)
                metrics = calculate_metrics(bt_result)

                results.append({
                    "entry_threshold": entry,
                    "exit_threshold": exit_val,
                    "lookback": lookback,
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                    "total_return": metrics.get("total_return", 0),
                    "max_drawdown": metrics.get("max_drawdown", 0),
                    "n_trades": metrics.get("n_trades", 0),
                    "win_rate": metrics.get("win_rate", 0),
                })

    return pd.DataFrame(results).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)


def walk_forward_split(
    data: pd.DataFrame,
    train_end: str = "2021-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and testing periods for walk-forward analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Full price dataset.
    train_end : str
        End date for training period (inclusive).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Training and testing DataFrames.
    """
    train = data.loc[:train_end]
    test = data.loc[train_end:].iloc[1:]  # Exclude the split date from test
    return train, test
