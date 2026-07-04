"""Pairs trading strategy: spread computation, z-score, and signal generation."""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def estimate_hedge_ratio(
    series1: pd.Series, series2: pd.Series
) -> float:
    """Estimate the hedge ratio (beta) via OLS regression.

    Regresses series1 on series2: series1 = alpha + beta * series2 + epsilon.

    Parameters
    ----------
    series1 : pd.Series
        Dependent variable (e.g., Visa prices).
    series2 : pd.Series
        Independent variable (e.g., Mastercard prices).

    Returns
    -------
    float
        The hedge ratio (slope coefficient).
    """
    x = add_constant(series2.values)
    model = OLS(series1.values, x).fit()
    return float(model.params[1])


def compute_spread(
    series1: pd.Series,
    series2: pd.Series,
    hedge_ratio: float | None = None,
) -> pd.Series:
    """Compute the spread between two price series.

    Parameters
    ----------
    series1 : pd.Series
        First asset prices.
    series2 : pd.Series
        Second asset prices.
    hedge_ratio : float or None
        If None, uses simple difference. Otherwise spread = series1 - hedge_ratio * series2.

    Returns
    -------
    pd.Series
        The spread series.
    """
    if hedge_ratio is None:
        hedge_ratio = estimate_hedge_ratio(series1, series2)
    spread = series1 - hedge_ratio * series2
    spread.name = "spread"
    return spread


def compute_zscore(
    spread: pd.Series, lookback: int = 60
) -> pd.Series:
    """Compute the rolling z-score of the spread.

    Parameters
    ----------
    spread : pd.Series
        The spread time series.
    lookback : int
        Rolling window size for mean and std calculation.

    Returns
    -------
    pd.Series
        Z-score series.
    """
    mean = spread.rolling(window=lookback).mean()
    std = spread.rolling(window=lookback).std()
    zscore = (spread - mean) / std
    zscore.name = "zscore"
    return zscore


def generate_signals(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> pd.Series:
    """Generate trading signals based on z-score thresholds.

    Signals:
        1  = Long spread (buy asset1, sell asset2)
        -1 = Short spread (sell asset1, buy asset2)
        0  = No position / exit

    Parameters
    ----------
    zscore : pd.Series
        Z-score of the spread.
    entry_threshold : float
        Absolute z-score level to enter a trade.
    exit_threshold : float
        Absolute z-score level to exit a trade.

    Returns
    -------
    pd.Series
        Signal series with values in {-1, 0, 1}.
    """
    signals = pd.Series(index=zscore.index, data=0.0, dtype=float)
    position = 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]

        if np.isnan(z):
            signals.iloc[i] = 0
            continue

        if position == 0:
            if z < -entry_threshold:
                position = 1  # Long spread
            elif z > entry_threshold:
                position = -1  # Short spread
        elif position == 1:
            if z > -exit_threshold:
                position = 0  # Exit long
        elif position == -1:
            if z < exit_threshold:
                position = 0  # Exit short

        signals.iloc[i] = position

    signals.name = "signal"
    return signals
