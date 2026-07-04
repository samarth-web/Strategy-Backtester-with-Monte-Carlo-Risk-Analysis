"""Backtesting engine for pairs trading strategy."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """Represents a single completed trade."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int  # 1 for long spread, -1 for short spread
    pnl: float
    duration: int  # number of trading days


@dataclass
class BacktestResult:
    """Container for backtest output."""

    equity_curve: pd.Series
    signals: pd.Series
    spread: pd.Series
    zscore: pd.Series
    trades: list = field(default_factory=list)
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    spread: pd.Series,
    zscore: pd.Series,
    initial_capital: float = 100000.0,
    transaction_cost: float = 0.001,
) -> BacktestResult:
    """Run the pairs trading backtest.

    The strategy allocates equal capital to each leg. PnL is computed
    from spread returns scaled by position direction.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with two columns of asset prices.
    signals : pd.Series
        Position signals (-1, 0, 1) aligned with prices index.
    spread : pd.Series
        The spread series.
    zscore : pd.Series
        The z-score series.
    initial_capital : float
        Starting capital.
    transaction_cost : float
        Proportional transaction cost per trade entry/exit.

    Returns
    -------
    BacktestResult
        Object containing equity curve, trades, and other data.
    """
    ticker1, ticker2 = prices.columns[0], prices.columns[1]
    price1 = prices[ticker1]
    price2 = prices[ticker2]

    # Compute daily spread returns (percent change of spread is tricky,
    # so we use the dollar PnL approach normalized by capital)
    capital = initial_capital
    equity = [capital]
    daily_returns = []
    trades: list[Trade] = []

    prev_signal = 0
    entry_date = None
    entry_spread = 0.0

    for i in range(1, len(signals)):
        current_signal = signals.iloc[i]
        prev_signal_val = signals.iloc[i - 1]

        spread_change = spread.iloc[i] - spread.iloc[i - 1]

        # Daily PnL based on position
        daily_pnl = prev_signal_val * spread_change

        # Transaction costs on position changes
        if current_signal != prev_signal_val:
            cost = abs(current_signal - prev_signal_val) * transaction_cost * capital
            daily_pnl -= cost

        # Scale PnL relative to capital (normalize spread PnL)
        # Use a fraction of capital allocated to the trade
        spread_std = spread.iloc[max(0, i - 60) : i].std()
        if spread_std > 0 and not np.isnan(spread_std):
            # Normalize: allocate capital such that 1 std move = ~2% of capital
            position_size = 0.02 * capital / spread_std
            scaled_pnl = prev_signal_val * spread_change * position_size
        else:
            scaled_pnl = 0.0

        capital += scaled_pnl
        equity.append(capital)
        daily_returns.append(scaled_pnl / equity[-2] if equity[-2] != 0 else 0)

        # Track trades
        if prev_signal_val == 0 and current_signal != 0:
            entry_date = signals.index[i]
            entry_spread = spread.iloc[i]
        elif prev_signal_val != 0 and current_signal == 0:
            if entry_date is not None:
                exit_date = signals.index[i]
                trade_pnl = prev_signal_val * (spread.iloc[i] - entry_spread)
                duration = (exit_date - entry_date).days
                trades.append(
                    Trade(
                        entry_date=entry_date,
                        exit_date=exit_date,
                        direction=int(prev_signal_val),
                        pnl=trade_pnl,
                        duration=duration,
                    )
                )
                entry_date = None

    equity_series = pd.Series(equity, index=signals.index[: len(equity)])
    daily_ret_series = pd.Series(
        daily_returns, index=signals.index[1 : len(daily_returns) + 1]
    )

    return BacktestResult(
        equity_curve=equity_series,
        signals=signals,
        spread=spread,
        zscore=zscore,
        trades=trades,
        daily_returns=daily_ret_series,
    )
