"""Performance metrics for strategy evaluation."""

import numpy as np
import pandas as pd

from .backtester import BacktestResult


def calculate_metrics(result: BacktestResult, risk_free_rate: float = 0.02) -> dict:
    """Calculate comprehensive performance metrics.

    Parameters
    ----------
    result : BacktestResult
        Output from the backtester.
    risk_free_rate : float
        Annual risk-free rate for Sharpe/Sortino calculations.

    Returns
    -------
    dict
        Dictionary of performance metrics.
    """
    equity = result.equity_curve
    returns = result.daily_returns.dropna()
    trades = result.trades

    if len(returns) == 0:
        return {"error": "No returns to calculate metrics"}

    # Total return
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

    # CAGR
    n_years = len(returns) / 252
    if n_years > 0 and equity.iloc[0] > 0:
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    else:
        cagr = 0.0

    # Annualized volatility
    annual_vol = returns.std() * np.sqrt(252)

    # Sharpe ratio
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    sharpe = (excess_returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
    sortino = ((returns.mean() - daily_rf) * 252 / downside_std) if downside_std > 0 else 0.0

    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Trade statistics
    n_trades = len(trades)
    if n_trades > 0:
        trade_pnls = [t.pnl for t in trades]
        winning_trades = [p for p in trade_pnls if p > 0]
        losing_trades = [p for p in trade_pnls if p <= 0]
        win_rate = len(winning_trades) / n_trades
        avg_trade = np.mean(trade_pnls)
        avg_holding_period = np.mean([t.duration for t in trades])

        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        win_rate = 0.0
        avg_trade = 0.0
        avg_holding_period = 0.0
        profit_factor = 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_trade_pnl": avg_trade,
        "profit_factor": profit_factor,
        "avg_holding_period_days": avg_holding_period,
    }


def print_metrics(metrics: dict) -> None:
    """Print metrics in a formatted table.

    Parameters
    ----------
    metrics : dict
        Dictionary of performance metrics.
    """
    print("\n" + "=" * 50)
    print("        PERFORMANCE METRICS")
    print("=" * 50)
    print(f"  Total Return:          {metrics['total_return']:.2%}")
    print(f"  CAGR:                  {metrics['cagr']:.2%}")
    print(f"  Annual Volatility:     {metrics['annual_volatility']:.2%}")
    print(f"  Sharpe Ratio:          {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio:         {metrics['sortino_ratio']:.3f}")
    print(f"  Max Drawdown:          {metrics['max_drawdown']:.2%}")
    print(f"  Number of Trades:      {metrics['n_trades']}")
    print(f"  Win Rate:              {metrics['win_rate']:.2%}")
    print(f"  Avg Trade PnL:         {metrics['avg_trade_pnl']:.4f}")
    print(f"  Profit Factor:         {metrics['profit_factor']:.3f}")
    print(f"  Avg Holding Period:    {metrics['avg_holding_period_days']:.1f} days")
    print("=" * 50 + "\n")
