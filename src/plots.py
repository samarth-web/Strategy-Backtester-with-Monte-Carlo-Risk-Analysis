"""Visualization functions for the pairs trading backtester."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtester import BacktestResult


def plot_prices(prices: pd.DataFrame, title: str = "Asset Prices") -> plt.Figure:
    """Plot both asset prices on the same chart.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with two columns of asset prices.
    title : str
        Chart title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ticker1, ticker2 = prices.columns[0], prices.columns[1]

    ax1.plot(prices.index, prices[ticker1], label=ticker1, linewidth=1.2)
    ax1.set_ylabel(ticker1, color="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(prices.index, prices[ticker2], label=ticker2, color="tab:orange", linewidth=1.2)
    ax2.set_ylabel(ticker2, color="tab:orange")

    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    return fig


def plot_spread_and_zscore(
    result: BacktestResult, entry_threshold: float = 2.0, exit_threshold: float = 0.5
) -> plt.Figure:
    """Plot the spread and z-score with entry/exit thresholds.

    Parameters
    ----------
    result : BacktestResult
        Backtest result object.
    entry_threshold : float
        Z-score entry threshold.
    exit_threshold : float
        Z-score exit threshold.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Spread
    ax1.plot(result.spread.index, result.spread.values, linewidth=0.8, color="steelblue")
    ax1.axhline(result.spread.mean(), color="black", linestyle="--", alpha=0.5, label="Mean")
    ax1.set_title("Spread")
    ax1.set_ylabel("Spread Value")
    ax1.legend()

    # Z-score
    ax2.plot(result.zscore.index, result.zscore.values, linewidth=0.8, color="steelblue")
    ax2.axhline(entry_threshold, color="red", linestyle="--", alpha=0.7, label=f"+{entry_threshold}σ")
    ax2.axhline(-entry_threshold, color="green", linestyle="--", alpha=0.7, label=f"-{entry_threshold}σ")
    ax2.axhline(exit_threshold, color="orange", linestyle=":", alpha=0.5)
    ax2.axhline(-exit_threshold, color="orange", linestyle=":", alpha=0.5)
    ax2.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax2.set_title("Z-Score with Thresholds")
    ax2.set_ylabel("Z-Score")
    ax2.set_xlabel("Date")
    ax2.legend()

    # Overlay signals
    long_entries = result.signals[
        (result.signals == 1) & (result.signals.shift(1) != 1)
    ]
    short_entries = result.signals[
        (result.signals == -1) & (result.signals.shift(1) != -1)
    ]

    ax2.scatter(
        long_entries.index,
        result.zscore.loc[long_entries.index],
        marker="^",
        color="green",
        s=50,
        zorder=5,
        label="Long Entry",
    )
    ax2.scatter(
        short_entries.index,
        result.zscore.loc[short_entries.index],
        marker="v",
        color="red",
        s=50,
        zorder=5,
        label="Short Entry",
    )
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_equity_curve(result: BacktestResult) -> plt.Figure:
    """Plot the equity curve.

    Parameters
    ----------
    result : BacktestResult
        Backtest result object.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(result.equity_curve.index, result.equity_curve.values, linewidth=1.2, color="darkblue")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_drawdown(result: BacktestResult) -> plt.Figure:
    """Plot the drawdown chart.

    Parameters
    ----------
    result : BacktestResult
        Backtest result object.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    returns = result.daily_returns.dropna()
    if len(returns) == 0:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.set_title("Drawdown (No data)")
        return fig

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
    ax.plot(drawdown.index, drawdown.values, color="red", linewidth=0.8)
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_monte_carlo(mc_results: dict, initial_capital: float = 100000.0) -> plt.Figure:
    """Plot Monte Carlo simulation results.

    Parameters
    ----------
    mc_results : dict
        Output from run_monte_carlo().
    initial_capital : float
        Starting capital.

    Returns
    -------
    plt.Figure
        Matplotlib figure with multiple subplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Equity paths
    ax = axes[0, 0]
    equity_paths = mc_results["equity_paths"]
    for i in range(min(50, len(equity_paths))):
        ax.plot(equity_paths[i], alpha=0.3, linewidth=0.5)
    ax.axhline(initial_capital, color="black", linestyle="--", alpha=0.5)
    ax.set_title("Monte Carlo Equity Paths (50 samples)")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Portfolio Value ($)")

    # Terminal wealth distribution
    ax = axes[0, 1]
    terminal = mc_results["terminal_wealth"]
    ax.hist(terminal, bins=50, color="steelblue", edgecolor="white", alpha=0.7)
    ax.axvline(initial_capital, color="red", linestyle="--", label="Initial Capital")
    ax.axvline(np.median(terminal), color="green", linestyle="--", label="Median")
    ax.set_title("Terminal Wealth Distribution")
    ax.set_xlabel("Final Portfolio Value ($)")
    ax.set_ylabel("Frequency")
    ax.legend()

    # Max drawdown distribution
    ax = axes[1, 0]
    ax.hist(mc_results["max_drawdowns"], bins=50, color="salmon", edgecolor="white", alpha=0.7)
    ax.set_title("Max Drawdown Distribution")
    ax.set_xlabel("Max Drawdown")
    ax.set_ylabel("Frequency")

    # Return distribution
    ax = axes[1, 1]
    returns = (terminal - initial_capital) / initial_capital
    ax.hist(returns, bins=50, color="lightgreen", edgecolor="white", alpha=0.7)
    ax.axvline(mc_results["var_95"], color="red", linestyle="--", label=f"95% VaR: {mc_results['var_95']:.2%}")
    ax.axvline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_title("Return Distribution")
    ax.set_xlabel("Total Return")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()
    return fig


def save_all_plots(
    prices: pd.DataFrame,
    result: BacktestResult,
    mc_results: dict,
    output_dir: str = "data",
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> None:
    """Save all plots to files.

    Parameters
    ----------
    prices : pd.DataFrame
        Asset prices.
    result : BacktestResult
        Backtest results.
    mc_results : dict
        Monte Carlo results.
    output_dir : str
        Directory to save plots.
    entry_threshold : float
        Entry threshold for spread/zscore plot.
    exit_threshold : float
        Exit threshold for spread/zscore plot.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    fig = plot_prices(prices)
    fig.savefig(f"{output_dir}/prices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_spread_and_zscore(result, entry_threshold, exit_threshold)
    fig.savefig(f"{output_dir}/spread_zscore.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_equity_curve(result)
    fig.savefig(f"{output_dir}/equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_drawdown(result)
    fig.savefig(f"{output_dir}/drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_monte_carlo(mc_results)
    fig.savefig(f"{output_dir}/monte_carlo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"All plots saved to {output_dir}/")
