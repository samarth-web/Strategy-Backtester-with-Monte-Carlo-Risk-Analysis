"""Monte Carlo risk analysis for pairs trading strategy."""

import numpy as np
import pandas as pd


def run_monte_carlo(
    trade_returns: list[float],
    n_simulations: int = 10000,
    n_trades_per_sim: int | None = None,
    initial_capital: float = 100000.0,
    random_seed: int | None = 42,
) -> dict:
    """Run Monte Carlo simulation by bootstrapping trade returns.

    Randomly resamples historical trade returns to simulate many
    possible paths the strategy could have taken.

    Parameters
    ----------
    trade_returns : list[float]
        List of individual trade PnL values from the backtest.
    n_simulations : int
        Number of Monte Carlo simulations to run.
    n_trades_per_sim : int or None
        Number of trades per simulation path. If None, uses the
        length of trade_returns.
    initial_capital : float
        Starting capital for each simulation.
    random_seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with simulation results and risk metrics.
    """
    if len(trade_returns) == 0:
        return {
            "terminal_wealth": np.array([initial_capital]),
            "max_drawdowns": np.array([0.0]),
            "prob_loss": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "median_return": 0.0,
            "mean_return": 0.0,
            "worst_drawdown": 0.0,
            "equity_paths": np.array([[initial_capital]]),
        }

    rng = np.random.default_rng(random_seed)

    if n_trades_per_sim is None:
        n_trades_per_sim = len(trade_returns)

    trade_returns_arr = np.array(trade_returns)
    terminal_wealth = np.zeros(n_simulations)
    max_drawdowns = np.zeros(n_simulations)

    # Store a subset of paths for visualization
    n_paths_to_store = min(100, n_simulations)
    equity_paths = np.zeros((n_paths_to_store, n_trades_per_sim + 1))

    for i in range(n_simulations):
        # Bootstrap: randomly sample trade returns with replacement
        sampled_returns = rng.choice(trade_returns_arr, size=n_trades_per_sim, replace=True)

        # Build equity curve
        equity = np.zeros(n_trades_per_sim + 1)
        equity[0] = initial_capital

        for j in range(n_trades_per_sim):
            equity[j + 1] = equity[j] + sampled_returns[j]

        terminal_wealth[i] = equity[-1]

        # Calculate max drawdown for this path
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        drawdown = np.where(running_max > 0, drawdown, 0)
        max_drawdowns[i] = drawdown.min()

        if i < n_paths_to_store:
            equity_paths[i] = equity

    # Calculate risk metrics
    total_returns = (terminal_wealth - initial_capital) / initial_capital
    prob_loss = np.mean(terminal_wealth < initial_capital)
    var_95 = np.percentile(total_returns, 5)  # 5th percentile = 95% VaR
    cvar_95 = total_returns[total_returns <= var_95].mean() if np.any(total_returns <= var_95) else var_95

    return {
        "terminal_wealth": terminal_wealth,
        "max_drawdowns": max_drawdowns,
        "prob_loss": float(prob_loss),
        "var_95": float(var_95),
        "cvar_95": float(cvar_95),
        "median_return": float(np.median(total_returns)),
        "mean_return": float(np.mean(total_returns)),
        "worst_drawdown": float(max_drawdowns.min()),
        "equity_paths": equity_paths,
    }


def print_monte_carlo_results(mc_results: dict) -> None:
    """Print Monte Carlo simulation results.

    Parameters
    ----------
    mc_results : dict
        Output from run_monte_carlo().
    """
    print("\n" + "=" * 50)
    print("     MONTE CARLO RISK ANALYSIS")
    print(f"     ({len(mc_results['terminal_wealth']):,} simulations)")
    print("=" * 50)
    print(f"  Probability of Loss:   {mc_results['prob_loss']:.2%}")
    print(f"  95% Value at Risk:     {mc_results['var_95']:.2%}")
    print(f"  95% CVaR (ES):         {mc_results['cvar_95']:.2%}")
    print(f"  Median Return:         {mc_results['median_return']:.2%}")
    print(f"  Mean Return:           {mc_results['mean_return']:.2%}")
    print(f"  Worst Max Drawdown:    {mc_results['worst_drawdown']:.2%}")
    print("=" * 50 + "\n")
