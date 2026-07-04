"""Main entry point for the Statistical Arbitrage Pairs Trading Backtester."""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots

from src.downloader import download_pair_data, load_csv_data, save_data
from src.strategy import estimate_hedge_ratio, compute_spread, compute_zscore, generate_signals
from src.backtester import run_backtest
from src.metrics import calculate_metrics, print_metrics
from src.montecarlo import run_monte_carlo, print_monte_carlo_results
from src.plots import save_all_plots
from src.utils import test_cointegration, test_stationarity, optimize_parameters, walk_forward_split


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Statistical Arbitrage Backtester using Pairs Trading"
    )
    parser.add_argument("--ticker1", type=str, default="V", help="First ticker symbol (default: V)")
    parser.add_argument("--ticker2", type=str, default="MA", help="Second ticker symbol (default: MA)")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (default: 2015-01-01)")
    parser.add_argument("--end", type=str, default="2025-01-01", help="End date (default: 2025-01-01)")
    parser.add_argument("--entry", type=float, default=2.0, help="Z-score entry threshold (default: 2.0)")
    parser.add_argument("--exit", type=float, default=0.5, help="Z-score exit threshold (default: 0.5)")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback window for z-score (default: 60)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital (default: 100000)")
    parser.add_argument("--mc-sims", type=int, default=10000, help="Monte Carlo simulations (default: 10000)")
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward test")
    parser.add_argument("--train-end", type=str, default="2021-12-31", help="Walk-forward train end date")
    parser.add_argument("--csv", type=str, default=None, help="Load data from CSV instead of downloading")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return parser.parse_args()


def main() -> None:
    """Run the full pairs trading backtest pipeline."""
    args = parse_args()

    print(f"\n{'='*60}")
    print("  STATISTICAL ARBITRAGE BACKTESTER - PAIRS TRADING")
    print(f"{'='*60}")
    print(f"  Pair: {args.ticker1} / {args.ticker2}")
    print(f"  Period: {args.start} to {args.end}")
    print(f"  Entry Z-Score: ±{args.entry}")
    print(f"  Exit Z-Score: ±{args.exit}")
    print(f"  Lookback Window: {args.lookback} days")
    print(f"  Initial Capital: ${args.capital:,.0f}")
    print(f"{'='*60}\n")

    # Step 1: Load data
    print("[1/7] Loading data...")
    if args.csv:
        prices = load_csv_data(args.csv)
    else:
        prices = download_pair_data(args.ticker1, args.ticker2, args.start, args.end)
        save_data(prices, f"{args.output_dir}/{args.ticker1}_{args.ticker2}_prices.csv")
    print(f"  Loaded {len(prices)} trading days\n")

    ticker1, ticker2 = prices.columns[0], prices.columns[1]

    # Step 2: Statistical tests
    print("[2/7] Running statistical tests...")
    coint_result = test_cointegration(prices[ticker1], prices[ticker2])
    print(f"  Cointegration p-value: {coint_result['p_value']:.4f}")
    print(f"  Cointegrated: {'Yes' if coint_result['is_cointegrated'] else 'No'}\n")

    # Step 3: Compute strategy signals
    print("[3/7] Computing strategy signals...")
    hedge_ratio = estimate_hedge_ratio(prices[ticker1], prices[ticker2])
    spread = compute_spread(prices[ticker1], prices[ticker2], hedge_ratio)
    zscore = compute_zscore(spread, lookback=args.lookback)
    signals = generate_signals(zscore, entry_threshold=args.entry, exit_threshold=args.exit)

    stationarity = test_stationarity(spread)
    print(f"  Hedge ratio (β): {hedge_ratio:.4f}")
    print(f"  Spread stationarity p-value: {stationarity['p_value']:.4f}")
    print(f"  Spread is stationary: {'Yes' if stationarity['is_stationary'] else 'No'}\n")

    # Step 4: Run backtest
    print("[4/7] Running backtest...")
    result = run_backtest(prices, signals, spread, zscore, initial_capital=args.capital)
    print(f"  Completed {len(result.trades)} trades\n")

    # Step 5: Calculate and display metrics
    print("[5/7] Calculating performance metrics...")
    metrics = calculate_metrics(result)
    print_metrics(metrics)

    # Step 6: Monte Carlo simulation
    print("[6/7] Running Monte Carlo simulation...")
    trade_pnls = [t.pnl for t in result.trades]
    mc_results = run_monte_carlo(
        trade_pnls,
        n_simulations=args.mc_sims,
        initial_capital=args.capital,
    )
    print_monte_carlo_results(mc_results)

    # Step 7: Generate plots
    if not args.no_plots:
        print("[7/7] Generating plots...")
        save_all_plots(
            prices, result, mc_results,
            output_dir=args.output_dir,
            entry_threshold=args.entry,
            exit_threshold=args.exit,
        )
    else:
        print("[7/7] Skipping plot generation (--no-plots)\n")

    # Optional: Parameter optimization
    if args.optimize:
        print("\n[OPT] Running parameter optimization...")
        opt_results = optimize_parameters(prices)
        print("\nTop 10 parameter combinations by Sharpe Ratio:")
        print(opt_results.head(10).to_string(index=False))
        opt_results.to_csv(f"{args.output_dir}/optimization_results.csv", index=False)
        print(f"\nFull results saved to {args.output_dir}/optimization_results.csv")

    # Optional: Walk-forward testing
    if args.walk_forward:
        print(f"\n[WF] Running walk-forward test (train until {args.train_end})...")
        train_data, test_data = walk_forward_split(prices, train_end=args.train_end)

        # Train: estimate parameters
        train_hedge = estimate_hedge_ratio(train_data[ticker1], train_data[ticker2])
        train_spread = compute_spread(train_data[ticker1], train_data[ticker2], train_hedge)
        train_zscore = compute_zscore(train_spread, lookback=args.lookback)
        train_signals = generate_signals(train_zscore, args.entry, args.exit)
        train_result = run_backtest(train_data, train_signals, train_spread, train_zscore, args.capital)
        train_metrics = calculate_metrics(train_result)

        # Test: apply same parameters to unseen data
        test_spread = compute_spread(test_data[ticker1], test_data[ticker2], train_hedge)
        test_zscore = compute_zscore(test_spread, lookback=args.lookback)
        test_signals = generate_signals(test_zscore, args.entry, args.exit)
        test_result = run_backtest(test_data, test_signals, test_spread, test_zscore, args.capital)
        test_metrics = calculate_metrics(test_result)

        print("\n  --- Training Period ---")
        print(f"  Sharpe: {train_metrics['sharpe_ratio']:.3f} | Return: {train_metrics['total_return']:.2%} | Trades: {train_metrics['n_trades']}")
        print("\n  --- Testing Period (Out-of-Sample) ---")
        print(f"  Sharpe: {test_metrics['sharpe_ratio']:.3f} | Return: {test_metrics['total_return']:.2%} | Trades: {test_metrics['n_trades']}")

    print("\n✓ Backtest complete!\n")


if __name__ == "__main__":
    main()
