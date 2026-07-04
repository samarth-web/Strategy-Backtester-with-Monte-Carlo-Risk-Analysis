"""Data downloading and preprocessing utilities."""

import pandas as pd
import yfinance as yf


def download_pair_data(
    ticker1: str,
    ticker2: str,
    start: str = "2015-01-01",
    end: str = "2025-01-01",
) -> pd.DataFrame:
    """Download adjusted closing prices for a pair of assets.

    Parameters
    ----------
    ticker1 : str
        First ticker symbol (e.g., 'V' for Visa).
    ticker2 : str
        Second ticker symbol (e.g., 'MA' for Mastercard).
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [ticker1, ticker2] of adjusted close prices.
    """
    data = yf.download([ticker1, ticker2], start=start, end=end, auto_adjust=True)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"][[ticker1, ticker2]].copy()
    else:
        prices = data[[ticker1, ticker2]].copy()

    prices.dropna(inplace=True)
    prices.columns = [ticker1, ticker2]
    return prices


def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load pair price data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to CSV file with columns for dates and two asset prices.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and two price columns.
    """
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    data.dropna(inplace=True)
    return data


def save_data(data: pd.DataFrame, filepath: str) -> None:
    """Save price data to CSV.

    Parameters
    ----------
    data : pd.DataFrame
        Price data to save.
    filepath : str
        Output file path.
    """
    data.to_csv(filepath)
