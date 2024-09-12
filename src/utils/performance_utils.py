from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns from a price series.

    Parameters
    ----------
    prices : pd.Series
        Price series.

    Returns
    -------
    pd.Series
        Log returns.
    """
    prices = prices.astype(float)
    log_ret = np.log(prices / prices.shift(1))
    return log_ret


def compute_simple_returns(prices: pd.Series) -> pd.Series:
    """
    Compute simple returns from a price series.

    Parameters
    ----------
    prices : pd.Series
        Price series.

    Returns
    -------
    pd.Series
        Simple returns (p_t / p_{t-1} - 1).
    """
    prices = prices.astype(float)
    return prices.pct_change()
