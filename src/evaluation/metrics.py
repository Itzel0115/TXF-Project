from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(
    equity: pd.Series,
    periods_per_year: int = 252 * 240,
) -> float:
    """
    Compute annualized return from equity curve.

    Parameters
    ----------
    equity : pd.Series
        Equity curve indexed by datetime.
    periods_per_year : int
        Number of bars per year (approx; for 1min day session you may customize).

    Returns
    -------
    float
        Annualized return.
    """
    equity = equity.dropna()
    if equity.empty:
        return 0.0

    total_return = float(equity.iloc[-1] / equity.iloc[0]) - 1.0
    n = equity.size
    if n <= 0:
        return 0.0
    years = n / float(periods_per_year)
    if years <= 0:
        return 0.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252 * 240,
) -> float:
    """
    Compute annualized volatility of returns.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    periods_per_year : int
        Number of bars per year.

    Returns
    -------
    float
        Annualized volatility.
    """
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    std = returns.std()
    if pd.isna(std) or std == 0:
        return 0.0
    return float(std * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 240,
) -> float:
    """
    Compute Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    risk_free_rate : float
        Annual risk-free rate.
    periods_per_year : int
        Number of bars per year.

    Returns
    -------
    float
        Sharpe ratio.
    """
    ann_ret = annualized_return(returns.cumsum() + 1.0, periods_per_year=periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year=periods_per_year)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def max_drawdown(equity: pd.Series) -> float:
    """
    Compute maximum drawdown of an equity curve.

    Returns
    -------
    float
        Maximum drawdown as a fraction (negative).
    """
    equity = equity.dropna()
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())
