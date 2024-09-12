from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 風險與績效計算（向量化）
# 1 分鐘日盤預設 bars_per_year ≈ 75_000（約 300 分鐘/日 × 250 日/年）
# -----------------------------------------------------------------------------

DEFAULT_BARS_PER_YEAR: int = 75_000


def compute_max_drawdown(equity: pd.Series) -> float:
    """
    計算淨值序列的最大回撤（以比例表示，負值）。

    Parameters
    ----------
    equity : pd.Series
        淨值序列（通常為累積淨值，index 為 datetime）。

    Returns
    -------
    float
        最大回撤比例，例如 -0.05 表示 5% 回撤。
    """
    peak = equity.cummax()
    drawdown_pct = (equity / peak) - 1.0
    return float(drawdown_pct.min())


def compute_annualized_return(
    returns: pd.Series,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
) -> float:
    """
    從報酬序列計算年化報酬率。

    Parameters
    ----------
    returns : pd.Series
        每根 K 線的簡單報酬率。
    bars_per_year : int, default 75_000
        每年 K 線數（1 分鐘日盤約 300×250）。

    Returns
    -------
    float
        年化報酬率。
    """
    n = returns.dropna().size
    if n == 0:
        return 0.0
    total_return = (1.0 + returns).prod() - 1.0
    years = n / bars_per_year
    if years <= 0:
        return 0.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def compute_annualized_volatility(
    returns: pd.Series,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
) -> float:
    """
    從報酬序列計算年化波動率。

    Parameters
    ----------
    returns : pd.Series
        每根 K 線的簡單報酬率。
    bars_per_year : int, default 75_000
        每年 K 線數。

    Returns
    -------
    float
        年化波動率（標準差）。
    """
    std = returns.std()
    if pd.isna(std) or std == 0:
        return 0.0
    return float(std * np.sqrt(bars_per_year))


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
) -> float:
    """
    從報酬序列計算 Sharpe ratio（年化）。

    Parameters
    ----------
    returns : pd.Series
        每根 K 線的簡單報酬率。
    risk_free_rate : float, default 0.0
        年化無風險利率。
    bars_per_year : int, default 75_000
        每年 K 線數。

    Returns
    -------
    float
        Sharpe ratio；若波動率為 0 則回傳 0。
    """
    ann_ret = compute_annualized_return(returns, bars_per_year)
    ann_vol = compute_annualized_volatility(returns, bars_per_year)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


# -----------------------------------------------------------------------------
# Portfolio 類別：淨值曲線與摘要
# -----------------------------------------------------------------------------


@dataclass
class Portfolio:
    """
    Portfolio object to hold equity curve and compute risk metrics.
    """

    equity_curve: pd.Series
    returns_curve: pd.Series | None = None

    def compute_drawdown(self) -> pd.DataFrame:
        """
        Compute drawdown series from equity curve.

        Returns
        -------
        pd.DataFrame
            Columns: ["equity", "peak", "drawdown", "drawdown_pct"].
        """
        peak = self.equity_curve.cummax()
        dd = self.equity_curve - peak
        dd_pct = (self.equity_curve / peak) - 1.0
        return pd.DataFrame(
            {
                "equity": self.equity_curve,
                "peak": peak,
                "drawdown": dd,
                "drawdown_pct": dd_pct,
            },
            index=self.equity_curve.index,
        )

    def summary(self) -> Dict[str, Any]:
        """
        Summarize basic portfolio statistics.

        Returns
        -------
        dict
            Basic stats: total_return, max_drawdown, annualized_return,
            annualized_volatility, sharpe_ratio (if returns_curve provided).
        """
        total_return = float(self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1.0
        max_dd = compute_max_drawdown(self.equity_curve)
        out: Dict[str, Any] = {
            "total_return": total_return,
            "max_drawdown": max_dd,
        }
        if self.returns_curve is not None:
            out["annualized_return"] = compute_annualized_return(self.returns_curve)
            out["annualized_volatility"] = compute_annualized_volatility(
                self.returns_curve
            )
            out["sharpe_ratio"] = compute_sharpe_ratio(self.returns_curve)
        return out
