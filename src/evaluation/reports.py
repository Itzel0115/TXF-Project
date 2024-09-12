from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.backtest.portfolio import (
    compute_annualized_return,
    compute_annualized_volatility,
    compute_max_drawdown,
    compute_sharpe_ratio,
)


def _count_trades(position: pd.Series) -> int:
    """
    依部位變化次數估算交易次數（部位由 t-1 到 t 發生變化即計為一次）。
    """
    if position is None or position.empty:
        return 0
    diff = position.diff().fillna(0).abs()
    return int((diff > 0).sum())


def summarize_strategies(
    strategies_results: Dict[str, pd.DataFrame],
    bars_per_year: int,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    彙總多個策略的回測結果，計算年化報酬、波動度、Sharpe、最大回撤與交易次數。

    使用 backtest.portfolio 的指標函式，不重複實作。

    Parameters
    ----------
    strategies_results : dict[str, pd.DataFrame]
        策略名稱 -> 回測結果 DataFrame。每個 DataFrame 至少需包含：
        - "returns" : 每根 K 線報酬
        - "equity" : 淨值曲線
        若有 "position" 則用來估算交易次數。
    bars_per_year : int
        每年 K 線數（例如 1 分鐘日盤約 75_000）。
    risk_free_rate : float, default 0.0
        年化無風險利率，用於 Sharpe 計算。

    Returns
    -------
    pd.DataFrame
        每列一個策略（index 為策略名稱），欄位為：
        - annualized_return
        - annualized_volatility
        - sharpe_ratio
        - max_drawdown
        - trade_count
        可直接在 notebook 中顯示或匯出 CSV。
    """
    rows = []
    for name, df in strategies_results.items():
        if "returns" not in df.columns or "equity" not in df.columns:
            raise KeyError(
                f"Strategy result '{name}' must contain 'returns' and 'equity' columns."
            )
        returns = df["returns"].dropna()
        equity = df["equity"].dropna()
        position = df["position"] if "position" in df.columns else None

        row = {
            "annualized_return": compute_annualized_return(returns, bars_per_year),
            "annualized_volatility": compute_annualized_volatility(
                returns, bars_per_year
            ),
            "sharpe_ratio": compute_sharpe_ratio(
                returns, risk_free_rate=risk_free_rate, bars_per_year=bars_per_year
            ),
            "max_drawdown": compute_max_drawdown(equity),
            "trade_count": _count_trades(position),
        }
        rows.append((name, row))

    summary = pd.DataFrame(
        [r[1] for r in rows],
        index=[r[0] for r in rows],
    )
    summary.index.name = "strategy"
    return summary


def aggregate_strategy_results(
    metrics_dict: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    將「策略名稱 -> 指標 dict」彙整成一個 DataFrame，方便輸出報表或 CSV。

    Parameters
    ----------
    metrics_dict : dict[str, dict]
        策略名稱 -> 指標名稱 -> 數值（例如已由 summarize_strategies 或自算的指標）。

    Returns
    -------
    pd.DataFrame
        每列一個策略，每欄一個指標。
    """
    if not metrics_dict:
        return pd.DataFrame()
    return pd.DataFrame(metrics_dict).T
