from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class StrategyProtocol(Protocol):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        ...


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    commission_per_contract: float = 50.0
    slippage_ticks: int = 1
    contract_multiplier: int = 200


def run_backtest(
    prices: pd.Series,
    positions: pd.Series,
    fee_rate: float,
    slippage_rate: float,
) -> pd.DataFrame:
    """
    Vectorized backtest using close-to-close returns.

    returns_t = position_{t-1} * price_return_t - (fee+slippage) * turnover_t
    """
    positions = positions.reindex(prices.index).ffill().bfill().fillna(0.0).astype(float)

    position_prev = positions.shift(1).fillna(0.0)
    price_ret = prices.pct_change().fillna(0.0)
    position_change = positions.diff().abs().fillna(positions.abs())

    cost_per_side = float(fee_rate) + float(slippage_rate)
    returns = position_prev * price_ret - cost_per_side * position_change
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    return pd.DataFrame(
        {
            "equity": equity,
            "returns": returns,
            "drawdown": drawdown,
            "position": positions,
        },
        index=prices.index,
    )


def apply_risk_overlay(
    raw_positions: pd.Series,
    prices: pd.Series,
    bars_per_year: int,
    target_vol_ann: float | None = None,
    max_leverage: float = 4.0,
    drawdown_deleverage_threshold: float = -0.20,
    min_leverage_after_deleverage: float = 0.3,
    vol_lookback: int = 120,
) -> pd.Series:
    """
    Apply volatility targeting and drawdown-based deleveraging to positions.
    """
    pos = raw_positions.reindex(prices.index).ffill().bfill().fillna(0.0).astype(float)
    ret = prices.pct_change().fillna(0.0)

    scale = pd.Series(1.0, index=prices.index, dtype=float)

    if target_vol_ann is not None and target_vol_ann > 0:
        realized_vol = ret.rolling(vol_lookback, min_periods=max(20, vol_lookback // 4)).std()
        realized_vol_ann = realized_vol * np.sqrt(float(bars_per_year))
        vol_scale = target_vol_ann / realized_vol_ann.replace(0.0, np.nan)
        vol_scale = vol_scale.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(1.0)
        scale = scale * vol_scale

    provisional_returns = pos.shift(1).fillna(0.0) * ret
    provisional_equity = (1.0 + provisional_returns).cumprod()
    provisional_drawdown = provisional_equity / provisional_equity.cummax() - 1.0

    dd_scale = pd.Series(1.0, index=prices.index, dtype=float)
    dd_scale.loc[provisional_drawdown < drawdown_deleverage_threshold] = float(
        min_leverage_after_deleverage
    )
    scale = scale * dd_scale

    scale = scale.clip(lower=0.0, upper=abs(max_leverage))
    adjusted = (pos * scale).clip(lower=-abs(max_leverage), upper=abs(max_leverage))
    return adjusted


class BacktestEngine:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run(self, data: pd.DataFrame, strategy: StrategyProtocol) -> pd.DataFrame:
        prices = data["Close"]
        positions = strategy.generate_signals(data)
        fee_rate = (
            self.config.commission_per_contract / (self.config.contract_multiplier * prices.mean())
            if self.config.contract_multiplier
            else 0.0
        )
        slippage_rate = (
            self.config.slippage_ticks / prices.mean() if self.config.slippage_ticks else 0.0
        )
        return run_backtest(
            prices=prices,
            positions=positions,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
