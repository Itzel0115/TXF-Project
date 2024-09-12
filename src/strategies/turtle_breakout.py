from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import BaseStrategy
from ..features.technical_indicators import atr


@dataclass
class TurtleBreakoutStrategy(BaseStrategy):
    """
    Turtle / Donchian channel breakout strategy.

    Typical logic (adapted to minute data):
    - Enter long when price breaks the rolling ``breakout_window`` high.
    - Exit long when price breaks the rolling ``exit_window`` low.
    - Enter short when price breaks the rolling ``breakout_window`` low.
    - Exit short when price breaks the rolling ``exit_window`` high.

    Note
    ----
    This implementation uses a simple, stateful rule applied on each bar,
    which is more readable than a fully vectorized version for this type
    of breakout / exit logic.
    """

    breakout_window: int = 20
    exit_window: int = 10
    price_high_column: str = "High"
    price_low_column: str = "Low"
    price_close_column: str = "Close"
    atr_window: int = 20

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate breakout-based position signals for TXF.

        Parameters
        ----------
        data : pd.DataFrame
            Minute-level price data with at least High/Low/Close columns.

        Returns
        -------
        pd.Series
            Position series (+1 long, 0 flat, -1 short) indexed by datetime.
        """
        required_cols = {self.price_high_column, self.price_low_column, self.price_close_column}
        missing = required_cols - set(data.columns)
        if missing:
            raise KeyError(f"Missing required columns for Turtle strategy: {sorted(missing)}")

        high = data[self.price_high_column]
        low = data[self.price_low_column]
        close = data[self.price_close_column]

        # Donchian channels for entry (breakout_window) and exit (exit_window)
        breakout_high = high.rolling(self.breakout_window, min_periods=self.breakout_window).max().shift(1)
        breakout_low = low.rolling(self.breakout_window, min_periods=self.breakout_window).min().shift(1)

        exit_high = high.rolling(self.exit_window, min_periods=self.exit_window).max().shift(1)
        exit_low = low.rolling(self.exit_window, min_periods=self.exit_window).min().shift(1)

        # ATR 可用於之後加減碼或風險控制，此處先計算但不強制使用
        _atr = atr(high=high, low=low, close=close, window=self.atr_window)

        index = data.index
        positions = np.zeros(len(index), dtype=int)

        for i in range(len(index)):
            if i == 0:
                positions[i] = 0
                continue

            prev_pos = positions[i - 1]
            c = close.iloc[i]

            bh = breakout_high.iloc[i]
            bl = breakout_low.iloc[i]
            eh = exit_high.iloc[i]
            el = exit_low.iloc[i]

            # 若尚未有足夠資料計算 Donchian，保持前一部位
            if pd.isna(bh) or pd.isna(bl) or pd.isna(eh) or pd.isna(el):
                positions[i] = prev_pos
                continue

            if prev_pos == 0:
                # 無部位 → 看是否突破進場
                if c > bh:
                    positions[i] = 1
                elif c < bl:
                    positions[i] = -1
                else:
                    positions[i] = 0
            elif prev_pos == 1:
                # 多單 → 跌破 exit_window 低點則出場
                if c < el:
                    positions[i] = 0
                else:
                    positions[i] = 1
            else:  # prev_pos == -1
                # 空單 → 突破 exit_window 高點則出場
                if c > eh:
                    positions[i] = 0
                else:
                    positions[i] = -1

        return pd.Series(positions, index=index, name="position")
