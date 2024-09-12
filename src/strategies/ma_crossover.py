from dataclasses import dataclass

import pandas as pd

from .base import Strategy
from ..features.technical_indicators import simple_moving_average


@dataclass
class MACrossoverStrategy(Strategy):
    """
    Moving average crossover strategy based on two simple moving averages.

    Default behavior:
    - When short MA > long MA: long (+1)
    - When short MA < long MA: short (-1) if ``allow_short`` is True,
      otherwise flat (0)
    - Otherwise: flat (0)
    """

    short_window: int = 10
    long_window: int = 20
    price_column: str = "Close"
    allow_short: bool = True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on simple moving average crossover.

        Parameters
        ----------
        data : pd.DataFrame
            Minute-level price data. Must contain the specified price column.

        Returns
        -------
        pd.Series
            Position series (+1, 0, -1) indexed by datetime.
        """
        if self.price_column not in data.columns:
            raise KeyError(f"Column '{self.price_column}' not found in input data.")

        price = data[self.price_column]

        short_ma = simple_moving_average(price, window=self.short_window)
        long_ma = simple_moving_average(price, window=self.long_window)

        signals = pd.Series(data=0, index=data.index, dtype=float)
        signals[short_ma > long_ma] = 1.0
        if self.allow_short:
            signals[short_ma < long_ma] = -1.0
        else:
            signals[short_ma < long_ma] = 0.0

        return signals.astype(int)
