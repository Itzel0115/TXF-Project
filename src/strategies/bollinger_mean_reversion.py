from dataclasses import dataclass

import pandas as pd

from .base import Strategy
from ..features.technical_indicators import bollinger_bands


@dataclass
class BollingerMeanReversionStrategy(Strategy):
    """
    Mean-reversion strategy based on Bollinger Bands.

    Example logic:
    - Long when price closes below lower band.
    - Short when price closes above upper band.
    """

    window: int = 20
    num_std: float = 2.0
    price_column: str = "Close"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate mean-reversion signals using Bollinger Bands.

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
        bands = bollinger_bands(price, window=self.window, num_std=self.num_std)

        upper = bands["upper_band"]
        lower = bands["lower_band"]

        signals = pd.Series(data=0, index=data.index, dtype=float)
        signals[price < lower] = 1.0
        signals[price > upper] = -1.0

        return signals.astype(int)
