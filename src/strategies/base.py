from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """
    Abstract base class for TXF trading strategies.

    Strategies are expected to consume a minute-level DataFrame that contains
    at least a ``Close`` column, and output a position series aligned with
    the input index (+1 long, 0 flat, -1 short).
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals (+1 long, 0 flat, -1 short).

        Parameters
        ----------
        data : pd.DataFrame
            Minute-level input data with necessary features.

        Returns
        -------
        pd.Series
            Position series indexed by datetime.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}()"


class BaseStrategy(Strategy):
    """
    Backward-compatible alias of :class:`Strategy`.

    This keeps older imports working, while new code can inherit from
    ``Strategy`` directly.
    """

    pass
