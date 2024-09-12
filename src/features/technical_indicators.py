import numpy as np
import pandas as pd


def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    """
    Compute simple moving average (SMA).

    Parameters
    ----------
    series : pd.Series
        Input price series (e.g. Close).
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Simple moving average values with the same index as input.
        The first ``window - 1`` points will be NaN.
    """
    return series.rolling(window=window, min_periods=window).mean()


def exponential_moving_average(series: pd.Series, window: int) -> pd.Series:
    """
    Compute exponential moving average (EMA).

    Parameters
    ----------
    series : pd.Series
        Input price series (e.g. Close).
    window : int
        EMA span parameter (commonly called window).

    Returns
    -------
    pd.Series
        Exponential moving average values with the same index as input.
    """
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def bollinger_bands(
    series: pd.Series,
    window: int,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Compute Bollinger Bands.

    Parameters
    ----------
    series : pd.Series
        Input price series (e.g. Close).
    window : int
        Rolling window size.
    num_std : float, default 2.0
        Number of standard deviations for the bands.

    Returns
    -------
    pd.DataFrame
        Columns:
        - ``middle_band`` : rolling simple moving average
        - ``upper_band`` : middle_band + num_std * rolling_std
        - ``lower_band`` : middle_band - num_std * rolling_std
    """
    middle = simple_moving_average(series, window=window)
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=0)

    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std

    return pd.DataFrame(
        {
            "middle_band": middle,
            "upper_band": upper,
            "lower_band": lower,
        },
        index=series.index,
    )


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI) using Wilder's smoothing.

    Parameters
    ----------
    series : pd.Series
        Input price series (e.g. Close).
    window : int, default 14
        RSI lookback window.

    Returns
    -------
    pd.Series
        RSI values in the range [0, 100].
    """
    delta = series.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    alpha = 1.0 / float(window)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_values = 100.0 - (100.0 / (1.0 + rs))

    return rsi_values


def true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    Compute True Range (TR) used in ATR and breakout systems.

    True range for bar t is defined as the maximum of:
    - high[t] - low[t]
    - |high[t] - close[t-1]|
    - |low[t] - close[t-1]|

    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.

    Returns
    -------
    pd.Series
        True range series.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    Compute Average True Range (ATR) as a simple moving average of true range.

    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    window : int, default 14
        ATR lookback window.

    Returns
    -------
    pd.Series
        ATR series.
    """
    tr = true_range(high=high, low=low, close=close)
    return tr.rolling(window=window, min_periods=window).mean()


# ---------------------------------------------------------------------------
# Backward-compatible helper names
# ---------------------------------------------------------------------------

def rolling_ma(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Backward-compatible wrapper for simple moving average.

    Parameters
    ----------
    series : pd.Series
        Input price series (e.g. Close).
    window : int
        Rolling window size.
    min_periods : int | None
        Ignored; kept for backward compatibility.

    Returns
    -------
    pd.Series
        Simple moving average values.
    """
    return simple_moving_average(series, window=window)
