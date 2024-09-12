from typing import Optional

import numpy as np
import pandas as pd

from src.utils.time_utils import localize_to_taipei


def clean_minute_data(
    df: pd.DataFrame,
    tz: str = "Asia/Taipei",
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Clean raw TXF minute data: handle timezone, duplicates, and column validation.

    Steps
    -----
    - Ensure columns: datetime, Open, High, Low, Close, Volume.
    - Convert ``datetime`` column to DatetimeIndex if necessary.
    - Localize / convert to the specified timezone (default Asia/Taipei).
    - Sort by datetime and optionally drop duplicated timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        Raw minute data.
    tz : str, default "Asia/Taipei"
        Target timezone.
    drop_duplicates : bool, default True
        Whether to drop duplicated timestamps.

    Returns
    -------
    pd.DataFrame
        Cleaned data prepared for feature engineering or backtesting.
    """
    data = df.copy()

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing_price_cols = required_cols - set(data.columns)
    if missing_price_cols:
        raise KeyError(f"Missing required OHLCV columns: {sorted(missing_price_cols)}")

    if not isinstance(data.index, pd.DatetimeIndex):
        if "datetime" not in data.columns:
            raise KeyError("Data must have a DatetimeIndex or a 'datetime' column.")
        data["datetime"] = pd.to_datetime(data["datetime"])
        data = data.set_index("datetime")

    # Localize / convert to target timezone (use helper for Asia/Taipei)
    if tz == "Asia/Taipei":
        idx = localize_to_taipei(data.index)
    else:
        idx = data.index
        if idx.tz is None:
            idx = idx.tz_localize(tz)
        else:
            idx = idx.tz_convert(tz)
    data.index = idx

    # Sort and optionally drop duplicated bars (keep last)
    data = data.sort_index()
    if drop_duplicates:
        data = data[~data.index.duplicated(keep="last")]

    return data


def fill_missing_bars(
    df: pd.DataFrame,
    method: str = "ffill",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fill missing minute bars within trading sessions.

    This function reconstructs a continuous 1-minute index between the first
    and last timestamps and then fills gaps according to the chosen method.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned minute data with datetime index.
    method : str, default "ffill"
        Method for filling OHLCV gaps. Supported:
        - "ffill": forward fill
        - "bfill": backward fill
        - "zero" : fill with zeros
        - "none" : do not fill, leave NaNs for missing bars
    limit : int | None
        Maximum number of successive NaNs to fill (for ffill/bfill).

    Returns
    -------
    pd.DataFrame
        Data with missing bars filled or left as NaN according to the method.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    if df.empty:
        return df

    data = df.sort_index().copy()
    full_index = pd.date_range(
        start=data.index[0],
        end=data.index[-1],
        freq="T",
        tz=data.index.tz,
    )
    reindexed = data.reindex(full_index)

    method = method.lower()
    if method == "ffill":
        filled = reindexed.ffill(limit=limit)
    elif method == "bfill":
        filled = reindexed.bfill(limit=limit)
    elif method == "zero":
        filled = reindexed.fillna(0)
    elif method == "none":
        filled = reindexed
    else:
        raise ValueError(f"Unsupported fill method: {method!r}")

    return filled

def resample_ohlcv(
    df: pd.DataFrame,
    rule: str,
) -> pd.DataFrame:
    """
    Resample minute OHLCV bars into lower-frequency OHLCV bars.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with columns ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.
    rule : str
        Pandas resample rule, e.g. ``"5min"``, ``"15min"``, ``"30min"``.

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV DataFrame with rows containing any NaN removed.
    """
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required OHLCV columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    out = pd.DataFrame(index=df.resample(rule).size().index)
    out["Open"] = df["Open"].resample(rule).first()
    out["High"] = df["High"].resample(rule).max()
    out["Low"] = df["Low"].resample(rule).min()
    out["Close"] = df["Close"].resample(rule).last()
    out["Volume"] = df["Volume"].resample(rule).sum()
    return out.dropna(how="any")
