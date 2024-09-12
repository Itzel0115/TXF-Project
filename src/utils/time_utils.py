from __future__ import annotations

from datetime import time
from typing import Iterable, Tuple

import pandas as pd
from zoneinfo import ZoneInfo


TAIPEI_TZ = ZoneInfo("Asia/Taipei")


def localize_to_taipei(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Localize or convert a DatetimeIndex to Asia/Taipei timezone.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Original datetime index.

    Returns
    -------
    pd.DatetimeIndex
        Timezone-aware index in Asia/Taipei.
    """
    if index.tz is None:
        # Naive -> localize to Taipei
        return index.tz_localize(TAIPEI_TZ)
    # Aware -> convert to Taipei
    return index.tz_convert(TAIPEI_TZ)


def filter_day_session(
    df: pd.DataFrame,
    start: time = time(8, 45),
    end: time = time(13, 45),
) -> pd.DataFrame:
    """
    Filter data to include only day session bars.

    Parameters
    ----------
    df : pd.DataFrame
        Minute-level data with datetime index in Asia/Taipei.
    start : datetime.time
        Session start time.
    end : datetime.time
        Session end time.

    Returns
    -------
    pd.DataFrame
        Data within specified session.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    idx = localize_to_taipei(df.index)
    df = df.copy()
    df.index = idx

    # 雿輻 between_time ?寞??啣???蝭??蕪?亦
    try:
        session_df = df.between_time(
            start_time=start,
            end_time=end,
            inclusive="both",
        )
    except TypeError:
        session_df = df.between_time(
            start_time=start,
            end_time=end,
            include_end=True,
        )
    return session_df


def split_by_trading_day(df: pd.DataFrame) -> Iterable[pd.DataFrame]:
    """
    Split continuous minute data into daily segments.

    Parameters
    ----------
    df : pd.DataFrame
        Minute-level data with datetime index.

    Returns
    -------
    Iterable[pd.DataFrame]
        Generator yielding one DataFrame per trading day.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    # ?Ⅱ靽?啣???嚗?敺???calendar ?亙???    idx = localize_to_taipei(df.index)
    df = df.copy()
    df.index = idx

    for _, group in df.groupby(df.index.date):
        yield group
