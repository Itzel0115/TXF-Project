from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from src.features.technical_indicators import (
    simple_moving_average,
    exponential_moving_average,
    bollinger_bands,
    rsi,
    atr,
)


def _ensure_datetime_index(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame is indexed by a DatetimeIndex.

    If a ``datetime`` column exists and the index is not datetime-like,
    the column will be converted to datetime and set as index.
    """
    if isinstance(data.index, pd.DatetimeIndex):
        return data.sort_index()

    if "datetime" in data.columns:
        df = data.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        return df.sort_index()

    # Fallback: keep original index but ensure sorted order
    return data.sort_index()


def build_ml_dataset(
    data: pd.DataFrame,
    horizon: int,
    feature_windows: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build an ML dataset for direction prediction using TXF 1-minute data.

    The function constructs common technical features and a binary label:
    ``y_t = 1`` if the future return over the next ``horizon`` minutes is
    positive, otherwise ``0``.

    Parameters
    ----------
    data : pd.DataFrame
        Minute-level OHLCV data. Must contain at least
        ``['Open', 'High', 'Low', 'Close', 'Volume']`` and optionally
        a ``datetime`` column (or a DatetimeIndex).
    horizon : int
        Forecast horizon in bars (here bars = 1-minute candles).
    feature_windows : dict | None, default None
        Configuration for feature lookback windows. Supported keys:
        - ``ma_windows``: list[int], e.g. [5, 10, 20, 60]
        - ``bb_windows``: list[int], e.g. [20]
        - ``rsi_windows``: list[int], e.g. [14]
        - ``atr_windows``: list[int], e.g. [14]
        - ``range_windows``: list[int], e.g. [5, 20]
        - ``volume_windows``: list[int], e.g. [20]

        If None, a reasonable default set will be used.

    Returns
    -------
    pd.DataFrame
        DataFrame containing engineered features and a binary label column
        ``'y'``. All rows with any NaN in features or label are removed.
    """
    if feature_windows is None:
        feature_windows = {
            "ma_windows": [5, 10, 20, 60],
            "bb_windows": [20],
            "rsi_windows": [14],
            "atr_windows": [14],
            "range_windows": [5, 20],
            "volume_windows": [20],
        }

    df = _ensure_datetime_index(data)

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    features: Dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # Price-based features
    # ------------------------------------------------------------------
    features["ret_1"] = close.pct_change()
    features["log_ret_1"] = np.log(close / close.shift(1))

    ma_windows = list(feature_windows.get("ma_windows", []))
    ma_windows = sorted({int(w) for w in ma_windows if w is not None and w > 1})

    for w in ma_windows:
        features[f"sma_{w}"] = simple_moving_average(close, window=w)
        features[f"ema_{w}"] = exponential_moving_average(close, window=w)

    # MA spread: short - long for adjacent window pairs
    for i in range(len(ma_windows) - 1):
        s, l = ma_windows[i], ma_windows[i + 1]
        features[f"sma_diff_{s}_{l}"] = features[f"sma_{s}"] - features[f"sma_{l}"]

    # ------------------------------------------------------------------
    # Bollinger features
    # ------------------------------------------------------------------
    bb_windows = list(feature_windows.get("bb_windows", []))
    bb_windows = sorted({int(w) for w in bb_windows if w is not None and w > 1})

    for w in bb_windows:
        bb = bollinger_bands(close, window=w, num_std=2.0)
        middle = bb["middle_band"]
        upper = bb["upper_band"]
        lower = bb["lower_band"]

        width = (upper - lower) / middle.replace(0.0, np.nan)
        percent_b = (close - lower) / (upper - lower).replace(0.0, np.nan)

        features[f"bb_width_{w}"] = width
        features[f"bb_percent_b_{w}"] = percent_b

    # ------------------------------------------------------------------
    # RSI features
    # ------------------------------------------------------------------
    rsi_windows = list(feature_windows.get("rsi_windows", []))
    rsi_windows = sorted({int(w) for w in rsi_windows if w is not None and w > 1})

    for w in rsi_windows:
        features[f"rsi_{w}"] = rsi(close, window=w)

    # ------------------------------------------------------------------
    # ATR features
    # ------------------------------------------------------------------
    atr_windows = list(feature_windows.get("atr_windows", []))
    atr_windows = sorted({int(w) for w in atr_windows if w is not None and w > 1})

    for w in atr_windows:
        features[f"atr_{w}"] = atr(high=high, low=low, close=close, window=w)
        # ATR as a fraction of price (volatility-like)
        features[f"atr_pct_{w}"] = features[f"atr_{w}"] / close.replace(0.0, np.nan)

    # ------------------------------------------------------------------
    # High-low range features over recent windows
    # ------------------------------------------------------------------
    range_windows = list(feature_windows.get("range_windows", []))
    range_windows = sorted({int(w) for w in range_windows if w is not None and w > 1})

    for w in range_windows:
        rolling_high = high.rolling(window=w, min_periods=w).max()
        rolling_low = low.rolling(window=w, min_periods=w).min()
        range_abs = rolling_high - rolling_low
        features[f"hl_range_{w}"] = range_abs
        features[f"hl_range_pct_{w}"] = range_abs / close.replace(0.0, np.nan)

    # ------------------------------------------------------------------
    # Volume-related features
    # ------------------------------------------------------------------
    volume_windows = list(feature_windows.get("volume_windows", []))
    volume_windows = sorted(
        {int(w) for w in volume_windows if w is not None and w > 1}
    )

    for w in volume_windows:
        vol_ma = volume.rolling(window=w, min_periods=w).mean()
        vol_std = volume.rolling(window=w, min_periods=w).std(ddof=0)
        features[f"vol_ma_{w}"] = vol_ma
        features[f"vol_zscore_{w}"] = (volume - vol_ma) / vol_std.replace(0.0, np.nan)
        features[f"vol_ratio_{w}"] = volume / vol_ma.replace(0.0, np.nan)

    # ------------------------------------------------------------------
    # Assemble feature DataFrame
    # ------------------------------------------------------------------
    feature_df = pd.DataFrame(features, index=df.index)

    # ------------------------------------------------------------------
    # Label: future direction over the next `horizon` bars
    # ------------------------------------------------------------------
    future_close = close.shift(-int(horizon))
    future_ret = future_close / close - 1.0
    y = (future_ret > 0).astype(int)

    dataset = feature_df.copy()
    dataset["y"] = y

    # Drop any rows with missing values in features or label
    dataset = dataset.dropna(how="any")

    return dataset

def triple_barrier_meta_label(
    prices: pd.Series,
    side: pd.Series,
    horizon: int = 20,
    take_profit: float = 0.01,
    stop_loss: float = 0.01,
) -> pd.Series:
    """
    Build meta labels (0/1) for candidate trades using a simple triple-barrier rule.

    Label = 1 if a trade reaches take-profit before stop-loss (or ends positive at timeout),
    otherwise 0.
    """
    px = prices.astype(float)
    sd = side.reindex(px.index).fillna(0.0).astype(float)
    out = pd.Series(index=px.index, dtype=float)

    idx = px.index
    vals = px.values
    sides = sd.values
    n = len(px)

    for i in range(n):
        s = sides[i]
        if s == 0:
            out.iloc[i] = np.nan
            continue

        p0 = vals[i]
        end = min(i + int(horizon), n - 1)
        y = 0.0
        decided = False

        for j in range(i + 1, end + 1):
            ret = (vals[j] / p0 - 1.0) * np.sign(s)
            if ret >= take_profit:
                y = 1.0
                decided = True
                break
            if ret <= -stop_loss:
                y = 0.0
                decided = True
                break

        if not decided:
            terminal_ret = (vals[end] / p0 - 1.0) * np.sign(s)
            y = 1.0 if terminal_ret > 0 else 0.0

        out.iloc[i] = y

    return out


def build_meta_label_dataset(
    data: pd.DataFrame,
    base_signal: pd.Series,
    horizon: int = 20,
    take_profit: float = 0.01,
    stop_loss: float = 0.01,
    feature_windows: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build dataset for meta-labeling:
    - features: technical features from build_ml_dataset
    - y_meta: whether taking the base signal is profitable under triple barrier
    - side: sign of base signal (for position reconstruction)
    """
    df = _ensure_datetime_index(data)
    features = build_ml_dataset(
        data=df,
        horizon=horizon,
        feature_windows=feature_windows,
    ).drop(columns=["y"])

    side = base_signal.reindex(df.index).fillna(0.0)
    y_meta = triple_barrier_meta_label(
        prices=df["Close"],
        side=side,
        horizon=horizon,
        take_profit=take_profit,
        stop_loss=stop_loss,
    )

    ds = features.join(
        pd.DataFrame(
            {
                "side": side,
                "abs_side": side.abs(),
                "y_meta": y_meta,
            },
            index=df.index,
        ),
        how="left",
    )

    # Meta-model only learns on timestamps where base strategy wants to trade.
    ds = ds[ds["abs_side"] > 0]
    ds = ds.dropna(how="any")
    ds["y_meta"] = ds["y_meta"].astype(int)
    return ds
