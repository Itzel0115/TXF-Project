from pathlib import Path
from typing import Union, List

import pandas as pd

from src.data.preprocess import clean_minute_data


def load_minute_data(
    path: Union[str, Path],
    tz: str = "Asia/Taipei",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Load raw 1-minute TXF day session data from a single CSV/parquet file.

    Expected columns: [\"datetime\", \"Open\", \"High\", \"Low\", \"Close\", \"Volume\"].

    Parameters
    ----------
    path : str | Path
        File path to the raw data.
    tz : str, default \"Asia/Taipei\"
        Timezone of the datetime column.
    parse_dates : bool, default True
        Whether to parse \"datetime\" column as pandas datetime.

    Returns
    -------
    pd.DataFrame
        Loaded and cleaned minute-level data.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension for load_minute_data: {suffix!r}")

    # 標準化欄位名稱：接受 datetime / date / time / timestamp 及 OHLCV 大小寫
    def _norm(s: str) -> str:
        n = s.strip().lower()
        if n in ("date", "datetime", "time", "timestamp"):
            return "datetime"
        if n in ("open", "high", "low", "close", "volume"):
            return n
        return s

    new_cols = []
    for c in df.columns:
        n = _norm(c)
        new_cols.append(n if n in ("datetime", "open", "high", "low", "close", "volume") else c)
    df.columns = new_cols
    df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "datetime": "datetime",
        },
        inplace=True,
    )
    if "datetime" not in df.columns:
        raise KeyError(
            "CSV 需包含時間欄位，名稱須為 datetime / date / time / timestamp 之一（大小寫不拘）。"
        )
    if parse_dates:
        df["datetime"] = pd.to_datetime(df["datetime"])

    # 交給 clean_minute_data 處理索引與時區
    cleaned = clean_minute_data(df, tz=tz)
    return cleaned


def load_multiple_files(
    paths: List[Union[str, Path]],
    tz: str = "Asia/Taipei",
    parse_dates: bool = True,
    sort_by_datetime: bool = True,
) -> pd.DataFrame:
    """
    Load and concatenate multiple raw files into one continuous DataFrame.

    Parameters
    ----------
    paths : list[str | Path]
        List of file paths.
    tz : str, default "Asia/Taipei"
        Timezone of the datetime column.
    parse_dates : bool, default True
        Whether to parse "datetime" column as pandas datetime.
    sort_by_datetime : bool, default True
        Whether to sort by datetime after concatenation.

    Returns
    -------
    pd.DataFrame
        Concatenated minute-level data.
    """
    if not paths:
        raise ValueError("paths must contain at least one file path.")

    dfs = [load_minute_data(p, tz=tz, parse_dates=parse_dates) for p in paths]
    combined = pd.concat(dfs, axis=0)

    if sort_by_datetime:
        combined = combined.sort_index()

    # 若多檔之間有重疊的時間戳，保留最後一筆
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined
