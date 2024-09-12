from __future__ import annotations

from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_equity_curve(
    equity: pd.Series,
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    繪製淨值曲線。

    Parameters
    ----------
    equity : pd.Series
        淨值曲線，index 為 datetime。
    title : str, optional
        圖標題，預設為 "Equity Curve"。
    ax : matplotlib.axes.Axes, optional
        若提供則在該 axes 上繪圖；否則建立新 figure。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    if not title:
        title = "Equity Curve"

    equity = equity.dropna()
    if equity.empty:
        return

    ax.plot(equity.index, equity.values, color="steelblue", linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.6)


def plot_drawdown(
    drawdown: pd.Series,
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    繪製回撤曲線（水下曲線）。

    drawdown 通常為 0 到負值，此函式以填滿方式顯示回撤區域。

    Parameters
    ----------
    drawdown : pd.Series
        回撤序列（例如 equity / equity.cummax() - 1），index 為 datetime。
    title : str, optional
        圖標題，預設為 "Drawdown"。
    ax : matplotlib.axes.Axes, optional
        若提供則在該 axes 上繪圖；否則建立新 figure。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    if not title:
        title = "Drawdown"

    drawdown = drawdown.dropna()
    if drawdown.empty:
        return

    ax.fill_between(
        drawdown.index,
        drawdown.values,
        0,
        color="coral",
        alpha=0.6,
    )
    ax.plot(drawdown.index, drawdown.values, color="darkred", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=min(drawdown.min(), 0) * 1.05, top=0.02)


def plot_monthly_return_heatmap(
    returns: pd.Series,
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    將分鐘報酬聚合成月度報酬，並以年份×月份 heatmap 顯示。

    Parameters
    ----------
    returns : pd.Series
        每根 K 線的報酬率，index 為 datetime。
    title : str, optional
        圖標題，預設為 "Monthly Returns"。
    ax : matplotlib.axes.Axes, optional
        若提供則在該 axes 上繪圖；否則建立新 figure。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, max(4, len(returns.index.year.unique()) * 0.35)))
    if not title:
        title = "Monthly Returns"

    returns = returns.dropna()
    if returns.empty:
        return

    # 以月末 resample（pandas 2.0 用 'ME'，舊版可用 'M'）
    try:
        monthly = (1.0 + returns).resample("ME").prod() - 1.0
    except TypeError:
        monthly = (1.0 + returns).resample("M").prod() - 1.0

    monthly = monthly.dropna()
    if monthly.empty:
        return

    df = pd.DataFrame({"ret": monthly}, index=monthly.index)
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(values="ret", index="year", columns="month")
    # 補齊 1–12 月（缺的月份為 NaN）
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    pivot = pivot.sort_index(ascending=False)  # 最近年度在上方

    data = pivot.values
    vmin = np.nanmin(data) if np.any(~np.isnan(data)) else 0
    vmax = np.nanmax(data) if np.any(~np.isnan(data)) else 0
    vabs = max(abs(vmin), abs(vmax), 1e-8)
    vmin, vmax = -vabs, vabs

    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )

    ax.set_xticks(np.arange(12))
    ax.set_xticklabels([f"{m}" for m in range(1, 13)])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int))
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title(title)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vabs * 0.5 else "black"
                ax.text(
                    j, i, f"{val:.1%}",
                    ha="center", va="center", fontsize=8, color=color,
                )

    plt.colorbar(im, ax=ax, label="Return", shrink=0.8)
