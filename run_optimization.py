from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    apply_risk_overlay,
    run_backtest,
)
from src.data.loader import load_minute_data
from src.data.preprocess import resample_ohlcv
from src.evaluation.reports import summarize_strategies
from src.ml.dataset import build_meta_label_dataset
from src.ml.models import train_meta_label_model, walk_forward_splits
from src.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.turtle_breakout import TurtleBreakoutStrategy
from src.utils.time_utils import filter_day_session


@dataclass
class Candidate:
    name: str
    strategy_type: str
    dataset_mode: str
    freq: str
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    trade_count: int


def _minutes_from_rule(rule: str) -> int:
    return int(rule.lower().replace("min", "").strip())


def _bars_per_year(rule: str) -> int:
    return int(75_000 / _minutes_from_rule(rule))


def _summarize_single(name: str, bt: pd.DataFrame, bars_per_year: int) -> Candidate:
    s = summarize_strategies({name: bt}, bars_per_year=bars_per_year, risk_free_rate=0.0).iloc[0]
    return Candidate(
        name=name,
        strategy_type=name.split("_", 1)[0],
        dataset_mode="day" if "_day_" in name else "all",
        freq=name.split("_")[2],
        annualized_return=float(s["annualized_return"]),
        annualized_volatility=float(s["annualized_volatility"]),
        sharpe_ratio=float(s["sharpe_ratio"]),
        max_drawdown=float(s["max_drawdown"]),
        trade_count=int(s["trade_count"]),
    )


def _get_source_df(df_all: pd.DataFrame, dataset_mode: str) -> pd.DataFrame:
    return filter_day_session(df_all) if dataset_mode == "day" else df_all


def _build_signal_from_candidate(candidate_name: str, dfr: pd.DataFrame) -> pd.Series:
    parts = candidate_name.split("_")
    if parts[0] == "MA":
        short_w = int(parts[3])
        long_w = int(parts[4])
        allow_short = parts[5] == "LS"
        return MACrossoverStrategy(short_w, long_w, allow_short=allow_short).generate_signals(dfr)
    if parts[0] == "BB":
        window = int(parts[3])
        num_std = float(parts[4])
        return BollingerMeanReversionStrategy(window=window, num_std=num_std).generate_signals(dfr)

    breakout_window = int(parts[3])
    exit_window = int(parts[4])
    atr_window = int(parts[5])
    return TurtleBreakoutStrategy(
        breakout_window=breakout_window,
        exit_window=exit_window,
        atr_window=atr_window,
    ).generate_signals(dfr)


def run_grid_search(df_all: pd.DataFrame, engine: BacktestEngine) -> pd.DataFrame:
    datasets = {
        "all": df_all,
        "day": filter_day_session(df_all),
    }

    ma_grid = [
        (5, 20, False),
        (10, 30, False),
        (20, 60, False),
        (30, 90, False),
        (50, 200, False),
        (10, 30, True),
        (20, 60, True),
        (50, 200, True),
    ]
    bb_grid = [
        (20, 1.5),
        (20, 2.0),
        (40, 2.0),
        (60, 2.0),
        (100, 2.5),
        (100, 3.0),
    ]
    turtle_grid = [
        (20, 10, 14),
        (50, 20, 20),
        (55, 20, 20),
        (80, 30, 20),
        (100, 50, 20),
        (200, 100, 40),
    ]
    freqs = ["5min", "10min", "15min", "30min", "60min"]

    rows: list[Candidate] = []

    for dataset_name, df in datasets.items():
        for freq in freqs:
            dfr = resample_ohlcv(df, freq)
            bpy = _bars_per_year(freq)

            for short_w, long_w, allow_short in ma_grid:
                if short_w >= long_w:
                    continue
                strategy = MACrossoverStrategy(
                    short_window=short_w,
                    long_window=long_w,
                    allow_short=allow_short,
                )
                bt = engine.run(dfr, strategy)
                side = "LS" if allow_short else "L"
                name = f"MA_{dataset_name}_{freq}_{short_w}_{long_w}_{side}"
                rows.append(_summarize_single(name, bt, bars_per_year=bpy))

            for window, num_std in bb_grid:
                strategy = BollingerMeanReversionStrategy(window=window, num_std=num_std)
                bt = engine.run(dfr, strategy)
                name = f"BB_{dataset_name}_{freq}_{window}_{num_std}"
                rows.append(_summarize_single(name, bt, bars_per_year=bpy))

            for breakout_window, exit_window, atr_window in turtle_grid:
                strategy = TurtleBreakoutStrategy(
                    breakout_window=breakout_window,
                    exit_window=exit_window,
                    atr_window=atr_window,
                )
                bt = engine.run(dfr, strategy)
                name = (
                    f"TURTLE_{dataset_name}_{freq}_"
                    f"{breakout_window}_{exit_window}_{atr_window}"
                )
                rows.append(_summarize_single(name, bt, bars_per_year=bpy))

    out = pd.DataFrame([r.__dict__ for r in rows]).sort_values(
        ["sharpe_ratio", "annualized_return"],
        ascending=False,
    )
    return out


def evaluate_candidate(
    df_all: pd.DataFrame,
    engine: BacktestEngine,
    candidate_row: pd.Series,
    use_risk_overlay: bool = False,
    target_vol_ann: float = 0.15,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    dataset_name = str(candidate_row["dataset_mode"])
    freq = str(candidate_row["freq"])
    source = _get_source_df(df_all, dataset_name)
    dfr = resample_ohlcv(source, freq)
    bpy = _bars_per_year(freq)

    base_signal = _build_signal_from_candidate(str(candidate_row["name"]), dfr)
    if use_risk_overlay:
        positions = apply_risk_overlay(
            raw_positions=base_signal,
            prices=dfr["Close"],
            bars_per_year=bpy,
            target_vol_ann=target_vol_ann,
            max_leverage=max_leverage,
            drawdown_deleverage_threshold=-0.20,
            min_leverage_after_deleverage=0.35,
            vol_lookback=120,
        )
        fee_rate = engine.config.commission_per_contract / (
            engine.config.contract_multiplier * dfr["Close"].mean()
        )
        slippage_rate = engine.config.slippage_ticks / dfr["Close"].mean()
        bt = run_backtest(
            prices=dfr["Close"],
            positions=positions,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        name = f"{candidate_row['name']}_risk_overlay"
    else:
        bt = engine.run(dfr, strategy=_build_strategy_object(str(candidate_row["name"])))
        name = str(candidate_row["name"])

    return summarize_strategies({name: bt}, bars_per_year=bpy, risk_free_rate=0.0)


def _build_strategy_object(candidate_name: str):
    parts = candidate_name.split("_")
    if parts[0] == "MA":
        return MACrossoverStrategy(
            short_window=int(parts[3]),
            long_window=int(parts[4]),
            allow_short=(parts[5] == "LS"),
        )
    if parts[0] == "BB":
        return BollingerMeanReversionStrategy(window=int(parts[3]), num_std=float(parts[4]))
    return TurtleBreakoutStrategy(
        breakout_window=int(parts[3]),
        exit_window=int(parts[4]),
        atr_window=int(parts[5]),
    )


def evaluate_walk_forward(
    df_all: pd.DataFrame,
    engine: BacktestEngine,
    candidate_row: pd.Series,
    use_risk_overlay: bool = True,
) -> pd.DataFrame:
    dataset_name = str(candidate_row["dataset_mode"])
    freq = str(candidate_row["freq"])
    source = _get_source_df(df_all, dataset_name)
    dfr = resample_ohlcv(source, freq)
    bpy = _bars_per_year(freq)

    signal = _build_signal_from_candidate(str(candidate_row["name"]), dfr)
    if use_risk_overlay:
        signal = apply_risk_overlay(
            raw_positions=signal,
            prices=dfr["Close"],
            bars_per_year=bpy,
            target_vol_ann=0.15,
            max_leverage=3.0,
            drawdown_deleverage_threshold=-0.20,
            min_leverage_after_deleverage=0.35,
            vol_lookback=120,
        )

    n = len(dfr)
    train_size = max(int(n * 0.5), 500)
    test_size = max(int(n * 0.1), 200)
    step_size = test_size
    splits = walk_forward_splits(dfr.index, train_size=train_size, test_size=test_size, step_size=step_size)

    if not splits:
        raise ValueError("Not enough data points to build walk-forward splits.")

    fee_rate = engine.config.commission_per_contract / (engine.config.contract_multiplier * dfr["Close"].mean())
    slippage_rate = engine.config.slippage_ticks / dfr["Close"].mean()

    fold_rows = []
    all_test_returns = []

    for fold_id, (_, test_idx) in enumerate(splits, start=1):
        p_test = dfr.loc[test_idx, "Close"]
        s_test = signal.loc[test_idx]
        bt_fold = run_backtest(
            prices=p_test,
            positions=s_test,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        sm_fold = summarize_strategies({f"fold_{fold_id}": bt_fold}, bars_per_year=bpy, risk_free_rate=0.0).iloc[0]
        fold_rows.append(
            {
                "fold": fold_id,
                "start": str(p_test.index.min()),
                "end": str(p_test.index.max()),
                "annualized_return": float(sm_fold["annualized_return"]),
                "annualized_volatility": float(sm_fold["annualized_volatility"]),
                "sharpe_ratio": float(sm_fold["sharpe_ratio"]),
                "max_drawdown": float(sm_fold["max_drawdown"]),
                "trade_count": int(sm_fold["trade_count"]),
            }
        )
        all_test_returns.append(bt_fold["returns"])

    combined_ret = pd.concat(all_test_returns).sort_index()
    combined_equity = (1.0 + combined_ret).cumprod()
    combined_bt = pd.DataFrame(
        {
            "equity": combined_equity,
            "returns": combined_ret,
            "drawdown": combined_equity / combined_equity.cummax() - 1.0,
            "position": 0.0,
        },
        index=combined_ret.index,
    )
    overall = summarize_strategies({"walk_forward": combined_bt}, bars_per_year=bpy, risk_free_rate=0.0)

    fold_df = pd.DataFrame(fold_rows)
    overall_row = overall.iloc[0].to_dict()
    overall_row["fold"] = "ALL_TEST"
    overall_row["start"] = str(combined_ret.index.min())
    overall_row["end"] = str(combined_ret.index.max())

    return pd.concat([fold_df, pd.DataFrame([overall_row])], ignore_index=True)


def evaluate_meta_labeling(
    df_all: pd.DataFrame,
    engine: BacktestEngine,
    candidate_row: pd.Series,
) -> pd.DataFrame:
    dataset_name = str(candidate_row["dataset_mode"])
    freq = str(candidate_row["freq"])
    source = _get_source_df(df_all, dataset_name)
    dfr = resample_ohlcv(source, freq)
    bpy = _bars_per_year(freq)

    base_signal = _build_signal_from_candidate(str(candidate_row["name"]), dfr)
    fee_rate = engine.config.commission_per_contract / (engine.config.contract_multiplier * dfr["Close"].mean())
    slippage_rate = engine.config.slippage_ticks / dfr["Close"].mean()

    # Baseline on test window (same split convention as meta model).
    base_bt_full = run_backtest(
        prices=dfr["Close"],
        positions=base_signal,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )

    meta_ds = build_meta_label_dataset(
        data=dfr,
        base_signal=base_signal,
        horizon=12,
        take_profit=0.008,
        stop_loss=0.008,
    )

    if len(meta_ds) < 500:
        raise ValueError("Meta-label dataset too small for reliable training.")

    meta_out = train_meta_label_model(
        meta_dataset=meta_ds,
        model_type="rf",
        test_ratio=0.2,
        decision_threshold=0.55,
    )

    exec_signal_test = meta_out["execution_signal"]
    p_test = dfr["Close"].reindex(exec_signal_test.index).ffill().bfill()

    meta_signal_risk = apply_risk_overlay(
        raw_positions=exec_signal_test,
        prices=p_test,
        bars_per_year=bpy,
        target_vol_ann=0.15,
        max_leverage=3.0,
        drawdown_deleverage_threshold=-0.20,
        min_leverage_after_deleverage=0.35,
        vol_lookback=80,
    )

    meta_bt = run_backtest(
        prices=p_test,
        positions=meta_signal_risk,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )

    base_test_bt = base_bt_full.reindex(exec_signal_test.index).dropna()

    base_summary = summarize_strategies({"base_test": base_test_bt}, bars_per_year=bpy, risk_free_rate=0.0).iloc[0]
    meta_summary = summarize_strategies({"meta_test": meta_bt}, bars_per_year=bpy, risk_free_rate=0.0).iloc[0]

    return pd.DataFrame(
        [
            {
                "mode": "base_test",
                "annualized_return": float(base_summary["annualized_return"]),
                "annualized_volatility": float(base_summary["annualized_volatility"]),
                "sharpe_ratio": float(base_summary["sharpe_ratio"]),
                "max_drawdown": float(base_summary["max_drawdown"]),
                "trade_count": int(base_summary["trade_count"]),
            },
            {
                "mode": "meta_test",
                "annualized_return": float(meta_summary["annualized_return"]),
                "annualized_volatility": float(meta_summary["annualized_volatility"]),
                "sharpe_ratio": float(meta_summary["sharpe_ratio"]),
                "max_drawdown": float(meta_summary["max_drawdown"]),
                "trade_count": int(meta_summary["trade_count"]),
                "meta_accuracy": float(meta_out["metrics"]["accuracy"]),
                "meta_f1": float(meta_out["metrics"]["f1"]),
                "meta_auc": None if meta_out["metrics"]["auc"] is None else float(meta_out["metrics"]["auc"]),
            },
        ]
    )


OUTPUT_DIR = Path("outputs")


def main() -> None:
    data_path = Path("TXF_R1_1min_data_combined.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    df = load_minute_data(data_path, tz="Asia/Taipei")
    print(f"Rows={len(df):,}, Range={df.index.min()} ~ {df.index.max()}")

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=1_000_000,
            commission_per_contract=50,
            slippage_ticks=1,
            contract_multiplier=200,
        )
    )

    print("Running grid search...")
    result = run_grid_search(df, engine)
    out_path = OUTPUT_DIR / "strategy_grid_search_results.csv"
    result.to_csv(out_path, index=False)

    print("\nTop 15 by Sharpe:")
    print(result.head(15).to_string(index=False))

    best_per_strategy = (
        result.sort_values(
            ["strategy_type", "sharpe_ratio", "annualized_return"],
            ascending=[True, False, False],
        )
        .groupby("strategy_type", as_index=False)
        .first()
        .sort_values("sharpe_ratio", ascending=False)
    )
    best_out_path = OUTPUT_DIR / "strategy_best_by_type.csv"
    best_per_strategy.to_csv(best_out_path, index=False)

    print("\nBest result for each strategy type (MA / BB / TURTLE):")
    print(best_per_strategy.to_string(index=False))

    print("\nEvaluating risk-overlay on each strategy winner...")
    risk_rows = []
    for _, row in best_per_strategy.iterrows():
        sm = evaluate_candidate(df, engine, row, use_risk_overlay=True)
        s = sm.iloc[0].to_dict()
        s["strategy"] = row["name"]
        risk_rows.append(s)
    risk_df = pd.DataFrame(risk_rows)
    risk_out_path = OUTPUT_DIR / "strategy_best_with_risk_overlay.csv"
    risk_df.to_csv(risk_out_path, index=False)
    print(risk_df[["strategy", "annualized_return", "sharpe_ratio", "max_drawdown", "trade_count"]].to_string(index=False))

    best = best_per_strategy.iloc[0]
    print(f"\nRunning walk-forward for best overall candidate: {best['name']}")
    wf_df = evaluate_walk_forward(df, engine, best, use_risk_overlay=True)
    wf_out_path = OUTPUT_DIR / "walk_forward_best_overall.csv"
    wf_df.to_csv(wf_out_path, index=False)
    print(wf_df.to_string(index=False))

    # Prefer MA winner for meta-labeling. If MA missing, use overall best.
    ma_best = best_per_strategy[best_per_strategy["strategy_type"] == "MA"]
    meta_target = ma_best.iloc[0] if not ma_best.empty else best
    print(f"\nRunning meta-labeling on candidate: {meta_target['name']}")
    meta_df = evaluate_meta_labeling(df, engine, meta_target)
    meta_out_path = OUTPUT_DIR / "meta_labeling_comparison.csv"
    meta_df.to_csv(meta_out_path, index=False)
    print(meta_df.to_string(index=False))

    print(f"\nAll results saved under: {OUTPUT_DIR}/")
    print(f"  - {out_path.name}")
    print(f"  - {best_out_path.name}")
    print(f"  - {risk_out_path.name}")
    print(f"  - {wf_out_path.name}")
    print(f"  - {meta_out_path.name}")


if __name__ == "__main__":
    main()
