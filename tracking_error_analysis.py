from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.ticker import PercentFormatter

from backtesting import (
    END_DATE,
    REBALANCE_FREQUENCY,
    START_DATE,
    TARGET_WEIGHTS,
    TICKERS,
    TRANSACTION_COST_BPS,
    TRADING_DAYS,
    cost_aware_backtest,
)


ROLLING_WINDOW_DAYS = TRADING_DAYS


def calculate_rolling_tracking_error(active_returns):
    """Calculate annualized tracking error over rolling one-year windows."""
    return active_returns.rolling(ROLLING_WINDOW_DAYS).std(ddof=1) * np.sqrt(
        TRADING_DAYS
    )


def main():
    prices = yf.download(TICKERS, start=START_DATE, end=END_DATE)["Close"]
    asset_returns = prices.loc[:, TICKERS].pct_change().dropna()
    portfolio_daily_returns, _, _ = cost_aware_backtest(
        asset_returns,
        TARGET_WEIGHTS,
        REBALANCE_FREQUENCY,
        TRANSACTION_COST_BPS,
    )
    portfolio_returns = pd.Series(
        portfolio_daily_returns, index=asset_returns.index, name="Portfolio"
    )
    spy_returns = asset_returns["SPY"].rename("SPY")
    active_returns = (portfolio_returns - spy_returns).rename("Active returns")

    daily_active_risk = active_returns.std(ddof=1)
    tracking_error = daily_active_risk * np.sqrt(TRADING_DAYS)
    information_ratio = (
        active_returns.mean() / daily_active_risk * np.sqrt(TRADING_DAYS)
        if daily_active_risk != 0
        else float("nan")
    )
    rolling_tracking_error = calculate_rolling_tracking_error(active_returns).dropna()

    weight_summary = ", ".join(
        f"{ticker}: {weight:.0%}"
        for ticker, weight in zip(TICKERS, TARGET_WEIGHTS)
    )
    result = f"""TRACKING ERROR AND INFORMATION RATIO ANALYSIS
Portfolio weights: {weight_summary}
Rebalancing: {REBALANCE_FREQUENCY}; transaction costs: {TRANSACTION_COST_BPS} bps per dollar traded
Benchmark: SPY, buy and hold
Analyzed dates: {active_returns.index[0].date()} to {active_returns.index[-1].date()}
Rolling window: {ROLLING_WINDOW_DAYS} trading days

Full-period annualized tracking error: {tracking_error:.2%}
Full-period information ratio: {information_ratio:.2f}

Rolling annualized tracking error
Average: {rolling_tracking_error.mean():.2%}
Highest: {rolling_tracking_error.max():.2%}
Lowest: {rolling_tracking_error.min():.2%}

Tracking error measures the volatility of daily portfolio returns minus SPY returns.
The information ratio scales average active returns by tracking error."""
    print(result)

    figures_dir = Path("results/figures")
    reports_dir = Path("results/reports")
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "tracking_error_analysis.txt").write_text(
        result + "\n", encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(
        rolling_tracking_error.index,
        rolling_tracking_error,
        label="Rolling one-year tracking error",
    )
    axis.axhline(
        tracking_error,
        color="black",
        linewidth=0.8,
        linestyle="--",
        label="Full-period tracking error",
    )
    axis.set_title("Rolling One-Year Annualized Tracking Error vs SPY")
    axis.set_ylabel("Tracking error")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        figures_dir / "tracking_error_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    main()
