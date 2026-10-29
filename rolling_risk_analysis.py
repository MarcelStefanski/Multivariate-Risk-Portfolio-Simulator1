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


def maximum_drawdown_in_window(daily_returns):
    """Return the largest peak-to-trough loss within one return window."""
    values = np.insert(np.cumprod(1 + daily_returns), 0, 1.0)
    drawdowns = values / np.maximum.accumulate(values) - 1
    return np.min(drawdowns)


def calculate_rolling_metrics(portfolio_returns, spy_returns):
    """Calculate one-year annualized volatility and maximum drawdown."""
    return pd.DataFrame(
        {
            "Portfolio volatility": portfolio_returns.rolling(
                ROLLING_WINDOW_DAYS
            ).std(ddof=1)
            * np.sqrt(TRADING_DAYS),
            "SPY volatility": spy_returns.rolling(ROLLING_WINDOW_DAYS).std(ddof=1)
            * np.sqrt(TRADING_DAYS),
            "Portfolio drawdown": portfolio_returns.rolling(
                ROLLING_WINDOW_DAYS
            ).apply(maximum_drawdown_in_window, raw=True),
            "SPY drawdown": spy_returns.rolling(ROLLING_WINDOW_DAYS).apply(
                maximum_drawdown_in_window, raw=True
            ),
        }
    ).dropna()


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
    metrics = calculate_rolling_metrics(portfolio_returns, spy_returns)

    weight_summary = ", ".join(
        f"{ticker}: {weight:.0%}"
        for ticker, weight in zip(TICKERS, TARGET_WEIGHTS)
    )
    result = f"""ROLLING ONE-YEAR RISK ANALYSIS
Portfolio weights: {weight_summary}
Rebalancing: {REBALANCE_FREQUENCY}; transaction costs: {TRANSACTION_COST_BPS} bps per dollar traded
Rolling window: {ROLLING_WINDOW_DAYS} trading days
Analyzed dates: {metrics.index[0].date()} to {metrics.index[-1].date()}

Portfolio, after costs
Average rolling annualized volatility: {metrics['Portfolio volatility'].mean():.2%}
Highest rolling annualized volatility: {metrics['Portfolio volatility'].max():.2%}
Worst rolling one-year drawdown: {metrics['Portfolio drawdown'].min():.2%}

SPY, buy and hold
Average rolling annualized volatility: {metrics['SPY volatility'].mean():.2%}
Highest rolling annualized volatility: {metrics['SPY volatility'].max():.2%}
Worst rolling one-year drawdown: {metrics['SPY drawdown'].min():.2%}"""
    print(result)

    figures_dir = Path("results/figures")
    reports_dir = Path("results/reports")
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "rolling_risk_analysis.txt").write_text(
        result + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(
        metrics.index,
        metrics["Portfolio volatility"],
        label="Portfolio, after costs",
    )
    axes[0].plot(metrics.index, metrics["SPY volatility"], label="SPY, buy and hold")
    axes[0].set_title("Rolling One-Year Annualized Volatility")
    axes[0].set_ylabel("Volatility")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend()

    axes[1].plot(
        metrics.index,
        metrics["Portfolio drawdown"],
        label="Portfolio, after costs",
    )
    axes[1].plot(metrics.index, metrics["SPY drawdown"], label="SPY, buy and hold")
    axes[1].set_title("Worst Drawdown Within Each Rolling One-Year Window")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(
        figures_dir / "rolling_risk_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    main()
