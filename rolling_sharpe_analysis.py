from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

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


def calculate_rolling_sharpe(portfolio_returns, spy_returns):
    """Calculate rolling annualized Sharpe ratios using a 0% risk-free rate."""
    portfolio_sharpe = (
        portfolio_returns.rolling(ROLLING_WINDOW_DAYS).mean()
        / portfolio_returns.rolling(ROLLING_WINDOW_DAYS).std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )
    spy_sharpe = (
        spy_returns.rolling(ROLLING_WINDOW_DAYS).mean()
        / spy_returns.rolling(ROLLING_WINDOW_DAYS).std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )
    return pd.DataFrame(
        {
            "Portfolio Sharpe ratio": portfolio_sharpe,
            "SPY Sharpe ratio": spy_sharpe,
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
    metrics = calculate_rolling_sharpe(portfolio_returns, spy_returns)

    weight_summary = ", ".join(
        f"{ticker}: {weight:.0%}"
        for ticker, weight in zip(TICKERS, TARGET_WEIGHTS)
    )
    result = f"""ROLLING ONE-YEAR SHARPE RATIO ANALYSIS
Portfolio weights: {weight_summary}
Rebalancing: {REBALANCE_FREQUENCY}; transaction costs: {TRANSACTION_COST_BPS} bps per dollar traded
Risk-free rate: 0%
Rolling window: {ROLLING_WINDOW_DAYS} trading days
Analyzed dates: {metrics.index[0].date()} to {metrics.index[-1].date()}

Portfolio, after costs
Average rolling annualized Sharpe ratio: {metrics['Portfolio Sharpe ratio'].mean():.2f}
Highest rolling annualized Sharpe ratio: {metrics['Portfolio Sharpe ratio'].max():.2f}
Lowest rolling annualized Sharpe ratio: {metrics['Portfolio Sharpe ratio'].min():.2f}

SPY, buy and hold
Average rolling annualized Sharpe ratio: {metrics['SPY Sharpe ratio'].mean():.2f}
Highest rolling annualized Sharpe ratio: {metrics['SPY Sharpe ratio'].max():.2f}
Lowest rolling annualized Sharpe ratio: {metrics['SPY Sharpe ratio'].min():.2f}"""
    print(result)

    figures_dir = Path("results/figures")
    reports_dir = Path("results/reports")
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "rolling_sharpe_analysis.txt").write_text(
        result + "\n", encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(
        metrics.index,
        metrics["Portfolio Sharpe ratio"],
        label="Portfolio, after costs",
    )
    axis.plot(
        metrics.index,
        metrics["SPY Sharpe ratio"],
        label="SPY, buy and hold",
    )
    axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
    axis.set_title("Rolling One-Year Annualized Sharpe Ratio (0% Risk-Free Rate)")
    axis.set_ylabel("Sharpe ratio")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        figures_dir / "rolling_sharpe_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    main()
