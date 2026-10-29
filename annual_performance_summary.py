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


def calculate_annual_summary(portfolio_returns, spy_returns):
    """Calculate calendar-year returns and annualized volatility."""
    daily_returns = pd.concat([portfolio_returns, spy_returns], axis=1).dropna()
    annual_returns = (1 + daily_returns).groupby(daily_returns.index.year).prod() - 1
    annual_volatility = (
        daily_returns.groupby(daily_returns.index.year).std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )

    annual_returns.columns = [f"{name} return" for name in annual_returns.columns]
    annual_volatility.columns = [
        f"{name} volatility" for name in annual_volatility.columns
    ]
    return pd.concat([annual_returns, annual_volatility], axis=1)


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
    summary = calculate_annual_summary(portfolio_returns, spy_returns)

    weight_summary = ", ".join(
        f"{ticker}: {weight:.0%}"
        for ticker, weight in zip(TICKERS, TARGET_WEIGHTS)
    )
    lines = [
        "ANNUAL PORTFOLIO PERFORMANCE SUMMARY",
        f"Portfolio weights: {weight_summary}",
        f"Portfolio rebalancing: {REBALANCE_FREQUENCY}; transaction costs: {TRANSACTION_COST_BPS} bps per dollar traded",
        "Portfolio returns include transaction costs; SPY is buy and hold.",
        "Volatility is annualized from daily returns.",
        "",
        "Year | Portfolio return | SPY return | Portfolio volatility | SPY volatility",
    ]
    for year, row in summary.iterrows():
        lines.append(
            f"{year} | "
            f"{row['Portfolio return']:>16.2%} | "
            f"{row['SPY return']:>10.2%} | "
            f"{row['Portfolio volatility']:>20.2%} | "
            f"{row['SPY volatility']:>13.2%}"
        )

    result = "\n".join(lines)
    print(result)

    figures_dir = Path("results/figures")
    reports_dir = Path("results/reports")
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "annual_performance_summary.txt").write_text(
        result + "\n", encoding="utf-8"
    )

    positions = np.arange(len(summary))
    bar_width = 0.38
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].bar(
        positions - bar_width / 2,
        summary["Portfolio return"],
        bar_width,
        label="Portfolio, after costs",
    )
    axes[0].bar(
        positions + bar_width / 2,
        summary["SPY return"],
        bar_width,
        label="SPY, buy and hold",
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Calendar-Year Return")
    axes[0].set_ylabel("Return")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend()

    axes[1].bar(
        positions - bar_width / 2,
        summary["Portfolio volatility"],
        bar_width,
        label="Portfolio, after costs",
    )
    axes[1].bar(
        positions + bar_width / 2,
        summary["SPY volatility"],
        bar_width,
        label="SPY, buy and hold",
    )
    axes[1].set_title("Annualized Volatility by Year")
    axes[1].set_ylabel("Volatility")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_xticks(positions, summary.index.astype(str))
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(
        figures_dir / "annual_performance_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
