from pathlib import Path

import matplotlib.pyplot as plt
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


def calculate_rolling_beta(portfolio_returns, spy_returns):
    """Calculate the portfolio's rolling beta relative to SPY."""
    rolling_covariance = portfolio_returns.rolling(ROLLING_WINDOW_DAYS).cov(
        spy_returns
    )
    rolling_spy_variance = spy_returns.rolling(ROLLING_WINDOW_DAYS).var()
    return (rolling_covariance / rolling_spy_variance).dropna()


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
    rolling_beta = calculate_rolling_beta(portfolio_returns, spy_returns)

    weight_summary = ", ".join(
        f"{ticker}: {weight:.0%}"
        for ticker, weight in zip(TICKERS, TARGET_WEIGHTS)
    )
    result = f"""ROLLING ONE-YEAR PORTFOLIO BETA ANALYSIS
Portfolio weights: {weight_summary}
Rebalancing: {REBALANCE_FREQUENCY}; transaction costs: {TRANSACTION_COST_BPS} bps per dollar traded
Benchmark: SPY
Rolling window: {ROLLING_WINDOW_DAYS} trading days
Analyzed dates: {rolling_beta.index[0].date()} to {rolling_beta.index[-1].date()}

Average rolling beta: {rolling_beta.mean():.2f}
Highest rolling beta: {rolling_beta.max():.2f}
Lowest rolling beta: {rolling_beta.min():.2f}

A beta of 1 indicates the portfolio moved with SPY at about the same magnitude over the window."""
    print(result)

    figures_dir = Path("results/figures")
    reports_dir = Path("results/reports")
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "rolling_beta_analysis.txt").write_text(
        result + "\n", encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(rolling_beta.index, rolling_beta, label="Portfolio beta to SPY")
    axis.axhline(
        1,
        color="black",
        linewidth=0.8,
        linestyle="--",
        label="Beta = 1 reference",
    )
    axis.set_title("Rolling One-Year Portfolio Beta to SPY")
    axis.set_ylabel("Beta")
    axis.set_xlabel("Date")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        figures_dir / "rolling_beta_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    main()
