from pathlib import Path

import matplotlib.pyplot as plt
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


CONFIDENCE_LEVEL = 0.95
LOOKBACK_DAYS = TRADING_DAYS


def historical_var_backtest(portfolio_returns, lookback_days, confidence_level):
    """Compare each day's loss with VaR estimated from earlier returns only."""
    if len(portfolio_returns) <= lookback_days:
        raise ValueError("There are not enough returns for the selected lookback.")

    return_threshold = portfolio_returns.shift(1).rolling(lookback_days).quantile(
        1 - confidence_level
    )
    forecasts = pd.DataFrame(
        {
            "VaR": -return_threshold,
            "Actual loss": -portfolio_returns,
        }
    ).dropna()
    forecasts["Breach"] = forecasts["Actual loss"] > forecasts["VaR"]
    return forecasts


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
    forecasts = historical_var_backtest(
        portfolio_returns, LOOKBACK_DAYS, CONFIDENCE_LEVEL
    )

    breach_count = int(forecasts["Breach"].sum())
    breach_rate = forecasts["Breach"].mean()
    weight_summary = ", ".join(
        f"{ticker}: {weight:.0%}"
        for ticker, weight in zip(TICKERS, TARGET_WEIGHTS)
    )
    result = f"""ROLLING HISTORICAL VaR BACKTEST
Portfolio weights: {weight_summary}
Rebalancing: {REBALANCE_FREQUENCY}; transaction costs: {TRANSACTION_COST_BPS} bps per dollar traded
VaR confidence level: {CONFIDENCE_LEVEL:.0%}
Historical lookback: {LOOKBACK_DAYS} trading days
Evaluation period: {forecasts.index[0].date()} to {forecasts.index[-1].date()}

Forecast observations: {len(forecasts)}
VaR breaches: {breach_count}
Observed breach rate: {breach_rate:.2%}
Expected breach rate: {1 - CONFIDENCE_LEVEL:.2%}
Average one-day VaR: {forecasts['VaR'].mean():.2%}

A breach occurs when the actual daily loss is greater than the VaR forecast.
The observed rate is a diagnostic and may differ from the expected rate."""
    print(result)

    figures_dir = Path("results/figures")
    reports_dir = Path("results/reports")
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "var_backtesting.txt").write_text(
        result + "\n", encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(
        forecasts.index,
        forecasts["Actual loss"],
        color="steelblue",
        linewidth=0.8,
        label="Actual daily loss",
    )
    axis.plot(
        forecasts.index,
        forecasts["VaR"],
        color="darkorange",
        label=f"{CONFIDENCE_LEVEL:.0%} VaR",
    )
    breaches = forecasts[forecasts["Breach"]]
    axis.scatter(
        breaches.index,
        breaches["Actual loss"],
        color="red",
        s=18,
        label="VaR breach",
        zorder=3,
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Rolling Historical VaR Backtest")
    axis.set_xlabel("Date")
    axis.set_ylabel("Daily loss")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        figures_dir / "var_backtesting.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    main()
