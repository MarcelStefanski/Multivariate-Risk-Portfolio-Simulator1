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
    cost_aware_backtest,
)


def analyze_drawdowns(daily_returns):
    """Return drawdowns and the duration of each period below a prior peak."""
    portfolio_values = (1 + daily_returns).cumprod()
    previous_peaks = portfolio_values.cummax().clip(lower=1.0)
    drawdowns = portfolio_values / previous_peaks - 1

    episodes = []
    position = 0
    while position < len(drawdowns):
        if drawdowns.iloc[position] >= 0:
            position += 1
            continue

        start = position
        trough = position
        while position < len(drawdowns) and drawdowns.iloc[position] < 0:
            if drawdowns.iloc[position] < drawdowns.iloc[trough]:
                trough = position
            position += 1

        recovered = position < len(drawdowns)
        episodes.append(
            {
                "start_date": drawdowns.index[start],
                "trough_date": drawdowns.index[trough],
                "trough_value": drawdowns.iloc[trough],
                "recovery_date": drawdowns.index[position] if recovered else None,
                "duration_days": position - start,
                "recovered": recovered,
            }
        )

    return drawdowns, episodes


def format_drawdown_summary(label, drawdowns, episodes):
    """Format the deepest drawdown and longest underwater period."""
    worst_date = drawdowns.idxmin()
    lines = [
        label,
        f"Worst peak-to-trough drawdown: {drawdowns.min():.2%} on {worst_date.date()}",
    ]
    if episodes:
        longest = max(episodes, key=lambda episode: episode["duration_days"])
        recovery_date = (
            longest["recovery_date"].date()
            if longest["recovered"]
            else "Not recovered by end of period"
        )
        lines.extend(
            [
                f"Longest period below a previous peak: {longest['duration_days']} trading days",
                f"Started: {longest['start_date'].date()}; lowest point: {longest['trough_date'].date()} ({longest['trough_value']:.2%})",
                f"Recovery: {recovery_date}",
            ]
        )
    else:
        lines.append("No drawdown periods during the analyzed dates.")
    return "\n".join(lines)


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
    portfolio_drawdowns, portfolio_episodes = analyze_drawdowns(portfolio_returns)
    spy_drawdowns, spy_episodes = analyze_drawdowns(spy_returns)

    weight_summary = ", ".join(
        f"{ticker}: {weight:.0%}"
        for ticker, weight in zip(TICKERS, TARGET_WEIGHTS)
    )
    result = "\n\n".join(
        [
            "DRAWDOWN AND RECOVERY ANALYSIS",
            f"Period: {asset_returns.index[0].date()} to {asset_returns.index[-1].date()}\nPortfolio weights: {weight_summary}\nRebalancing: {REBALANCE_FREQUENCY}; transaction costs: {TRANSACTION_COST_BPS} bps per dollar traded",
            format_drawdown_summary(
                "Portfolio, after costs", portfolio_drawdowns, portfolio_episodes
            ),
            format_drawdown_summary("SPY, buy and hold", spy_drawdowns, spy_episodes),
        ]
    )
    print(result)

    figures_dir = Path("results/figures")
    reports_dir = Path("results/reports")
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "drawdown_recovery_analysis.txt").write_text(
        result + "\n", encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(
        portfolio_drawdowns.index,
        portfolio_drawdowns,
        label="Portfolio, after costs",
    )
    axis.plot(spy_drawdowns.index, spy_drawdowns, label="SPY, buy and hold")
    axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
    axis.set_title("Portfolio and SPY Drawdowns from Previous Peaks")
    axis.set_ylabel("Drawdown")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        figures_dir / "drawdown_recovery_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    main()
