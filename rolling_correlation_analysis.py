from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from backtesting import END_DATE, START_DATE, TICKERS, TRADING_DAYS


ROLLING_WINDOW_DAYS = TRADING_DAYS
COMPARISON_TICKERS = [ticker for ticker in TICKERS if ticker != "SPY"]


def calculate_rolling_correlations(asset_returns):
    """Calculate rolling correlations between SPY and the other assets."""
    spy_returns = asset_returns["SPY"]
    return pd.DataFrame(
        {
            ticker: spy_returns.rolling(ROLLING_WINDOW_DAYS).corr(
                asset_returns[ticker]
            )
            for ticker in COMPARISON_TICKERS
        }
    ).dropna()


def main():
    prices = yf.download(TICKERS, start=START_DATE, end=END_DATE)["Close"]
    asset_returns = prices.loc[:, TICKERS].pct_change().dropna()
    correlations = calculate_rolling_correlations(asset_returns)

    result_lines = [
        "ROLLING ONE-YEAR CORRELATION ANALYSIS",
        f"Rolling window: {ROLLING_WINDOW_DAYS} trading days",
        f"Analyzed dates: {correlations.index[0].date()} to {correlations.index[-1].date()}",
    ]
    for ticker in COMPARISON_TICKERS:
        series = correlations[ticker]
        result_lines.extend(
            [
                "",
                f"SPY and {ticker}",
                f"Average rolling correlation: {series.mean():.2f}",
                f"Highest rolling correlation: {series.max():.2f}",
                f"Lowest rolling correlation: {series.min():.2f}",
            ]
        )
    result = "\n".join(result_lines)
    print(result)

    figures_dir = Path("results/figures")
    reports_dir = Path("results/reports")
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "rolling_correlation_analysis.txt").write_text(
        result + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(
        len(COMPARISON_TICKERS), 1, figsize=(11, 9), sharex=True
    )
    for axis, ticker in zip(axes, COMPARISON_TICKERS):
        axis.plot(correlations.index, correlations[ticker])
        axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(f"SPY and {ticker}")
        axis.set_ylabel("Correlation")
        axis.set_ylim(-1, 1)
    axes[-1].set_xlabel("Date")
    figure.suptitle("Rolling One-Year Asset Correlations", y=1.02)
    figure.tight_layout()
    figure.savefig(
        figures_dir / "rolling_correlation_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
