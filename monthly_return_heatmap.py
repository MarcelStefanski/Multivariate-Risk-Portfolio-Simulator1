import calendar
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
    cost_aware_backtest,
)


def calculate_monthly_returns(daily_returns):
    """Compound daily returns into calendar-month returns."""
    monthly_returns = (1 + daily_returns).groupby(
        daily_returns.index.to_period("M")
    ).prod() - 1
    monthly_returns.index = monthly_returns.index.to_timestamp()
    return monthly_returns


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
    monthly_returns = calculate_monthly_returns(portfolio_returns)

    monthly_data = monthly_returns.rename("Return").to_frame()
    monthly_data["Year"] = monthly_data.index.year
    monthly_data["Month"] = monthly_data.index.month
    monthly_table = monthly_data.pivot(
        index="Year", columns="Month", values="Return"
    ).reindex(columns=range(1, 13))
    month_names = [calendar.month_abbr[month] for month in range(1, 13)]

    weight_summary = ", ".join(
        f"{ticker}: {weight:.0%}"
        for ticker, weight in zip(TICKERS, TARGET_WEIGHTS)
    )
    lines = [
        "MONTHLY PORTFOLIO RETURNS",
        f"Portfolio weights: {weight_summary}",
        f"Rebalancing: {REBALANCE_FREQUENCY}; transaction costs: {TRANSACTION_COST_BPS} bps per dollar traded",
        "Returns are compounded within each calendar month and include transaction costs.",
        f"Period: {monthly_returns.index[0]:%Y-%m} to {monthly_returns.index[-1]:%Y-%m}",
        "",
        "Year | " + " | ".join(month_names),
    ]
    for year, row in monthly_table.iterrows():
        month_values = [
            f"{row[month]:>6.1%}" if pd.notna(row[month]) else "    --"
            for month in range(1, 13)
        ]
        lines.append(f"{year} | " + " | ".join(month_values))

    result = "\n".join(lines)
    print(result)

    figures_dir = Path("results/figures")
    reports_dir = Path("results/reports")
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "monthly_return_heatmap.txt").write_text(
        result + "\n", encoding="utf-8"
    )

    values = monthly_table.to_numpy(dtype=float)
    color_limit = max(np.nanmax(np.abs(values)), 0.01)
    figure, axis = plt.subplots(figsize=(13, 7))
    image = axis.imshow(
        np.ma.masked_invalid(values),
        cmap="RdYlGn",
        vmin=-color_limit,
        vmax=color_limit,
        aspect="auto",
    )
    axis.set_xticks(range(12), month_names)
    axis.set_yticks(range(len(monthly_table.index)), monthly_table.index)
    axis.set_xlabel("Month")
    axis.set_ylabel("Year")
    axis.set_title("Monthly Portfolio Returns After Transaction Costs")

    for row_index, row in enumerate(values):
        for month_index, value in enumerate(row):
            if np.isfinite(value):
                axis.text(
                    month_index,
                    row_index,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Monthly return")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    figure.tight_layout()
    figure.savefig(
        figures_dir / "monthly_return_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    main()
