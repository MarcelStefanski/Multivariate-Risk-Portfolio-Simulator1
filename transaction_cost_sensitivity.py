from pathlib import Path

import matplotlib.pyplot as plt
import yfinance as yf

from backtesting import (
    END_DATE,
    START_DATE,
    TARGET_WEIGHTS,
    TICKERS,
    cost_aware_backtest,
    performance_summary,
)


FREQUENCIES = ("monthly", "quarterly")
COST_LEVELS_BPS = (0, 5, 10, 20)


def main():
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE)
    prices = data["Close"].loc[:, TICKERS]
    asset_returns = prices.pct_change().dropna()

    results = {}
    for frequency in FREQUENCIES:
        for cost_bps in COST_LEVELS_BPS:
            daily_returns, portfolio_values, turnover = cost_aware_backtest(
                asset_returns, TARGET_WEIGHTS, frequency, cost_bps
            )
            summary = performance_summary(daily_returns, portfolio_values)
            summary["turnover"] = turnover
            results[(frequency, cost_bps)] = summary

    lines = [
        "TRANSACTION COST SENSITIVITY",
        f"Period: {START_DATE} to {END_DATE}",
        "Monthly and quarterly rebalancing; returns shown after trading costs.",
        "",
        "Schedule   Cost (bps)   Final Value   Total Return   Annual Return   Volatility   Max Drawdown   Turnover",
    ]
    for frequency in FREQUENCIES:
        for cost_bps in COST_LEVELS_BPS:
            summary = results[(frequency, cost_bps)]
            lines.append(
                f"{frequency.title():<10}"
                f"{cost_bps:>8}"
                f"${summary['final_value']:>12.4f}"
                f"{summary['total_return']:>14.2%}"
                f"{summary['annualized_return']:>15.2%}"
                f"{summary['annualized_volatility']:>12.2%}"
                f"{summary['maximum_drawdown']:>15.2%}"
                f"{summary['turnover']:>11.2%}"
            )

    result_text = "\n".join(lines)
    print(result_text)

    for frequency in FREQUENCIES:
        final_values = [
            results[(frequency, cost_bps)]["final_value"]
            for cost_bps in COST_LEVELS_BPS
        ]
        plt.plot(COST_LEVELS_BPS, final_values, marker="o", label=frequency.title())

    plt.title("Portfolio Value by Transaction Cost")
    plt.xlabel("Transaction Cost (basis points)")
    plt.ylabel("Final Value of $1")
    plt.legend(title="Rebalancing")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        results_dir / "transaction_cost_sensitivity.png",
        dpi=300,
        bbox_inches="tight",
    )
    (results_dir / "transaction_cost_sensitivity.txt").write_text(
        result_text + "\n", encoding="utf-8"
    )
    plt.show()


if __name__ == "__main__":
    main()
