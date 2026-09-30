from pathlib import Path

import matplotlib.pyplot as plt
import yfinance as yf

from backtesting import (
    END_DATE,
    START_DATE,
    TARGET_WEIGHTS,
    TICKERS,
    TRANSACTION_COST_BPS,
    cost_aware_backtest,
    performance_summary,
)


FREQUENCIES = ("monthly", "quarterly")


def main():
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE)
    prices = data["Close"].loc[:, TICKERS]
    asset_returns = prices.pct_change().dropna()

    results = {}
    plt.figure(figsize=(10, 6))

    for frequency in FREQUENCIES:
        daily_returns, portfolio_values, turnover = cost_aware_backtest(
            asset_returns, TARGET_WEIGHTS, frequency, TRANSACTION_COST_BPS
        )
        results[frequency] = performance_summary(daily_returns, portfolio_values)
        results[frequency]["turnover"] = turnover
        plt.plot(asset_returns.index, portfolio_values, label=frequency.title())

    result_text = f"""REBALANCING SCHEDULE COMPARISON
Period: {START_DATE} to {END_DATE}
Transaction cost: {TRANSACTION_COST_BPS} bps per dollar traded

Monthly
Final Value: ${results['monthly']['final_value']:.4f}
Total Return: {results['monthly']['total_return']:.2%}
Annualized Arithmetic Return: {results['monthly']['annualized_return']:.2%}
Annualized Volatility: {results['monthly']['annualized_volatility']:.2%}
Maximum Drawdown: {results['monthly']['maximum_drawdown']:.2%}
Cumulative Two-Way Turnover: {results['monthly']['turnover']:.2%}

Quarterly
Final Value: ${results['quarterly']['final_value']:.4f}
Total Return: {results['quarterly']['total_return']:.2%}
Annualized Arithmetic Return: {results['quarterly']['annualized_return']:.2%}
Annualized Volatility: {results['quarterly']['annualized_volatility']:.2%}
Maximum Drawdown: {results['quarterly']['maximum_drawdown']:.2%}
Cumulative Two-Way Turnover: {results['quarterly']['turnover']:.2%}"""
    print(result_text)

    plt.title("Monthly vs Quarterly Rebalancing")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        results_dir / "rebalancing_comparison.png", dpi=300, bbox_inches="tight"
    )
    (results_dir / "rebalancing_comparison.txt").write_text(
        result_text + "\n", encoding="utf-8"
    )
    plt.show()


if __name__ == "__main__":
    main()
