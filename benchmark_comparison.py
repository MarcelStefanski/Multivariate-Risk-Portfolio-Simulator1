from pathlib import Path

import matplotlib.pyplot as plt
import yfinance as yf

from backtesting import (
    END_DATE,
    REBALANCE_FREQUENCY,
    START_DATE,
    TARGET_WEIGHTS,
    TRANSACTION_COST_BPS,
    cost_aware_backtest,
    performance_summary,
)


TICKERS = ["SPY", "QQQ", "GLD", "TLT"]


def main():
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE)
    prices = data["Close"].loc[:, TICKERS]
    asset_returns = prices.pct_change().dropna()

    portfolio_returns, portfolio_values, turnover = cost_aware_backtest(
        asset_returns, TARGET_WEIGHTS, REBALANCE_FREQUENCY, TRANSACTION_COST_BPS
    )

    # SPY is the buy-and-hold benchmark over the same dates.
    spy_returns = asset_returns["SPY"].to_numpy()
    spy_values = (1 + spy_returns).cumprod()
    portfolio = performance_summary(portfolio_returns, portfolio_values)
    spy = performance_summary(spy_returns, spy_values)

    result = f"""COST-AWARE PORTFOLIO VS SPY BENCHMARK
Period: {START_DATE} to {END_DATE}
Portfolio rebalancing: {REBALANCE_FREQUENCY}
Portfolio transaction cost: {TRANSACTION_COST_BPS} bps per dollar traded
SPY benchmark: buy and hold

Portfolio
Final Value: ${portfolio['final_value']:.4f}
Total Return: {portfolio['total_return']:.2%}
Annualized Arithmetic Return: {portfolio['annualized_return']:.2%}
Annualized Volatility: {portfolio['annualized_volatility']:.2%}
Maximum Drawdown: {portfolio['maximum_drawdown']:.2%}
Cumulative Two-Way Turnover: {turnover:.2%}

SPY
Final Value: ${spy['final_value']:.4f}
Total Return: {spy['total_return']:.2%}
Annualized Arithmetic Return: {spy['annualized_return']:.2%}
Annualized Volatility: {spy['annualized_volatility']:.2%}
Maximum Drawdown: {spy['maximum_drawdown']:.2%}"""
    print(result)

    plt.figure(figsize=(10, 6))
    plt.plot(
        asset_returns.index,
        portfolio_values,
        label=f"Portfolio: {REBALANCE_FREQUENCY}, {TRANSACTION_COST_BPS} bps costs",
    )
    plt.plot(asset_returns.index, spy_values, label="SPY buy and hold")
    plt.title("Cost-Aware Portfolio vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        results_dir / "benchmark_comparison.png", dpi=300, bbox_inches="tight"
    )
    (results_dir / "benchmark_comparison.txt").write_text(
        result + "\n", encoding="utf-8"
    )
    plt.show()


if __name__ == "__main__":
    main()
