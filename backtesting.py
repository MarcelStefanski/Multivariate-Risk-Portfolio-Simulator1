from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf


TICKERS = ["SPY", "QQQ", "GLD", "TLT"]
TARGET_WEIGHTS = np.array([0.4, 0.3, 0.15, 0.15])
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

# Edit these settings to compare another schedule or transaction-cost rate.
REBALANCE_FREQUENCY = "monthly"  # "monthly" or "quarterly"
TRANSACTION_COST_BPS = 10  # Cost per dollar traded, in basis points.
TRADING_DAYS = 252


def annualized_return(daily_returns):
    return np.mean(daily_returns) * TRADING_DAYS


def annualized_volatility(daily_returns):
    return np.std(daily_returns, ddof=1) * np.sqrt(TRADING_DAYS)


def maximum_drawdown(portfolio_values):
    values_with_initial = np.insert(portfolio_values, 0, 1.0)
    running_max = np.maximum.accumulate(values_with_initial)
    drawdowns = values_with_initial / running_max - 1
    return np.min(drawdowns)


def cost_aware_backtest(asset_returns, target_weights, frequency, cost_bps):
    period_aliases = {"monthly": "M", "quarterly": "Q"}
    if frequency not in period_aliases:
        raise ValueError("REBALANCE_FREQUENCY must be 'monthly' or 'quarterly'.")

    period_ids = asset_returns.index.to_period(period_aliases[frequency])
    rebalance_dates = ~period_ids.duplicated(keep="last")
    cost_rate = cost_bps / 10_000

    current_weights = target_weights.copy()
    portfolio_value = 1.0
    portfolio_values = []
    daily_returns = []
    cumulative_turnover = 0.0

    for asset_daily_returns, should_rebalance in zip(
        asset_returns.to_numpy(), rebalance_dates
    ):
        gross_return = np.dot(current_weights, asset_daily_returns)
        value_after_market_return = portfolio_value * (1 + gross_return)
        drifted_weights = (
            current_weights * (1 + asset_daily_returns) / (1 + gross_return)
        )

        if should_rebalance:
            # Sum of absolute weight changes counts both buys and sells.
            turnover = np.abs(target_weights - drifted_weights).sum()
            transaction_cost = cost_rate * turnover
            value_after_market_return *= 1 - transaction_cost
            current_weights = target_weights.copy()
            cumulative_turnover += turnover
        else:
            current_weights = drifted_weights

        daily_returns.append(value_after_market_return / portfolio_value - 1)
        portfolio_value = value_after_market_return
        portfolio_values.append(portfolio_value)

    return (
        np.asarray(daily_returns),
        np.asarray(portfolio_values),
        cumulative_turnover,
    )


data = yf.download(TICKERS, start=START_DATE, end=END_DATE)
prices = data["Close"].loc[:, TICKERS]
asset_returns = prices.pct_change().dropna()

# Daily frictionless rebalancing remains the baseline for comparison.
daily_baseline_returns = asset_returns.to_numpy() @ TARGET_WEIGHTS
daily_baseline_values = (1 + daily_baseline_returns).cumprod()

cost_aware_returns, cost_aware_values, cumulative_turnover = cost_aware_backtest(
    asset_returns,
    TARGET_WEIGHTS,
    REBALANCE_FREQUENCY,
    TRANSACTION_COST_BPS,
)

baseline_summary = {
    "final_value": daily_baseline_values[-1],
    "total_return": daily_baseline_values[-1] - 1,
    "annualized_return": annualized_return(daily_baseline_returns),
    "annualized_volatility": annualized_volatility(daily_baseline_returns),
    "maximum_drawdown": maximum_drawdown(daily_baseline_values),
}
cost_aware_summary = {
    "final_value": cost_aware_values[-1],
    "total_return": cost_aware_values[-1] - 1,
    "annualized_return": annualized_return(cost_aware_returns),
    "annualized_volatility": annualized_volatility(cost_aware_returns),
    "maximum_drawdown": maximum_drawdown(cost_aware_values),
}

lines = [
    "PORTFOLIO BACKTEST COMPARISON",
    f"Period: {START_DATE} to {END_DATE}",
    "",
    "Daily rebalancing baseline (no transaction costs)",
    f"Final Portfolio Value: ${baseline_summary['final_value']:.4f}",
    f"Total Return: {baseline_summary['total_return']:.2%}",
    f"Annualized Arithmetic Return: {baseline_summary['annualized_return']:.2%}",
    f"Annualized Volatility: {baseline_summary['annualized_volatility']:.2%}",
    f"Maximum Drawdown: {baseline_summary['maximum_drawdown']:.2%}",
    "",
    f"{REBALANCE_FREQUENCY.title()} rebalancing with transaction costs",
    f"Transaction Cost: {TRANSACTION_COST_BPS} bps per dollar traded",
    f"Final Portfolio Value: ${cost_aware_summary['final_value']:.4f}",
    f"Total Return: {cost_aware_summary['total_return']:.2%}",
    f"Annualized Arithmetic Return: {cost_aware_summary['annualized_return']:.2%}",
    f"Annualized Volatility: {cost_aware_summary['annualized_volatility']:.2%}",
    f"Maximum Drawdown: {cost_aware_summary['maximum_drawdown']:.2%}",
    f"Cumulative Two-Way Turnover: {cumulative_turnover:.2%}",
]
result = "\n".join(lines)
print(result)

plt.figure(figsize=(10, 6))
plt.plot(asset_returns.index, daily_baseline_values, label="Daily, no costs")
plt.plot(
    asset_returns.index,
    cost_aware_values,
    label=f"{REBALANCE_FREQUENCY.title()}, {TRANSACTION_COST_BPS} bps costs",
)
plt.title("Portfolio Backtest: Rebalancing and Transaction Costs")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(results_dir / "backtest.png", dpi=300, bbox_inches="tight")

output_path = results_dir / "backtest.txt"
output_path.write_text(result + "\n", encoding="utf-8")

plt.show()
