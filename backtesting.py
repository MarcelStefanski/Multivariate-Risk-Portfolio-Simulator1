from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf


TICKERS = ["SPY", "QQQ", "GLD", "TLT"]
TARGET_WEIGHTS = np.array([0.4, 0.3, 0.15, 0.15])
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
REBALANCE_FREQUENCY = "monthly"  # Change to "quarterly" if desired.
TRANSACTION_COST_BPS = 10
TRADING_DAYS = 252


def cost_aware_backtest(asset_returns, target_weights, frequency, cost_bps):
    period_code = {"monthly": "M", "quarterly": "Q"}.get(frequency)
    if period_code is None:
        raise ValueError("Frequency must be 'monthly' or 'quarterly'.")

    rebalance_dates = ~asset_returns.index.to_period(period_code).duplicated(
        keep="last"
    )
    cost_rate = cost_bps / 10_000
    current_weights = target_weights.copy()
    portfolio_value = 1.0
    daily_returns, portfolio_values = [], []
    cumulative_turnover = 0.0

    for daily_asset_returns, should_rebalance in zip(
        asset_returns.to_numpy(), rebalance_dates
    ):
        previous_value = portfolio_value
        portfolio_return = np.dot(current_weights, daily_asset_returns)
        portfolio_value *= 1 + portfolio_return
        current_weights *= 1 + daily_asset_returns
        current_weights /= 1 + portfolio_return

        if should_rebalance:
            turnover = np.abs(target_weights - current_weights).sum()
            portfolio_value *= 1 - cost_rate * turnover
            cumulative_turnover += turnover
            current_weights = target_weights.copy()

        daily_returns.append(portfolio_value / previous_value - 1)
        portfolio_values.append(portfolio_value)

    return np.array(daily_returns), np.array(portfolio_values), cumulative_turnover


def annualized_return(daily_returns):
    return np.mean(daily_returns) * TRADING_DAYS


def annualized_volatility(daily_returns):
    return np.std(daily_returns, ddof=1) * np.sqrt(TRADING_DAYS)


def maximum_drawdown(portfolio_values):
    values = np.insert(portfolio_values, 0, 1.0)
    drawdowns = values / np.maximum.accumulate(values) - 1
    return np.min(drawdowns)


def performance_summary(daily_returns, portfolio_values):
    return {
        "final_value": portfolio_values[-1],
        "total_return": portfolio_values[-1] - 1,
        "annualized_return": annualized_return(daily_returns),
        "annualized_volatility": annualized_volatility(daily_returns),
        "maximum_drawdown": maximum_drawdown(portfolio_values),
    }


def main():
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE)
    prices = data["Close"].loc[:, TICKERS]
    asset_returns = prices.pct_change().dropna()

    # The original daily, zero-cost strategy is kept as a baseline.
    baseline_returns = asset_returns.to_numpy() @ TARGET_WEIGHTS
    baseline_values = (1 + baseline_returns).cumprod()
    cost_returns, cost_values, turnover = cost_aware_backtest(
        asset_returns, TARGET_WEIGHTS, REBALANCE_FREQUENCY, TRANSACTION_COST_BPS
    )

    baseline = performance_summary(baseline_returns, baseline_values)
    cost_aware = performance_summary(cost_returns, cost_values)

    result = f"""PORTFOLIO BACKTEST COMPARISON
Period: {START_DATE} to {END_DATE}

Daily rebalancing baseline (no transaction costs)
Final Portfolio Value: ${baseline['final_value']:.4f}
Total Return: {baseline['total_return']:.2%}
Annualized Arithmetic Return: {baseline['annualized_return']:.2%}
Annualized Volatility: {baseline['annualized_volatility']:.2%}
Maximum Drawdown: {baseline['maximum_drawdown']:.2%}

{REBALANCE_FREQUENCY.title()} rebalancing with transaction costs
Transaction Cost: {TRANSACTION_COST_BPS} bps per dollar traded
Final Portfolio Value: ${cost_aware['final_value']:.4f}
Total Return: {cost_aware['total_return']:.2%}
Annualized Arithmetic Return: {cost_aware['annualized_return']:.2%}
Annualized Volatility: {cost_aware['annualized_volatility']:.2%}
Maximum Drawdown: {cost_aware['maximum_drawdown']:.2%}
Cumulative Two-Way Turnover: {turnover:.2%}"""
    print(result)

    plt.figure(figsize=(10, 6))
    plt.plot(asset_returns.index, baseline_values, label="Daily, no costs")
    plt.plot(
        asset_returns.index,
        cost_values,
        label=f"{REBALANCE_FREQUENCY.title()}, {TRANSACTION_COST_BPS} bps costs",
    )
    plt.title("Portfolio Backtest: Rebalancing and Transaction Costs")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "backtest.png", dpi=300, bbox_inches="tight")
    (results_dir / "backtest.txt").write_text(result + "\n", encoding="utf-8")
    plt.show()


if __name__ == "__main__":
    main()
