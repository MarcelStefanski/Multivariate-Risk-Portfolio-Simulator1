from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from backtesting import (
    END_DATE,
    REBALANCE_FREQUENCY,
    START_DATE,
    TARGET_WEIGHTS,
    TICKERS,
    TRANSACTION_COST_BPS,
    cost_aware_backtest,
    performance_summary,
)


HOLDOUT_START_DATE = "2021-01-01"


def main():
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE)
    prices = data["Close"].loc[:, TICKERS]
    asset_returns = prices.pct_change().dropna()

    # Use earlier years for context and the later years as the holdout period.
    training_returns = asset_returns.loc[asset_returns.index < HOLDOUT_START_DATE]
    holdout_returns = asset_returns.loc[asset_returns.index >= HOLDOUT_START_DATE]

    training_daily_returns, training_values, training_turnover = cost_aware_backtest(
        training_returns,
        TARGET_WEIGHTS,
        REBALANCE_FREQUENCY,
        TRANSACTION_COST_BPS,
    )
    holdout_daily_returns, holdout_values, holdout_turnover = cost_aware_backtest(
        holdout_returns,
        TARGET_WEIGHTS,
        REBALANCE_FREQUENCY,
        TRANSACTION_COST_BPS,
    )

    spy_returns = holdout_returns["SPY"].to_numpy()
    spy_values = np.cumprod(1 + spy_returns)
    training = performance_summary(training_daily_returns, training_values)
    holdout = performance_summary(holdout_daily_returns, holdout_values)
    spy = performance_summary(spy_returns, spy_values)

    result = f"""OUT-OF-SAMPLE BACKTEST
Portfolio weights are fixed at 40% SPY, 30% QQQ, 15% GLD, and 15% TLT.
Rebalancing: {REBALANCE_FREQUENCY}; transaction cost: {TRANSACTION_COST_BPS} bps per dollar traded.
The weights are not optimized using holdout-period data.

Earlier period: {START_DATE} to {HOLDOUT_START_DATE}
Portfolio final value: ${training['final_value']:.4f}
Total return: {training['total_return']:.2%}
Annualized arithmetic return: {training['annualized_return']:.2%}
Annualized volatility: {training['annualized_volatility']:.2%}
Maximum drawdown: {training['maximum_drawdown']:.2%}
Cumulative two-way turnover: {training_turnover:.2%}

Holdout period: {HOLDOUT_START_DATE} to {END_DATE}
Portfolio final value: ${holdout['final_value']:.4f}
Total return: {holdout['total_return']:.2%}
Annualized arithmetic return: {holdout['annualized_return']:.2%}
Annualized volatility: {holdout['annualized_volatility']:.2%}
Maximum drawdown: {holdout['maximum_drawdown']:.2%}
Cumulative two-way turnover: {holdout_turnover:.2%}

SPY buy-and-hold during holdout
Final value: ${spy['final_value']:.4f}
Total return: {spy['total_return']:.2%}
Annualized arithmetic return: {spy['annualized_return']:.2%}
Annualized volatility: {spy['annualized_volatility']:.2%}
Maximum drawdown: {spy['maximum_drawdown']:.2%}"""
    print(result)

    plt.figure(figsize=(10, 6))
    plt.plot(holdout_returns.index, holdout_values, label="Portfolio, after costs")
    plt.plot(holdout_returns.index, spy_values, label="SPY buy and hold")
    plt.title("Holdout Backtest: Portfolio vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        results_dir / "out_of_sample_backtest.png", dpi=300, bbox_inches="tight"
    )
    (results_dir / "out_of_sample_backtest.txt").write_text(
        result + "\n", encoding="utf-8"
    )
    plt.show()


if __name__ == "__main__":
    main()
