from pathlib import Path

import numpy as np
import yfinance as yf


TICKERS = ["SPY", "QQQ", "GLD", "TLT"]
WEIGHTS = np.array([0.4, 0.3, 0.15, 0.15])
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
TRADING_DAYS = 252
ANNUAL_TARGET_RETURN = 0.0

data = yf.download(TICKERS, start=START_DATE, end=END_DATE)
prices = data["Close"].loc[:, TICKERS]
returns = prices.pct_change().dropna()
portfolio_returns = returns @ WEIGHTS

daily_target_return = ANNUAL_TARGET_RETURN / TRADING_DAYS
excess_returns = portfolio_returns - daily_target_return
downside_deviation = np.sqrt(
    np.mean(np.minimum(excess_returns, 0) ** 2)
)

annual_excess_return = excess_returns.mean() * TRADING_DAYS
annual_downside_deviation = downside_deviation * np.sqrt(TRADING_DAYS)
sortino_ratio = (
    annual_excess_return / annual_downside_deviation
    if annual_downside_deviation > 0
    else np.nan
)

result = "\n".join(
    [
        "SORTINO RATIO",
        "-" * 30,
        f"Annual Target Return: {ANNUAL_TARGET_RETURN:.2%}",
        f"Annualized Arithmetic Return: {portfolio_returns.mean() * TRADING_DAYS:.2%}",
        f"Annualized Downside Deviation: {annual_downside_deviation:.2%}",
        f"Sortino Ratio: {sortino_ratio:.2f}",
    ]
)

print(result)

output_path = Path("results/sortino_ratio.txt")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(result + "\n", encoding="utf-8")
