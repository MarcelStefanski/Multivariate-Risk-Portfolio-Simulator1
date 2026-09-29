import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

tickers = ["SPY", "QQQ", "GLD", "TLT"]

data = yf.download(
    tickers,
    start="2015-01-01",
    end="2025-01-01"
)

prices = data["Close"].loc[:, tickers]

returns = prices.pct_change().dropna()

weights = np.array([0.4, 0.3, 0.15, 0.15])

# This applies the target weights daily, so it models frictionless daily
# rebalancing and excludes transaction costs.
portfolio_returns = returns @ weights

#If invested $1 at the start
portfolio_value = (1 + portfolio_returns).cumprod()

print("Final Portfolio Value:")
print(portfolio_value.iloc[-1])

plt.figure(figsize=(10, 6))

plt.plot(
    portfolio_value,
    label="Portfolio Value"
)

plt.title("Portfolio Backtest")
plt.xlabel("Date")
plt.ylabel("Portfolio Growth")

plt.legend()

plt.savefig(
    "results/backtest.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
