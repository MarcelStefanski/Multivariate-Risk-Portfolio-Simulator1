import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

tickers = ["SPY", "QQQ", "GLD", "TLT"]

data = yf.download(
    tickers,
    start="2015-01-01",
    end="2025-01-01"
)

prices = data["Close"]

returns = prices.pct_change().dropna()

mean_returns = returns.mean()
cov_matrix = returns.cov()

portfolio_returns = []
portfolio_risks = []

for i in range(5000):

    weights = np.random.random(len(tickers))

    weights = weights / np.sum(weights)

    annual_return = np.sum(weights * mean_returns) * 252

    annual_risk = np.sqrt(
        np.dot(
            weights.T,
            np.dot(cov_matrix * 252, weights)
        )
    )

    portfolio_returns.append(annual_return)
    portfolio_risks.append(annual_risk)

plt.scatter(portfolio_risks, portfolio_returns, alpha=0.5)

plt.xlabel("Risk (Volatility)")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier")

plt.savefig(
    "results/efficient_frontier.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()