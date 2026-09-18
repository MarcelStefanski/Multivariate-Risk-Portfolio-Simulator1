import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

tickers = ["SPY", "QQQ", "GLD", "TLT"]

portfolio_returns = []
portfolio_risks = []
sharpe_ratios = []
all_weights = []

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

    sharpe_ratio = annual_return / annual_risk

    portfolio_returns.append(annual_return)
    portfolio_risks.append(annual_risk)
    sharpe_ratios.append(sharpe_ratio)
    all_weights.append(weights)

# Find the best portfolio
max_sharpe_index = np.argmax(sharpe_ratios)
best_return = portfolio_returns[max_sharpe_index]
best_risk = portfolio_risks[max_sharpe_index]
best_weights = all_weights[max_sharpe_index]

print("Maximum Sharpe Portfolio")
print("Return:", best_return)
print("Risk:", best_risk)
print("Weights:")
print(best_weights)

plt.scatter(portfolio_risks, portfolio_returns, alpha=0.5)

min_risk_index = np.argmin(portfolio_risks)

print("\nMinimum Risk Portfolio")
print("Return:", portfolio_returns[min_risk_index])
print("Risk:", portfolio_risks[min_risk_index])
print("Weights:")
print(all_weights[min_risk_index])

plt.scatter(
    best_risk,
    best_return,
    color="red",
    s=120,
    label="Max Sharpe"
)

plt.scatter(
    portfolio_risks[min_risk_index],
    portfolio_returns[min_risk_index],
    color="green",
    s=120,
    label="Min Risk"
)
plt.xlabel("Risk (Volatility)")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier")
plt.legend()

plt.savefig(
    "results/optimal_portfolios.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()