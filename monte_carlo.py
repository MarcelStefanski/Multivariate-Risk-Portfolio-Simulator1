import yfinance as yf
import numpy as np

tickers = ["SPY", "QQQ", "GLD", "TLT"]

data = yf.download(
    tickers,
    start="2015-01-01",
    end="2025-01-01"
)

prices = data["Close"]

returns = prices.pct_change().dropna()

# Portfolio weights
weights = np.array([0.4, 0.3, 0.15, 0.15])

# Historical means and covariance matrix
mean_returns = returns.mean()
covariance_matrix = returns.cov()

# Number of simulations
num_simulations = 10000

# Generate random return scenarios
simulated_returns = np.random.multivariate_normal(
    mean_returns,
    covariance_matrix,
    num_simulations
)

# Calculate portfolio return for each simulation
portfolio_returns = simulated_returns @ weights

# Basic statistics
average_return = np.mean(portfolio_returns)
minimum_return = np.min(portfolio_returns)
maximum_return = np.max(portfolio_returns)

print("Average Portfolio Return:")
print(average_return)

print("\nMinimum Simulated Return:")
print(minimum_return)

print("\nMaximum Simulated Return:")
print(maximum_return)

# Value-at-Risk
var_95 = np.percentile(portfolio_returns, 5)

print("\n95% Value at Risk:")
print(var_95)

# Expected Shortfall
worst_returns = portfolio_returns[portfolio_returns <= var_95]

expected_shortfall = np.mean(worst_returns)

print("\nExpected Shortfall:")
print(expected_shortfall)