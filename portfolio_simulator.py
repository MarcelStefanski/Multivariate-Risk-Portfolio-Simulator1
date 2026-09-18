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

# Average daily returns
mean_returns = returns.mean()

# Covariance matrix
covariance_matrix = returns.cov()

# Portfolio expected daily return
portfolio_return = np.dot(weights, mean_returns)

# Portfolio variance
portfolio_variance = np.dot(
    weights.T,
    np.dot(covariance_matrix, weights)
)

# Portfolio volatility
portfolio_volatility = np.sqrt(portfolio_variance)

print("Portfolio Daily Return:")
print(portfolio_return)

print("\nPortfolio Daily Volatility:")
print(portfolio_volatility)

annual_return = portfolio_return * 252
annual_volatility = portfolio_volatility * np.sqrt(252)

print("\nAnnualized Return:")
print(annual_return)

print("\nAnnualized Volatility:")
print(annual_volatility)