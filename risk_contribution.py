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

# Covariance matrix
cov_matrix = returns.cov()

# Portfolio variance
portfolio_variance = np.dot(
    weights.T,
    np.dot(cov_matrix, weights)
)

# Risk contribution of each asset
risk_contributions = (
    weights * np.dot(cov_matrix, weights)
) / portfolio_variance

print("RISK CONTRIBUTIONS")
print("-" * 30)

for ticker, contribution in zip(tickers, risk_contributions):
    print(f"{ticker}: {contribution:.2%}")

print("\nTotal:")

total = np.sum(risk_contributions)
print(f"{total:.2%}")