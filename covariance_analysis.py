import yfinance as yf

tickers = ["SPY", "QQQ", "GLD", "TLT"]

data = yf.download(
    tickers,
    start="2015-01-01",
    end="2025-01-01"
)

prices = data["Close"]

returns = prices.pct_change().dropna()

covariance_matrix = returns.cov()

correlation_matrix = returns.corr()

print("Covariance Matrix:")
print(covariance_matrix)

print("\nCorrelation Matrix:")
print(correlation_matrix)