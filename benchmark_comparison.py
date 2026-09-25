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

# Portfolio

weights = np.array([0.4, 0.3, 0.15, 0.15])

portfolio_returns = returns @ weights

# Benchmark

spy_returns = returns["SPY"]

# Cumulative growth

portfolio_growth = (1 + portfolio_returns).cumprod()

spy_growth = (1 + spy_returns).cumprod()

# Metrics

portfolio_return = portfolio_returns.mean() * 252
spy_return = spy_returns.mean() * 252

portfolio_volatility = (
    portfolio_returns.std() * np.sqrt(252)
)

spy_volatility = (
    spy_returns.std() * np.sqrt(252)
)

print("PORTFOLIO")
print(f"Annual Return: {portfolio_return:.2%}")
print(f"Annual Volatility: {portfolio_volatility:.2%}")

print("\nSPY")
print(f"Annual Return: {spy_return:.2%}")
print(f"Annual Volatility: {spy_volatility:.2%}")

# Plot

plt.figure(figsize=(10, 6))

plt.plot(
    portfolio_growth,
    label="Portfolio"
)

plt.plot(
    spy_growth,
    label="SPY"
)

plt.title("Portfolio vs SPY Benchmark")

plt.xlabel("Date")

plt.ylabel("Growth of $1")

plt.legend()

plt.savefig(
    "results/benchmark_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
