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

weights = np.array([0.4, 0.3, 0.15, 0.15])

portfolio_returns = returns @ weights

annual_return = portfolio_returns.mean() * 252

annual_volatility = portfolio_returns.std() * np.sqrt(252)

sharpe_ratio = annual_return / annual_volatility

print("PERFORMANCE METRICS")
print("-" * 30)

print(f"Annual Return: {annual_return:.2%}")
print(f"Annual Volatility: {annual_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

portfolio_value = (1 + portfolio_returns).cumprod()

running_max = portfolio_value.cummax()

drawdown = (portfolio_value - running_max) / running_max

max_drawdown = drawdown.min()

print(f"Maximum Drawdown: {max_drawdown:.2%}")