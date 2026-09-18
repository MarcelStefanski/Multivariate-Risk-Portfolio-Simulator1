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

weights = np.array([0.4, 0.3, 0.15, 0.15])

mean_returns = returns.mean()
covariance_matrix = returns.cov()

num_simulations = 10000

simulated_returns = np.random.multivariate_normal(
    mean_returns,
    covariance_matrix,
    num_simulations
)

portfolio_returns = simulated_returns @ weights

var_95 = np.percentile(portfolio_returns, 5)

plt.hist(portfolio_returns, bins=50)

plt.axvline(
    var_95,
    color="red",
    linestyle="--",
    label="95% VaR"
)

plt.title("Simulated Portfolio Returns")
plt.xlabel("Return")
plt.ylabel("Frequency")

plt.legend()
plt.savefig("results/monte_carlo_distribution.png")

plt.show()