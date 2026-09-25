import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

tickers = ["SPY", "QQQ", "GLD", "TLT"]

data = yf.download(
    tickers,
    start="2015-01-01",
    end="2025-01-01"
)

prices = data["Close"]

returns = prices.pct_change().dropna()

correlation_matrix = returns.corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "results/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
