import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

tickers = ["SPY", "QQQ", "GLD", "TLT"]

data = yf.download(
    tickers,
    start="2015-01-01",
    end="2025-01-01"
)

prices = data["Close"]

returns = prices.pct_change().dropna()

pca = PCA()

pca.fit(returns)

explained_variance = pca.explained_variance_ratio_

print("PCA Risk Decomposition")
print("-" * 30)

for i, variance in enumerate(explained_variance):
    print(
        f"Principal Component {i+1}: "
        f"{variance:.2%}"
    )

components = range(1,len(explained_variance) + 1)

plt.bar(components,explained_variance)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance")
plt.title("PCA Risk Decomposition")

plt.savefig(
    "results/pca_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()